"""Decoy generation with a non-protein residue in the pose.

Before this, pose.sequence() handed the shuffle a ligand's one-letter code -- 'Z' for
ATP -- the permutation moved it, and make_decoy died with a bare IndexError. The crash was
the lucky outcome: the same mechanism would otherwise try to mutate the ligand into an
amino acid and some amino acid into ATP.

The load-bearing tests are the two that assert the ligand did not MOVE, and the one that
asserts a ligand-free pose is unchanged. Everything else is scaffolding.
"""

import random

import numpy as np
import pytest

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.decoys import (backbone_coords, frozen_coords, is_designable,  # noqa: E402
                           is_frozen, make_decoy, native_reference,
                           shuffle_sequence, shuffled_target_sequence)
from frustx.energies import init_rosetta, make_score_function  # noqa: E402


def _helix(seq="ACDEFGHIKL"):
    pose = pyrosetta.pose_from_sequence(seq)
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    return pose


@pytest.fixture(scope="module")
def sf():
    init_rosetta()
    return make_score_function(remove_fa_rep=False)


@pytest.fixture(scope="module")
def protein_only():
    init_rosetta()
    return _helix()


@pytest.fixture(scope="module")
def with_ligand():
    """ATP appended by jump. Module-scoped, so every test that uses it must treat it as
    READ-ONLY -- make_decoy and native_reference both clone, so they are safe."""
    init_rosetta()
    pose = _helix()
    rts = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance() \
        .residue_type_set("fa_standard")
    pose.append_residue_by_jump(
        pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(
            rts.name_map("ATP")), 5)
    return pose


# --- the no-regression guarantee ---------------------------------------------------

def test_a_ligand_free_pose_shuffles_exactly_as_before(protein_only):
    """The whole change must be invisible without a ligand. Same seed, same string."""
    for seed in range(5):
        assert (shuffled_target_sequence(protein_only, random.Random(seed))
                == shuffle_sequence(protein_only.sequence(), random.Random(seed)))


def test_backbone_coords_unchanged_for_a_protein_only_pose(protein_only):
    n = protein_only.total_residue()
    assert backbone_coords(protein_only).shape == (3 * n, 3)


# --- the shuffle ------------------------------------------------------------------

def test_the_ligand_letter_is_not_permuted(with_ligand):
    """ATP's one-letter code is 'Z'. If it moves, the design tries to mutate the ligand
    into an amino acid."""
    assert with_ligand.sequence().endswith("Z")
    for seed in range(5):
        got = shuffled_target_sequence(with_ligand, random.Random(seed))
        assert len(got) == with_ligand.total_residue()
        assert got.endswith("Z"), f"ligand letter moved: {got}"
        assert got.count("Z") == 1


def test_composition_of_the_protein_part_is_preserved(with_ligand):
    """The paper requires a permutation, not random mutation -- composition must be
    exact over the positions whose identity can change."""
    native = with_ligand.sequence()[:-1]
    for seed in range(5):
        got = shuffled_target_sequence(with_ligand, random.Random(seed))[:-1]
        assert sorted(got) == sorted(native)


def test_the_shuffle_actually_shuffles(with_ligand):
    """Guard against the exclusion accidentally freezing everything."""
    got = {shuffled_target_sequence(with_ligand, random.Random(s)) for s in range(8)}
    assert len(got) > 1
    assert any(g[:-1] != with_ligand.sequence()[:-1] for g in got)


# --- the freeze -------------------------------------------------------------------

def test_the_ligand_does_not_move_in_a_decoy(with_ligand, sf):
    """ALL atoms, not just heavy: a rotated hydroxyl changes the energies too."""
    before = frozen_coords(with_ligand)
    assert before.shape[0] > 0, "fixture must actually contain a frozen residue"
    for seed in (1, 2, 3):
        decoy = make_decoy(with_ligand, sf, seed=seed, protocol="min")
        assert np.array_equal(frozen_coords(decoy), before), \
            f"ligand moved in decoy seed={seed}"


def test_the_ligand_does_not_move_in_the_native_reference(with_ligand, sf):
    """Freezing decoys but not the native is the silent version of this bug: E0 shifts
    against an ensemble prepared differently and nothing raises. repack() and
    apply_sequence() are separate code paths, so this needs its own test."""
    before = frozen_coords(with_ligand)
    nat = native_reference(with_ligand, sf, protocol="min")
    assert np.array_equal(frozen_coords(nat), before)


def test_freeze_ligand_false_actually_lets_it_move(with_ligand, sf):
    """Proves the freeze is doing work rather than the ligand being immobile anyway. If
    this ever stops holding, every test above is passing for the wrong reason."""
    before = frozen_coords(with_ligand)
    loose = make_decoy(with_ligand, sf, seed=1, protocol="min", freeze_ligand=False)
    assert not np.array_equal(frozen_coords(loose), before), \
        "with freeze_ligand=False the ligand should be free to move"


def test_the_backbone_is_still_fixed_with_a_ligand_present(with_ligand, sf):
    before = backbone_coords(with_ligand)
    decoy = make_decoy(with_ligand, sf, seed=4, protocol="min")
    assert np.array_equal(backbone_coords(decoy), before)


def test_backbone_coords_skips_the_ligand(with_ligand):
    """ATP has no N/CA/C. The count must be protein residues only."""
    n_protein = sum(1 for i in range(1, with_ligand.total_residue() + 1)
                    if with_ligand.residue(i).is_protein())
    assert backbone_coords(with_ligand).shape == (3 * n_protein, 3)
    assert n_protein == with_ligand.total_residue() - 1


# --- the predicates ---------------------------------------------------------------

def test_the_two_predicates_are_not_the_same_set(with_ligand):
    """is_designable and is_frozen split the residues three ways, and conflating them is
    how a modified residue gets silently mutated."""
    lig = with_ligand.residue(with_ligand.total_residue())
    ala = with_ligand.residue(1)
    assert is_designable(ala) and not is_frozen(ala)
    assert not is_designable(lig) and is_frozen(lig)


def test_water_is_frozen_even_though_rosetta_calls_it_a_ligand():
    """is_ligand() returns True for HOH, so it cannot be the discriminator. is_protein()
    is, which is what residue_kinds_from_pose chose for the same reason."""
    init_rosetta()
    rts = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance() \
        .residue_type_set("fa_standard")
    water = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(
        rts.name_map("HOH"))
    assert water.is_ligand(), "fixture assumption: Rosetta calls water a ligand"
    assert is_frozen(water)
    assert not is_designable(water)


# --- end to end --------------------------------------------------------------------

def test_a_full_frustration_run_on_a_complex(with_ligand, sf):
    """Everything joined up: kind-aware contact map, frozen ligand, decoy ensemble,
    Eq. 1, and the output table.

    The assertion that matters is that protein-LIGAND contacts appear with FINITE
    indices. Every earlier failure mode in this area -- the NaN CA coordinate, the
    shuffled ligand letter -- ends with those contacts either absent or NaN, so a run
    that merely completes proves nothing.
    """
    from frustx.energies import make_score_function as _sf
    from frustx.frustration import compute_frustration, residue_kinds_from_pose
    from frustx.output import contact_table

    result = compute_frustration(
        with_ligand, sf_pack=sf, sf_measure=_sf(remove_fa_rep=True),
        n_decoys=6, seed=3, protocol="min", packing_seed=11, n_jobs=1, min_seq_sep=2)

    kinds = residue_kinds_from_pose(with_ligand)
    lig = len(kinds) - 1
    involving_ligand = [(i, j) for i, j in result.contacts.tolist()
                        if lig in (i, j)]
    assert involving_ligand, "no protein-ligand contacts were found at all"

    idx = np.array([result.index[i, j] for i, j in involving_ligand])
    assert np.isfinite(idx).all(), f"ligand contacts produced non-finite indices: {idx}"

    df = contact_table(result)
    assert len(df) == len(result.contacts)
    assert df["frustration_index"].notna().all()


def test_the_ligand_never_contacts_itself_or_another_frozen_residue(with_ligand, sf):
    """include_ligand_ligand defaults False. With one ligand this is trivially true, but
    it pins the default so a change has to be deliberate."""
    from frustx.frustration import compute_frustration, residue_kinds_from_pose
    from frustx.energies import make_score_function as _sf

    result = compute_frustration(
        with_ligand, sf_pack=sf, sf_measure=_sf(remove_fa_rep=True),
        n_decoys=4, seed=1, protocol="min", packing_seed=2, n_jobs=1)
    kinds = residue_kinds_from_pose(with_ligand)
    for i, j in result.contacts.tolist():
        assert kinds[i] or kinds[j], f"pair ({i},{j}) has no protein endpoint"
