"""Tests for arbitrary novel ligand parametrisation.

Every fixture in tests/fixtures/ is hand-built and under 1.2 KB -- code, not data, in the
sense CLAUDE.md allows. The real EGFR complexes are 300 KB each and are deliberately NOT
here; they are exercised by scripts/egfr_ligand_check.py, whose output lands in results/.

The molecules were chosen because they can FAIL, not because they are convenient:
methanol in three protonation states that all parse without complaint, a bromide whose
virtual atoms outnumber its real one, and two files whose names collide with Rosetta's
curated set in the two different ways Rosetta indexes names.
"""

import numpy as np
import pytest

from frustx.ligand_params import (
    LigandParametrisationError,
    build_residue_type_set,
    check_chemistry,
    make_spec,
    missing_hydrogens,
    real_heavy_atom_count,
    single_mutable_type,
    ligands_setting,
)

pyrosetta = pytest.importorskip("pyrosetta")

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def rosetta():
    from frustx.energies import init_rosetta

    init_rosetta()


def _mutable(name):
    return single_mutable_type(FIXTURES / name)


# ---------------------------------------------------------------- the hydrogen gate


def test_a_ligand_with_no_hydrogens_is_refused():
    with pytest.raises(LigandParametrisationError, match="below valence"):
        make_and_register("methanol_noH.sdf")


def test_a_ligand_with_only_polar_hydrogens_is_refused():
    """The test a global "are there hydrogens" count would fail.

    Polar-hydrogens-only is what AutoDock and PDBQT-derived pipelines emit, and it is the
    realistic corruption -- far more common than a fully stripped file. methanol_polarH
    has one hydrogen, so any `n_hydrogens == 0` gate passes it through while its carbon is
    typed CH1 instead of CH3.
    """
    with pytest.raises(LigandParametrisationError, match="below valence"):
        make_and_register("methanol_polarH.sdf")


def test_the_hydrogen_gate_is_not_vacuous_because_the_typing_really_does_change():
    """Prove the refusals above are protecting against something.

    A gate that refuses files is only worth having if the files it refuses would have been
    scored WRONGLY. Same molecule, three protonation states, all of which Rosetta parses
    without raising -- and the carbon type and both partial charges move:

        methanol.sdf         C1 CH3 -0.234   O1 OH  -0.624
        methanol_polarH.sdf  C1 CH1 +0.017   O1 OH  -0.553
        methanol_noH.sdf     C1 CH1 +0.335   O1 OOC -0.335
    """
    def heavy(mutable):
        out = []
        for vd in mutable.all_atoms():
            atom = mutable.atom(vd)
            if not atom.is_hydrogen() and not atom.is_virtual():
                out.append((atom.name().strip(), mutable.atom_type(vd).name().strip(),
                            round(atom.charge(), 3)))
        return out

    good = heavy(_mutable("methanol.sdf"))
    polar = heavy(_mutable("methanol_polarH.sdf"))
    stripped = heavy(_mutable("methanol_noH.sdf"))

    assert [a[0] for a in good] == [a[0] for a in polar] == [a[0] for a in stripped]
    # The carbon is mistyped in both corrupt files ...
    assert good[0][1] == "CH3" and polar[0][1] == "CH1" and stripped[0][1] == "CH1"
    # ... and every charge moves, on both heavy atoms, in both corrupt files.
    assert good[0][2] != polar[0][2] != stripped[0][2]
    assert good[1][2] != polar[1][2] != stripped[1][2]


def test_a_correctly_protonated_ligand_passes_the_gate():
    assert missing_hydrogens(_mutable("methanol.sdf")) == []


def test_a_halide_ion_is_not_flagged_for_missing_hydrogens():
    """A bromide has no hydrogens and needs none. The gate must not refuse it.

    The valence caps cover C/N/O/S only, precisely so that ions, halogens, metals and
    phosphorus -- which have no single neutral valence worth assuming -- are left alone.
    """
    assert missing_hydrogens(_mutable("bromide.sdf")) == []


# ---------------------------------------------------------------- virtual atoms


def test_a_bromide_is_one_atom_however_many_rosetta_counts():
    """Assert against chemistry, not against the API under suspicion.

    `nheavyatoms()` is a prefix count that INCLUDES the virtual atoms Rosetta adds to give
    a small molecule a coordinate frame. For a bromide it says 3. A bromide is one atom.
    An earlier version of this project shipped a bug by comparing an extracted count
    against `nheavyatoms()` itself -- a circular test that passed while the bug was live.
    """
    mutable = _mutable("bromide.sdf")
    n_virtual = sum(mutable.atom(vd).is_virtual() for vd in mutable.all_atoms())

    assert real_heavy_atom_count(mutable) == 1
    assert n_virtual == 2, "fixture no longer exercises the virtual-atom path"
    assert mutable.natoms() == 3


# ---------------------------------------------------------------- file-level refusals


def test_a_multi_molecule_sdf_is_refused_rather_than_silently_taking_the_first():
    """The normal output of a docking run is one entry per pose. Guessing is not allowed."""
    with pytest.raises(LigandParametrisationError, match="holds 2 molecules"):
        make_spec(FIXTURES / "multi.sdf")
        single_mutable_type(FIXTURES / "multi.sdf")


def test_mol2_is_refused_by_extension_not_by_trying_it():
    """Rosetta's reader returns an EMPTY result for mol2 and raises nothing at all.

    So "try it and catch the error" does not work here -- there is no error. Refusing by
    extension is what turns a later IndexError about a vector1 into a message that names
    the actual problem.
    """
    with pytest.raises(LigandParametrisationError, match="mol2"):
        make_spec(FIXTURES / "unsupported.mol2")


# ---------------------------------------------------------------- names


def test_name_and_name3_are_kept_separate_because_rosetta_indexes_them_separately():
    """The bug that a name3-keyed implementation has, on every real docking output.

    An SDF title longer than three characters gets a name3 that is only a truncation, and
    the truncation is not a lookup key. `name_map('lig')` raises; `name_map('ligand_1')`
    works.
    """
    mutable = _mutable("longtitle.sdf")
    assert mutable.name() == "ligand_1"
    assert mutable.name3() == "lig"

    pose_set, report = build_residue_type_set([make_spec(FIXTURES / "longtitle.sdf")])
    assert pose_set.has_name("ligand_1")
    assert not pose_set.has_name("lig")
    assert report[0]["name"] == "ligand_1" and report[0]["name3"] == "lig"
    assert pose_set.name_map("ligand_1").natoms() == 6


def test_a_name_collision_with_a_curated_type_is_refused():
    """ATP collides on `name`, which a naive guard would catch."""
    with pytest.raises(LigandParametrisationError, match="already defined"):
        build_residue_type_set([make_spec(FIXTURES / "atpname.sdf")])


def test_a_name3_only_collision_with_a_curated_type_is_also_refused():
    """SAH collides on name3 ONLY, which a name-keyed guard misses.

    This is the test that fails against the obvious implementation. Registering over SAH
    is accepted by Rosetta without complaint and the new type wins the lookup, so a user
    who fetched SAH_ideal.sdf from RCSB would silently score a hand-typed molecule in
    place of Rosetta's fitted one.
    """
    global_set = (
        pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
        .residue_type_set("fa_standard")
    )
    assert not global_set.has_name("SAH"), "fixture no longer exercises the name3-only path"
    assert global_set.has_name3("SAH")

    with pytest.raises(LigandParametrisationError, match="already defined"):
        build_residue_type_set([make_spec(FIXTURES / "sahname.sdf")])


def test_two_ligands_sharing_a_name_are_refused_as_our_error_not_rosettas():
    """Vina, Glide and Boltz name every hit UNL or LIG, so this is the normal case.

    Rosetta does refuse the duplicate, but as a C++ exit surfacing as a bare RuntimeError,
    which a caller catching LigandParametrisationError would not handle.
    """
    spec = make_spec(FIXTURES / "methanol.sdf")
    with pytest.raises(LigandParametrisationError, match="both declare"):
        build_residue_type_set([spec, spec])


# ---------------------------------------------------------------- chemistry warnings


def test_a_saturated_ligand_produces_no_chemistry_warnings():
    assert check_chemistry(_mutable("methanol.sdf")) == []


# ---------------------------------------------------------------- provenance


def test_the_provenance_value_is_content_keyed_and_order_independent():
    """`--ligand-sdf hit.sdf` is a different molecule tomorrow if the screen was re-run.

    Keyed on content so a re-docked file invalidates a cached ensemble; sorted so the
    order the flags were typed in does not.
    """
    a = make_spec(FIXTURES / "methanol.sdf")
    b = make_spec(FIXTURES / "longtitle.sdf")
    _, ab = build_residue_type_set([a, b])
    _, ba = build_residue_type_set([b, a])
    assert ligands_setting(ab) == ligands_setting(ba)

    keys = {e["key"] for e in ab}
    assert len(keys) == 2, "two different files must not share a content key"


# ---------------------------------------------------------------- pose integration


@pytest.fixture(scope="module")
def peptide_with_ligand(tmp_path_factory):
    """A poly-glycine peptide with methanol appended, built entirely at test time.

    Generated rather than stored: a PDB is data, and CLAUDE.md keeps data out of git.
    """
    from frustx.ligand_params import append_ligand

    pose_set, _ = build_residue_type_set([make_spec(FIXTURES / "methanol.sdf")])
    pose = pyrosetta.pose_from_sequence("GGGGGGGG")
    pose.conformation().reset_residue_type_set_for_conf(pose_set)
    index = append_ligand(pose, pose_set.name_map("MOH"))
    return pose, index


def test_an_appended_ligand_keeps_its_files_coordinates(peptide_with_ligand):
    """Residue(rt, True) is built from the type's own conformer, which for an SDF-derived
    type is the SDF coordinate block, atom for atom. No coordinate-setting loop needed."""
    pose, index = peptide_with_ligand
    residue = pose.residue(index)
    assert np.allclose(list(residue.xyz(1)), [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(list(residue.xyz(2)), [1.43, 0.0, 0.0], atol=1e-6)


def test_an_appended_ligand_gets_a_real_pdb_identity(peptide_with_ligand):
    """append_residue_by_jump leaves the record at Rosetta's unknown-chain sentinel.

    Measured without the fix: `chain='^' num=0`, while dump_pdb invents a DIFFERENT
    identity at write time. contacts.csv reads pdb_info and frustration.pdb is written by
    dump_pdb, so the two disagree about which residue is which -- and with two ligands the
    ('^', 0) key is duplicated and write_bfactor_pdb dies on a non-unique index, after the
    whole decoy ensemble has been computed.
    """
    pose, index = peptide_with_ligand
    info = pose.pdb_info()
    assert info.chain(index) != "^"
    assert info.number(index) != 0

    keys = {(info.chain(i), info.number(i)) for i in range(1, pose.total_residue() + 1)}
    assert len(keys) == pose.total_residue(), "PDB keys must stay unique"


def test_the_ligand_is_a_ligand_and_is_not_designable(peptide_with_ligand):
    """The two predicates the whole spectator treatment rests on, on a RUNTIME-built type.

    Everything in decoys.py was verified against ATP and ZN, which come from the curated
    fa_standard set. A type built at runtime from an SDF could differ, and if
    is_canonical_aa() or is_protein() answered differently the ligand would be mutated
    like an amino acid with no error anywhere.
    """
    from frustx import decoys

    pose, index = peptide_with_ligand
    residue = pose.residue(index)
    assert residue.is_ligand()
    assert not residue.is_protein()
    assert not decoys.is_designable(residue)
    assert decoys.is_frozen(residue)
    assert decoys.is_designable(pose.residue(1)), "control: glycine IS designable"


def test_the_packer_task_excludes_the_ligand_and_the_operation_is_not_a_no_op(
    peptide_with_ligand,
):
    """The non-vacuous control for the freeze on a small fixture.

    The obvious control -- freeze_ligand=False must move the ligand -- is DEGENERATE here:
    methanol has nchi=0, so it cannot move whatever the setting. So this test checks the
    mechanism instead, and pins it with a plain RestrictToRepacking that DOES leave the
    ligand packable. (The displacement control is exercised for real in
    scripts/egfr_ligand_check.py, on lapatinib, which has 11 chi angles and moves 2.29 A
    when unfrozen.)
    """
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from frustx import decoys

    pose, index = peptide_with_ligand
    task = decoys._relax_task_factory(pose).create_task_and_apply_taskoperations(pose)
    assert task.being_packed(1), "control: a protein residue is packed"
    assert not task.being_packed(index)
    assert not task.being_designed(index)

    plain = TaskFactory()
    plain.push_back(operation.RestrictToRepacking())
    bare = plain.create_task_and_apply_taskoperations(pose)
    assert bare.being_packed(index), "RestrictToRepacking alone leaves a ligand packable"


def test_a_runtime_ligand_type_contributes_exactly_its_real_heavy_atoms(
    peptide_with_ligand,
):
    """The virtual-atom filter is load-bearing on the SDF path too, not only for ions."""
    from frustx.frustration import heavy_atom_coords_from_pose

    pose, index = peptide_with_ligand
    coords, offsets = heavy_atom_coords_from_pose(pose)
    n_rows = offsets[index] - offsets[index - 1]
    assert n_rows == 2, "methanol has exactly two heavy atoms"


def make_and_register(name):
    """Register a fixture, so the gates that run at registration time actually run."""
    return build_residue_type_set([make_spec(FIXTURES / name)])


def _ligand_contacts(pose, index):
    """Protein-ligand contacts for one pose, through the real contact machinery."""
    from frustx.contacts import (min_heavy_distances, pairwise_distances,
                                 select_pairs_by_kind)
    from frustx.frustration import (contact_coords_from_pose, heavy_atom_coords_from_pose,
                                    residue_kinds_from_pose, residues_from_pose)

    coords, offsets = heavy_atom_coords_from_pose(pose)
    kinds = residue_kinds_from_pose(pose)
    pairs = select_pairs_by_kind(
        residues_from_pose(pose),
        pairwise_distances(contact_coords_from_pose(pose, "CA", missing="nan")),
        min_heavy_distances(coords, offsets), kinds,
        cutoff=10.0, ligand_cutoff=6.0, min_seq_sep=2)
    return sum(1 for i, j in pairs if index - 1 in (i, j))


def test_a_ligand_placed_from_the_wrong_frame_silently_makes_no_contacts(
    peptide_with_ligand,
):
    """The silent failure --ligand-placed invites, and the reason the CLI warns.

    An RCSB `_ideal.sdf` is an idealised conformer generated near the origin, not the
    crystallographic or docked pose. Appending one puts the ligand nowhere near the
    protein -- and nothing raises. The run completes and reports a perfectly clean
    protein-only answer, which is exactly the failure class the kind-aware contact rule
    was written to stop.

    Two-sided on purpose. In place the ligand contacts the peptide; translated 100 A it
    contacts nothing, with no error anywhere along the way. Without the first assertion
    the second would also pass if the contact machinery simply never saw the ligand.
    """
    pose, index = peptide_with_ligand
    assert _ligand_contacts(pose, index) > 0, "control: in place, the ligand IS in contact"

    moved = pose.clone()
    residue = moved.residue(index)
    for k in range(1, residue.natoms() + 1):
        xyz = residue.xyz(k)
        moved.set_xyz(pyrosetta.rosetta.core.id.AtomID(k, index),
                      pyrosetta.rosetta.numeric.xyzVector_double_t(
                          xyz.x + 100.0, xyz.y, xyz.z))

    assert _ligand_contacts(moved, index) == 0
