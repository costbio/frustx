"""Tests for the geometry superset: chunked distance kernels and ContactGeometry.

Coordinates are hand-built and mostly on a line or a grid, so the expected distances are
checkable by mental arithmetic rather than by trusting the code under test.

The chunking tests matter more than they look. This repo holds serial and parallel runs
to BIT-identical agreement, so a kernel that is "the same to 1e-15" is not the same. Each
kernel is therefore checked against the naive form it replaces, with array_equal rather
than allclose.
"""

import numpy as np
import pytest

from frustx.contacts import (ContactGeometry, Residue, contact_pairs,
                             min_heavy_distances, pairwise_distances, select_pairs)


def _res(i, chain="A"):
    return Residue(index=i, chain=chain, resseq=i + 1, icode=" ", resname="ALA")


def _line(xs):
    return np.array([[x, 0.0, 0.0] for x in xs])


# --- pairwise_distances ------------------------------------------------------------

def test_pairwise_distances_is_bit_identical_to_the_naive_broadcast():
    """The kernel exists to bound memory, not to change arithmetic. Every entry is an
    independent reduction over three terms, so blocking must not move a single bit."""
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(300, 3)) * 20.0
    delta = coords[:, None, :] - coords[None, :, :]
    naive = np.sqrt((delta ** 2).sum(axis=-1))
    assert np.array_equal(pairwise_distances(coords), naive)


@pytest.mark.parametrize("block", [1, 2, 7, 64, 1000])
def test_pairwise_distances_does_not_depend_on_block_size(block):
    coords = _line([0.0, 5.0, 11.0, 14.0, 20.0])
    assert np.array_equal(pairwise_distances(coords, block=block),
                          pairwise_distances(coords, block=4096))


def test_pairwise_distances_on_a_line_is_arithmetic():
    d = pairwise_distances(_line([0.0, 5.0, 11.0]))
    assert d[0, 1] == 5.0 and d[0, 2] == 11.0 and d[1, 2] == 6.0
    assert np.array_equal(np.diag(d), np.zeros(3))


# --- min_heavy_distances -----------------------------------------------------------

def _brute_min_heavy(coords, offsets):
    n = len(offsets) - 1
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a, b = coords[offsets[i]:offsets[i + 1]], coords[offsets[j]:offsets[j + 1]]
            out[i, j] = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min()
    return out


def test_min_heavy_picks_the_closest_atom_pair_not_the_centroid():
    """The whole point of the definition. Residue 0 has one atom at x=0; residue 1 has
    atoms at x=10 and x=3. The minimum is 3, not the distance between any centre."""
    coords = np.array([[0.0, 0, 0], [10.0, 0, 0], [3.0, 0, 0]])
    offsets = np.array([0, 1, 3])
    d = min_heavy_distances(coords, offsets)
    assert d[0, 1] == 3.0
    assert d[0, 0] == 0.0


def test_min_heavy_matches_brute_force_bitwise():
    """Ragged residues of very different sizes, which is the ligand case -- glycine has
    4 heavy atoms, ATP has 33."""
    rng = np.random.default_rng(1)
    sizes = [4, 9, 33, 1, 7]
    coords = rng.normal(size=(sum(sizes), 3)) * 10.0
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    assert np.array_equal(min_heavy_distances(coords, offsets),
                          _brute_min_heavy(coords, offsets))


@pytest.mark.parametrize("block", [1, 2, 3, 1000])
def test_min_heavy_does_not_depend_on_block_size(block):
    rng = np.random.default_rng(2)
    sizes = [3, 8, 2, 11]
    coords = rng.normal(size=(sum(sizes), 3)) * 5.0
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    assert np.array_equal(min_heavy_distances(coords, offsets, block=block),
                          _brute_min_heavy(coords, offsets))


def test_min_heavy_is_symmetric():
    rng = np.random.default_rng(3)
    sizes = [5, 2, 9]
    coords = rng.normal(size=(sum(sizes), 3))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    d = min_heavy_distances(coords, offsets)
    assert np.array_equal(d, d.T)


def test_a_residue_with_no_atoms_raises():
    """np.minimum.reduceat does NOT return the identity for an empty segment -- it
    returns the element at that index. An atomless residue would silently report some
    other residue's distance, so it has to be refused."""
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="no heavy atoms"):
        min_heavy_distances(coords, np.array([0, 1, 1, 2]))


def test_offsets_that_do_not_span_the_coordinates_raise():
    coords = np.zeros((5, 3))
    with pytest.raises(ValueError, match="must span"):
        min_heavy_distances(coords, np.array([0, 2, 3]))


# --- select_pairs ------------------------------------------------------------------

def test_select_pairs_agrees_with_contact_pairs():
    """The refactor must not have changed the contact set."""
    residues = [_res(i) for i in range(4)]
    coords = _line([0.0, 5.0, 11.0, 14.0])
    direct = contact_pairs(residues, coords, cutoff=10.0, min_seq_sep=2)
    viaman = select_pairs(residues, pairwise_distances(coords), cutoff=10.0, min_seq_sep=2)
    assert np.array_equal(direct, viaman)


def test_select_pairs_refuses_nan_but_allows_inf():
    """inf is how a per-kind rule marks a pair it does not apply to, and it correctly
    compares False. NaN also compares False but by accident, silently, so it raises."""
    residues = [_res(0), _res(1)]
    d = np.array([[0.0, np.inf], [np.inf, 0.0]])
    assert select_pairs(residues, d, cutoff=10.0).shape == (0, 2)

    d_nan = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="NaN"):
        select_pairs(residues, d_nan, cutoff=10.0)


def test_select_pairs_checks_the_matrix_shape():
    with pytest.raises(ValueError, match=r"expected \(3, 3\)"):
        select_pairs([_res(i) for i in range(3)], np.zeros((2, 2)))


# --- ContactGeometry ---------------------------------------------------------------

def test_geometry_accepts_coordinates_or_precomputed_distances():
    residues = [_res(i) for i in range(4)]
    coords = _line([0.0, 5.0, 11.0, 14.0])
    g = ContactGeometry(residues, {"CA": coords,
                                   "heavy": pairwise_distances(coords)})
    assert g.definitions == ("CA", "heavy")
    assert np.array_equal(g.distances("CA"), g.distances("heavy"))


def test_geometry_selects_many_cutoffs_from_one_distance_pass():
    """The motivating waste: contact_definition.py sweeps 10.0/9.5/9.0 and pays for a
    full distance pass each time."""
    residues = [_res(i) for i in range(4)]
    coords = _line([0.0, 5.0, 11.0, 14.0])
    g = ContactGeometry(residues, {"CA": coords})
    for cutoff in (10.0, 9.5, 6.0, 5.0):
        assert np.array_equal(g.pairs("CA", cutoff=cutoff, min_seq_sep=1),
                              contact_pairs(residues, coords, cutoff=cutoff))


def test_geometry_rejects_a_wrongly_shaped_column():
    with pytest.raises(ValueError, match="expected"):
        ContactGeometry([_res(i) for i in range(3)], {"CA": np.zeros((5, 3))})


def test_geometry_names_a_missing_definition():
    g = ContactGeometry([_res(i) for i in range(2)], {"CA": _line([0.0, 1.0])})
    with pytest.raises(KeyError, match="heavy"):
        g.distances("heavy")


# --- pose side ---------------------------------------------------------------------

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.energies import init_rosetta  # noqa: E402
from frustx.frustration import (heavy_atom_coords_from_pose,  # noqa: E402
                                residue_kinds_from_pose)


@pytest.fixture(scope="module")
def peptide_with_atp():
    """ATP is used as the test ligand because Rosetta SHIPS its params -- no vendoring,
    no -extra_res_fa, nothing to install. 33 heavy atoms and no CA, which is exactly the
    shape the protein-only code paths cannot handle."""
    init_rosetta()
    pose = pyrosetta.pose_from_sequence("ACDEFGHIKL")
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    rts = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance() \
        .residue_type_set("fa_standard")
    lig = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(
        rts.name_map("ATP"))
    pose.append_residue_by_jump(lig, 5)
    return pose


def test_heavy_atom_counts_are_per_residue_and_ragged(peptide_with_atp):
    xyz, off = heavy_atom_coords_from_pose(peptide_with_atp)
    counts = np.diff(off).tolist()
    assert len(counts) == peptide_with_atp.total_residue()
    assert counts[5] == 4, "glycine should have 4 heavy atoms (N, CA, C, O)"
    # 31, not the 33 that nheavyatoms() reports: two of ATP's are VIRTUAL. They sit
    # 0.00 A from real atoms so they cannot move a minimum distance, which is why an
    # ATP-only test could not see the problem that a metal ion makes obvious.
    assert counts[-1] == 31, "ATP has 33 heavy-indexed atoms, 2 of which are virtual"
    assert off[-1] == len(xyz)


def test_no_hydrogens_are_included(peptide_with_atp):
    """Rosetta orders heavy atoms first, so nheavyatoms() is a prefix count."""
    pose = peptide_with_atp
    xyz, _ = heavy_atom_coords_from_pose(pose)
    total_atoms = sum(pose.residue(i).natoms()
                      for i in range(1, pose.total_residue() + 1))
    assert len(xyz) < total_atoms, "hydrogens were not excluded"


@pytest.fixture(scope="module")
def peptide_with_zinc():
    """ZN is the adversarial case, not ATP. Rosetta's metal params carry a shell of
    VIRTUAL atoms at the coordination positions: 4 of zinc's 5 'heavy' atoms are
    fictitious, against 2 of ATP's 33 which happen to sit 0.00 A from real atoms and so
    cannot move a minimum. Testing only with ATP is what hid this."""
    init_rosetta()
    pose = pyrosetta.pose_from_sequence("ACDEFGHIKL")
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    rts = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance() \
        .residue_type_set("fa_standard")
    pose.append_residue_by_jump(
        pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(
            rts.name_map("ZN")), 5)
    return pose


def test_virtual_atoms_are_excluded(peptide_with_zinc):
    """nheavyatoms() COUNTS VIRTUAL ATOMS. A zinc reports 5 and has exactly 1 real atom.

    Deliberately NOT asserted against nheavyatoms(), which is the quantity under
    suspicion -- comparing to it is circular and is exactly how this shipped. The count
    is asserted against the ion's chemistry: a zinc is one atom.
    """
    pose = peptide_with_zinc
    zn = pose.residue(pose.total_residue())
    assert zn.name3().strip() == "ZN"
    assert zn.nheavyatoms() == 5, "fixture assumption: Rosetta's ZN reports 5 heavy atoms"

    _, off = heavy_atom_coords_from_pose(pose)
    assert off[-1] - off[-2] == 1, "a zinc ion has one atom; the rest are virtual"


def test_virtual_atoms_would_have_invented_contacts(peptide_with_zinc):
    """Why it matters, rather than just that it happens.

    The virtual shell sits 1-2 A off the metal, so including it shrinks every distance
    to that ion and manufactures contacts no atom supports. This test builds the buggy
    coordinates explicitly and asserts the two disagree -- if virtual atoms ever stopped
    being placed off-centre, this fails and says the guard is no longer needed.
    """
    pose = peptide_with_zinc
    buggy, off = [], [0]
    for i in range(1, pose.total_residue() + 1):
        r = pose.residue(i)
        for k in range(1, r.nheavyatoms() + 1):
            buggy.append(np.array(r.xyz(k)))
        off.append(len(buggy))
    d_buggy = min_heavy_distances(np.asarray(buggy, float), np.asarray(off, np.intp))

    xyz, offsets = heavy_atom_coords_from_pose(pose)
    d_real = min_heavy_distances(xyz, offsets)

    inflation = (d_real[-1, :-1] - d_buggy[-1, :-1]).max()
    assert inflation > 1.0, f"expected >1 A of phantom shortening, got {inflation:.2f}"
    assert (d_buggy[-1, :-1] <= 6.0).sum() > (d_real[-1, :-1] <= 6.0).sum(), \
        "the virtual shell should manufacture at least one extra contact at 6 A"


def test_the_ligand_is_flagged_as_not_protein(peptide_with_atp):
    kinds = residue_kinds_from_pose(peptide_with_atp)
    assert kinds[:-1].all(), "every amino acid should be protein"
    assert not kinds[-1], "ATP should not be protein"


def test_min_heavy_gives_the_ligand_real_distances(peptide_with_atp):
    """The point of the whole exercise: under a CA rule the ligand has no coordinate at
    all, and NaN <= cutoff silently drops every pair. Under a heavy-atom minimum it has
    finite distances to every protein residue."""
    xyz, off = heavy_atom_coords_from_pose(peptide_with_atp)
    d = min_heavy_distances(xyz, off)
    ligand_row = d[-1, :-1]
    assert np.isfinite(ligand_row).all()
    assert (ligand_row >= 0).all()
    assert np.array_equal(d, _brute_min_heavy(xyz, off))
