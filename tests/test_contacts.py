"""Tests for the contact-map definition.

Everything here uses hand-built coordinates placed on a straight line, so the
expected contacts can be checked by mental arithmetic rather than by trusting
the code that is under test.
"""

import numpy as np
import pytest

from frustx.contacts import Residue, contact_pairs, load_ca


def _atom(serial, name, resname, chain, resseq, xyz, element="C"):
    """Format one PDB ATOM record in fixed-column layout."""
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5d} {name:^4s}{'':1s}{resname:>3s} {chain:1s}"
        f"{resseq:>4d}{'':1s}   {x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}{'':10s}{element:>2s}"
    )


def _write_pdb(path, atoms):
    path.write_text("\n".join(atoms) + "\nEND\n")
    return path


# Four CA atoms on the x axis at x = 0, 5, 11, 14.
#
#   pair   distance   <= 10 A ?
#   0-1        5         yes
#   0-2       11         no
#   0-3       14         no
#   1-2        6         yes
#   1-3        9         yes
#   2-3        3         yes
#
# So: 4 contacts, and (0,2) / (0,3) must be absent.
LINE_X = [0.0, 5.0, 11.0, 14.0]


@pytest.fixture
def line_pdb(tmp_path):
    atoms = [
        _atom(i + 1, "CA", "ALA", "A", i + 1, (x, 0.0, 0.0))
        for i, x in enumerate(LINE_X)
    ]
    return _write_pdb(tmp_path / "line.pdb", atoms)


def test_load_ca_reads_every_residue(line_pdb):
    residues, coords = load_ca(line_pdb)
    assert len(residues) == 4
    assert coords.shape == (4, 3)
    # Coordinates come back in file order, and index is the 0-based position.
    assert [r.index for r in residues] == [0, 1, 2, 3]
    assert [r.resseq for r in residues] == [1, 2, 3, 4]
    assert residues[0].label == "A:ALA1"
    np.testing.assert_allclose(coords[:, 0], LINE_X)


def test_contacts_at_10A_cutoff(line_pdb):
    residues, coords = load_ca(line_pdb)
    pairs = contact_pairs(residues, coords, cutoff=10.0)
    assert {tuple(p) for p in pairs} == {(0, 1), (1, 2), (1, 3), (2, 3)}


def test_cutoff_is_inclusive_and_tightening_it_drops_pairs(line_pdb):
    residues, coords = load_ca(line_pdb)
    # At exactly 5.0 A the 0-1 pair (d = 5) is kept: the paper's cutoff is "<=".
    pairs = {tuple(p) for p in contact_pairs(residues, coords, cutoff=5.0)}
    assert (0, 1) in pairs
    assert (1, 2) not in pairs  # d = 6
    # At 4.9 A only the 2-3 pair (d = 3) survives.
    pairs = {tuple(p) for p in contact_pairs(residues, coords, cutoff=4.9)}
    assert pairs == {(2, 3)}


def test_min_seq_sep_excludes_sequential_neighbours(line_pdb):
    residues, coords = load_ca(line_pdb)
    # min_seq_sep=2 drops |i-j| == 1, i.e. (0,1), (1,2), (2,3), leaving (1,3).
    pairs = contact_pairs(residues, coords, cutoff=10.0, min_seq_sep=2)
    assert {tuple(p) for p in pairs} == {(1, 3)}


def test_seq_sep_does_not_apply_across_chains(tmp_path):
    """Two chains 5 A apart: the inter-chain pair must survive any min_seq_sep.

    Residue indices 0 (chain A) and 1 (chain B) differ by 1, so a naive
    index-only sequence-separation filter would wrongly discard this contact.
    """
    atoms = [
        _atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)),
        _atom(2, "CA", "ALA", "B", 1, (5.0, 0.0, 0.0)),
    ]
    pdb = _write_pdb(tmp_path / "two_chains.pdb", atoms)
    residues, coords = load_ca(pdb)
    assert [r.chain for r in residues] == ["A", "B"]
    pairs = contact_pairs(residues, coords, cutoff=10.0, min_seq_sep=5)
    assert {tuple(p) for p in pairs} == {(0, 1)}


def test_waters_ligands_and_ca_less_residues_are_dropped(tmp_path):
    """Only standard amino acids carrying a CA atom should be parsed."""
    atoms = [
        _atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)),
        # A glycine whose CA was not resolved -- only N present. Must be dropped,
        # otherwise it would silently shift every downstream residue index.
        _atom(2, "N", "GLY", "A", 2, (5.0, 0.0, 0.0), element="N"),
        _atom(3, "CA", "ALA", "A", 3, (8.0, 0.0, 0.0)),
    ]
    atoms += [
        "HETATM    4  O   HOH A 101       1.000   0.000   0.000  1.00  0.00           O",
        "HETATM    5  C1  STI A 102       2.000   0.000   0.000  1.00  0.00           C",
    ]
    pdb = _write_pdb(tmp_path / "mixed.pdb", atoms)
    residues, coords = load_ca(pdb)
    assert [r.resseq for r in residues] == [1, 3]
    # Indices are renumbered contiguously after dropping, which is exactly why
    # Residue.index and Residue.resseq are kept as separate fields.
    assert [r.index for r in residues] == [0, 1]


def test_empty_structure_raises(tmp_path):
    pdb = _write_pdb(
        tmp_path / "water_only.pdb",
        ["HETATM    1  O   HOH A 101       0.000   0.000   0.000  1.00  0.00           O"],
    )
    with pytest.raises(ValueError, match="No standard amino-acid residues"):
        load_ca(pdb)


def _res(index, resseq):
    """A minimal Residue, for tests that only need labels and count."""
    return Residue(index=index, chain="A", resseq=resseq, icode=" ", resname="ALA")


def test_a_nan_coordinate_raises_instead_of_silently_dropping_pairs():
    """The whole point of the guard.

    `np.nan <= 10.0` is False with no warning, so without this check residue 1 would
    simply have no contacts and the run would report a clean, wrong answer over an
    incomplete map. Nothing downstream can tell that apart from a residue that
    genuinely touches nothing, so it has to raise here.
    """
    residues = [_res(0, 1), _res(1, 2), _res(2, 3)]
    coords = np.array([[0.0, 0.0, 0.0],
                       [np.nan, np.nan, np.nan],   # e.g. a ligand with no CA
                       [5.0, 0.0, 0.0]])

    # First: confirm the hazard is real, i.e. that the pair really would vanish.
    # If numpy ever changed this, the guard would be pointless and this test says so.
    d = np.abs(coords[:, None, 0] - coords[None, :, 0])
    assert not (d[0, 1] <= 10.0), "NaN comparison no longer False; revisit the guard"

    with pytest.raises(ValueError, match="non-finite contact coordinate"):
        contact_pairs(residues, coords, cutoff=10.0)


def test_the_error_names_the_offending_residues():
    """A bare 'bad coordinate' is not actionable on a 500-residue pose."""
    residues = [_res(0, 1), _res(1, 2)]
    coords = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match=r"A:ALA2"):
        contact_pairs(residues, coords)


def test_infinity_is_rejected_too():
    """isfinite, not isnan: an inf coordinate compares True and would invent contacts
    with everything, which is the same class of silent wrongness in the other direction."""
    residues = [_res(0, 1), _res(1, 2)]
    coords = np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        contact_pairs(residues, coords)


def test_the_bad_residue_list_is_truncated():
    """Eight bad residues must not print eight labels; five plus a count."""
    residues = [_res(i, i + 1) for i in range(8)]
    coords = np.full((8, 3), np.nan)
    with pytest.raises(ValueError, match=r"and 3 more"):
        contact_pairs(residues, coords)


def test_a_clean_structure_is_unaffected_by_the_guard(line_pdb):
    """The guard must not change any answer for well-formed input."""
    residues, coords = load_ca(line_pdb)
    assert len(contact_pairs(residues, coords, cutoff=10.0)) == 4


def test_a_coordinate_list_is_accepted_not_just_an_array():
    """contact_pairs is called with plain lists in scripts/, and the guard runs
    np.isfinite, so the asarray has to happen before it."""
    residues = [_res(0, 1), _res(1, 2)]
    pairs = contact_pairs(residues, [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], cutoff=10.0)
    assert pairs.tolist() == [[0, 1]]
