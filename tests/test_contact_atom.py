"""Contact representative atom: CA vs CB, on coordinates checkable by hand.

The CB option exists because a Ca-Ca cutoff admits pairs whose side chains point away
from each other and for which no energy term fires (23.7% of contacts across the six
GTPase runs). The behaviour that must not regress is the glycine fallback: Gly has no CB,
and silently dropping it or crashing would both be wrong.
"""
import numpy as np
import pytest

from frustx.contacts import Residue, contact_pairs


def residues(names):
    return [Residue(index=k, chain="A", resseq=k + 1, icode=" ", resname=n)
            for k, n in enumerate(names)]


def test_ca_and_cb_can_disagree_on_the_same_pair():
    """Two residues 9.4 A apart at CA, whose CBs point in opposite directions.

    Placed on a line so the arithmetic is trivial:
        res0 CA at x=0,   CB at x=-1   (side chain points away, to the left)
        res1 CA at x=9.4, CB at x=10.4 (side chain points away, to the right)
    CA-CA  = 9.4  -> inside a 10 A cutoff
    CB-CB  = 11.4 -> outside it
    """
    res = residues(["ALA", "ALA"])
    ca = np.array([[0.0, 0, 0], [9.4, 0, 0]])
    cb = np.array([[-1.0, 0, 0], [10.4, 0, 0]])
    assert len(contact_pairs(res, ca, cutoff=10.0, min_seq_sep=1)) == 1
    assert len(contact_pairs(res, cb, cutoff=10.0, min_seq_sep=1)) == 0


def test_cb_pointing_toward_can_create_a_contact_ca_misses():
    """The converse: CA just outside, CBs leaning together and inside."""
    res = residues(["ALA", "ALA"])
    ca = np.array([[0.0, 0, 0], [10.6, 0, 0]])
    cb = np.array([[1.0, 0, 0], [9.6, 0, 0]])       # CB-CB = 8.6
    assert len(contact_pairs(res, ca, cutoff=10.0, min_seq_sep=1)) == 0
    assert len(contact_pairs(res, cb, cutoff=10.0, min_seq_sep=1)) == 1


def test_contact_pairs_is_agnostic_to_which_atom_it_was_given():
    """contact_pairs takes coordinates, not atom names -- the choice is the caller's.

    This is deliberate: it keeps the geometry unit-testable without a PDB, and means
    adding CB required no change to the contact rule itself.
    """
    res = residues(["ALA", "ALA", "ALA"])
    xyz = np.array([[0.0, 0, 0], [5.0, 0, 0], [50.0, 0, 0]])
    p = contact_pairs(res, xyz, cutoff=10.0, min_seq_sep=1)
    assert p.tolist() == [[0, 1]]


def test_load_ca_rejects_an_unknown_atom(tmp_path):
    from frustx.contacts import load_ca
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CB  ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
        "END\n")
    with pytest.raises(ValueError, match="atom must be"):
        load_ca(pdb, atom="CG")


def test_glycine_falls_back_to_ca_rather_than_vanishing(tmp_path):
    """Gly has no CB. It must keep its CA coordinate, not be dropped or NaN."""
    from frustx.contacts import load_ca
    pdb = tmp_path / "g.pdb"
    pdb.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  N   ALA A   2       3.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      5  CA  ALA A   2       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  CB  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      7  C   ALA A   2       6.000   0.000   0.000  1.00  0.00           C\n"
        "END\n")
    res, xyz = load_ca(pdb, atom="CB")
    assert [r.resname for r in res] == ["GLY", "ALA"]
    assert xyz[0][0] == pytest.approx(1.0)      # Gly kept its CA
    assert xyz[1][0] == pytest.approx(5.0)      # Ala used its CB
