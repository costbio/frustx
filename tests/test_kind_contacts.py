"""The per-residue-kind contact rule.

Protein pairs keep the paper's Ca-Ca criterion; anything involving a non-protein residue
falls back to a minimum heavy-atom distance, because a ligand has no CA and the paper's
rule is not merely inaccurate for it but undefined.

Coordinates and distances are hand-built so every expected answer is arithmetic.
"""

import numpy as np
import pytest

from frustx.config import DEFAULT_LIGAND_CUTOFF
from frustx.contacts import Residue, select_pairs_by_kind


def _res(i, chain="A", name="ALA"):
    return Residue(index=i, chain=chain, resseq=i + 1, icode=" ", resname=name)


def _dists(xs):
    """Distances along a line, so d(i,j) = |x_i - x_j|."""
    x = np.asarray(xs, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def test_protein_pairs_use_the_ca_cutoff_and_ligand_pairs_use_the_heavy_one():
    """Residues on a line at 0, 8, 15. Index 2 is the ligand.

        pair   d    both protein?   rule            in?
        0-1    8    yes             <= 10.0         yes
        0-2   15    no              <= 6.0          no
        1-2    7    no              <= 6.0          no
    """
    residues = [_res(0), _res(1), _res(2, name="ATP")]
    d = _dists([0.0, 8.0, 15.0])
    is_protein = [True, True, False]
    got = select_pairs_by_kind(residues, d, d, is_protein,
                               cutoff=10.0, ligand_cutoff=6.0)
    assert got.tolist() == [[0, 1]]


def test_a_ligand_close_enough_on_the_heavy_rule_is_a_contact():
    """Same layout, ligand pulled in to 5 A of residue 1: now inside 6.0."""
    residues = [_res(0), _res(1), _res(2, name="ATP")]
    d = _dists([0.0, 8.0, 13.0])
    got = select_pairs_by_kind(residues, d, d, [True, True, False],
                               cutoff=10.0, ligand_cutoff=6.0)
    assert got.tolist() == [[0, 1], [1, 2]]


def test_the_two_cutoffs_are_independent():
    """A pair at 9 A is a contact between two protein residues and NOT a contact if one
    endpoint is a ligand. Same distance, different rule -- which is the whole point."""
    d = _dists([0.0, 9.0])
    both = select_pairs_by_kind([_res(0), _res(1)], d, d, [True, True])
    assert both.tolist() == [[0, 1]]
    mixed = select_pairs_by_kind([_res(0), _res(1, name="ATP")], d, d, [True, False])
    assert mixed.tolist() == []


def test_the_protein_branch_never_consults_a_nan():
    """A ligand has no CA, so dist_protein is legitimately NaN in its row and column.
    That must not raise and must not leak into the answer."""
    residues = [_res(0), _res(1), _res(2, name="ATP")]
    heavy = _dists([0.0, 8.0, 11.0])
    ca = heavy.copy()
    ca[2, :] = np.nan
    ca[:, 2] = np.nan
    got = select_pairs_by_kind(residues, ca, heavy, [True, True, False],
                               cutoff=10.0, ligand_cutoff=6.0)
    # 0-1 by the protein rule (8 <= 10); 1-2 by the heavy rule (3 <= 6); 0-2 is 11, out.
    assert got.tolist() == [[0, 1], [1, 2]]


def test_a_nan_in_the_heavy_matrix_raises():
    """dist_heavy is the fallback; a NaN there has nothing behind it, and NaN <= cutoff
    is False, so it would silently drop the pair."""
    residues = [_res(0), _res(1, name="ATP")]
    heavy = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="dist_heavy contains NaN"):
        select_pairs_by_kind(residues, heavy, heavy, [True, False])


# --- ligand-ligand -----------------------------------------------------------------

def test_ligand_ligand_pairs_are_excluded_by_default():
    """Not because they are uninteresting: Eq. 1 cannot be computed for them. A decoy
    differs from the native by a shuffled protein sequence and a repack, so a pair with
    no protein endpoint has no shuffleable identity and its decoy spread is float noise.
    """
    residues = [_res(0), _res(1, name="ATP"), _res(2, name="ZN")]
    d = _dists([0.0, 3.0, 4.0])
    got = select_pairs_by_kind(residues, d, d, [True, False, False],
                               cutoff=10.0, ligand_cutoff=6.0)
    assert got.tolist() == [[0, 1], [0, 2]], "the ligand-ion pair (1,2) must be absent"


def test_ligand_ligand_can_be_opted_into():
    residues = [_res(0), _res(1, name="ATP"), _res(2, name="ZN")]
    d = _dists([0.0, 3.0, 4.0])
    got = select_pairs_by_kind(residues, d, d, [True, False, False],
                               cutoff=10.0, ligand_cutoff=6.0,
                               include_ligand_ligand=True)
    assert got.tolist() == [[0, 1], [0, 2], [1, 2]]


def test_protein_ligand_pairs_are_kept():
    """The distinction that matters: a protein-ligand pair HAS a varying endpoint."""
    residues = [_res(0), _res(1, name="ATP")]
    d = _dists([0.0, 3.0])
    assert select_pairs_by_kind(residues, d, d, [True, False]).tolist() == [[0, 1]]


# --- sequence separation -----------------------------------------------------------

def test_sequence_separation_counts_polymer_position_not_list_position():
    """The trap, and it is silent and in the WRONG DIRECTION.

    Rosetta gives a HETATM the same chain letter as the protein it sits in. Separation
    measured on flat list position therefore inflates when a ligand is parsed between two
    covalently adjacent residues -- and min_seq_sep=2, which exists to DROP adjacent
    pairs, stops dropping them.

    Both orderings below hold the same three protein residues at the same coordinates.
    P0 and P1 are covalently adjacent in both, so both must drop the pair.

        ordering A: [P0, P1, P2, L]   flat sep(P0,P1) = 1  -> dropped either way
        ordering B: [P0, L, P1, P2]   flat sep(P0,P1) = 2  -> WRONGLY KEPT on flat index
    """
    xs = [0.0, 3.0, 6.0]          # protein residues, all mutually within 10 A
    lig_x = 50.0                  # far away, so it forms no contacts of its own

    a_res = [_res(0), _res(1), _res(2), _res(3, name="ATP")]
    a_d = _dists(xs + [lig_x])
    a_prot = [True, True, True, False]

    b_res = [_res(0), _res(1, name="ATP"), _res(2), _res(3)]
    b_d = _dists([xs[0], lig_x, xs[1], xs[2]])
    b_prot = [True, False, True, True]

    a = select_pairs_by_kind(a_res, a_d, a_d, a_prot, cutoff=10.0, min_seq_sep=2)
    b = select_pairs_by_kind(b_res, b_d, b_d, b_prot, cutoff=10.0, min_seq_sep=2)

    # Translate both back to "which protein residues are in contact", 0/1/2 in order.
    a_pairs = {(i, j) for i, j in a.tolist()}
    b_map = {0: 0, 2: 1, 3: 2}
    b_pairs = {(b_map[i], b_map[j]) for i, j in b.tolist()}

    assert a_pairs == {(0, 2)}, "only the i,i+2 pair survives min_seq_sep=2"
    assert b_pairs == a_pairs, (
        f"contact set changed with the ligand's file position: {a_pairs} vs {b_pairs}"
    )


def test_a_ligand_is_never_dropped_for_sequence_adjacency():
    """A ligand has no covalent sequence relationship to anything, so min_seq_sep must
    not remove a protein-ligand contact just because their indices are adjacent."""
    residues = [_res(0), _res(1, name="ATP"), _res(2)]
    d = _dists([0.0, 2.0, 4.0])
    got = select_pairs_by_kind(residues, d, d, [True, False, True],
                               cutoff=10.0, ligand_cutoff=6.0, min_seq_sep=2)
    assert [0, 1] in got.tolist(), "protein-ligand contact dropped for index adjacency"
    assert [1, 2] in got.tolist()


def test_separation_still_does_not_apply_across_chains():
    """The pre-existing invariant must survive the ordinal change."""
    residues = [_res(0, chain="A"), _res(1, chain="B")]
    d = _dists([0.0, 3.0])
    got = select_pairs_by_kind(residues, d, d, [True, True],
                               cutoff=10.0, min_seq_sep=5)
    assert got.tolist() == [[0, 1]]


# --- validation --------------------------------------------------------------------

def test_shapes_are_checked():
    residues = [_res(i) for i in range(3)]
    d = np.zeros((3, 3))
    with pytest.raises(ValueError, match=r"is_protein is \(2,\)"):
        select_pairs_by_kind(residues, d, d, [True, True])
    with pytest.raises(ValueError, match=r"dist_protein is \(2, 2\)"):
        select_pairs_by_kind(residues, np.zeros((2, 2)), d, [True] * 3)


def test_the_default_ligand_cutoff_is_the_documented_one():
    """Pins that the default comes from config rather than a literal in the signature."""
    assert DEFAULT_LIGAND_CUTOFF == 6.0
    residues = [_res(0), _res(1, name="ATP")]
    d = _dists([0.0, 6.0])
    assert select_pairs_by_kind(residues, d, d, [True, False]).tolist() == [[0, 1]]
    d2 = _dists([0.0, 6.001])
    assert select_pairs_by_kind(residues, d2, d2, [True, False]).tolist() == []
