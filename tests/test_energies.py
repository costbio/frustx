"""Tests for pairwise energy extraction.

The central test here is `test_decomposition_is_complete`: one-body + pairwise must
reproduce the REF2015 total exactly.  That is the guarantee that no interaction energy
is being silently dropped from e_ij, which was a real failure mode -- backbone-backbone
hbonds are off the energy graph by default and would have vanished without notice.

No PDB fixture is used.  Poses are built from sequence and given explicit backbone
torsions, so the tests are self-contained, deterministic, and carry no data files.
"""

import numpy as np
import pytest

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.energies import (  # noqa: E402
    init_rosetta,
    make_score_function,
    onebody_energy_vector,
    pair_energy_matrix,
)


@pytest.fixture(scope="module")
def helix():
    """A 12-residue poly-alanine alpha helix.

    Built by setting ideal helical torsions rather than loading a structure: it is
    reproducible from nothing, and unlike an extended chain it has genuine i,i+4
    backbone hydrogen bonds -- which is the interaction class most at risk of being
    lost from the pairwise decomposition.
    """
    init_rosetta()
    pose = pyrosetta.pose_from_sequence("A" * 12)
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)     # ideal right-handed alpha-helix
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    return pose


def test_decomposition_is_complete(helix):
    """one-body + sum of pair energies == REF2015 total, to machine precision.

    Uses the unmodified force field (fa_rep kept) so the target is the real REF2015
    score.  If a future Rosetta release moves a term off the energy graph, this fails
    rather than quietly biasing every frustration index.
    """
    sf = make_score_function(remove_fa_rep=False)
    total = sf(helix)

    onebody = onebody_energy_vector(helix, sf).sum()
    # e is symmetric with both (i,j) and (j,i) populated, so the upper triangle is the
    # sum over unique pairs.
    pairwise = np.triu(pair_energy_matrix(helix, sf), k=1).sum()

    assert onebody + pairwise == pytest.approx(total, abs=1e-9)


def test_bb_hbonds_would_be_lost_without_the_option(helix):
    """Guards the fix, not just the outcome.

    A score function without decompose_bb_hb_into_pair_energies must capture strictly
    LESS pairwise energy than ours, because its backbone hbonds sit outside the graph.
    If this ever stops being true the option has become a no-op and the comment in
    energies.py is lying.
    """
    sf_fixed = make_score_function(remove_fa_rep=False)

    sf_naive = pyrosetta.create_score_function("ref2015")  # option deliberately not set
    total_naive = sf_naive(helix)
    onebody_naive = onebody_energy_vector(helix, sf_naive).sum()
    pairwise_naive = np.triu(pair_energy_matrix(helix, sf_naive), k=1).sum()

    missing = total_naive - onebody_naive - pairwise_naive
    # Backbone hbonds are stabilising, so the unaccounted remainder is negative.
    assert missing < -0.5, f"expected lost bb-hbond energy, got {missing}"

    # And the fixed score function loses nothing.
    total_fixed = sf_fixed(helix)
    complete = (
        onebody_energy_vector(helix, sf_fixed).sum()
        + np.triu(pair_energy_matrix(helix, sf_fixed), k=1).sum()
    )
    assert complete == pytest.approx(total_fixed, abs=1e-9)


def test_rama_prepro_is_carried_on_i_iplus1_pairs(helix):
    """rama_prepro must reach the matrix, on strictly sequential pairs.

    It lives in a long-range container, not the energy graph. Zeroing its weight must
    change only the (i, i+1) entries -- if it leaked elsewhere, our generic long-range
    sweep is mis-assigning residues.
    """
    from pyrosetta.rosetta.core.scoring import ScoreType

    sf = make_score_function(remove_fa_rep=False)
    with_rama = pair_energy_matrix(helix, sf)

    sf.set_weight(ScoreType.rama_prepro, 0.0)
    without_rama = pair_energy_matrix(helix, sf)

    diff = np.abs(with_rama - without_rama)
    assert diff.sum() > 0, "rama_prepro never reached the pair matrix"

    n = helix.total_residue()
    off_diagonal_band = diff.copy()
    for i in range(n - 1):
        off_diagonal_band[i, i + 1] = 0.0
        off_diagonal_band[i + 1, i] = 0.0
    assert off_diagonal_band.sum() == pytest.approx(0.0, abs=1e-9), (
        "rama_prepro was assigned to non-sequential pairs"
    )


def test_matrix_is_symmetric_with_zero_diagonal(helix):
    sf = make_score_function()
    e = pair_energy_matrix(helix, sf)
    n = helix.total_residue()
    assert e.shape == (n, n)
    np.testing.assert_allclose(e, e.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(e), 0.0, atol=1e-12)


def test_removing_fa_rep_changes_pair_energies(helix):
    """The paper's fa_rep removal must actually take effect.

    fa_rep is purely repulsive (positive), so dropping it can only lower pair energies.
    """
    e_with = pair_energy_matrix(helix, make_score_function(remove_fa_rep=False))
    e_without = pair_energy_matrix(helix, make_score_function(remove_fa_rep=True))

    assert not np.allclose(e_with, e_without), "fa_rep removal had no effect"
    assert e_without.sum() < e_with.sum()
