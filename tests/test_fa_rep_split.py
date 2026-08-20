"""Tests for carrying the fa_rep term alongside the measured energy.

sf_measure zeroes fa_rep at BUILD time, so historically the repulsive part of e_ij was
gone by the time anything could look at it, and asking "what would this run look like as
packing frustration?" meant rebuilding the whole decoy ensemble. compute_frustration now
records that part separately.

The load-bearing test is `test_recombination_matches_a_real_second_run`: it checks the
stored parts against an actual run measured WITH fa_rep, rather than against the same
arithmetic that produced them.
"""

import numpy as np
import pytest

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.energies import (  # noqa: E402
    init_rosetta,
    make_fa_rep_score_function,
    make_score_function,
    pair_energy_matrix,
)
from frustx.frustration import compute_frustration  # noqa: E402
from frustx.output import contact_table  # noqa: E402


@pytest.fixture(scope="module")
def peptide():
    """A heterogeneous 14-mer, NOT poly-alanine.

    Poly-alanine decoys come out near-identical, so decoy_std collapses to ~1e-9 and
    every index is a 0/0 -- which makes any agreement test here meaningless. A mixed
    sequence gives an ensemble with real spread.
    """
    init_rosetta()
    pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQ")
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    return pose


def _run(pose, remove_fa_rep):
    return compute_frustration(
        pose,
        sf_pack=make_score_function(remove_fa_rep=False),
        sf_measure=make_score_function(remove_fa_rep=remove_fa_rep),
        n_decoys=8, seed=5, protocol="min", packing_seed=7, n_jobs=1,
    )


@pytest.fixture(scope="module")
def measured(peptide):
    """The default run: fa_rep removed from the measurement, as the paper prescribes."""
    return _run(peptide, remove_fa_rep=True)


def test_the_fa_rep_only_score_function_isolates_exactly_that_term(peptide):
    """full - measure must equal the fa_rep-only matrix, or the column is mislabelled."""
    full = pair_energy_matrix(peptide, make_score_function(remove_fa_rep=False))
    meas = pair_energy_matrix(peptide, make_score_function(remove_fa_rep=True))
    rep = pair_energy_matrix(peptide, make_fa_rep_score_function())
    assert np.abs((full - meas) - rep).max() < 1e-12


def test_fa_rep_is_actually_nonzero(measured):
    """A guard against the whole feature silently recording zeros -- which is what a
    wrongly-built score function would produce, and every other test here would still
    pass."""
    assert np.abs(measured.fa_rep_native).max() > 0
    assert np.abs(measured.fa_rep_mean).max() > 0


def test_recombination_matches_a_real_second_run(peptide, measured):
    """The claim the feature exists to support.

    index_at_fa_rep(+1) reconstructs, from one run, what a second run measuring WITH
    fa_rep would have reported. Checked against that second run, not against itself.
    """
    recombined = measured.index_at_fa_rep(+1.0)
    packing = _run(peptide, remove_fa_rep=False)
    m = np.isfinite(recombined) & np.isfinite(packing.index)
    assert m.sum() > 50, "too few finite contacts to be a meaningful comparison"
    rel = np.abs((recombined[m] - packing.index[m]) / packing.index[m])
    assert rel.max() < 1e-10


def test_recombination_works_in_the_other_direction(peptide):
    """A --packing-frustration run can have fa_rep taken back OUT with weight=-1."""
    packing = _run(peptide, remove_fa_rep=False)
    stripped = packing.index_at_fa_rep(-1.0)
    plain = _run(peptide, remove_fa_rep=True)
    m = np.isfinite(stripped) & np.isfinite(plain.index)
    rel = np.abs((stripped[m] - plain.index[m]) / plain.index[m])
    assert rel.max() < 1e-10


def test_weight_zero_is_the_unmodified_index(measured):
    """Sanity on the algebra: adding none of it changes nothing."""
    got = measured.index_at_fa_rep(0.0)
    m = np.isfinite(got) & np.isfinite(measured.index)
    assert np.allclose(got[m], measured.index[m], rtol=0, atol=1e-12)


def test_the_covariance_is_actually_used(peptide, measured):
    """Var(E + R) = Var(E) + Var(R) + 2Cov(E, R).

    Asserting that fa_rep_cov is merely nonzero would pass even if index_at_fa_rep
    ignored it entirely. So instead: recompute the same thing WITHOUT the cross term and
    require that it disagrees with the real second run by far more than the tolerance
    the correct version meets. That pins both that the term is material on this ensemble
    and that the implementation uses it.
    """
    packing = _run(peptide, remove_fa_rep=False)

    mean = measured.decoy_mean + measured.fa_rep_mean
    var_no_cross = measured.decoy_std ** 2 + measured.fa_rep_std ** 2   # cross dropped
    std = np.sqrt(np.maximum(var_no_cross, 0.0))
    e0 = measured.native_energy + measured.fa_rep_native
    with np.errstate(divide="ignore", invalid="ignore"):
        wrong = np.where(std > 0, (mean - e0) / std, np.nan)

    m = np.isfinite(wrong) & np.isfinite(packing.index)
    rel = np.abs((wrong[m] - packing.index[m]) / packing.index[m])
    assert rel.max() > 1e-3, (
        "dropping the covariance changed nothing, so this ensemble cannot detect "
        "whether the cross term is used at all"
    )
    # and the real one, on the same entries, is exact
    good = measured.index_at_fa_rep(+1.0)
    assert np.abs((good[m] - packing.index[m]) / packing.index[m]).max() < 1e-10


def test_the_parts_reach_the_contact_table(measured):
    """Values, not just column names: writing fa_rep_mean into the fa_rep_native column
    would pass a presence-only check."""
    df = contact_table(measured)
    ij = (measured.contacts[:, 0], measured.contacts[:, 1])
    for col, arr in (("fa_rep_native", measured.fa_rep_native),
                     ("fa_rep_decoy_mean", measured.fa_rep_mean),
                     ("fa_rep_decoy_std", measured.fa_rep_std)):
        assert col in df.columns
        assert np.allclose(df[col].to_numpy(), arr[ij], rtol=0, atol=0)
    assert len(df) == len(measured.contacts)


def test_adding_fa_rep_to_a_run_that_already_has_it_is_refused(peptide):
    """The failure this guard exists for is not loud on its own: double-counting gives
    an index that correlates with the truth at r = 0.99 with twice the scale."""
    packing = _run(peptide, remove_fa_rep=False)
    assert packing.fa_rep_in_measure is True
    with pytest.raises(ValueError, match="twice"):
        packing.index_at_fa_rep(+1.0)


def test_subtracting_fa_rep_from_a_run_that_never_had_it_is_refused(measured):
    assert measured.fa_rep_in_measure is False
    with pytest.raises(ValueError, match="not in the energy"):
        measured.index_at_fa_rep(-1.0)


def test_the_default_weight_flips_the_run_to_its_complement(peptide, measured):
    """No argument means "the other reading", which is the only direction that is
    correct without knowing which kind of run this was."""
    packing = _run(peptide, remove_fa_rep=False)
    assert np.array_equal(np.nan_to_num(measured.index_at_fa_rep(), nan=-9),
                          np.nan_to_num(measured.index_at_fa_rep(+1.0), nan=-9))
    assert np.array_equal(np.nan_to_num(packing.index_at_fa_rep(), nan=-9),
                          np.nan_to_num(packing.index_at_fa_rep(-1.0), nan=-9))


def test_a_mismatched_weights_set_is_refused(peptide):
    """sf_measure is caller-supplied and make_score_function takes its own weights, so
    the fa_rep weight can silently disagree -- score12 is 0.44 against ref2015's 0.55.
    Only detectable when fa_rep survived in sf_measure, which is exactly when it is
    checked."""
    from frustx.frustration import compute_frustration as cf
    with pytest.raises(ValueError, match="pass the weights set"):
        cf(peptide,
           sf_pack=make_score_function(remove_fa_rep=False),
           sf_measure=make_score_function(remove_fa_rep=False, weights="score12"),
           n_decoys=2, seed=1, protocol="min", n_jobs=1)


def test_a_result_without_the_decomposition_says_so():
    """Older pickled results have these as None; asking for a recombination must raise
    rather than produce NaNs that look like real output."""
    from frustx.frustration import FrustrationResult
    r = FrustrationResult(residues=[], contacts=np.zeros((0, 2), int),
                          native_energy=np.zeros((0, 0)), decoy_mean=np.zeros((0, 0)),
                          decoy_std=np.zeros((0, 0)), index=np.zeros((0, 0)), n_decoys=0)
    with pytest.raises(ValueError, match="no fa_rep decomposition"):
        r.index_at_fa_rep()
