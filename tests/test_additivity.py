"""Additive decomposition: cases where the right answer is known by construction.

The decomposition is reported per contact, so the properties that must hold are that the
FITTED and RESIDUAL parts are correct and unique, and that a purely one-body signal is
recovered exactly while a purely pair-specific one is not absorbed.
"""
import numpy as np
import pytest

from frustx.additivity import additive_decomposition


def all_pairs(n):
    return np.array([(i, j) for i in range(n) for j in range(i + 1, n)])


def test_a_purely_additive_signal_is_recovered_exactly():
    """If the index really IS c + a_i + a_j, R2 must be 1 and the residual must vanish.

    Built with known coefficients, so this is checkable without trusting the fit.
    """
    n = 8
    pairs = all_pairs(n)
    a = np.array([0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, -0.3])
    y = np.array([3.0 + a[i] + a[j] for i, j in pairs])
    d = additive_decomposition(y, pairs, n)
    assert d.r2 == pytest.approx(1.0)
    assert np.allclose(d.residual, 0.0, atol=1e-10)
    assert np.allclose(d.fitted, y)
    # Coefficients are identified only up to a constant, so DIFFERENCES are what to check.
    assert d.coefficients[2] - d.coefficients[1] == pytest.approx(a[2] - a[1])
    assert d.coefficients[5] - d.coefficients[0] == pytest.approx(a[5] - a[0])


def test_a_purely_pair_specific_signal_is_not_absorbed():
    """One contact perturbed, everything else flat.

    The additive part cannot represent 'this pair specifically', so the perturbation must
    survive into the residual. If it were absorbed, the decomposition would be useless
    for the question it exists to answer.
    """
    n = 8
    pairs = all_pairs(n)
    y = np.zeros(len(pairs))
    y[7] = 5.0
    d = additive_decomposition(y, pairs, n)
    assert d.r2 < 0.5
    assert d.residual[7] == pytest.approx(max(d.residual), abs=1e-12)
    assert d.residual[7] > 2.0


def test_fitted_plus_residual_reconstructs_the_index():
    rng = np.random.default_rng(0)
    n, pairs = 9, all_pairs(9)
    y = rng.normal(size=len(pairs))
    d = additive_decomposition(y, pairs, n)
    assert np.allclose(d.fitted + d.residual, y)


def test_nan_contacts_are_excluded_not_imputed():
    """sigma=0 contacts carry NaN and are reported 'undefined' elsewhere. Quietly giving
    them a fitted value here would undo that."""
    n, pairs = 8, all_pairs(8)
    rng = np.random.default_rng(1)
    y = rng.normal(size=len(pairs))
    y[3] = np.nan
    d = additive_decomposition(y, pairs, n)
    assert np.isnan(d.fitted[3]) and np.isnan(d.residual[3])
    assert d.n_used == len(pairs) - 1
    assert np.isfinite(d.fitted[np.arange(len(pairs)) != 3]).all()


def test_underdetermined_fit_reports_nothing_rather_than_a_perfect_one():
    """With fewer contacts than parameters the fit is exact by construction and means
    nothing. Reporting R2=1.0 there would be actively misleading."""
    n = 10
    pairs = np.array([(0, 1), (2, 3), (4, 5)])
    d = additive_decomposition(np.array([1.0, 2.0, 3.0]), pairs, n)
    assert np.isnan(d.r2)
    assert np.isnan(d.fitted).all()
    assert d.n_used == 3


def test_coefficients_are_the_minimum_norm_solution_and_reproducible():
    """The fit is rank deficient -- adding t to c and subtracting t/2 from every a_i
    leaves every fitted value unchanged. lstsq's minimum-norm choice must at least be
    stable, or per-residue numbers would jitter between runs."""
    n, pairs = 7, all_pairs(7)
    rng = np.random.default_rng(2)
    y = rng.normal(size=len(pairs))
    d1 = additive_decomposition(y, pairs, n)
    d2 = additive_decomposition(y, pairs, n)
    assert np.array_equal(d1.coefficients, d2.coefficients)
    assert d1.intercept == d2.intercept
