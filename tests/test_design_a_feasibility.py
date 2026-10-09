"""Regression tests for the pooled permutation test shared by the Design A scripts.

The bug these pin down (fixed 2026-10-06): pooled_power compared the statistic against
the null distribution's (1 - alpha) quantile instead of computing a permutation p-value.
Both give the same answer when the null is continuous, so the large scopes were right,
but on a COARSE null -- few distinct attainable values, which is what small scopes give --
the quantile route declares significance where no p-value could ever reach alpha. It had
reported power 1.0 for a scope of one two-complex ligand, and 0.385 instead of 0.241 for
the 18-complex negative control.

Offline, no network, no data files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import design_a_feasibility as da  # noqa: E402
import extended_set as es  # noqa: E402

ALPHA = da.ALPHA
# Both scripts must use the p-value rule; the tests run against each implementation.
POOLED = pytest.mark.parametrize("pooled_power", [da.pooled_power, es.pooled_power],
                                 ids=["design_a", "extended_set"])


# --- the decision rule itself ------------------------------------------------------

def quantile_verdict(null: np.ndarray, obs: np.ndarray, alpha: float = ALPHA) -> float:
    """The OLD, wrong rule: reject when |stat| >= the null's (1 - alpha) quantile."""
    return float((obs >= np.quantile(null, 1 - alpha)).mean())


def pvalue_verdict(null: np.ndarray, obs: np.ndarray, alpha: float = ALPHA) -> float:
    """The correct rule: reject when P(null >= |stat|) <= alpha."""
    ns = np.sort(null)
    p = 1.0 - np.searchsorted(ns, obs - 1e-12, side="left") / len(ns)
    return float((p <= alpha).mean())


def test_degenerate_null_is_where_the_two_rules_diverge():
    # One two-complex ligand: |stat| is always exactly 1, under the null and under any
    # alternative. Nothing is distinguishable, so the only honest power is 0.
    null = np.ones(5000)
    obs = np.ones(1000)
    assert quantile_verdict(null, obs) == 1.0      # the bug: "perfect power"
    assert pvalue_verdict(null, obs) == 0.0        # p = 1 for every draw


def test_coarse_null_two_attainable_values():
    # A null taking two values: the 95th percentile is the larger one, so every draw at
    # that value "beats" it, yet its p-value is 0.5.
    null = np.repeat([0.5, 1.0], 2500)
    obs = np.ones(1000)
    assert quantile_verdict(null, obs) == 1.0
    assert pvalue_verdict(null, obs) == 0.0


def test_continuous_null_the_two_rules_agree():
    rng = np.random.default_rng(0)
    null = rng.uniform(size=20000)
    obs = rng.uniform(size=4000)
    assert pvalue_verdict(null, obs) == pytest.approx(quantile_verdict(null, obs), abs=0.02)


# --- the rule as wired into both scripts -------------------------------------------

@POOLED
def test_two_complex_scope_cannot_be_significant(pooled_power):
    """A scope of one two-point ligand: power 0 at every effect size, and the minimum
    attainable p says why. This is the case the old rule reported as power 1.0."""
    res = pooled_power([np.array([9.0, 6.0])], (0.5, 2.0, 10.0), 200,
                       np.random.default_rng(1))
    assert res["complexes_used"] == 2
    assert res["min_attainable_p"] == 1.0
    assert set(res["power"].values()) == {0.0}


@POOLED
def test_small_scope_power_stays_under_the_quantile_estimate(pooled_power):
    """Three two-point ligands: attainable p values are coarse (2^-3 arrangements per
    sign), so a huge effect still cannot pass alpha = 0.05."""
    gs = [np.array([9.0, 6.0]), np.array([8.0, 7.0]), np.array([9.5, 5.5])]
    res = pooled_power(gs, (2.0, 10.0), 200, np.random.default_rng(2))
    assert res["min_attainable_p"] > ALPHA
    assert set(res["power"].values()) == {0.0}


@POOLED
def test_large_scope_is_unaffected_and_well_powered(pooled_power):
    """A scope with enough spread has a fine-grained null, where the fix changes nothing:
    a strong effect is detected, a null effect sits near alpha."""
    rng = np.random.default_rng(3)
    gs = [np.sort(rng.uniform(5, 10, 5))[::-1] for _ in range(10)]
    res = pooled_power(gs, (0.0, 3.0), 400, rng)
    assert res["complexes_used"] == 50
    assert res["min_attainable_p"] < ALPHA
    assert res["power"][3.0] > 0.9
    assert res["power"][0.0] <= 0.10      # no signal: around alpha, never inflated


@POOLED
def test_power_rises_with_effect_size(pooled_power):
    rng = np.random.default_rng(4)
    gs = [np.sort(rng.uniform(5, 10, 4))[::-1] for _ in range(8)]
    res = pooled_power(gs, (0.1, 1.0, 5.0), 400, rng)
    p = [res["power"][r] for r in (0.1, 1.0, 5.0)]
    assert p[0] < p[1] <= p[2]        # 32 complexes saturate at 1.0 well before 5.0
    assert p[0] < 0.2 and p[2] == 1.0


@POOLED
def test_ligands_without_spread_are_dropped(pooled_power):
    gs = [np.array([8.0, 8.0]), np.array([9.0, 6.0]), np.array([7.0])]
    res = pooled_power(gs, (1.0,), 100, np.random.default_rng(5))
    assert (res["ligands_used"], res["complexes_used"]) == (1, 2)


@POOLED
def test_empty_scope(pooled_power):
    res = pooled_power([np.array([8.0, 8.0])], (1.0,), 100, np.random.default_rng(6))
    assert res["ligands_used"] == 0 and res["power"][1.0] == 0.0


def test_both_implementations_agree():
    """design_a and extended_set must give the same answer; only the inner loop differs
    (extended_set vectorises the simulations)."""
    gs = [np.array([9.0, 7.5, 6.0, 5.0]), np.array([8.0, 7.0, 6.5]),
          np.array([9.5, 8.0, 7.0, 6.0, 5.5]), np.array([8.5, 6.5])]
    ratios = (0.5, 2.0)
    a = da.pooled_power(gs, ratios, 1500, np.random.default_rng(7))
    b = es.pooled_power(gs, ratios, 1500, np.random.default_rng(7))
    assert (a["ligands_used"], a["complexes_used"]) == (b["ligands_used"],
                                                        b["complexes_used"])
    for r in ratios:
        assert a["power"][r] == pytest.approx(b["power"][r], abs=0.05)


# --- the exploratory role rule (extended_set) --------------------------------------

def _ligands(rows):
    """rows: (ligand, range_pKd, range_from_single)"""
    import pandas as pd
    return pd.DataFrame(rows, columns=["ligand_group", "range_pKd", "range_from_single"])


def test_exploratory_role_demotes_only_wide_ligands_resting_on_single_reports():
    tbl = _ligands([("wide_solid", 2.0, False),      # ranking
                    ("wide_single", 2.0, True),      # demoted
                    ("narrow_solid", 0.5, False),    # negative control
                    ("narrow_single", 0.5, True),    # stays a negative control
                    ("exactly_one", 1.0, False)])    # the boundary is inclusive
    assert es.assign_roles(tbl).tolist() == [
        "ranking", "exploratory", "negative_control", "negative_control", "ranking"]
