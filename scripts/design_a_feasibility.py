"""Is Design A -- ranking kinases WITHIN one ligand by affinity -- statistically possible?

Design A asks, for a fixed ligand with co-crystal structures against several kinases,
whether a frustration index computed per complex tracks the measured affinity across
those kinases. This script measures whether the data can carry that test at all: how
wide each ligand's affinity range is, and what power a within-ligand rank test would
have. It answers feasibility only; it computes no frustration and fits no model.

    .venv/bin/python scripts/design_a_feasibility.py

Input (no network, nothing refetched):
    data/kinome/davis/activities.csv       the primary affinities (one assay, Kd)
    data/kinome/klifs/klifs_manifest.csv   which (kinase, ligand) pairs have a structure,
                                           and each kinase's family/group

A BindingDB cross-check was dropped on purpose: 63% of BindingDB's primary Kd values for
these complexes are identical to Davis (its own copy, imported via ChEMBL -- see
scripts/davis_activity.py), so the comparison mostly compared Davis with itself.

Output:
    data/kinome/analysis/design_a_feasibility.csv   one row per ligand
    data/kinome/analysis/design_a_power_curve.csv   power vs effect size, per ligand + pooled
    data/kinome/analysis/design_a_points.csv        one row per complex
    data/kinome/analysis/design_a_pairs.csv         paralog pairs with their delta pKd
    data/kinome/analysis/design_a_feasibility.json  settings, assumptions, summary
    summary tables on stdout; no figures.

Scope: the Davis comparable set -- per ligand, the kinases that have a co-crystal
structure of THAT ligand in the manifest and a wild-type, uncensored Davis Kd. Censored
cells ("no binding") are excluded rather than set to the 10 uM bound: they would stretch
every range by an amount the assay never measured. One structure per (kinase, ligand) is
what the manifest holds, so a "complex" here is one (kinase domain, ligand) pair.
"""

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import davis_activity as dav  # noqa: E402

ALPHA = 0.05
# Effect size for the power simulation, as the ratio slope/noise_sd (see simulate_power:
# only this ratio matters). The two scenarios reported before -- slope 1 with noise sd 0.5
# and 1.0 -- are ratios 2.0 and 1.0 of this grid.
RATIOS = (0.1, 0.2, 0.33, 0.5, 1.0, 2.0)
HEADLINE_RATIOS = (2.0, 1.0)
TARGET_POWER = 0.80
# A ligand whose affinity window is narrower than this cannot be used for a ranking test;
# it is kept, under a different role, as a negative control (see assign_roles).
MIN_RANGE_FOR_RANKING = 1.0
N_SIMS = 4000
# Exact permutation enumeration up to this n (7! = 5040); sampled above it.
EXACT_MAX_N = 7
N_PERMS = 5000
SEED = 20261005


# --- the comparable set ------------------------------------------------------------

def davis_points(davis: pd.DataFrame) -> pd.DataFrame:
    """One usable Design A point per (ligand, kinase domain): wild type, domain known
    from the measurement, ligand matched to a manifest group, uncensored value, and a
    co-crystal structure of this ligand on this kinase domain."""
    d = dav.primary_subset(davis)
    d = d[(d["n"] == 1) & d["has_structure"] & d["median_pX"].notna()]
    return (d[["ligand_group", "uniprot", "klifs_kinase_id", "kinase_name", "median_pX"]]
            .rename(columns={"median_pX": "pKd"})
            .sort_values(["ligand_group", "pKd"], ascending=[True, False]))


def comparable(points: pd.DataFrame, min_proteins: int = 2) -> pd.DataFrame:
    """Ligands whose points cover at least `min_proteins` distinct UniProts (two domains
    of one protein are not a selectivity comparison -- the convention of
    bindingdb_activity.coverage)."""
    keep = points.groupby("ligand_group")["uniprot"].nunique()
    return points[points["ligand_group"].isin(keep[keep >= min_proteins].index)]


# --- spread ------------------------------------------------------------------------

def spread(points: pd.DataFrame) -> pd.DataFrame:
    """Per ligand: how many complexes/proteins, and how wide the affinity window is."""
    g = points.groupby("ligand_group")["pKd"]
    out = pd.DataFrame({
        "n_complexes": g.size(),
        "n_proteins": points.groupby("ligand_group")["uniprot"].nunique(),
        "min_pKd": g.min().round(3), "max_pKd": g.max().round(3),
        "range_pKd": (g.max() - g.min()).round(3),
        "iqr_pKd": (g.quantile(0.75) - g.quantile(0.25)).round(3),
        "std_pKd": g.std(ddof=1).round(3)})
    out["proteins"] = points.groupby("ligand_group").apply(
        lambda d: "; ".join(f"{k}={v:.2f}" for k, v in zip(d["kinase_name"], d["pKd"])),
        include_groups=False)
    return out.reset_index()


# --- power -------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def _rho(rx: np.ndarray, ry: np.ndarray) -> float:
    """Spearman rho as Pearson on ranks (pandas' spearman needs scipy)."""
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den else 0.0


def permutation_null(pkd: np.ndarray, rng: random.Random) -> np.ndarray:
    """|rho| under the permutation null for this x: all n! permutations when n is small,
    otherwise N_PERMS random ones. This is what makes the small-n verdicts honest: with
    n = 4 the smallest attainable two-sided p is 2/24 = 0.083, so no result can reach
    alpha = 0.05 however strong the true relationship."""
    rx = _rank(pkd)
    n = len(rx)
    if n <= EXACT_MAX_N:
        perms = itertools.permutations(range(n))
    else:
        perms = (rng.sample(range(n), n) for _ in range(N_PERMS))
    return np.array([abs(_rho(rx, rx[list(p)])) for p in perms])


def simulate_power(pkd: np.ndarray, ratio: float, n_sims: int, null: np.ndarray,
                   nprng: np.random.Generator) -> dict:
    """Power of a within-ligand Spearman test on this ligand's affinity values.

    Model: frustration_i = slope * pKd_i + e_i, e_i ~ N(0, noise_sd^2), independent per
    complex. Both are in "log units" of the affinity scale: slope = 1 means one unit of
    frustration per log unit of Kd, so noise_sd = 0.5 is scatter half as large as the
    affinity shift of one log unit. ONLY THE RATIO slope/noise_sd matters, so the
    frustration index's own units never enter and `ratio` is the single effect size here
    (simulated as slope = ratio, noise_sd = 1). x is the OBSERVED pKd vector, so the real
    spacing (and ties) of each ligand are kept.

    The test is a two-sided permutation test on Spearman rho at ALPHA, against the
    precomputed `null` from permutation_null() -- which depends on the data, not on the
    effect size, so one null serves the whole power curve. Power is the share of
    simulations whose p <= ALPHA; `min_attainable_p` says whether that is possible at all
    for this n.
    """
    p_min = float((null >= null.max() - 1e-12).mean())
    if p_min > ALPHA:        # no arrangement of the data could ever be significant
        return {"power": 0.0, "min_attainable_p": round(p_min, 4),
                "median_rho": None, "attainable": False}
    rx = _rank(pkd)
    y = ratio * pkd + nprng.normal(0.0, 1.0, size=(n_sims, len(pkd)))
    rhos = np.array([_rho(rx, _rank(row)) for row in y])
    pvals = np.array([(null >= abs(r) - 1e-12).mean() for r in rhos])
    return {"power": round(float((pvals <= ALPHA).mean()), 3),
            "min_attainable_p": round(p_min, 4),
            "median_rho": round(float(np.median(rhos)), 3), "attainable": True}


def pooled_power(groups: list[np.ndarray], ratios, n_sims: int,
                 nprng: np.random.Generator) -> dict:
    """Power of one test over all ligands at once, as a stand-in for the mixed model
    (frustration ~ pKd + (1 | ligand)), at each effect size in `ratios`.

    Statistic: Pearson correlation of within-ligand centred ranks -- ranks are taken
    inside each ligand and centred, so only the within-ligand signal counts and a
    ligand's overall frustration level cannot contribute. That is the fixed-effect slope
    a random-intercept model tests, without fitting one. Null: ranks permuted inside each
    ligand independently (same data-generating model as simulate_power). Ligands whose
    affinity values are all tied contribute nothing and are dropped.

    Significance is the permutation p-value, P(null >= |stat|) <= ALPHA. It must NOT be
    a comparison against the null's (1 - ALPHA) quantile: the two agree only when the
    null is continuous, and on a coarse null the quantile route overstates power. The
    extreme case is a scope of one two-complex ligand, where |stat| is always 1, so the
    quantile is 1, every draw "beats" it and power reads 1.0 -- while the real p-value is
    1 and nothing could ever be significant. `min_attainable_p` reports whether the scope
    can reach ALPHA at all. (This is the bug fixed on 2026-10-06; it had inflated the
    small-scope figures, e.g. the 18-complex negative control at ratio 2.0: 0.385 -> 0.241.
    Large scopes were unaffected.)
    """
    gs = [g for g in groups if len(g) >= 2 and np.ptp(g) > 0]
    if not gs:                      # nothing with any spread to test
        return {"ligands_used": 0, "complexes_used": 0, "min_attainable_p": None,
                "power": {r: 0.0 for r in ratios}}
    xs = [_rank(g) - _rank(g).mean() for g in gs]
    xcat = np.concatenate(xs)

    def stat(ys):
        yc = np.concatenate([_rank(y) - _rank(y).mean() for y in ys])
        den = math.sqrt(float((xcat ** 2).sum() * (yc ** 2).sum()))
        return float((xcat * yc).sum() / den) if den else 0.0

    # The null depends on the data only, so it is built once for the whole curve.
    null = np.sort(np.array([abs(stat([nprng.permutation(g) for g in gs]))
                             for _ in range(N_PERMS)]))
    p_value = lambda obs: 1.0 - np.searchsorted(  # noqa: E731
        null, np.asarray(obs) - 1e-12, side="left") / len(null)
    power = {}
    for ratio in ratios:
        obs = np.array([abs(stat([ratio * g + nprng.normal(0.0, 1.0, len(g)) for g in gs]))
                        for _ in range(n_sims)])
        power[ratio] = round(float((p_value(obs) <= ALPHA).mean()), 3)
    return {"ligands_used": len(gs), "complexes_used": int(sum(len(g) for g in gs)),
            "min_attainable_p": round(float(p_value([null[-1]])[0]), 4), "power": power}


def min_detectable_ratio(power: dict, target: float = TARGET_POWER) -> dict:
    """Smallest effect size on the grid that still reaches `target` power, and the next
    one down, where power falls below it -- the "smallest detectable effect" to report."""
    ok = sorted(r for r, p in power.items() if p >= target)
    below = sorted((r for r, p in power.items() if p < target), reverse=True)
    return {"target_power": target,
            "min_detectable_ratio": ok[0] if ok else None,
            "power_at_min_detectable": power[ok[0]] if ok else None,
            "largest_ratio_below_target": below[0] if below else None,
            "power_there": power[below[0]] if below else None}


def assign_roles(tbl: pd.DataFrame) -> pd.Series:
    """Each ligand's role in Design A. A ligand whose affinity window is narrower than
    MIN_RANGE_FOR_RANKING cannot support a ranking test -- its kinases are effectively
    equipotent -- so it is NOT dropped but tests a different hypothesis: a frustration
    index that ranks these should find no ordering. Hence "negative_control"."""
    return np.where(tbl["range_pKd"] < MIN_RANGE_FOR_RANKING, "negative_control", "ranking")


# --- paralogs ----------------------------------------------------------------------

def paralog_pairs(points: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Pairs of kinases with a structure of the same ligand that sit in the same KLIFS
    family (or, failing that, the same group). Closest-relative pairs: the two pockets
    differ least, so the affinity difference is the hardest case for a frustration index
    and the most interesting one. delta_pKd is always >= 0 (stronger minus weaker)."""
    meta = (manifest.drop_duplicates("klifs_kinase_id")
            .set_index("klifs_kinase_id")[["kinase_family", "kinase_group"]])
    p = points.join(meta, on="klifs_kinase_id")
    rows = []
    for lig, d in p.groupby("ligand_group"):
        for (_, a), (_, b) in itertools.combinations(d.iterrows(), 2):
            if a["uniprot"] == b["uniprot"]:
                continue                      # two domains of one protein, not a pair
            same = ("family" if a["kinase_family"] == b["kinase_family"] else
                    "group" if a["kinase_group"] == b["kinase_group"] else None)
            if not same:
                continue
            hi, lo = sorted([a, b], key=lambda r: -r["pKd"])
            rows.append({"ligand_group": lig, "relation": same,
                         "kinase_family": a["kinase_family"] if same == "family" else "",
                         "kinase_group": a["kinase_group"],
                         "kinase_hi": hi["kinase_name"], "pKd_hi": round(hi["pKd"], 2),
                         "kinase_lo": lo["kinase_name"], "pKd_lo": round(lo["pKd"], 2),
                         "delta_pKd": round(hi["pKd"] - lo["pKd"], 2)})
    cols = ["ligand_group", "relation", "kinase_family", "kinase_group", "kinase_hi",
            "pKd_hi", "kinase_lo", "pKd_lo", "delta_pKd"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["relation", "ligand_group", "delta_pKd"])


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kin = Path("data/kinome")
    p.add_argument("--davis", type=Path, default=kin / "davis/activities.csv")
    p.add_argument("--manifest", type=Path, default=kin / "klifs/klifs_manifest.csv")
    p.add_argument("--out", type=Path, default=kin / "analysis")
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS),
                   metavar="R", help="effect sizes slope/noise_sd (default: %(default)s)")
    p.add_argument("--sims", type=int, default=N_SIMS)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    ratios = sorted(a.ratios)

    davis = pd.read_csv(a.davis, keep_default_na=False)
    for c in ("median_pX", "n"):
        davis[c] = pd.to_numeric(davis[c], errors="coerce")
    for c in ("primary_eligible", "has_structure"):
        davis[c] = davis[c].astype(str).isin(["True", "true"])
    manifest = pd.read_csv(a.manifest, keep_default_na=False)

    points = comparable(davis_points(davis))
    tbl = spread(points)
    tbl["role"] = assign_roles(tbl)
    tbl["range_under_1_log"] = tbl["range_pKd"] < MIN_RANGE_FOR_RANKING

    rng, nprng = random.Random(SEED), np.random.default_rng(SEED)
    groups = {lig: d["pKd"].to_numpy() for lig, d in points.groupby("ligand_group")}
    # One permutation null per ligand, reused across every effect size.
    nulls = {lig: permutation_null(v, rng) for lig, v in groups.items()}
    per_ligand = {lig: {r: simulate_power(v, r, a.sims, nulls[lig], nprng) for r in ratios}
                  for lig, v in groups.items()}
    tbl["min_attainable_p"] = tbl["ligand_group"].map(
        {lig: res[ratios[0]]["min_attainable_p"] for lig, res in per_ligand.items()})
    tbl["testable_alone"] = tbl["min_attainable_p"] <= ALPHA
    # Power columns in the per-ligand table: the headline effect sizes when the grid
    # includes them (it does by default), otherwise whatever grid was asked for.
    headline = [r for r in HEADLINE_RATIOS if r in ratios] or ratios
    for r in headline:
        tbl[f"power_ratio{r:g}"] = tbl["ligand_group"].map(
            {lig: res[r]["power"] for lig, res in per_ligand.items()})

    roles = dict(zip(tbl["ligand_group"], tbl["role"]))
    subsets = {"pooled_all": list(groups.values()),
               "pooled_ranking": [v for lig, v in groups.items() if roles[lig] == "ranking"],
               "pooled_negative_control": [v for lig, v in groups.items()
                                           if roles[lig] == "negative_control"]}
    pooled = {name: pooled_power(gs, ratios, a.sims, nprng) for name, gs in subsets.items()}
    mde = {name: min_detectable_ratio(res["power"]) for name, res in pooled.items()}

    curve = [{"scope": lig, "role": roles[lig], "n_complexes": len(groups[lig]),
              "ratio": r, "power": res[r]["power"],
              "min_attainable_p": res[r]["min_attainable_p"],
              "median_rho": res[r]["median_rho"]}
             for lig, res in per_ligand.items() for r in ratios]
    curve += [{"scope": name, "role": "pooled", "n_complexes": res["complexes_used"],
               "ratio": r, "power": pw, "min_attainable_p": None, "median_rho": None}
              for name, res in pooled.items() for r, pw in res["power"].items()]
    curve = pd.DataFrame(curve).sort_values(["role", "n_complexes", "scope", "ratio"],
                                            ascending=[True, False, True, True])
    curve.to_csv(a.out / "design_a_power_curve.csv", index=False)

    pairs = paralog_pairs(points, manifest)
    points = points.join(tbl.set_index("ligand_group")["role"], on="ligand_group")
    cols = ["ligand_group", "role", "n_complexes", "n_proteins", "min_pKd", "max_pKd",
            "range_pKd", "iqr_pKd", "std_pKd", "range_under_1_log", "min_attainable_p",
            "testable_alone", *[f"power_ratio{r:g}" for r in headline], "proteins"]
    tbl = tbl.sort_values(["role", "n_complexes", "range_pKd"],
                          ascending=[True, False, False])[cols]
    tbl.to_csv(a.out / "design_a_feasibility.csv", index=False)
    pairs.to_csv(a.out / "design_a_pairs.csv", index=False)
    points.to_csv(a.out / "design_a_points.csv", index=False)

    neg = tbl[tbl["role"] == "negative_control"]
    rank = tbl[tbl["role"] == "ranking"]
    summary = {
        "inputs": {"davis": str(a.davis), "manifest": str(a.manifest)},
        "scope": {"ligands": len(tbl), "complexes": int(tbl["n_complexes"].sum()),
                  "proteins": int(points["uniprot"].nunique()),
                  "selection": "Davis wild-type uncensored Kd, primary_eligible, manifest "
                               "structure for that (kinase domain, ligand); >= 2 proteins"},
        "roles": {"ranking": {"ligands": len(rank),
                              "complexes": int(rank["n_complexes"].sum()),
                              "list": rank["ligand_group"].tolist()},
                  "negative_control": {
                      "ligands": len(neg), "complexes": int(neg["n_complexes"].sum()),
                      "list": neg["ligand_group"].tolist(),
                      "why": f"affinity range < {MIN_RANGE_FOR_RANKING} log unit: the "
                             "kinases are effectively equipotent, so these test the "
                             "opposite hypothesis (a sound index should find NO ordering) "
                             "and are kept, not dropped"}},
        "assumptions": {
            "model": "frustration_i = ratio * pKd_i + N(0, 1), independent per complex",
            "effect_size": "ratio = slope / noise_sd, both in log units of the affinity "
                           "scale; only the ratio matters, so the frustration index's own "
                           "scale is irrelevant. ratio 2.0 and 1.0 are the previously "
                           "reported noise sd 0.5 and 1.0 at slope 1",
            "test": f"two-sided permutation test on Spearman rho, alpha = {ALPHA}; "
                    f"exact enumeration for n <= {EXACT_MAX_N}, else {N_PERMS} samples",
            "pooled": "Pearson on within-ligand centred ranks, null permuted within each "
                      "ligand: the fixed-effect slope of frustration ~ pKd + (1|ligand), "
                      "without fitting the model",
            "ignored": ["affinity measurement error (Davis Kd taken as exact)",
                        "censored 'no binding' cells (excluded, not set to the 10 uM bound)",
                        "that one structure represents the complex",
                        "any frustration-index bias shared within a ligand (absorbed by "
                        "the ligand intercept) or across kinases"],
            "ratios": ratios, "sims": a.sims, "seed": SEED},
        "range": {"median_range_pKd": round(float(tbl["range_pKd"].median()), 3),
                  "total_range_span": [round(float(tbl["range_pKd"].min()), 3),
                                       round(float(tbl["range_pKd"].max()), 3)]},
        "testable_alone": {"ligands": int(tbl["testable_alone"].sum()),
                           "list": tbl.loc[tbl["testable_alone"], "ligand_group"].tolist()},
        "power_curve": {name: {"ligands_used": res["ligands_used"],
                               "complexes_used": res["complexes_used"],
                               "min_attainable_p": res["min_attainable_p"],
                               "power": res["power"], **mde[name]}
                        for name, res in pooled.items()},
        "paralog_pairs": {"total": len(pairs),
                          "same_family": int((pairs["relation"] == "family").sum()),
                          "same_group_only": int((pairs["relation"] == "group").sum())},
    }
    (a.out / "design_a_feasibility.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")

    pd.set_option("display.width", 200)
    print(f"\nDesign A feasibility: {len(tbl)} ligands, {int(tbl['n_complexes'].sum())} "
          f"complexes, {points['uniprot'].nunique()} proteins")
    print(tbl[["ligand_group", "role", "n_complexes", "n_proteins", "range_pKd",
               "iqr_pKd", "min_attainable_p", "testable_alone",
               *[f"power_ratio{r:g}" for r in headline]]].to_string(index=False))
    print(f"\nroles: {len(rank)} ranking ({int(rank['n_complexes'].sum())} complexes), "
          f"{len(neg)} negative_control ({int(neg['n_complexes'].sum())} complexes: "
          f"{', '.join(neg['ligand_group'])})")
    print(f"testable alone (n large enough for p <= {ALPHA}): "
          f"{summary['testable_alone']['ligands']} ligands "
          f"({', '.join(summary['testable_alone']['list'])})")

    print("\npower curve (effect size = slope / noise sd):")
    wide = curve.pivot_table(index=["role", "scope", "n_complexes"], columns="ratio",
                             values="power").reset_index()
    wide = wide.sort_values(["role", "n_complexes", "scope"],
                            ascending=[True, False, True])
    print(wide.to_string(index=False))
    for name, m in mde.items():
        print(f"\n{name} ({pooled[name]['complexes_used']} complexes): smallest effect "
              f"with power >= {TARGET_POWER:.0%} is ratio "
              f"{m['min_detectable_ratio']} (power {m['power_at_min_detectable']}); "
              f"at ratio {m['largest_ratio_below_target']} power is {m['power_there']}")

    print(f"\nparalog pairs: {len(pairs)} "
          f"({summary['paralog_pairs']['same_family']} same family, "
          f"{summary['paralog_pairs']['same_group_only']} same group only)")
    print(pairs[pairs["relation"] == "family"].to_string(index=False))
    print(f"\nwrote {a.out}/design_a_feasibility.csv, design_a_power_curve.csv, "
          f"design_a_points.csv, design_a_pairs.csv, design_a_feasibility.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
