"""Extend the Davis primary set with a BindingDB replication set, and redo the power analysis.

scripts/design_a_feasibility.py measured whether Design A (ranking kinases within one
ligand by affinity) is feasible on Davis alone: 20 ligands, 96 complexes, and a pooled
minimum detectable effect around slope/noise 0.55-0.6. This script asks how much the
BindingDB ligands that Davis never measured move those numbers. It computes no
frustration and fits no model; the test statistic, permutation method and role rule are
reused from design_a_feasibility.

    .venv/bin/python scripts/extended_set.py

Input (no network, nothing refetched):
    data/kinome/davis/activities.csv        the primary source (one assay, one measure)
    data/kinome/bindingdb/activities.csv    the replication source
    data/kinome/bindingdb/ligand_ids.csv    ligand_code -> ligand_group
    data/kinome/klifs/klifs_manifest.csv    the structures (pdb, chain, ligand code)

Output, in data/kinome/analysis/:
    extended_set.csv            one row per complex
    extended_set_ligands.csv    one row per ligand
    extended_set_power.csv      the power curve, per ligand and per pooled scope
    extended_set.json           assumptions, funnel counts, summary

Rules that decide what enters, each of which the report states the cost of:

  - A ligand takes ALL its values from ONE source. A ligand Davis measured uses Davis,
    never a mixture: the two sources differ in assay, in aggregation and in how many
    reports back a value, and mixing them inside a ligand would put that difference
    inside the very comparison Design A makes.
  - On the BindingDB side IC50 is not used at all (a functional readout, not a binding
    constant), and a ligand uses either Kd or Ki, never both -- chosen as whichever
    covers more proteins, Kd on a tie. The alternative's protein count is reported.
  - Aggregated BindingDB values flagged `inconsistent` (spread > 1 log unit over the
    reports behind them) are dropped.
  - Censored values never enter a median: aggregate() in bindingdb_activity.py medians
    uncensored values only, so a value exists exactly when n >= 1. The script asserts
    this rather than assuming it, and counts what censoring removes.
  - A ligand wide enough to rank whose window rests on single-report values at BOTH
    ends becomes `role = "exploratory"`: still in the set, still run, but not evidence
    for the ranking hypothesis (see assign_roles).
  - `single_measurement` marks a BindingDB value aggregated from exactly ONE report.
    These are kept, flagged, and every power figure is computed twice: with and without
    them. Davis values are NOT flagged -- each is one KINOMEscan Kd by construction, and
    Davis is the designated primary source -- so the flag measures how much the
    EXTENSION leans on thinly supported values.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_a_feasibility as da  # noqa: E402

RATIOS = (0.1, 0.2, 0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0)
MEASURE_PRIORITY = ("Kd", "Ki")          # tie-break when both cover the same proteins
SEED = 20261006


# --- assembling the two sources ----------------------------------------------------

def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def davis_side(davis: pd.DataFrame) -> pd.DataFrame:
    """Davis points, exactly as design_a_feasibility selects them: wild type, domain
    known from the measurement, ligand matched to a manifest group, uncensored, and a
    co-crystal structure for that (kinase domain, ligand)."""
    pts = da.davis_points(davis)
    pts = da.comparable(pts)
    pts["measure"] = "Kd"
    pts["source"] = "davis"
    pts["single_measurement"] = False     # see the module docstring
    pts["n_reports"] = 1
    return pts


def bindingdb_side(bdb: pd.DataFrame, exclude: set[str]) -> tuple[pd.DataFrame, dict]:
    """BindingDB points for ligands NOT in `exclude` (the Davis ligands).

    Returns (points, funnel). Filters, in order, each counted in the funnel: primary
    rows only (domain known from the measurement), main match levels, binding constants
    (no IC50), a value present (n >= 1, which is where censored-only rows go), not
    `inconsistent`, a structure for that (kinase domain, ligand), one measure per ligand,
    and at least two distinct proteins.
    """
    b = bdb.copy()
    for c in ("median_pX", "n", "n_censored"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    for c in ("primary_eligible", "inconsistent", "has_structure"):
        b[c] = _truthy(b[c])
    b = b[b["klifs_kinase_id"].astype(str).str.strip() != ""]
    b["klifs_kinase_id"] = b["klifs_kinase_id"].astype(float).astype(int)

    # Censored values must never have reached a median: a value exists iff n >= 1.
    bad = b[(b["n"] == 0) & b["median_pX"].notna()]
    if len(bad):
        raise AssertionError(f"{len(bad)} aggregated rows have n == 0 but a median_pX; "
                             "censored values would be entering the medians")

    funnel = [{"step": "bindingdb activities.csv", "rows": len(b),
               "ligands": int(b["ligand_group"].nunique())}]

    def step(name, d):
        funnel.append({"step": name, "rows": len(d),
                       "ligands": int(d["ligand_group"].nunique())})
        return d

    b = step("ligand not measured by Davis", b[~b["ligand_group"].isin(exclude)])
    b = step("primary_eligible", b[b["primary_eligible"]])
    b = step("match level full / no_stereo", b[b["match_level"].isin(da.dav.MAIN_LEVELS)])
    n_ic50 = int((b["measure"] == "IC50").sum())
    b = step("measure is Kd or Ki (IC50 dropped)",
             b[b["measure"].isin(da.dav.bdb.BINDING_CONSTANTS)])
    censored_only = int(((b["n"] == 0) & (b["n_censored"] > 0)).sum())
    b = step("uncensored value present (n >= 1)", b[(b["n"] >= 1) & b["median_pX"].notna()])
    n_inconsistent = int(b["inconsistent"].sum())
    b = step("not inconsistent (std <= 1 log)", b[~b["inconsistent"]])
    b = step("structure for this (kinase domain, ligand)", b[b["has_structure"]])

    # one measure per ligand: whichever covers more proteins, Kd on a tie
    cover = (b.groupby(["ligand_group", "measure"])["uniprot"].nunique()
             .rename("proteins").reset_index())
    cover["rank"] = cover["measure"].map({m: i for i, m in enumerate(MEASURE_PRIORITY)})
    chosen = (cover.sort_values(["ligand_group", "proteins", "rank"],
                                ascending=[True, False, True])
              .drop_duplicates("ligand_group"))
    measure_choice = cover.pivot(index="ligand_group", columns="measure",
                                 values="proteins").reindex(
        columns=list(MEASURE_PRIORITY)).fillna(0).astype(int)
    measure_choice["chosen"] = chosen.set_index("ligand_group")["measure"]
    b = step("one measure per ligand",
             b.merge(chosen[["ligand_group", "measure"]], on=["ligand_group", "measure"]))

    # One row per (ligand, kinase domain). Aggregation keys that are not the complex
    # (match level, InChIKey, domain_source) can leave two rows for one complex; keeping
    # both would count that complex twice. Prefer the full InChIKey match, then the value
    # backed by more reports.
    key = ["ligand_group", "klifs_kinase_id"]
    n_dup = int(b.duplicated(key).sum())
    b = step("one row per (ligand, kinase domain)",
             b.assign(_lv=b["match_level"].map(da.dav.bdb.LEVEL_RANK))
             .sort_values(["_lv", "n"], ascending=[True, False])
             .drop_duplicates(key).drop(columns="_lv"))
    b = step(">= 2 distinct proteins",
             b.groupby("ligand_group").filter(lambda d: d["uniprot"].nunique() >= 2))

    pts = pd.DataFrame({
        "ligand_group": b["ligand_group"], "uniprot": b["uniprot"],
        "klifs_kinase_id": b["klifs_kinase_id"], "kinase_name": b["kinase_name"],
        "pKd": b["median_pX"], "measure": b["measure"], "source": "bindingdb",
        "single_measurement": b["n"] == 1, "n_reports": b["n"].astype(int)})
    stats = {"funnel": funnel, "ic50_rows_dropped": n_ic50,
             "censored_only_rows_dropped": censored_only,
             "inconsistent_rows_dropped": n_inconsistent,
             "duplicate_complex_rows_collapsed": n_dup,
             "measure_choice": measure_choice.reset_index().to_dict("records")}
    return pts.sort_values(["ligand_group", "pKd"], ascending=[True, False]), stats


def attach_structures(points: pd.DataFrame, manifest: pd.DataFrame,
                      code_to_group: dict) -> pd.DataFrame:
    """Add the manifest's pdb / chain / ligand_code for each (kinase domain, ligand).
    A ligand group with two codes can have two structures on one kinase; both are listed
    and n_structures says so, rather than one being picked silently here."""
    m = manifest.copy()
    m["klifs_kinase_id"] = m["klifs_kinase_id"].astype(int)
    m["ligand_group"] = m["ligand_code"].map(code_to_group)
    agg = (m.groupby(["klifs_kinase_id", "ligand_group"])
           .agg(pdb=("pdb", lambda s: ";".join(s)), chain=("chain", lambda s: ";".join(s)),
                ligand_code=("ligand_code", lambda s: ";".join(s)),
                n_structures=("pdb", "size")))
    return points.join(agg, on=["klifs_kinase_id", "ligand_group"])


# --- power (same statistic and null as design_a_feasibility, vectorised) -----------

def assign_roles(tbl: pd.DataFrame) -> pd.Series:
    """Each ligand's role in Design A, in two steps.

    First design_a_feasibility's rule: an affinity window of at least
    MIN_RANGE_FOR_RANKING log units -> "ranking", narrower -> "negative_control" (those
    kinases are effectively equipotent, so they test the opposite hypothesis).

    Then a ranking candidate whose window rests on single-report values at BOTH ends --
    its highest and its lowest value are each `single_measurement` -- is demoted to
    "exploratory": the range that qualified it could be measurement noise, so it must not
    carry the primary ranking test. Such ligands stay in the set and frustx still runs
    their complexes; they are simply not evidence for the ranking hypothesis. The rule is
    general; no ligand is named anywhere.
    """
    role = pd.Series(da.assign_roles(tbl), index=tbl.index)
    role[(role == "ranking") & tbl["range_from_single"].astype(bool)] = "exploratory"
    return role


def _row_ranks(y: np.ndarray) -> np.ndarray:
    """Ranks along axis 1. Continuous simulated values have no ties with probability 1."""
    order = np.argsort(y, axis=1, kind="stable")
    out = np.empty(order.shape, dtype=float)
    np.put_along_axis(out, order, np.arange(1, y.shape[1] + 1, dtype=float), axis=1)
    return out


def ligand_power(pkd: np.ndarray, ratios, n_sims: int, null: np.ndarray,
                 nprng: np.random.Generator) -> dict:
    """Power of the within-ligand Spearman test at each effect size.

    Model and test are design_a_feasibility.simulate_power's: frustration = ratio * pKd +
    N(0, 1), two-sided permutation test on Spearman rho at design_a's ALPHA, against the
    precomputed exact/sampled `null`. Vectorised over simulations; rho is Pearson on
    ranks, with average ranks on the x side so ties in pKd count correctly.
    """
    p_min = float((null >= null.max() - 1e-12).mean())
    if p_min > da.ALPHA:           # no arrangement of this n could ever be significant
        return {"min_attainable_p": round(p_min, 4), "attainable": False,
                "power": {r: 0.0 for r in ratios}, "median_rho": {r: None for r in ratios}}
    rx = da._rank(pkd) - da._rank(pkd).mean()
    ssx = float((rx ** 2).sum())
    nullsort = np.sort(null)
    power, med = {}, {}
    for r in ratios:
        y = r * pkd + nprng.normal(0.0, 1.0, size=(n_sims, len(pkd)))
        ry = _row_ranks(y)
        ry -= ry.mean(axis=1, keepdims=True)
        rho = (ry @ rx) / np.sqrt(ssx * (ry ** 2).sum(axis=1))
        p = 1.0 - np.searchsorted(nullsort, np.abs(rho) - 1e-12, side="left") / len(nullsort)
        power[r] = round(float((p <= da.ALPHA).mean()), 3)
        med[r] = round(float(np.median(rho)), 3)
    return {"min_attainable_p": round(p_min, 4), "attainable": True,
            "power": power, "median_rho": med}


def _pooled_stat(groups, xcs, ssx, ratio, n_sims, nprng) -> np.ndarray:
    """|statistic| for n_sims draws: Pearson on within-ligand centred ranks, the
    fixed-effect slope of frustration ~ pKd + (1|ligand) without fitting the model."""
    num = np.zeros(n_sims)
    ssy = np.zeros(n_sims)
    for g, xc in zip(groups, xcs):
        y = ratio * g + nprng.normal(0.0, 1.0, size=(n_sims, len(g)))
        ry = _row_ranks(y)
        ry -= ry.mean(axis=1, keepdims=True)
        num += ry @ xc
        ssy += (ry ** 2).sum(axis=1)
    return np.abs(num / np.sqrt(ssx * ssy))


def pooled_power(groups: list[np.ndarray], ratios, n_sims: int,
                 nprng: np.random.Generator) -> dict:
    """Pooled power at each effect size.

    The permutation null is the same computation at ratio 0: with no signal the ranks of
    y inside a ligand are a uniformly random permutation of that ligand's ranks, which is
    exactly design_a_feasibility.pooled_power's within-ligand permutation null.

    Significance is a permutation p-value, P(null >= |stat|) <= ALPHA, NOT a comparison
    against the null's (1 - ALPHA) quantile. The two agree when the null is continuous,
    but the quantile route reports power 1.0 on a degenerate null -- a scope of one
    two-complex ligand always gives |stat| = 1, so the quantile is 1 and every draw
    "beats" it, while the real p-value is 1 and nothing can be significant.
    `min_attainable_p` says whether the scope can reach ALPHA at all.
    """
    gs = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2 and np.ptp(g) > 0]
    if not gs:
        return {"ligands_used": 0, "complexes_used": 0, "min_attainable_p": None,
                "power": {r: 0.0 for r in ratios}}
    xcs = [da._rank(g) - da._rank(g).mean() for g in gs]
    ssx = float(sum((xc ** 2).sum() for xc in xcs))
    null = np.sort(_pooled_stat(gs, xcs, ssx, 0.0, da.N_PERMS, nprng))
    p_of = lambda obs: 1.0 - np.searchsorted(  # noqa: E731
        null, obs - 1e-12, side="left") / len(null)
    p_min = float(p_of(np.array([null[-1]]))[0])
    power = {}
    for r in ratios:
        obs = _pooled_stat(gs, xcs, ssx, r, n_sims, nprng)
        power[r] = round(float((p_of(obs) <= da.ALPHA).mean()), 3)
    return {"ligands_used": len(gs), "complexes_used": int(sum(len(g) for g in gs)),
            "min_attainable_p": round(p_min, 4), "power": power}


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kin = Path("data/kinome")
    p.add_argument("--davis", type=Path, default=kin / "davis/activities.csv")
    p.add_argument("--bindingdb", type=Path, default=kin / "bindingdb/activities.csv")
    p.add_argument("--ligand-ids", type=Path, default=kin / "bindingdb/ligand_ids.csv")
    p.add_argument("--manifest", type=Path, default=kin / "klifs/klifs_manifest.csv")
    p.add_argument("--out", type=Path, default=kin / "analysis")
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS), metavar="R")
    p.add_argument("--sims", type=int, default=da.N_SIMS)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    ratios = sorted(a.ratios)

    rd = lambda f: pd.read_csv(f, keep_default_na=False, dtype=str)  # noqa: E731
    davis = pd.read_csv(a.davis, keep_default_na=False)
    for c in ("median_pX", "n"):
        davis[c] = pd.to_numeric(davis[c], errors="coerce")
    for c in ("primary_eligible", "has_structure"):
        davis[c] = _truthy(davis[c])
    bdb, manifest, lig_ids = rd(a.bindingdb), rd(a.manifest), rd(a.ligand_ids)
    code_to_group = dict(zip(lig_ids["ligand_code"], lig_ids["ligand_group"]))

    dav_pts = davis_side(davis)
    davis_ligands = set(dav_pts["ligand_group"])
    bdb_pts, bstats = bindingdb_side(bdb, davis_ligands)
    points = pd.concat([dav_pts, bdb_pts], ignore_index=True)

    # A ligand must never draw on both sources.
    mixed = (points.groupby("ligand_group")["source"].nunique() > 1)
    mixed_ligands = sorted(mixed[mixed].index)
    mixed_measure = points.groupby("ligand_group")["measure"].nunique()
    mixed_measure = sorted(mixed_measure[mixed_measure > 1].index)

    points = attach_structures(points, manifest, code_to_group)
    tbl = da.spread(points)
    src = points.groupby("ligand_group").agg(
        source=("source", lambda s: "/".join(sorted(set(s)))),
        measure=("measure", lambda s: "/".join(sorted(set(s)))),
        n_single=("single_measurement", "sum"))
    tbl = tbl.join(src, on="ligand_group")

    # Does the ligand's window rest on single-measurement values?
    def single_range(d):
        hi, lo = d.loc[d["pKd"].idxmax()], d.loc[d["pKd"].idxmin()]
        rest = d[~d["single_measurement"]]
        return pd.Series({
            "range_from_single": bool(hi["single_measurement"] and lo["single_measurement"]),
            "range_endpoint_single": bool(hi["single_measurement"] or lo["single_measurement"]),
            "n_excl_single": len(rest),
            "range_excl_single": round(float(rest["pKd"].max() - rest["pKd"].min()), 3)
                                 if len(rest) >= 2 else None})
    tbl = tbl.join(points.groupby("ligand_group").apply(single_range, include_groups=False),
                   on="ligand_group")
    # Roles need range_from_single, so they are assigned after it (see assign_roles).
    tbl["role"] = assign_roles(tbl)
    # Kept as the record of which ligands the single-measurement rule caught.
    tbl["needs_decision"] = tbl["role"] == "exploratory"

    points = points.join(tbl.set_index("ligand_group")["role"], on="ligand_group")

    # --- power, with and without the single-measurement values
    rng, nprng = random.Random(SEED), np.random.default_rng(SEED)
    roles = dict(zip(tbl["ligand_group"], tbl["role"]))
    srcs = dict(zip(tbl["ligand_group"], tbl["source"]))
    curve, pooled, mde, scope_names = [], {}, {}, []
    for variant, sel in (("with_single", points),
                         ("without_single", points[~points["single_measurement"]])):
        groups = {lig: d["pKd"].to_numpy() for lig, d in sel.groupby("ligand_group")
                  if len(d) >= 2}
        for lig, v in groups.items():
            res = ligand_power(v, ratios, a.sims, da.permutation_null(v, rng), nprng)
            curve += [{"variant": variant, "scope": lig, "kind": "ligand",
                       "role": roles[lig], "source": srcs[lig], "n_complexes": len(v),
                       "ratio": r, "power": res["power"][r],
                       "min_attainable_p": res["min_attainable_p"],
                       "median_rho": res["median_rho"][r]} for r in ratios]
        pick = lambda f: [v for lig, v in groups.items() if f(lig)]  # noqa: E731
        scopes = {
            "davis_only": pick(lambda l: srcs[l] == "davis"),
            "extended_all": list(groups.values()),
            "extended_ranking": pick(lambda l: roles[l] == "ranking"),
            "extended_negative_control": pick(lambda l: roles[l] == "negative_control"),
            "davis_ranking": pick(lambda l: srcs[l] == "davis" and roles[l] == "ranking"),
            "bindingdb_ranking": pick(lambda l: srcs[l] == "bindingdb"
                                      and roles[l] == "ranking"),
            "extended_exploratory": pick(lambda l: roles[l] == "exploratory"),
        }
        scope_names = list(scopes)
        for name, gs in scopes.items():
            res = pooled_power(gs, ratios, a.sims, nprng)
            pooled[(variant, name)] = res
            mde[(variant, name)] = da.min_detectable_ratio(res["power"])
            curve += [{"variant": variant, "scope": name, "kind": "pooled", "role": "pooled",
                       "source": "", "n_complexes": res["complexes_used"], "ratio": r,
                       "power": pw, "min_attainable_p": None, "median_rho": None}
                      for r, pw in res["power"].items()]
    curve = pd.DataFrame(curve).sort_values(
        ["variant", "kind", "n_complexes", "scope", "ratio"],
        ascending=[True, True, False, True, True])
    curve.to_csv(a.out / "extended_set_power.csv", index=False)

    pcols = ["ligand_group", "uniprot", "klifs_kinase_id", "kinase_name", "pdb", "chain",
             "ligand_code", "n_structures", "pKd", "measure", "source",
             "single_measurement", "n_reports", "role"]
    points = points.sort_values(["role", "ligand_group", "pKd"],
                                ascending=[True, True, False])[pcols]
    points.to_csv(a.out / "extended_set.csv", index=False)
    lcols = ["ligand_group", "role", "source", "measure", "n_complexes", "n_proteins",
             "min_pKd", "max_pKd", "range_pKd", "iqr_pKd", "std_pKd", "n_single",
             "range_from_single", "range_endpoint_single", "n_excl_single",
             "range_excl_single", "needs_decision", "min_attainable_p", "proteins"]
    lig_power = {lig: g for lig, g in curve[(curve["variant"] == "with_single")
                                            & (curve["kind"] == "ligand")]
                 .groupby("scope")["min_attainable_p"].first().items()}
    tbl["min_attainable_p"] = tbl["ligand_group"].map(lig_power)
    tbl = tbl.sort_values(["role", "source", "n_complexes", "range_pKd"],
                          ascending=[True, True, False, False])[lcols]
    tbl.to_csv(a.out / "extended_set_ligands.csv", index=False)

    by_src = points.groupby("source")["pKd"]
    summary = {
        "inputs": {k: str(v) for k, v in (("davis", a.davis), ("bindingdb", a.bindingdb),
                                          ("manifest", a.manifest),
                                          ("ligand_ids", a.ligand_ids))},
        "scope": {
            "davis": {"ligands": int((tbl["source"] == "davis").sum()),
                      "complexes": int(tbl.loc[tbl["source"] == "davis",
                                               "n_complexes"].sum())},
            "bindingdb_added": {"ligands": int((tbl["source"] == "bindingdb").sum()),
                                "complexes": int(tbl.loc[tbl["source"] == "bindingdb",
                                                         "n_complexes"].sum())},
            "extended": {"ligands": len(tbl), "complexes": int(tbl["n_complexes"].sum()),
                         "proteins": int(points["uniprot"].nunique())}},
        "roles": {r: {"ligands": int((tbl["role"] == r).sum()),
                      "complexes": int(tbl.loc[tbl["role"] == r, "n_complexes"].sum()),
                      "list": tbl.loc[tbl["role"] == r, "ligand_group"].tolist()}
                  for r in ("ranking", "exploratory", "negative_control")},
        "role_rule": {
            "ranking": f"affinity range >= {da.MIN_RANGE_FOR_RANKING} log unit",
            "negative_control": "narrower than that",
            "exploratory": "a ranking candidate whose highest AND lowest value are both "
                           "single-report (range_from_single): the window may be "
                           "measurement noise, so it is excluded from the primary "
                           "ranking test but kept in the set"},
        "bindingdb_filters": bstats,
        "checks": {"ligands_drawing_on_both_sources": mixed_ligands,
                   "ligands_mixing_two_measures": mixed_measure,
                   "censored_values_in_a_median": 0,
                   "complexes_with_several_structures":
                       int((points["n_structures"] > 1).sum())},
        "single_measurement": {
            "values": int(points["single_measurement"].sum()),
            "of_which_bindingdb": int(points.loc[points["source"] == "bindingdb",
                                                 "single_measurement"].sum()),
            "ligands_range_from_single": tbl.loc[tbl["range_from_single"],
                                                 "ligand_group"].tolist(),
            "ligands_needing_a_decision": tbl.loc[tbl["needs_decision"],
                                                  "ligand_group"].tolist(),
            "note": "Davis values are never flagged: each is one KINOMEscan Kd and Davis "
                    "is the primary source, so the flag measures how much the BindingDB "
                    "extension leans on values backed by a single report"},
        "pkd_by_source": {s: {"n": int(len(g)), "median": round(float(g.median()), 3),
                              "mean": round(float(g.mean()), 3),
                              "std": round(float(g.std(ddof=1)), 3),
                              "min": round(float(g.min()), 3),
                              "max": round(float(g.max()), 3),
                              "q25": round(float(g.quantile(0.25)), 3),
                              "q75": round(float(g.quantile(0.75)), 3)}
                          for s, g in [(s, by_src.get_group(s)) for s in by_src.groups]},
        "within_ligand_range_by_source": {
            s: {"ligands": int((tbl["source"] == s).sum()),
                "median_range": round(float(tbl.loc[tbl["source"] == s,
                                                    "range_pKd"].median()), 3)}
            for s in sorted(tbl["source"].unique())},
        "power": {v: {name: {"ligands_used": pooled[(v, name)]["ligands_used"],
                             "complexes_used": pooled[(v, name)]["complexes_used"],
                             "min_attainable_p": pooled[(v, name)]["min_attainable_p"],
                             "power": pooled[(v, name)]["power"], **mde[(v, name)]}
                      for name in scope_names}
                  for v in ("with_single", "without_single")},
    }
    summary["assumptions"] = {
        "model": "frustration_i = ratio * pKd_i + N(0, 1), independent per complex",
        "effect_size": "ratio = slope / noise_sd in log units of the affinity scale; only "
                       "the ratio matters, so the frustration index's own scale is irrelevant",
        "test": f"two-sided permutation test on Spearman rho, alpha = {da.ALPHA}; exact "
                f"enumeration for n <= {da.EXACT_MAX_N}, else {da.N_PERMS} samples",
        "pooled": "Pearson on within-ligand centred ranks, null = the same statistic at "
                  "ratio 0 (within-ligand permutation), as in design_a_feasibility",
        "ignored": ["affinity measurement error on both sources (values taken as exact)",
                    "that the two sources' values are comparable in kind beyond being "
                    "binding constants",
                    "censored values (excluded, not set to a bound)",
                    "that one structure represents the complex"],
        "ratios": ratios, "sims": a.sims, "seed": SEED,
    }
    (a.out / "extended_set.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    # --- console
    pd.set_option("display.width", 220)
    sc = summary["scope"]
    print(f"\nDavis: {sc['davis']['ligands']} ligands / {sc['davis']['complexes']} complexes"
          f"  +  BindingDB: {sc['bindingdb_added']['ligands']} / "
          f"{sc['bindingdb_added']['complexes']}"
          f"  =  extended: {sc['extended']['ligands']} ligands / "
          f"{sc['extended']['complexes']} complexes / {sc['extended']['proteins']} proteins")
    for r, d in summary["roles"].items():
        print(f"  {r}: {d['ligands']} ligands, {d['complexes']} complexes")
    print("\nper-ligand (extended set):")
    show = ["ligand_group", "role", "source", "measure", "n_complexes", "n_proteins",
            "range_pKd", "n_single", "range_excl_single", "needs_decision",
            "min_attainable_p"]
    print(tbl[show].to_string(index=False))

    print("\npooled power (effect size = slope / noise sd):")
    pw = curve[curve["kind"] == "pooled"].pivot_table(
        index=["variant", "scope", "n_complexes"], columns="ratio", values="power")
    print(pw.to_string())
    print(f"\nsmallest effect reaching {da.TARGET_POWER:.0%} power:")
    for v in ("with_single", "without_single"):
        for name in scope_names:
            m = mde[(v, name)]
            print(f"  {v:15} {name:38} n={pooled[(v, name)]['complexes_used']:4}  "
                  f"ratio {str(m['min_detectable_ratio']):5} "
                  f"(power {m['power_at_min_detectable']}); "
                  f"below it {m['largest_ratio_below_target']} -> {m['power_there']}")

    print("\npKd by source:")
    print(pd.DataFrame(summary["pkd_by_source"]).T.to_string())
    print("\nchecks:", json.dumps(summary["checks"]))
    if summary["single_measurement"]["ligands_needing_a_decision"]:
        print("NEEDS A DECISION (range rests on single-measurement values, currently in "
              "the ranking set):",
              ", ".join(summary["single_measurement"]["ligands_needing_a_decision"]))
    print(f"\nwrote {a.out}/extended_set.csv, extended_set_ligands.csv, "
          f"extended_set_power.csv, extended_set.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
