"""Does ligand-contact frustration follow measured affinity across one ligand's kinases?

The analysis of the KLIFS-first complex set (scripts/run_complex_set.sh, 91 complexes,
8 ligands, 1000 decoys, relax). For each ligand, its co-crystal complexes with several
kinases are compared against the measured affinities in DRUG_KINASE_COMPLEXES.csv.

    .venv/bin/python scripts/complex_set_compare.py
    .venv/bin/python scripts/plot_complex_set.py

THE RULES BELOW WERE FIXED BEFORE ANY ASSOCIATION WAS COMPUTED (2026-10-25). Only
descriptive numbers had been looked at: ligand contacts per complex 20-37, 19% of the
2,495 ligand contacts with |e_ij| < 0.1 kcal/mol, 10% with decoy sigma <= 0.02.

  Unit       one complex = one ligand bound to one kinase domain (one crystal).
  Contacts   the complex's protein-ligand contacts (frustx's ligand cutoff, 6 A minimum
             heavy-atom distance). Contacts whose index is undefined (decoy sigma 0,
             class "undefined", 0.2%) are dropped and counted.
  Measure    Kd is primary (rows with in_Kd_set); IC50 is secondary (in_IC50_set). They
             are never pooled. Ki is not analysed.
  Hypothesis a tighter binder's contacts are less frustrated: pKd up -> mean frustration
             index up (frustratometeR sign: high positive = minimally frustrated). Tests
             are two-sided; the sign is reported.
  Primary    mean_F: mean frustration index over the ligand's contacts.
  Secondary  frac_minimally, frac_highly (classes at frustx's 0.78 / -1.0 cut points).
  Controls   n_contacts (pocket size) and interaction (= -sum of native e_ij over the
             ligand's contacts, so larger = stronger): Rosetta's interaction energy needs
             no decoys, and in the COX series it explained the frustration signal away.
             A frustration result counts only if it survives both.
  Floors     0.0, 0.1, 0.3, 0.5 kcal/mol on |native e_ij|: descriptors are recomputed
             on the contacts at or above each floor (pilot_compare.py's sweep). A floor is
             not chosen; a result has to hold at EVERY floor.
  Per ligand Spearman rho(pKd, descriptor), permutation p (exact for n <= 7).
  Pooled     Pearson correlation of within-ligand centred ranks (the fixed-effect slope of
             descriptor ~ pKd + (1 | ligand), as in design_a_feasibility.py); null:
             pKd ranks permuted inside each ligand, N_PERMS times.
  Partial    the same pooled statistic after regressing the descriptor's centred ranks
             on the centred ranks of n_contacts and interaction (OLS, no intercept):
             what frustration says beyond pocket size and raw energy.
  Robustness the pooled tests without STU, which holds 35 of the 79 Kd complexes.
  Verdict    a descriptor SUPPORTS the hypothesis only if, for Kd, the pooled AND the
             partial p are < 0.05 at all four floors, with the same sign. Everything else
             is reported as not supported, whatever single cell looks significant.

Output, under results/complex_set_analysis/:
  descriptors.csv   one row per (complex, floor): descriptors, controls, pKd, pIC50
  per_ligand.csv    rho and p per (measure, floor, ligand, descriptor)
  pooled.csv        pooled and partial statistics per (measure, scope, floor, descriptor)
  positions.csv     per (complex, KLIFS pocket position): the ligand contact's F and e_ij
  verdict.json      the rule above applied, plus counts and settings
"""

import argparse
import itertools
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_a_feasibility import _rank, _rho  # noqa: E402

FLOORS = (0.0, 0.1, 0.3, 0.5)
DESCRIPTORS = ("mean_F", "frac_minimally", "frac_highly")
CONTROLS = ("n_contacts", "interaction")
MEASURES = {"Kd": ("pKd", "in_Kd_set"), "IC50": ("pIC50", "in_IC50_set")}
ALPHA = 0.05
EXACT_MAX_N = 7
N_PERMS = 10000
SEED = 20261025


# --- per-complex descriptors (pure) ---------------------------------------------------

def ligand_contacts(contacts: pd.DataFrame, name3: str) -> pd.DataFrame:
    """The protein-ligand contacts of one run, with the protein side made explicit. The
    ligand is usually the j side but not always, so both are tested."""
    on_j = contacts["resname_j"] == name3
    on_i = contacts["resname_i"] == name3
    lig = contacts[on_i | on_j].copy()
    lig["protein_resnum"] = np.where(on_j[lig.index], lig["resnum_i"], lig["resnum_j"])
    lig["protein_resname"] = np.where(on_j[lig.index], lig["resname_i"], lig["resname_j"])
    return lig


def descriptors(lig: pd.DataFrame, floor: float) -> dict:
    """Interface descriptors over the contacts with |native e_ij| >= floor."""
    d = lig[lig["frustration_index"].notna()]
    d = d[d["native_energy"].abs() >= floor]
    n = len(d)
    if n == 0:
        return {"n_contacts": 0, **{k: np.nan for k in DESCRIPTORS}, "interaction": np.nan}
    cls = d["frustration_class"]
    return {"n_contacts": n,
            "mean_F": d["frustration_index"].mean(),
            "frac_minimally": (cls == "minimally").mean(),
            "frac_highly": (cls == "highly").mean(),
            "interaction": -d["native_energy"].sum()}


# --- statistics (pure) ----------------------------------------------------------------

def spearman_test(x: np.ndarray, y: np.ndarray, rng: random.Random) -> dict:
    """Spearman rho and a two-sided permutation p (exact for n <= EXACT_MAX_N), plus the
    smallest p this n could ever produce."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": n, "rho": np.nan, "p": np.nan, "min_p": np.nan}
    rx, ry = _rank(x), _rank(y)
    obs = _rho(rx, ry)
    perms = (itertools.permutations(range(n)) if n <= EXACT_MAX_N
             else (rng.sample(range(n), n) for _ in range(N_PERMS)))
    null = np.array([abs(_rho(rx[list(p)], ry)) for p in perms])
    return {"n": n, "rho": obs, "p": float((null >= abs(obs) - 1e-12).mean()),
            "min_p": float((null >= null.max() - 1e-12).mean())}


def _centred_ranks(df: pd.DataFrame, col: str) -> np.ndarray:
    return df.groupby("ligand_code")[col].transform(
        lambda s: s.rank() - s.rank().mean()).to_numpy()


def pooled_test(df: pd.DataFrame, x: str, y: str, controls: tuple, nprng) -> dict:
    """Pooled within-ligand rank correlation of y with x, optionally after removing the
    controls from y. x (the affinity) is permuted inside each ligand for the null."""
    d = df.dropna(subset=[x, y, *controls])
    d = d[d.groupby("ligand_code")[x].transform("size") >= 3]
    if d.empty:
        return {"n": 0, "ligands": 0, "stat": np.nan, "p": np.nan}
    xc = _centred_ranks(d, x)
    yc = _centred_ranks(d, y)
    if controls:
        Z = np.column_stack([_centred_ranks(d, c) for c in controls])
        beta, *_ = np.linalg.lstsq(Z, yc, rcond=None)
        yc = yc - Z @ beta

    def stat(xv):
        den = math.sqrt(float((xv ** 2).sum() * (yc ** 2).sum()))
        return float((xv * yc).sum() / den) if den else 0.0

    obs = stat(xc)
    groups = [np.flatnonzero((d["ligand_code"] == g).to_numpy())
              for g in d["ligand_code"].unique()]
    null = np.empty(N_PERMS)
    for k in range(N_PERMS):
        xp = xc.copy()
        for idx in groups:
            xp[idx] = xc[nprng.permutation(idx)]
        null[k] = abs(stat(xp))
    return {"n": len(d), "ligands": d["ligand_code"].nunique(), "stat": obs,
            "p": float((null >= abs(obs) - 1e-12).mean())}


def verdict(pooled: pd.DataFrame) -> dict:
    """The pre-registered rule, applied: Kd, all scope, every floor, pooled and partial
    both p < ALPHA with one sign."""
    out = {}
    k = pooled[(pooled["measure"] == "Kd") & (pooled["scope"] == "all")]
    for desc in DESCRIPTORS:
        d = k[k["descriptor"] == desc]
        raw, part = d[d["test"] == "pooled"], d[d["test"] == "partial"]
        signs = set(np.sign(pd.concat([raw["stat"], part["stat"]])).dropna())
        ok = (len(raw) == len(FLOORS) and len(part) == len(FLOORS)
              and (raw["p"] < ALPHA).all() and (part["p"] < ALPHA).all() and len(signs) == 1)
        out[desc] = {"supported": bool(ok),
                     "pooled_p_by_floor": dict(zip(raw["floor"], raw["p"].round(4))),
                     "partial_p_by_floor": dict(zip(part["floor"], part["p"].round(4))),
                     "pooled_stat_by_floor": dict(zip(raw["floor"], raw["stat"].round(3))),
                     "partial_stat_by_floor": dict(zip(part["floor"], part["stat"].round(3)))}
    return out


# --- driver ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", type=Path, default=Path("data/kinome/DRUG_KINASE_COMPLEXES.csv"))
    p.add_argument("--runs", type=Path, default=Path("data/kinome/complex_set/runs.csv"))
    p.add_argument("--pocket-map", type=Path,
                   default=Path("data/kinome/complex_set/pocket_map.csv"))
    p.add_argument("--results", type=Path, default=Path("results/complex_set"))
    p.add_argument("--out", type=Path, default=Path("results/complex_set_analysis"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rng, nprng = random.Random(SEED), np.random.default_rng(SEED)

    runs = pd.read_csv(a.runs)
    table = pd.read_csv(a.table)
    pmap = pd.read_csv(a.pocket_map, dtype={"xray_residue": str})
    aff = table.set_index(["ligand_code", "kinase_name"])[
        ["uniprot", "klifs_structure_id", "pKd", "pIC50", "in_Kd_set", "in_IC50_set"]]

    rows, pos_rows, undefined = [], [], 0
    for r in runs.itertuples():
        settings = json.loads((a.results / r.complex / "run.json").read_text())
        assert settings["n_decoys"] == 1000 and settings["protocol"] == "relax", r.complex
        lig = ligand_contacts(pd.read_csv(a.results / r.complex / "contacts.csv"),
                              r.ligand_name3)
        undefined += int(lig["frustration_index"].isna().sum())
        meta = {"complex": r.complex, "ligand_code": r.ligand_code,
                "kinase_name": r.kinase_name,
                **aff.loc[(r.ligand_code, r.kinase_name)].to_dict()}
        for f in FLOORS:
            rows.append({**meta, "floor": f, **descriptors(lig, f)})
        # KLIFS pocket position of each contacted residue, for the position-level view
        pm = pmap[(pmap["klifs_structure_id"] == meta["klifs_structure_id"])
                  & pmap["resolved"]].set_index("xray_residue")
        for c in lig.itertuples():
            key = str(c.protein_resnum)
            pos_rows.append({**{k: meta[k] for k in ("complex", "ligand_code",
                                                     "kinase_name", "pKd", "pIC50",
                                                     "in_Kd_set", "in_IC50_set")},
                             "resnum": c.protein_resnum, "resname": c.protein_resname,
                             "pocket_position": (int(pm.loc[key, "pocket_position"])
                                                 if key in pm.index else np.nan),
                             "klifs_position": (pm.loc[key, "klifs_position"]
                                                if key in pm.index else ""),
                             "native_energy": c.native_energy, "decoy_std": c.decoy_std,
                             "frustration_index": c.frustration_index,
                             "frustration_class": c.frustration_class})
    desc = pd.DataFrame(rows)
    desc.to_csv(a.out / "descriptors.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(a.out / "positions.csv", index=False)

    per_lig, pooled = [], []
    for m, (pcol, flag) in MEASURES.items():
        sub = desc[desc[flag]]
        for f in FLOORS:
            sf = sub[sub["floor"] == f]
            for lig_code, g in sf.groupby("ligand_code"):
                for d_ in DESCRIPTORS + CONTROLS:
                    per_lig.append({"measure": m, "floor": f, "ligand_code": lig_code,
                                    "descriptor": d_,
                                    **spearman_test(g[pcol].to_numpy(), g[d_].to_numpy(), rng)})
            for scope, s in (("all", sf), ("without_STU", sf[sf["ligand_code"] != "STU"])):
                for d_ in DESCRIPTORS + CONTROLS:
                    pooled.append({"measure": m, "scope": scope, "floor": f,
                                   "descriptor": d_, "test": "pooled",
                                   **pooled_test(s, pcol, d_, (), nprng)})
                for d_ in DESCRIPTORS:
                    pooled.append({"measure": m, "scope": scope, "floor": f,
                                   "descriptor": d_, "test": "partial",
                                   **pooled_test(s, pcol, d_, CONTROLS, nprng)})
    per_lig, pooled = pd.DataFrame(per_lig), pd.DataFrame(pooled)
    per_lig.to_csv(a.out / "per_ligand.csv", index=False)
    pooled.to_csv(a.out / "pooled.csv", index=False)

    v = verdict(pooled)
    (a.out / "verdict.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "Kd, all ligands: pooled AND partial (controls: n_contacts, interaction) "
                "p < 0.05 at every floor, one sign",
        "floors": list(FLOORS), "n_perms": N_PERMS, "seed": SEED,
        "complexes": int(desc["complex"].nunique()),
        "kd_complexes": int(desc[desc["in_Kd_set"] & (desc["floor"] == 0)].shape[0]),
        "ic50_complexes": int(desc[desc["in_IC50_set"] & (desc["floor"] == 0)].shape[0]),
        "undefined_contacts_dropped": undefined,
        "verdict": v}, indent=2, default=float) + "\n")

    pd.set_option("display.width", 200)
    show = pooled[(pooled["scope"] == "all")].pivot_table(
        index=["measure", "descriptor", "test"], columns="floor", values=["stat", "p"])
    print(show.round(3).to_string())
    print("\nverdict:", {k: x["supported"] for k, x in v.items()})


if __name__ == "__main__":
    main()
