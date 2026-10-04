#!/usr/bin/env python
"""Does ligand-interface frustration separate COX-2-selective inhibitors?

Reads results/cox_selectivity/selectivity_delta.csv (built by
scripts/cox_selectivity.py) and, for each COX2-minus-COX1 descriptor, reports:

  AUC        rank-based, P(Selective ranks above Non-Selective). 0.5 = nothing.
  p_perm     label-permutation test on the AUC. No distributional assumption,
             which matters at n = 17 vs 31 where a normal approximation is thin.
  q          Benjamini-Hochberg across the descriptors tested here.

and then the only question that decides whether this is worth anything: does
the descriptor still separate the classes AFTER the docking score is regressed
out of it? Docking ddG is already a selectivity predictor, so a frustration
descriptor that merely re-expresses it adds nothing.

    .venv/bin/python scripts/cox_selectivity_stats.py
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

N_PERM = 20000
SEED = 0


def auc(values: np.ndarray, positive: np.ndarray) -> float:
    """P(random positive ranks above random negative), ties counted as half.

    Equivalent to the Mann-Whitney U statistic scaled to [0, 1]; computed from
    average ranks so tied descriptor values do not bias it either way.
    """
    ranks = pd.Series(values).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def perm_p(values: np.ndarray, positive: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided permutation p-value for |AUC - 0.5|."""
    observed = abs(auc(values, positive) - 0.5)
    labels = positive.copy()
    hits = sum(
        abs(auc(values, rng.permutation(labels)) - 0.5) >= observed
        for _ in range(N_PERM)
    )
    # +1/+1 so a p-value is never reported as exactly 0, which permutation
    # testing cannot establish -- it can only bound it by 1/(N_PERM+1).
    return (hits + 1) / (N_PERM + 1)


def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR across the descriptors tested."""
    n = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(n)
    running = 1.0
    for rank, idx in reversed(list(enumerate(pvals[order], start=1))):
        pass
    # walk from largest p downwards, enforcing monotonicity
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        running = min(running, pvals[idx] * n / rank)
        q[idx] = running
    return q


def residualise(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Least-squares residual of y on [1, covariates]: y with them removed."""
    design = np.column_stack([np.ones(len(y)), covariates])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=pathlib.Path,
                    default=pathlib.Path("results/cox_selectivity/selectivity_delta.csv"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/cox_selectivity/stats.csv"))
    # Sensitivity check. pd-138387 and sulindac-sulfide have the same formula
    # and bond count in both targets but differ in stereochemistry / bond-order
    # perception, so their COX2-COX1 difference is slightly contaminated.
    ap.add_argument("--exclude", nargs="*", default=[], metavar="LIGAND",
                    help="ligand names to drop before testing")
    args = ap.parse_args()

    df = pd.read_csv(args.delta).dropna(subset=["selectivity_class"])
    if args.exclude:
        dropped = df["ligand"].isin(args.exclude)
        print(f"excluded {dropped.sum()}: {sorted(df.loc[dropped, 'ligand'])}")
        df = df[~dropped]
    positive = (df["selectivity_class"] == "Selective").to_numpy()
    rng = np.random.default_rng(SEED)

    descriptors = [c for c in df.columns if c.startswith("d_")]
    print(f"n = {len(df)} inhibitors "
          f"({positive.sum()} Selective / {(~positive).sum()} Non-Selective)\n")

    # The baseline every frustration descriptor has to beat.
    dock = df["dock_ddG"].to_numpy()
    print(f"BASELINE  docking ddG (COX2-COX1):  AUC = {auc(dock, positive):.3f}  "
          f"p_perm = {perm_p(dock, positive, rng):.4f}\n")

    # Two things could explain any separation for free. The docking score is a
    # pure baseline: a frustration descriptor that just re-expresses it is
    # worthless. Contact count is more arguable -- COX-2's larger pocket really
    # does admit more contacts, so that may be signal rather than confound --
    # so the two adjustments are reported separately instead of pooled.
    contacts = df["d_n_ligand_contacts"].to_numpy()
    covariate_sets = {
        "dock": np.column_stack([dock]),
        "dock_contacts": np.column_stack([dock, contacts]),
    }

    rows = []
    for name in descriptors:
        raw = df[name].to_numpy()
        row = {
            "descriptor": name,
            "mean_selective": raw[positive].mean(),
            "mean_nonselective": raw[~positive].mean(),
            "auc": auc(raw, positive),
            "p_perm": perm_p(raw, positive, rng),
        }
        for label, covs in covariate_sets.items():
            # Residualising a covariate against itself leaves only numerical
            # noise, so the adjusted AUC of d_n_ligand_contacts under the
            # contact-count adjustment is undefined, not 0.11.
            if name == "d_n_ligand_contacts" and "contacts" in label:
                row[f"auc_adj_{label}"] = np.nan
                row[f"p_adj_{label}"] = np.nan
                continue
            adj = residualise(raw, covs)
            row[f"auc_adj_{label}"] = auc(adj, positive)
            row[f"p_adj_{label}"] = perm_p(adj, positive, rng)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["q"] = bh(out["p_perm"].to_numpy())
    out = out.sort_values("p_perm")

    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\nAUC below 0.5 means the descriptor runs the OTHER way "
          "(Non-Selective scores higher).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
