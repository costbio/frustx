#!/usr/bin/env python
"""Residue-resolved COX-1/COX-2 selectivity: does frustration localise to the side pocket?

The interface-aggregate analysis (scripts/cox_selectivity_stats.py) found that the
whole signal reduces to the ligand contact count -- how MANY contacts the ligand
makes in COX-2 versus COX-1. This script asks the question the aggregate cannot:
WHERE. Two separate tests per residue position, which answer different things:

  presence  does the ligand touch this position at all? Summed over positions this
            IS the contact count, so a presence result is a decomposition of the
            known effect, not a new one. Reported to show which residues carry it.

  delta-F   among ligands that contact the position in BOTH targets, is the contact
            more or less frustrated in COX-2? This is conditional on contact, so it
            is orthogonal to the count -- it is the one result here that could not
            be obtained from contacts.py alone.

Numbering. 1EQG uses COX-1 numbering; 3LN1 uses its own, offset by 14 (3LN1 523 is
Asn; the famous Val523 sits at 509). Comparing resnum across the two targets without
that shift silently misaligns everything by 14 residues. The offset is asserted
against eight anchors, not assumed -- the script exits if any fails.

    .venv/bin/python scripts/cox_pocket_analysis.py
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cox_selectivity_stats import auc, bh, perm_p, residualise

# 3LN1 resnum + 14 == COX-1 resnum. Derived by scanning for the offset that makes
# the conserved catalytic set line up, then pinned here.
OFFSET = {"cox1": 0, "cox2": 14}

# Anchors on the shared axis: (position, expected COX-1 residue, expected COX-2).
# The last two are the diagnostic differences -- 523 Ile->Val is the substitution
# that opens the COX-2 side pocket, and it is what makes this analysis worth doing.
ANCHORS = [(120, "ARG", "ARG"), (355, "TYR", "TYR"), (530, "SER", "SER"),
           (385, "TYR", "TYR"), (352, "LEU", "LEU"), (518, "PHE", "PHE"),
           (513, "HIS", "ARG"), (523, "ILE", "VAL")]

# Stated before looking at the per-position results: the residues that define the
# COX-2 side pocket and its entrance. Reported separately from the 43-position scan
# so a hypothesis test is never confused with a screen.
SIDE_POCKET = [523, 513, 518, 352, 90]

LIGAND = "UNL"
MIN_ELIGIBLE = 12      # ligands contacting in both targets, for a delta-F test
MIN_PER_CLASS = 4      # and at least this many on each side of the label


def load_contacts(results_dir: pathlib.Path) -> pd.DataFrame:
    """One row per (target, ligand, shared-axis position) protein-ligand contact."""
    rows = []
    for path in sorted(glob.glob(str(results_dir / "cox*_*" / "contacts.csv"))):
        name = os.path.basename(os.path.dirname(path))
        target, _, ligand = name.partition("_")
        c = pd.read_csv(path)
        # The ligand is appended last so it is always the j side; assert rather than
        # assume, because an i-side ligand would silently halve the interface.
        if (c["resname_i"] == LIGAND).any():
            raise SystemExit(f"{name}: ligand appears on the i side; the extraction "
                             f"below assumes it is always j")
        lig = c[c["resname_j"] == LIGAND]
        for _, r in lig.iterrows():
            rows.append({
                "target": target, "ligand": ligand,
                "pos": int(r["resnum_i"]) + OFFSET[target],
                "resname": r["resname_i"],
                "F": r["frustration_index"],
                "cls": r["frustration_class"],
            })
    return pd.DataFrame(rows)


def check_anchors(df: pd.DataFrame) -> None:
    """Fail loudly if the numbering offset does not reproduce known residue identities."""
    names = (df.groupby(["pos", "target"])["resname"]
               .agg(lambda s: s.mode().iloc[0]).unstack())
    bad = []
    for pos, want1, want2 in ANCHORS:
        got1 = names["cox1"].get(pos)
        got2 = names["cox2"].get(pos)
        if got1 != want1 or got2 != want2:
            bad.append(f"  {pos}: expected {want1}/{want2}, got {got1}/{got2}")
    if bad:
        raise SystemExit("numbering offset check FAILED -- positions are not aligned:\n"
                         + "\n".join(bad))
    print(f"numbering offset +{OFFSET['cox2']} verified on {len(ANCHORS)} anchors "
          f"(incl. 513 HIS/ARG and 523 ILE/VAL)")


def test_positions(df, labels, rng):
    """Per position: the presence test on all ligands, the delta-F test on the paired subset."""
    positive_of = dict(zip(labels["ligand"], labels["selectivity_class"] == "Selective"))
    # Per-ligand covariates, for the same two controls the aggregate analysis used.
    dock_of = dict(zip(labels["ligand"], -labels["dock_ddG"]))
    cont_of = dict(zip(labels["ligand"], labels["d_n_ligand_contacts"]))
    all_ligands = sorted(set(df["ligand"]) & set(positive_of))

    # (ligand, target) -> F, and the set of contacted positions, for O(1) lookup.
    fmap = {(r.ligand, r.target, r.pos): r.F for r in df.itertuples()}
    # (pos, target) -> set of ligands contacting it. A plain dict, not an unstacked
    # frame: a missing combination must come back as an empty set, and unstack fills
    # it with NaN instead.
    touched = df.groupby(["pos", "target"])["ligand"].apply(set).to_dict()

    rows = []
    for pos in sorted(set(df["pos"])):
        c1 = touched.get((pos, "cox1"), set())
        c2 = touched.get((pos, "cox2"), set())
        if not c1 or not c2:
            continue   # position reached in only one target; no paired comparison

        # --- presence: +1 gained in COX-2, -1 lost, 0 either both or neither
        pres = np.array([(l in c2) - (l in c1) for l in all_ligands], dtype=float)
        pos_mask = np.array([positive_of[l] for l in all_ligands])
        pres_auc = auc(pres, pos_mask)
        pres_p = perm_p(pres, pos_mask, rng) if len(set(pres)) > 1 else 1.0

        # --- delta-F: only ligands present in BOTH targets
        eligible = [l for l in all_ligands if l in c1 and l in c2]
        elig_pos = np.array([positive_of[l] for l in eligible])
        if (len(eligible) < MIN_ELIGIBLE or elig_pos.sum() < MIN_PER_CLASS
                or (~elig_pos).sum() < MIN_PER_CLASS):
            d_auc = d_p = adj_auc = adj_p = np.nan
            d_sel = d_non = np.nan
        else:
            d = np.array([fmap[(l, "cox2", pos)] - fmap[(l, "cox1", pos)] for l in eligible])
            d_auc = auc(d, elig_pos)
            d_p = perm_p(d, elig_pos, rng)
            d_sel, d_non = d[elig_pos].mean(), d[~elig_pos].mean()
            # The control that killed every aggregate descriptor. Run it here too,
            # on the same ligand subset, so a per-position hit is not just the
            # docking score or the contact count showing up again locally.
            cov = np.column_stack([[dock_of[l] for l in eligible],
                                   [cont_of[l] for l in eligible]])
            d_adj = residualise(d, cov)
            adj_auc = auc(d_adj, elig_pos)
            adj_p = perm_p(d_adj, elig_pos, rng)

        rows.append({
            "pos": pos,
            "cox1_res": df[(df["pos"] == pos) & (df.target == "cox1")]["resname"].mode().iloc[0],
            "cox2_res": df[(df["pos"] == pos) & (df.target == "cox2")]["resname"].mode().iloc[0],
            "n_cox1": len(c1), "n_cox2": len(c2),
            "presence_auc": pres_auc, "presence_p": pres_p,
            "n_paired": len(eligible),
            "dF_selective": d_sel, "dF_nonselective": d_non,
            "dF_auc": d_auc, "dF_p": d_p,
            "dF_adj_auc": adj_auc, "dF_adj_p": adj_p,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path,
                    default=pathlib.Path("data/docking/cox_results/decoy_results"))
    ap.add_argument("--delta", type=pathlib.Path,
                    default=pathlib.Path("results/cox_selectivity/selectivity_delta.csv"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/cox_selectivity/pocket_positions.csv"))
    args = ap.parse_args()

    df = load_contacts(args.results)
    check_anchors(df)
    labels = pd.read_csv(args.delta).dropna(subset=["selectivity_class"])
    rng = np.random.default_rng(0)

    out = test_positions(df, labels, rng)
    # BH over the scan. The presence and delta-F families are corrected separately:
    # they are different questions, and pooling them would penalise both for the
    # other's multiplicity.
    out["presence_q"] = bh(out["presence_p"].to_numpy())
    mask = out["dF_p"].notna()
    for src, dst in (("dF_p", "dF_q"), ("dF_adj_p", "dF_adj_q")):
        q = np.full(len(out), np.nan)
        q[mask.to_numpy()] = bh(out.loc[mask, src].to_numpy())
        out[dst] = q

    pd.set_option("display.width", 220)
    cols = ["pos", "cox1_res", "cox2_res", "n_cox1", "n_cox2",
            "presence_auc", "presence_p", "presence_q",
            "n_paired", "dF_selective", "dF_nonselective", "dF_auc", "dF_p", "dF_q",
            "dF_adj_auc", "dF_adj_p", "dF_adj_q"]

    print(f"\n{len(out)} positions contacted in both targets, "
          f"{mask.sum()} with enough paired ligands for a delta-F test\n")

    print("=== PRE-REGISTERED: COX-2 side pocket and entrance ===")
    pre = out[out["pos"].isin(SIDE_POCKET)]
    print(pre[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
          if len(pre) else "  none of these positions is contacted in both targets")

    print("\n=== SCAN: strongest delta-F positions, AFTER removing docking ddG and "
          "contact count ===")
    print("(dF_adj_q is corrected across the "
          f"{int(mask.sum())} positions tested -- the raw dF_p of the top hit is the "
          "maximum of a scan and is optimistic on its own)")
    scan = out[mask].sort_values("dF_adj_p").head(10)
    print(scan[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== SCAN: strongest presence positions (this IS the contact count, "
          "decomposed) ===")
    print(out.sort_values("presence_p").head(10)[cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("dF_auc > 0.5 = the contact is LESS frustrated in COX-2 for selective compounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
