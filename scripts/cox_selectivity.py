#!/usr/bin/env python
"""Collate the COX-1/COX-2 ligand-interface frustration into a selectivity table.

Pairs each inhibitor's two runs (produced by scripts/run_cox_batch.sh) and
reduces the protein-ligand contacts of each to a handful of descriptors, then
differences them COX2 - COX1. The class labels and docking scores come from
data/COX_Docking_Selectivity_Scores_Table.xlsx.

Statistics live in a separate step; this script only builds the table and is
honest about what is missing.

    .venv/bin/python scripts/cox_selectivity.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

# Settings that change the *scale* of the frustration index, so a COX1 run and a
# COX2 run are only comparable if they agree on all of them. --protocol sets
# Eq. 1's denominator; n_decoys sets its sampling error; background_weight and
# readout change what the index even is. Mismatch here silently poisons a ddF.
COMPARABLE_SETTINGS = (
    "protocol", "n_decoys", "background_weight", "readout",
    "cutoff", "ligand_cutoff", "contact_atom", "min_seq_sep",
)

LIGAND_NAME = "UNL"


def interface_descriptors(contacts: pd.DataFrame) -> dict:
    """Reduce one run's protein-ligand contacts to comparable scalars."""
    # The ligand is appended last, so it is always the j side in practice --
    # but test both so a future reordering fails loudly instead of silently
    # returning an empty interface.
    lig = contacts[
        (contacts["resname_i"] == LIGAND_NAME) | (contacts["resname_j"] == LIGAND_NAME)
    ]
    if lig.empty:
        raise ValueError(f"no {LIGAND_NAME} contacts in this run")

    cls = lig["frustration_class"].value_counts()
    return {
        "n_ligand_contacts": len(lig),
        # Mean is the pocket-size-independent reading; sum is what you would
        # quote as a total interface frustration but scales with contact count.
        "mean_frustration": lig["frustration_index"].mean(),
        "sum_frustration": lig["frustration_index"].sum(),
        # frustration_index_specific is an OLS residual against per-residue
        # coefficients, so by the normal equations it sums to ZERO over any one
        # residue's contacts. The ligand is one residue, so its mean residual is
        # identically 0 in every run (verified: 1.5e-14 on cox1_celecoxib) and
        # carries no signal. Only the spread is informative -- it says how
        # unevenly the interface is frustrated once burial is regressed out.
        "std_frustration_specific": lig["frustration_index_specific"].std(),
        # (No mean_frustration_onebody: the same identity makes it equal to
        # mean_frustration to machine precision, since total = onebody +
        # residual and the residual averages to zero over the ligand.)
        "n_minimally": int(cls.get("minimally", 0)),
        "n_neutral": int(cls.get("neutral", 0)),
        "n_highly": int(cls.get("highly", 0)),
        "frac_minimally": cls.get("minimally", 0) / len(lig),
        # Rosetta's own interaction energy over the interface: the in-house
        # analogue of the docking score, and the control that any frustration
        # result has to beat.
        "sum_native_energy": lig["native_energy"].sum(),
    }


def load_runs(results_dir: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    """One row per completed (target, ligand) run."""
    rows, problems = [], []
    for run_dir in sorted(results_dir.glob("cox*_*")):
        target, _, ligand = run_dir.name.partition("_")
        run_json, contacts_csv = run_dir / "run.json", run_dir / "contacts.csv"
        if not run_json.exists() or not contacts_csv.exists():
            continue  # not finished; coverage is reported by the caller
        settings = json.loads(run_json.read_text())
        try:
            desc = interface_descriptors(pd.read_csv(contacts_csv))
        except ValueError as exc:
            problems.append(f"{run_dir.name}: {exc}")
            continue
        rows.append(
            {"target": target, "ligand": ligand, **desc,
             **{k: settings.get(k) for k in COMPARABLE_SETTINGS}}
        )
    return pd.DataFrame(rows), problems


def check_settings(runs: pd.DataFrame) -> list[str]:
    """Every run must agree on the settings that set the index's scale."""
    return [
        f"{col}: {sorted(runs[col].dropna().unique().tolist())}"
        for col in COMPARABLE_SETTINGS
        if runs[col].nunique(dropna=False) > 1
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path,
                    default=pathlib.Path("data/docking/cox_results/decoy_results"))
    ap.add_argument("--table", type=pathlib.Path,
                    default=pathlib.Path("data/COX_Docking_Selectivity_Scores_Table.xlsx"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/cox_selectivity"))
    args = ap.parse_args()

    runs, problems = load_runs(args.results)
    if runs.empty:
        print(f"No completed runs under {args.results}", file=sys.stderr)
        return 1
    for p in problems:
        print(f"SKIPPED {p}", file=sys.stderr)

    mismatched = check_settings(runs)
    if mismatched:
        print("Runs disagree on settings that set the frustration scale; "
              "the COX2-COX1 difference would be meaningless:", file=sys.stderr)
        for m in mismatched:
            print(f"  {m}", file=sys.stderr)
        return 2

    labels = pd.read_excel(args.table)
    labels["ligand"] = labels["Inhibitor"].str.lower().str.strip()
    labels = labels.rename(columns={
        "inhibitor type": "selectivity_class",
        "COX1_Best_Score_kcal_mol": "dock_cox1",
        "COX2_Best_Score_kcal_mol": "dock_cox2",
        "Selectivity (COX2_minus_COX1)": "dock_ddG",
    })[["ligand", "selectivity_class", "dock_cox1", "dock_cox2", "dock_ddG"]]

    # Wide: one row per inhibitor, only where BOTH targets finished. A ddF from
    # a half-finished pair is not a number, it is a missing value.
    descriptors = [c for c in runs.columns
                   if c not in ("target", "ligand", *COMPARABLE_SETTINGS)]
    wide = runs.pivot(index="ligand", columns="target", values=descriptors)
    # Until the batch has run both halves, one target is absent from the pivot
    # entirely; reindex so the "no complete pairs yet" case is an empty table
    # rather than a KeyError.
    wide = wide.reindex(
        columns=pd.MultiIndex.from_product([descriptors, ["cox1", "cox2"]])
    )
    paired = wide.dropna(how="any")
    delta = pd.DataFrame(
        {f"d_{d}": paired[(d, "cox2")] - paired[(d, "cox1")] for d in descriptors}
    ).reset_index()
    delta = delta.merge(labels, on="ligand", how="left")

    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "per_run_interface.csv", index=False)
    delta.to_csv(args.out / "selectivity_delta.csv", index=False)

    ligands = set(runs["ligand"])
    print(f"runs collated        : {len(runs)}")
    print(f"inhibitors seen      : {len(ligands)}")
    print(f"complete pairs (ddF) : {len(delta)}")
    for target in ("cox1", "cox2"):
        print(f"  {target}: {(runs['target'] == target).sum()} runs")
    missing_label = set(delta["ligand"]) - set(labels["ligand"])
    if missing_label:
        print(f"no label in table    : {sorted(missing_label)}", file=sys.stderr)
    unrun = set(labels["ligand"]) - ligands
    if unrun:
        print(f"in table, never run  : {sorted(unrun)}", file=sys.stderr)
    print(f"wrote {args.out}/per_run_interface.csv and selectivity_delta.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
