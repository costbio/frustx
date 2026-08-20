"""Is the GDP->GTP frustration change real, or decoy noise?

Two controls, both necessary before reading anything into the profile difference.

CONTROL 1 -- the noise floor, measured not assumed. Split one structure's own 500 decoys
into two halves and build the profile from each. Both halves describe the SAME structure,
so any difference between them is pure decoy sampling noise. If the between-structure
difference is not clearly larger than this, there is nothing to interpret.

CONTROL 2 -- region selection. The P-loop and switches were chosen a priori from
small-GTPase biology, which is legitimate, but the regions are contiguous and the profile
is spatially autocorrelated, so a randomly placed block of the same size is the right null.
We slide a same-length window over all positions rather than permuting residue labels,
which would destroy that autocorrelation and understate the null.

Usage:
    .venv/bin/python scripts/fig2_significance.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fig2_profile import contacts_from_tensor
from vicinity_profile import load_atoms, profile

REGIONS = {"P-loop": (12, 20), "switch I": (32, 41), "switch II": (63, 79)}


def profile_from_subset(npz, pdb, decoy_slice):
    """Vicinity profile built from a subset of the decoy ensemble."""
    d = np.load(npz, allow_pickle=True)
    sub = dict(decoys=d["decoys"][decoy_slice], native=d["native"],
               resseq=d["resseq"], chain=d["chain"], resname=d["resname"])
    tmp = npz.replace(".npz", f".__half{decoy_slice.start}.npz")
    np.savez_compressed(tmp, **sub)
    con = contacts_from_tensor(tmp, pdb)
    ca, sel = load_atoms(pdb, atom="CB")
    return profile(con, ca, sel)


def main():
    npzq, pdbq = "results/fig2/decoys/1XTQ_min500.npz", "results/fig2/1XTQ_A.pdb"
    npzs, pdbs = "results/fig2/decoys/1XTS_min500.npz", "results/fig2/1XTS_A.pdb"

    # ---- control 1: split-half noise floor, within each structure ------------------
    print("[control 1] within-structure split-half (same structure, different decoys)")
    floors = {}
    for name, npz, pdb in (("1XTQ", npzq, pdbq), ("1XTS", npzs, pdbs)):
        a = profile_from_subset(npz, pdb, slice(0, 250))
        b = profile_from_subset(npz, pdb, slice(250, 500))
        j = a.merge(b, on=["Res", "ChainRes"], suffixes=("_a", "_b"))
        for col in ("nHighlyFrst", "nMinimallyFrst"):
            d = (j[f"{col}_b"] - j[f"{col}_a"])
            floors[(name, col)] = d.abs().mean()
            print(f"   {name} {col:>15}: mean |split-half diff| = {d.abs().mean():.3f}, "
                  f"sd = {d.std():.3f}, profile r = {np.corrcoef(j[f'{col}_a'], j[f'{col}_b'])[0,1]:.3f}")

    # ---- the real difference, for comparison --------------------------------------
    m = pd.read_csv("results/fig2/rheb_profile_comparison.csv")
    print("\n[signal] between-structure (GDP vs GTP, full 500 decoys each)")
    for col in ("nHighlyFrst", "nMinimallyFrst"):
        d = m[f"{col}_gtp"] - m[f"{col}_gdp"]
        fl = np.mean([floors[("1XTQ", col)], floors[("1XTS", col)]])
        print(f"   {col:>15}: mean |diff| = {d.abs().mean():.3f}  vs noise floor "
              f"{fl:.3f}  -> ratio {d.abs().mean()/fl:.2f}x")

    # ---- control 2: sliding-window null for the region means ----------------------
    print("\n[control 2] sliding-window null (same-length contiguous block, all positions)")
    res = m["Res"].values
    for label, col in (("dHigh", "nHighlyFrst"), ("dMin", "nMinimallyFrst")):
        d = (m[f"{col}_gtp"] - m[f"{col}_gdp"]).values
        print(f"  {label}:")
        for rname, (a, b) in REGIONS.items():
            sel = (res >= a) & (res <= b)
            obs = d[sel].mean()
            L = int(sel.sum())
            null = np.array([d[k:k + L].mean() for k in range(len(d) - L + 1)])
            # two-sided: how often does a random block reach this magnitude?
            p = (np.abs(null) >= abs(obs)).mean()
            print(f"     {rname:>10} (n={L:>2}): observed {obs:>+6.2f}, "
                  f"null mean {null.mean():>+5.2f} sd {null.std():.2f}, p = {p:.3f}")


if __name__ == "__main__":
    main()
