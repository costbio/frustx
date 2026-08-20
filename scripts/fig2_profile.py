"""Build Fig. 2's right-column quantity from a FrustX decoy tensor.

Chain: tensor -> Eq. 1 per-contact index -> XAdens vicinity profile -> CSV + plot.
The vicinity rule and its 5 A radius come from vicinity_profile.py, which reproduces
frustratometeR's own output exactly; see its module docstring.

What is and is not comparable to the published figure. The profile SHAPE is -- which
regions carry minimally frustrated contacts and which carry highly frustrated ones.
Absolute counts are not, because they scale with the classification thresholds, and
Chen et al. state neither their thresholds nor their radius. We use frustratometeR's
0.78 / -1.0, and -1.0 is on record here as provisional for REF2015.

Usage:
    .venv/bin/python scripts/fig2_profile.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from vicinity_profile import load_atoms, profile

PAIRS = [("1XTQ", "Rheb-GDP"), ("1XTS", "Rheb-GTP")]


def contacts_from_tensor(npz_path, pdb_path, cutoff=10.0, min_seq_sep=2):
    """Per-contact Eq. 1 index at w = 0, plus the residue labels to join on."""
    d = np.load(npz_path, allow_pickle=True)
    e_dec, e_nat = d["decoys"], d["native"]
    resseq, chain = d["resseq"], d["chain"]
    n = e_nat.shape[0]

    # Contacts from CA geometry, same rule as the production path.
    ca, _ = load_atoms(pdb_path, atom="CA")
    xyz = np.array([ca[(c, int(r))] for c, r in zip(chain, resseq)])
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    ii, jj = np.triu_indices(n, k=min_seq_sep)
    keep = dist[ii, jj] <= cutoff
    i, j = ii[keep], jj[keep]

    dec = e_dec[:, i, j]
    sig = dec.std(axis=0)
    num = dec.mean(axis=0) - e_nat[i, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.where(sig > 0, num / sig, np.nan)   # sigma = 0 -> undefined, NOT neutral

    return pd.DataFrame({
        "ci": chain[i], "i": resseq[i], "cj": chain[j], "j": resseq[j],
        "index": idx, "sigma": sig,
    })


def main():
    out = {}
    for pdb_id, label in PAIRS:
        npz = f"results/fig2/decoys/{pdb_id}_min500.npz"
        pdb = f"results/fig2/{pdb_id}_A.pdb"
        con = contacts_from_tensor(npz, pdb)
        nan = int(con["index"].isna().sum())
        ca, sel = load_atoms(pdb, atom="CB")     # CB convention, verified in vicinity_profile
        prof = profile(con, ca, sel)
        con.to_csv(f"results/fig2/{pdb_id}_contacts.csv", index=False)
        prof.to_csv(f"results/fig2/{pdb_id}_5adens.csv", index=False)
        out[pdb_id] = prof
        v = con["index"].dropna()
        print(f"{pdb_id} ({label}): {len(con)} contacts, {nan} with sigma=0 dropped")
        print(f"   index: min {v.min():.2f}  median {v.median():.2f}  max {v.max():.2f}"
              f"   | highly {(v <= -1).sum()}  minimally {(v >= 0.78).sum()}")
        print(f"   vicinity: mean Total {prof['Total'].mean():.1f}, "
              f"residues with >=1 highly {int((prof['nHighlyFrst'] > 0).sum())}/{len(prof)}")
    return out


if __name__ == "__main__":
    main()
