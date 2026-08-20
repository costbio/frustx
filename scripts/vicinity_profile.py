"""The Fig. 2 quantity: frustrated-contact counts "in the vicinity of each residue".

Chen et al. never define "vicinity" -- the word appears exactly once in the paper, in the
Fig. 2 caption, and no radius is given. Rather than invent one, this reproduces
frustratometeR's XAdens(), which computes precisely the quantity the caption describes and
is the reference implementation the figure is almost certainly mirroring. Dumped from the
installed package with `frustratometeR:::XAdens`; the rule is:

  - each CONTACT is given a position: the midpoint of its two interacting atoms
  - for each residue i, count contacts whose midpoint lies within `radius` of i's CA
  - strictly `< radius` (the R code uses `<`, not `<=`), default 5 A
  - split by the usual cut points: highly <= -1, minimally >= 0.78, neutral between

Note what this is NOT. It is not "the contacts involving residue i" -- it is a spatial
density of contact midpoints near i. A residue can therefore accumulate vicinity contacts
it takes no part in. That is what "in the vicinity of" buys over "of", and getting it wrong
would change the profile shape, which is the only thing Fig. 2 lets us compare.

AMBIGUITY, exposed rather than silently chosen: which atom positions define the midpoint.
frustratometeR is AWSEM-based so its contacts sit on CB (CA for glycine); FrustX defines
contacts by CA-CA distance. `atom` selects between them. Verified in main(): the CB
convention reproduces frustratometeR's own 1UBQ file exactly, CA does not.
"""
import sys

import numpy as np
import pandas as pd


def load_atoms(pdb_path, atom="CB"):
    """Per-residue coordinates keyed by (chain, resseq).

    Returns (ca, sel) dicts. `sel` falls back to CA where the requested atom is absent --
    glycine always, and any residue with an unresolved side chain.
    """
    ca, sel = {}, {}
    for line in open(pdb_path):
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        key = (line[21], int(line[22:26]))
        name = line[12:16].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if name == "CA":
            ca[key] = xyz
        if name == atom:
            sel[key] = xyz
    for k, v in ca.items():
        sel.setdefault(k, v)          # CB missing (Gly, or disordered) -> use CA
    return ca, sel


def profile(contacts, ca, sel, radius=5.0, minimally=0.78, highly=-1.0):
    """XAdens: count contact midpoints within `radius` of each residue's CA.

    `contacts` needs columns ci, i, cj, j, index. Returns one row per residue in `ca`.
    """
    mid, val = [], []
    for r in contacts.itertuples():
        a, b = sel.get((r.ci, r.i)), sel.get((r.cj, r.j))
        if a is None or b is None or not np.isfinite(r.index):
            continue
        mid.append(0.5 * (a + b))
        val.append(r.index)
    mid, val = np.array(mid), np.array(val)

    rows = []
    for (chain, resseq), c in ca.items():
        d = np.linalg.norm(mid - c, axis=1)
        near = d < radius                       # strict, matching the R code
        v = val[near]
        tot = int(near.sum())
        nh = int((v <= highly).sum())
        nm = int((v >= minimally).sum())
        rows.append({
            "Res": resseq, "ChainRes": chain, "Total": tot,
            "nHighlyFrst": nh, "nNeutrallyFrst": tot - nh - nm, "nMinimallyFrst": nm,
            "relHighlyFrustrated": nh / tot if tot else 0.0,
            "relMinimallyFrustrated": nm / tot if tot else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["ChainRes", "Res"]).reset_index(drop=True)


def main():
    """Mini-test: reproduce frustratometeR's own 1UBQ 5adens file from its own contacts.

    If this matches, the vicinity rule and the coordinate convention are both right, and
    the same code can then be pointed at FrustX output with confidence.
    """
    base = "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb"
    ref = pd.read_csv(base + "_mutational_5adens", sep=r"\s+")
    m = pd.read_csv(base + "_mutational", sep=r"\s+")
    contacts = pd.DataFrame({
        "ci": m["ChainRes1"], "i": m["Res1"], "cj": m["ChainRes2"], "j": m["Res2"],
        "index": m["FrstIndex"],
    })
    for atom in ("CB", "CA"):
        ca, sel = load_atoms(base, atom=atom)
        got = profile(contacts, ca, sel)
        j = ref.merge(got, on=["Res", "ChainRes"], suffixes=("_ref", "_got"))
        assert len(j) == len(ref), f"{len(j)} vs {len(ref)} residues joined"
        eq = {c: int((j[f"{c}_ref"] == j[f"{c}_got"]).sum())
              for c in ("Total", "nHighlyFrst", "nNeutrallyFrst", "nMinimallyFrst")}
        print(f"atom={atom}: exact-match residues out of {len(ref)} -> {eq}")
        print(f"          max |Total diff| = {(j['Total_ref'] - j['Total_got']).abs().max()}")


if __name__ == "__main__":
    main()
