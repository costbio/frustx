"""Replicate the Rheb result across independent small-GTPase conformer pairs.

The Rheb finding was that P-loop / switch I / switch II shift toward frustration on
activation (joint contrast p = 0.010), at 3.6-4.1x the decoy noise floor. It carried two
confounders that a single pair cannot address: the two crystal structures differ in
resolution (2.00 vs 2.80 A), and the marginal significance rested on one test.

The two additional Fig. 2 pairs have COMPLEMENTARY weaknesses, which is why both are run:

    1KAO / 2RAP  Rap2A GDP/GTP    sequences identical (0 mismatches), but 1.70 vs 2.60 A
    1OIV / 1OIW  Rab11A GDP/GSP   1.98 vs 2.05 A, but 1OIW carries Q70L -- IN switch II

A result that survives in all three is not easily explained by either confounder, since no
single one is shared by all three pairs. Residue 70 is additionally excluded from the
Rab11A test as a sensitivity check, since a mutated residue has different frustration by
construction.

REGIONS ARE DERIVED, NOT EYEBALLED. Each protein is numbered differently, so hardcoding
residue ranges per structure would invite fitting them to the answer. Instead the two
sequence motifs are located and the canonical Ras offsets applied uniformly:

    P-loop (G1)   = the GxxxxGK[ST] (Walker A) match span          Ras 10-17
    switch I (G2) = P-loop start + 20 .. + 28                      Ras 30-38
    switch II     = first DxxG (G3) start + 3 .. + 19              Ras 60-76

Checked against the Rheb regions used before this script existed: the rule returns
switch I 33-41 (32-41 used) and switch II 63-79 (63-79 used), so it reproduces the earlier
choice rather than redefining it.

Usage:
    .venv/bin/python scripts/gtpase_replication.py
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fig2_profile import contacts_from_tensor
from vicinity_profile import load_atoms, profile

THREE2ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))

PAIRS = [
    ("Rheb",   "1XTQ", "1XTS", None),
    ("Rap2A",  "1KAO", "2RAP", None),
    ("Rab11A", "1OIV", "1OIW", 70),     # Q70L: excluded as a sensitivity check
]


def ca_sequence(path, chain="A"):
    return [(int(l[22:26]), THREE2ONE.get(l[17:20].strip(), "X"))
            for l in open(path)
            if l.startswith("ATOM") and l[12:16] == " CA " and l[21] == chain
            and l[16] in (" ", "A")]


def regions(pdb_path):
    """P-loop / switch I / switch II resseq ranges, from sequence motifs."""
    s = ca_sequence(pdb_path)
    nums = [n for n, _ in s]
    txt = "".join(c for _, c in s)
    p = re.search(r"G.{4}GK[ST]", txt)
    if p is None:
        raise SystemExit(f"{pdb_path}: no Walker A motif found")
    # First DxxG downstream of the P-loop; later DxxG hits are unrelated turns.
    d = next(m.start() for m in re.finditer(r"D..G", txt) if m.start() > p.end())
    return {
        "P-loop":    (nums[p.start()], nums[p.end() - 1]),
        "switch I":  (nums[p.start() + 20], nums[p.start() + 28]),
        "switch II": (nums[d + 3], nums[d + 19]),
    }


def build_profile(pdb_id):
    npz = f"results/fig2/decoys/{pdb_id}_min500.npz"
    pdb = f"results/fig2/{pdb_id}_A.pdb"
    con = contacts_from_tensor(npz, pdb)
    ca, sel = load_atoms(pdb, atom="CB")
    return profile(con, ca, sel)


def joint_test(m, reg, exclude, seed=0, draws=20000):
    """Contrast between functional residues and the rest, against a block null.

    The null places three non-overlapping contiguous blocks of the SAME lengths at random
    positions. Contiguous blocks, not permuted labels: the profile is spatially
    autocorrelated, and permuting labels would destroy that and understate the null.
    """
    rng = np.random.default_rng(seed)
    res = m["Res"].values
    n = len(m)
    sizes = [b - a + 1 for a, b in reg.values()]
    sel = np.zeros(n, bool)
    for a, b in reg.values():
        sel |= (res >= a) & (res <= b)
    keep = np.ones(n, bool)
    if exclude is not None:
        keep = res != exclude

    def rand_blocks():
        for _ in range(300):
            st = rng.integers(0, n - max(sizes), size=len(sizes))
            idx = [set(range(s, s + L)) for s, L in zip(st, sizes)]
            if len(set().union(*idx)) == sum(sizes):
                return np.array(sorted(set().union(*idx)))

    out = {}
    for label, col in (("d highly", "nHighlyFrst"), ("d minimally", "nMinimallyFrst")):
        d = (m[f"{col}_b"] - m[f"{col}_a"]).values.astype(float)
        obs = d[sel & keep].mean() - d[~sel & keep].mean()
        null = []
        for _ in range(draws):
            i = rand_blocks()
            mk = np.zeros(n, bool)
            mk[i] = True
            null.append(d[mk & keep].mean() - d[~mk & keep].mean())
        null = np.array(null)
        out[label] = (obs, (np.abs(null) >= abs(obs)).mean())
    return out, int(sel.sum())


def main():
    for name, a, b, excl in PAIRS:
        reg = regions(f"results/fig2/{a}_A.pdb")
        reg_b = regions(f"results/fig2/{b}_A.pdb")
        pa, pb = build_profile(a), build_profile(b)
        m = pa.merge(pb, on=["Res", "ChainRes"], suffixes=("_a", "_b"))
        agree = "same" if reg == reg_b else f"DIFFER: {reg_b}"
        print(f"\n=== {name}: {a} (inactive) -> {b} (active) ===")
        print(f"  regions {dict(reg)}   [motifs in {b}: {agree}]")
        print(f"  {len(m)} shared residues" + (f", excluding {excl}" if excl else ""))
        for rn, (lo, hi) in reg.items():
            sub = m[(m.Res >= lo) & (m.Res <= hi)]
            if excl is not None:
                sub = sub[sub.Res != excl]
            print(f"    {rn:>10}: d highly {(sub.nHighlyFrst_b - sub.nHighlyFrst_a).mean():>+6.2f}"
                  f"   d minimally {(sub.nMinimallyFrst_b - sub.nMinimallyFrst_a).mean():>+6.2f}")
        res, nfunc = joint_test(m, reg, excl)
        for label, (obs, p) in res.items():
            print(f"  joint {label:>12}: contrast {obs:>+6.2f}   p = {p:.4f}")
        m.to_csv(f"results/fig2/{name}_profile_comparison.csv", index=False)


if __name__ == "__main__":
    main()
