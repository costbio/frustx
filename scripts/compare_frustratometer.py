"""Compare FrustX output against frustratometeR on the same structure.

frustratometeR computes configurational frustration with the AWSEM coarse-grained
energy function; FrustX uses atomistic Rosetta REF2015. The numbers are not expected to
match. What we are testing is whether they agree on the PATTERN -- which contacts are
minimally frustrated and which are frustrated -- and whether our sign convention and
classification thresholds transfer.

Usage:
    python scripts/compare_frustratometer.py <frustx_out_dir> <frustratometeR_file>

where <frustratometeR_file> is
    <ResultsDir>/<pdb>.done/FrustrationData/<pdb>.pdb_configurational

No scipy dependency: Spearman is computed as Pearson on ranks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def spearman(a, b):
    """Rank correlation without scipy: Pearson on the ranks."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_p(rho, n):
    """Two-sided p-value for a rank correlation, via the usual t approximation.

    t = rho * sqrt((n-2)/(1-rho^2)) on n-2 df; the tail is the regularised incomplete
    beta I_x(df/2, 1/2) with x = df/(df+t^2), evaluated by Lentz continued fraction.
    Spelled out here only to avoid taking a scipy dependency for one number.
    """
    if n < 4 or abs(rho) >= 1:
        return float("nan")
    from math import sqrt, lgamma, log, exp
    t = rho * sqrt((n - 2) / (1 - rho**2))
    df = n - 2
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    front = exp(a * log(x) + b * log(1 - x)
                - (lgamma(a) + lgamma(b) - lgamma(a + b))) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m_ = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m_ * (b - m_) * x) / ((a + 2*m_ - 1) * (a + 2*m_))
        else:
            num = -((a + m_) * (a + b + m_) * x) / ((a + 2*m_) * (a + 2*m_ + 1))
        d = 1.0 / (1e-30 if abs(1.0 + num * d) < 1e-30 else 1.0 + num * d)
        c = 1e-30 if abs(1.0 + num / c) < 1e-30 else 1.0 + num / c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def _canonical_pair(df):
    """Order each contact as (chain, resnum)_i <= (chain, resnum)_j.

    The join key must include the chain. Ordering on resnum alone would be wrong for
    complexes -- residue 40 of chain A and residue 40 of chain B are different residues,
    and two distinct contacts could collapse onto the same key. Harmless on the 1UBQ
    single-chain validation, silently wrong on anything else, which is exactly the kind
    of bug that surfaces as an unexplained low correlation.
    """
    key_i = list(zip(df["ci"], df["i"]))
    key_j = list(zip(df["cj"], df["j"]))
    swap = np.array([a > b for a, b in zip(key_i, key_j)])
    out = df.copy()
    out["ci"] = np.where(swap, df["cj"], df["ci"])
    out["cj"] = np.where(swap, df["ci"], df["cj"])
    out["i"] = np.where(swap, df["j"], df["i"])
    out["j"] = np.where(swap, df["i"], df["j"])
    return out


def load_frustratometer(path):
    """Their per-contact table: whitespace-separated with a header line."""
    df = pd.read_csv(path, sep=r"\s+")
    df = df.rename(columns={"Res1": "i", "Res2": "j", "FrstIndex": "their_index",
                            "FrstState": "their_class", "AA1": "aa1", "AA2": "aa2",
                            "ChainRes1": "ci", "ChainRes2": "cj"})
    df = _canonical_pair(df)
    return df[["ci", "i", "cj", "j", "aa1", "aa2",
               "their_index", "their_class", "Welltype"]]


def load_frustx(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"resnum_i": "i", "resnum_j": "j",
                            "chain_i": "ci", "chain_j": "cj",
                            "frustration_index": "our_index",
                            "frustration_class": "our_class"})
    df = _canonical_pair(df)
    return df[["ci", "i", "cj", "j", "resname_i", "resname_j", "our_index", "our_class",
               "native_energy", "decoy_mean", "decoy_std"]]


def main(frustx_dir, frustra_file):
    ours = load_frustx(Path(frustx_dir) / "contacts.csv")
    theirs = load_frustratometer(frustra_file)

    print(f"FrustX contacts          : {len(ours)}")
    print(f"frustratometeR contacts  : {len(theirs)}")

    merged = ours.merge(theirs, on=["ci", "i", "cj", "j"], how="inner")
    only_ours = len(ours) - len(merged)
    only_theirs = len(theirs) - len(merged)
    print(f"shared contacts          : {len(merged)}"
          f"   (only FrustX: {only_ours}, only frustratometeR: {only_theirs})")

    if merged.empty:
        print("\nNo shared contacts -- residue numbering probably disagrees.")
        return 1

    m = merged.dropna(subset=["our_index", "their_index"])
    print(f"\n--- correlation on {len(m)} shared contacts ---")
    print(f"Spearman : {spearman(m['our_index'], m['their_index']):+.3f}")
    print(f"Pearson  : {np.corrcoef(m['our_index'], m['their_index'])[0,1]:+.3f}")

    print("\n--- index distributions ---")
    for name, col in (("FrustX", "our_index"), ("frustratometeR", "their_index")):
        v = m[col]
        print(f"{name:<16} min {v.min():+.2f}  median {v.median():+.2f}  "
              f"max {v.max():+.2f}  sd {v.std():.2f}")

    print("\n--- classification agreement (rows FrustX, cols frustratometeR) ---")
    print(pd.crosstab(m["our_class"], m["their_class"]).to_string())
    agree = (m["our_class"] == m["their_class"]).mean()
    print(f"\nexact agreement: {agree:.1%}")

    # Correlation by AWSEM well type.
    #
    # What Welltype actually means, from RenumFiles.pl:50-64 -- it is assigned purely by
    # CB-CB distance and local density, NOT by which energy term fired:
    #     r < 6.5 A                          -> "short"           (the direct well)
    #     r >= 6.5 A, both densities < 2.6   -> "water-mediated"
    #     r >= 6.5 A, otherwise              -> "long"            (protein-mediated)
    # So "long" is NOT a direct contact: it sits in the same >=6.5 A mediated shell as
    # water-mediated and differs only by local density. An earlier version of this script
    # grouped short+long as "DIRECT", which was wrong. Only "short" is direct.
    #
    # Read these numbers with care. frustratometeR's configurational FrstIndex is an
    # affine function of its NativeEnergy (one global decoy mean/sd for the whole
    # protein -- fix_backbone.cpp:5095-5101), and over half its variance is a one-body
    # burial term that FrustX excludes at w=0. Correlation within a well type is largely
    # carried by that shared one-body component, not by contact physics. See
    # docs/method.md, "The reference implementation".
    print("\n--- Spearman by AWSEM well type ---")
    for well, grp in m.groupby("Welltype"):
        if len(grp) > 5:
            r = spearman(grp["our_index"], grp["their_index"])
            print(f"  {well:<16} n={len(grp):<4} rho={r:+.3f}  p={spearman_p(r, len(grp)):.4f}")

    direct = m[m["Welltype"] == "short"]
    if len(direct) > 5 and len(direct) < len(m):
        r = spearman(direct["our_index"], direct["their_index"])
        print(f"  {'DIRECT (short only)':<16} n={len(direct):<4} rho={r:+.3f}  "
              f"p={spearman_p(r, len(direct)):.4f}")
        # Does our index rank THEIR classes correctly on the direct-well subset?
        # Positive separation = yes. Note this is NOT evidence about pair physics on its
        # own, for the one-body reason given above.
        hi = direct[direct["their_class"] == "highly"]["our_index"]
        lo = direct[direct["their_class"] == "minimally"]["our_index"]
        if len(hi) and len(lo):
            print(f"    our index on their classes: highly {hi.mean():+.2f} (n={len(hi)})"
                  f"  minimally {lo.mean():+.2f} (n={len(lo)})"
                  f"  separation {lo.mean() - hi.mean():+.2f}")

    # Contacts where Eq. 1 is undefined: sigma == 0 because no REF2015 term fires for the
    # pair in the native OR any decoy. These sit at the outer edge of the Ca-Ca cutoff,
    # where a Ca-based contact criterion admits pairs with no atomistic interaction.
    undef = ours["our_index"].isna().sum()
    if undef:
        print(f"\n--- {undef} of {len(ours)} FrustX contacts are undefined (decoy sigma = 0) ---")
        print(f"    shared with frustratometeR: "
              f"{merged['our_index'].isna().sum()}  (excluded from all statistics above)")

    # The glycine/terminus artifact: does frustratometeR show it too?
    print("\n--- contacts involving GLY (the suspected shuffling artifact) ---")
    gly = m[(m["resname_i"] == "GLY") | (m["resname_j"] == "GLY")]
    if not gly.empty:
        print(f"n={len(gly)}   FrustX median {gly['our_index'].median():+.2f}   "
              f"frustratometeR median {gly['their_index'].median():+.2f}")
        print(f"non-GLY  FrustX median {m[~m.index.isin(gly.index)]['our_index'].median():+.2f}   "
              f"frustratometeR median {m[~m.index.isin(gly.index)]['their_index'].median():+.2f}")
        print("\n  worst offenders by our index:")
        for _, r in gly.nlargest(5, "our_index").iterrows():
            print(f"    {r.resname_i}{r.i:<4}-{r.resname_j}{r.j:<4} "
                  f"ours {r.our_index:+.2f}  theirs {r.their_index:+.2f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
