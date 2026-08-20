"""Category ratios: the one aggregate number Chen et al. actually publishes.

The paper reports three quantitative fractions, all in the protein-protein INTERFACE
analysis (Supplementary Fig. 8):

    atomistic frustratometer, interface minimally frustrated  ->  14.2%
    AWSEM frustratometer,     interface minimally frustrated  ->  26.0%
    AWSEM, water-mediated interactions minimally frustrated   ->  ~24%

plus two claims with no number attached:

    "short-range and long-range contacts that are not water-mediated do display similar
     fractions of minimally frustrating interactions in both frustratometers"
    "the atomistic frustratometer shows only a much lower fraction of minimally frustrated
     water-mediated interactions and a higher level of high frustration in the
     water-mediated contacts than does the AWSEM"

This script tests the two unnumbered claims on 1UBQ, where we hold all three calculations.
It does NOT test 14.2% / 26.0%: those are interface fractions and 1UBQ is a monomer. See
the caveat printed at the end.

Welltype comes from frustratometeR, which is the only source that assigns it, and is joined
onto the other two by contact identity so all three are classified identically.

Usage:
    .venv/bin/python scripts/category_ratios.py
"""
import numpy as np
import pandas as pd

FR = ("results/validation/frustra_mutational/1ubq_A.done/FrustrationData/"
      "1ubq_A.pdb_mutational")
SETS = {
    "FrustX / REF2015": "results/validation/frustx_min500/contacts.csv",
    "FrustX / AWSEM": "results/validation/frustx_awsem/contacts.csv",
}


def load_frustx(path):
    d = pd.read_csv(path)
    return pd.DataFrame({
        "ci": d.chain_i, "i": d.resnum_i, "cj": d.chain_j, "j": d.resnum_j,
        "index": d.frustration_index,
    })


def fractions(df, label):
    """Minimally / neutral / highly fractions, dropping undefined indices.

    sigma = 0 contacts carry a NaN index; counting them as neutral would inflate the
    neutral fraction and deflate the other two.
    """
    v = df["index"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    return {
        "set": label, "n": n,
        "minimally %": 100 * (v >= 0.78).mean(),
        "neutral %": 100 * ((v > -1.0) & (v < 0.78)).mean(),
        "highly %": 100 * (v <= -1.0).mean(),
    }


def main():
    fr = pd.read_csv(FR, sep=r"\s+")
    ref = pd.DataFrame({
        "ci": fr.ChainRes1, "i": fr.Res1, "cj": fr.ChainRes2, "j": fr.Res2,
        "index": fr.FrstIndex, "Welltype": fr.Welltype,
    })

    frames = {"frustratometeR / AWSEM": ref[["ci", "i", "cj", "j", "index"]]}
    for label, path in SETS.items():
        frames[label] = load_frustx(path)

    # Join every set onto frustratometeR's contact list so all three are compared on the
    # SAME contacts with the SAME welltype label. Contacts only one tool finds are
    # excluded rather than counted, since a fraction over different denominators is not
    # a comparison.
    key = ["ci", "i", "cj", "j"]
    shared = ref[key + ["Welltype"]].copy()
    for label, df in frames.items():
        shared = shared.merge(df.rename(columns={"index": label}), on=key, how="inner")
    print(f"shared contacts across all three calculations: {len(shared)}\n")

    print("=== overall (1UBQ monomer -- NOT the paper's interface number) ===")
    rows = [fractions(shared.rename(columns={l: "index"}), l) for l in frames]
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.1f"))

    print("\n=== by welltype: the paper's comparative claim ===")
    for wt in ("short", "long", "water-mediated"):
        sub = shared[shared.Welltype == wt]
        rows = [fractions(sub.rename(columns={l: "index"}), l) for l in frames]
        print(f"\n-- {wt} (n={len(sub)}) --")
        print(pd.DataFrame(rows).drop(columns=["n"]).to_string(index=False,
                                                              float_format="%.1f"))

    print("\nCAVEAT: 1UBQ is a monomer. The paper's 14.2% / 26.0% are fractions over"
          "\nprotein-protein INTERFACE contacts in complexes, so they are not the same"
          "\nquantity and must not be compared against the 'overall' block above.")


if __name__ == "__main__":
    main()
