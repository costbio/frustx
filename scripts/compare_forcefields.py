"""Separate the force-field effect from the protocol effect.

FrustX and frustratometeR disagree at the contact level (rho = +0.32). Two things
differ at once: the energy function and the protocol. Running AWSEM's energy through
FrustX's own machinery gives the missing cell of a 2x2 and separates them:

                     FrustX protocol        frustratometeR protocol
    REF2015          frustx_min500          (infeasible)
    AWSEM            frustx_awsem           frustra_mutational

    REF2015-FrustX vs AWSEM-FrustX   -> force field alone (protocol held fixed)
    AWSEM-FrustX   vs frustratometeR -> protocol alone   (force field held fixed)

Also reports how one-body each index is, by fitting X ~ c + a_i + a_j -- the exact
functional form of a sum of two residue terms. A high R^2 means the "contact" index is
really a residue property, which is what makes contact-level agreement uninterpretable.

Usage:
    .venv/bin/python scripts/compare_forcefields.py [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_frustratometer import (_canonical_pair, load_frustratometer, load_frustx,
                                    spearman, spearman_p)

B = Path("results/validation")


def additive_r2(df, col):
    """R^2 of fitting `col` to c + a_i + a_j, one free parameter per residue.

    The design matrix puts a 1 in the column of each of the contact's two residues, so
    the fit can express any sum of two residue-level terms and nothing else. Whatever
    R^2 it reaches is the share of the index that carries no pair-specific information.
    Solved with lstsq because the system is rank-deficient by construction (a constant
    can move between the intercept and the a_i).
    """
    res = sorted(set(zip(df["ci"], df["i"])) | set(zip(df["cj"], df["j"])))
    pos = {r: k for k, r in enumerate(res)}
    X = np.zeros((len(df), len(res) + 1))
    X[:, 0] = 1.0
    for row, (ci, i, cj, j) in enumerate(zip(df["ci"], df["i"], df["cj"], df["j"])):
        X[row, pos[(ci, i)] + 1] += 1.0
        X[row, pos[(cj, j)] + 1] += 1.0
    y = df[col].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid.var() / y.var(), resid


def load_awsem(path):
    """Our AWSEM run writes the same schema as the FrustX CLI, so reuse that loader."""
    return load_frustx(path)


def report(name, a, b):
    m = pd.concat([a, b], axis=1).dropna()
    rho = spearman(m.iloc[:, 0], m.iloc[:, 1])
    print(f"  {name:<46} rho = {rho:+.4f}  p = {spearman_p(rho, len(m)):.2g}  n = {len(m)}")
    return rho


def main(out_dir):
    ref = load_frustx(B / "frustx_min500" / "contacts.csv").rename(
        columns={"our_index": "ref2015"})
    aws = load_awsem(B / "frustx_awsem" / "contacts.csv").rename(
        columns={"our_index": "awsem"})
    fr = load_frustratometer(
        B / "frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb_mutational").rename(
        columns={"their_index": "frustra"})

    key = ["ci", "i", "cj", "j"]
    m = (ref[key + ["ref2015"]]
         .merge(aws[key + ["awsem"]], on=key, how="inner")
         .merge(fr[key + ["frustra", "Welltype"]], on=key, how="inner"))
    print(f"contacts shared by all three: {len(m)}\n")

    print("=== the 2x2 decomposition ===")
    d = m.dropna(subset=["ref2015", "awsem", "frustra"])
    r_ff = report("force field alone  (REF2015 vs AWSEM, FrustX protocol)",
                  d["ref2015"], d["awsem"])
    r_pr = report("protocol alone     (AWSEM-FrustX vs frustratometeR)",
                  d["awsem"], d["frustra"])
    r_bo = report("both differ        (REF2015-FrustX vs frustratometeR)",
                  d["ref2015"], d["frustra"])

    print("\n=== how one-body is each index? (X ~ c + a_i + a_j) ===")
    for label, col in [("FrustX REF2015", "ref2015"), ("FrustX AWSEM", "awsem"),
                       ("frustratometeR mutational", "frustra")]:
        print(f"  {label:<28} additive R2 = {additive_r2(d, col)[0]:.3f}")

    print("\n=== by AWSEM well type ===")
    print(f"  {'welltype':<16} {'n':>4} {'force field':>12} {'protocol':>10} {'both':>8}")
    for w, g in d.groupby("Welltype"):
        if len(g) >= 10:
            print(f"  {w:<16} {len(g):>4} "
                  f"{spearman(g['ref2015'], g['awsem']):>+12.3f} "
                  f"{spearman(g['awsem'], g['frustra']):>+10.3f} "
                  f"{spearman(g['ref2015'], g['frustra']):>+8.3f}")

    # The withdrawn water-mediated conclusion died because the direct-contact agreement
    # turned out to be carried entirely by shared one-body burial. Redo that control
    # here: strip the additive component from every index and re-correlate. If the
    # welltype split survives, it is a genuine pair-level force-field difference.
    print("\n=== one-body component removed (residuals of X ~ c + a_i + a_j) ===")
    r = d.copy()
    for col in ("ref2015", "awsem", "frustra"):
        r[col] = additive_r2(d, col)[1]
    print(f"  {'welltype':<16} {'n':>4} {'force field':>12} {'protocol':>10} {'both':>8}")
    print(f"  {'ALL':<16} {len(r):>4} "
          f"{spearman(r['ref2015'], r['awsem']):>+12.3f} "
          f"{spearman(r['awsem'], r['frustra']):>+10.3f} "
          f"{spearman(r['ref2015'], r['frustra']):>+8.3f}")
    for w, g in r.groupby("Welltype"):
        if len(g) >= 10:
            print(f"  {w:<16} {len(g):>4} "
                  f"{spearman(g['ref2015'], g['awsem']):>+12.3f} "
                  f"{spearman(g['awsem'], g['frustra']):>+10.3f} "
                  f"{spearman(g['ref2015'], g['frustra']):>+8.3f}")

    if out_dir:
        plot(d, (r_ff, r_pr, r_bo), Path(out_dir) / "forcefield_vs_protocol.png")


def plot(d, rhos, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("ref2015", "awsem", "Force field alone\nFrustX-REF2015 vs FrustX-AWSEM",
         "FrustX index (REF2015)", "FrustX index (AWSEM)", rhos[0]),
        ("awsem", "frustra", "Protocol alone\nFrustX-AWSEM vs frustratometeR",
         "FrustX index (AWSEM)", "frustratometeR mutational", rhos[1]),
        ("ref2015", "frustra", "Both differ\nFrustX-REF2015 vs frustratometeR",
         "FrustX index (REF2015)", "frustratometeR mutational", rhos[2]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (x, y, title, xl, yl, rho) in zip(axes, panels):
        ax.axhline(0, color="0.88", lw=0.8, zorder=0)
        ax.axvline(0, color="0.88", lw=0.8, zorder=0)
        ax.scatter(d[x], d[y], s=15, alpha=0.55, c="#2c7fb8",
                   edgecolors="none", zorder=3)
        k, c = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 2)
        ax.plot(xs, k * xs + c, color="#d7301f", lw=1.4, zorder=4)
        ax.set_title(f"{title}\nSpearman {rho:+.3f}  (n={len(d)})", fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, zorder=0)
    fig.suptitle("Same equations, different force field: 1UBQ", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/validation/plots_mutational")
