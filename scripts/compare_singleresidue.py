"""Residue-level comparison: FrustX against frustratometeR's `singleresidue` mode.

Two FrustX quantities are compared, because they are not the same thing and the
difference is the point:

  A. `mean_frustration` from residues.csv -- the mean of that residue's *contact*
     indices (frustx/output.py:67). It is a summary of pair frustration, not a
     residue Z-score. This is what the CLI already emits.

  B. A true per-residue Z-score, computed here from the dumped decoy tensor:

         R_i = sum_j E_ij          (residue i's total contact energy)
         Z_i = ( <R_i>_decoys - R_i_native ) / sd(R_i)_decoys

     Same sign convention as the contact index: positive = native better than decoys.
     This is the like-for-like analogue of what their singleresidue mode reports,
     which is why it needs the full tensor -- production keeps only per-contact
     running sums, from which sd(R_i) cannot be recovered (the contact energies
     within a residue are correlated, so the variances do not simply add).

Caveat on B, which is real and does not go away: our decoy ensemble shuffles the
*whole* sequence, while their singleresidue decoys mutate only residue i and leave
every other position native. Both are "randomise identity, keep geometry", but ours
perturbs residue i's entire environment at the same time. Expect ours to be noisier
and to run smaller in magnitude.

Usage:
    .venv/bin/python scripts/compare_singleresidue.py <frustx_dir> <singleresidue_file> \\
        [samples.npz] [pdb] [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_frustratometer import spearman, spearman_p

from frustx.contacts import contact_pairs, load_ca


def load_singleresidue(path):
    df = pd.read_csv(path, sep=r"\s+")
    return df.rename(columns={"Res": "resnum", "ChainRes": "chain",
                              "FrstIndex": "their_index", "AA": "aa"})[
        ["chain", "resnum", "aa", "their_index"]]


def residue_zscores(npz_path, pdb_path, min_seq_sep=2):
    """Quantity B: a genuine per-residue Z-score from the full decoy tensor."""
    z = np.load(npz_path)
    decoys, native = z["decoys"], z["native"]
    residues, ca = load_ca(pdb_path)
    pairs = contact_pairs(residues, ca, min_seq_sep=min_seq_sep)

    n = native.shape[0]
    # Sum only over pairs that count as contacts, so the residue total matches the
    # same contact definition the rest of the pipeline uses.
    mask = np.zeros((n, n), dtype=bool)
    for i, j in pairs:
        mask[i, j] = mask[j, i] = True

    R_dec = (decoys * mask).sum(axis=2)          # (n_decoys, n_res)
    R_nat = (native * mask).sum(axis=1)          # (n_res,)
    sd = R_dec.std(axis=0)
    ok = sd > 1e-9
    zs = np.full(n, np.nan)
    zs[ok] = (R_dec.mean(axis=0)[ok] - R_nat[ok]) / sd[ok]

    return pd.DataFrame({"chain": z["chain"], "resnum": z["resseq"], "our_z": zs})


def report(name, m, a, b):
    m = m.dropna(subset=[a, b])
    rho = spearman(m[a], m[b])
    print(f"{name:<34} n={len(m):>3}  Spearman {rho:+.4f}  p={spearman_p(rho, len(m)):.2g}"
          f"  Pearson {np.corrcoef(m[a], m[b])[0, 1]:+.4f}")
    return m, rho


def main(frustx_dir, sr_file, npz, pdb, out_dir):
    theirs = load_singleresidue(sr_file)
    ours = pd.read_csv(Path(frustx_dir) / "residues.csv")

    merged = ours.merge(theirs, on=["chain", "resnum"], how="inner")
    print(f"FrustX residues {len(ours)}, theirs {len(theirs)}, shared {len(merged)}\n")

    panels = []
    m_a, rho_a = report("A. mean of contact indices", merged, "mean_frustration", "their_index")
    panels.append(("A. mean of contact indices", m_a, "mean_frustration", rho_a))

    if npz:
        merged_b = merged.merge(residue_zscores(npz, pdb), on=["chain", "resnum"], how="inner")
        m_b, rho_b = report("B. true residue Z-score", merged_b, "our_z", "their_index")
        panels.append(("B. true residue Z-score", m_b, "our_z", rho_b))
        # Do the two FrustX quantities even agree with each other?
        report("   (A vs B, internal check)", merged_b, "mean_frustration", "our_z")

    if out_dir:
        plot(panels, Path(out_dir) / "singleresidue_scatter.png")


def plot(panels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.4), squeeze=False)
    for ax, (title, m, col, rho) in zip(axes[0], panels):
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.scatter(m[col], m["their_index"], s=34, alpha=0.75,
                   c="#2c7fb8", edgecolors="white", linewidths=0.5, zorder=3)
        # Least-squares line, drawn only to make the trend legible -- the reported
        # statistic is Spearman, which does not assume linearity.
        if len(m) > 2:
            k, c = np.polyfit(m[col], m["their_index"], 1)
            xs = np.linspace(m[col].min(), m[col].max(), 2)
            ax.plot(xs, k * xs + c, color="#d7301f", lw=1.4, zorder=4)
        ax.set_xlabel(f"FrustX -- {title}")
        ax.set_ylabel("frustratometeR singleresidue index")
        ax.set_title(f"{title}\nSpearman {rho:+.3f}  (n={len(m)})", fontsize=10)
        ax.grid(alpha=0.25, zorder=0)
    fig.suptitle("Residue-level frustration, 1UBQ: FrustX vs frustratometeR", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 3:
        print(__doc__)
        sys.exit(2)
    main(a[1], a[2],
         a[3] if len(a) > 3 else None,
         a[4] if len(a) > 4 else "results/validation/1ubq.pdb",
         a[5] if len(a) > 5 else "results/validation/plots_mutational")
