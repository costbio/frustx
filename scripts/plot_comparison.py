"""Visual comparison of FrustX against frustratometeR on the same structure.

**Point this at the `mutational` table, not `configurational`.** FrustX builds its
decoys by randomising residue identities at fixed native geometry, which is exactly
what AWSEM calls a mutational decoy (fix_backbone.cpp:5255-5277). Configurational
decoys redraw the geometry too, so their index is the raw contact energy rescaled by
two protein-wide constants and is not a per-contact Z-score at all. Comparing against
it is a category error -- see docs/method.md.

Produces three figures:

  1. <out>/class_ratios.png -- how each tool distributes contacts across the three
     frustration classes (minimally / neutral / highly), as relative ratios so the
     differing contact counts do not confound the comparison.

  2. <out>/scatter.png -- per-contact frustration index, ours vs theirs, on the shared
     contacts only, split by AWSEM well type.

     NOTE on the well types: `short` (r < 6.5 A) is the only *direct* well. `long` is
     protein-mediated, not direct -- an earlier reading of these plots got that
     backwards (RenumFiles.pl:50-64).

  3. <out>/index_distributions.png -- the marginal distributions.

Usage:
    python scripts/plot_comparison.py <frustx_out_dir> <frustratometeR_file> <out_dir>

Reuses the loaders in compare_frustratometer.py so there is exactly one definition of
how the two tables are joined.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display in this environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_frustratometer import load_frustx, load_frustratometer, spearman, spearman_p

# Their labels are lowercase already; ours match. Fixed order so the bars line up.
CLASSES = ["minimally", "neutral", "highly"]
COLORS = {"minimally": "#2c7fb8", "neutral": "#bdbdbd", "highly": "#d7301f"}
WELL_COLORS = {"short": "#1b7837", "long": "#762a83", "water-mediated": "#e08214"}


def plot_class_ratios(ours, theirs, out_path):
    """Grouped bars: fraction of contacts in each class, per tool.

    Ratios rather than counts -- FrustX finds 475 contacts and frustratometeR 389 (a
    contact-definition difference, not a frustration difference), so raw counts would
    make the tools look more different than they are.
    """
    # Undefined contacts (decoy sigma == 0) are excluded: they are not a frustration
    # class, they are pairs where Eq. 1 has no denominator.
    o = ours.dropna(subset=["our_index"])["our_class"].value_counts(normalize=True)
    t = theirs["their_class"].value_counts(normalize=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(CLASSES))
    w = 0.38
    ov = [o.get(c, 0.0) for c in CLASSES]
    tv = [t.get(c, 0.0) for c in CLASSES]
    b1 = ax.bar(x - w/2, ov, w, label=f"FrustX (REF2015), n={len(ours.dropna(subset=['our_index']))}",
                color=[COLORS[c] for c in CLASSES], edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, tv, w, label=f"frustratometeR (AWSEM), n={len(theirs)}",
                color=[COLORS[c] for c in CLASSES], edgecolor="black",
                linewidth=0.8, hatch="///")
    for bars, vals in ((b1, ov), (b2, tv)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.1%}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("fraction of contacts")
    ax.set_title("Frustration class distribution\n(solid = FrustX, hatched = frustratometeR)")
    ax.set_ylim(0, max(ov + tv) * 1.22)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return dict(zip(CLASSES, ov)), dict(zip(CLASSES, tv))


def plot_scatter(m, out_path):
    """Per-contact index, ours vs theirs, coloured by AWSEM well type.

    Left panel: all shared contacts. Right panel: the direct-contact subset.

    AWSEM's well types are assigned in RenumFiles.pl:50-64: r < 6.5 A is `short`,
    r >= 6.5 A with both densities < 2.6 is `water-mediated`, and everything else is
    `long`. Only `short` is a direct contact -- `long` is protein-mediated. An earlier
    version of this figure grouped short+long and labelled it "direct", which is wrong.
    """
    direct = m[m["Welltype"] == "short"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)

    for ax, sub, title in (
        (axes[0], m, "All shared contacts"),
        (axes[1], direct, "Direct contacts only (short, r < 6.5 A)"),
    ):
        for well, grp in sub.groupby("Welltype"):
            ax.scatter(grp["their_index"], grp["our_index"], s=22, alpha=0.65,
                       color=WELL_COLORS.get(well, "#666666"),
                       edgecolor="none", label=f"{well} (n={len(grp)})")
        r = spearman(sub["our_index"], sub["their_index"])
        p = spearman_p(r, len(sub))

        # Least-squares trend line, drawn only to make the direction visible; the
        # reported statistic is the rank correlation, which does not assume linearity.
        if len(sub) > 2:
            k, b = np.polyfit(sub["their_index"], sub["our_index"], 1)
            xs = np.linspace(sub["their_index"].min(), sub["their_index"].max(), 10)
            ax.plot(xs, k*xs + b, "k--", linewidth=1.2, alpha=0.8)

        # Threshold lines: the classification cutoffs both tools use.
        for v in (0.78, -1.0):
            ax.axhline(v, color="grey", linewidth=0.6, linestyle=":")
            ax.axvline(v, color="grey", linewidth=0.6, linestyle=":")

        ax.set_title(f"{title}\nSpearman = {r:+.3f}  (n={len(sub)}, p={p:.4f})", fontsize=11)
        ax.set_xlabel("frustratometeR index (AWSEM)")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("FrustX index (REF2015)")
    fig.suptitle("Per-contact frustration index: FrustX vs frustratometeR  (1UBQ)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_index_distributions(ours, theirs, out_path):
    """Overlaid histograms of the raw index, with the classification thresholds drawn.

    This is the explanation for the class-ratio gap. The two distributions have similar
    WIDTH but different CENTRE, and the thresholds (+0.78 / -1.0) are absolute cuts
    borrowed from AWSEM. A shifted centre therefore moves large numbers of contacts
    across a fixed threshold even when the underlying ranking is unchanged.
    """
    o = ours.dropna(subset=["our_index"])["our_index"]
    t = theirs["their_index"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bins = np.linspace(-3.5, 4.0, 46)
    ax.hist(o, bins=bins, density=True, alpha=0.55, color="#2166ac",
            label=f"FrustX (REF2015)   median {o.median():+.2f}, sd {o.std():.2f}")
    ax.hist(t, bins=bins, density=True, alpha=0.55, color="#b2182b",
            label=f"frustratometeR (AWSEM)   median {t.median():+.2f}, sd {t.std():.2f}")
    ax.axvline(o.median(), color="#2166ac", linestyle="-", linewidth=2)
    ax.axvline(t.median(), color="#b2182b", linestyle="-", linewidth=2)
    for v, lab in ((0.78, "minimally >= +0.78"), (-1.0, "highly <= -1.0")):
        ax.axvline(v, color="black", linestyle="--", linewidth=1.2)
        ax.text(v, ax.get_ylim()[1]*0.94, f" {lab}", fontsize=8, rotation=90,
                va="top", ha="right" if v < 0 else "left")

    ax.set_xlabel("frustration index")
    ax.set_ylabel("density")
    ax.set_title("Index distributions and the borrowed thresholds\n"
                 "similar spread, different centre -- so fixed cuts land differently",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main(frustx_dir, frustra_file, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ours = load_frustx(Path(frustx_dir) / "contacts.csv")
    theirs = load_frustratometer(frustra_file)
    m = ours.merge(theirs, on=["ci", "i", "cj", "j"], how="inner").dropna(
        subset=["our_index", "their_index"])

    ov, tv = plot_class_ratios(ours, theirs, out / "class_ratios.png")
    plot_scatter(m, out / "scatter.png")
    plot_index_distributions(ours, theirs, out / "index_distributions.png")

    # Quantify the centring effect: reclassify our contacts using thresholds shifted by
    # the difference in medians, i.e. ask what the ratios would be if only the centre
    # were matched and nothing else changed.
    o = ours.dropna(subset=["our_index"])["our_index"]
    shift = theirs["their_index"].median() - o.median()
    shifted = o + shift
    n = len(shifted)
    print(f"\nif our index were re-centred on theirs (shift {shift:+.2f}):")
    print(f"  minimally  {(shifted >= 0.78).sum()/n:6.1%}    (was {ov['minimally']:.1%},"
          f" theirs {tv['minimally']:.1%})")
    print(f"  highly     {(shifted <= -1.0).sum()/n:6.1%}    (was {ov['highly']:.1%},"
          f" theirs {tv['highly']:.1%})")

    print("\nclass ratios (all contacts each tool found):")
    for c in CLASSES:
        print(f"  {c:<10} FrustX {ov[c]:6.1%}    frustratometeR {tv[c]:6.1%}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
