"""Figures for the COX-1/COX-2 selectivity analysis.

Reads results/cox_selectivity/selectivity_delta.csv and per_run_interface.csv
(scripts/cox_selectivity.py) and draws five views of the same question: does
ligand-interface frustration separate COX-2-selective inhibitors from
non-selective ones?

**The figure set is built to show the negative result, not to hide it.** The raw
separation is strong (AUC ~0.87) and would make a convincing single panel. It does
not survive controlling for the docking score and the ligand contact count, so
`auc_ladder.png` and `confound.png` are the load-bearing figures here -- the first
two panels are the claim, the last two are why it does not hold.

    .venv/bin/python scripts/plot_cox_selectivity.py

Colours are slots 1 and 2 of the reference categorical palette (blue / orange),
used in fixed order. They are a pre-validated adjacent pair; no custom hues are
invented here, and node was unavailable to re-run the validator.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # headless, same as the other plot scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cox_selectivity_stats import auc, residualise

SEL, NON = "#2a78d6", "#eb6834"      # categorical slots 1, 2
INK, MUTED = "#0b0b0b", "#52514e"

# The control stages (raw -> minus docking -> minus docking and contacts) are an
# ordered progression, not identities, so they get a single-hue sequential ramp
# rather than the categorical palette. Keeping them off blue/orange means those
# two hues mean "Selective" and "Non-Selective" everywhere in the figure set and
# nowhere else -- reusing them for a stage would make one colour encode two
# different things across panels.
STAGE = ["#b3aee0", "#7a6cc4", "#4a3aa7"]   # violet, light -> dark
OUT = pathlib.Path("results/cox_selectivity/figures")

# Two pairs whose cox1/cox2 chemistry still differs by stereo / bond-order
# perception only. Marked in the figures rather than dropped, so the reader can
# see they are not driving anything.
SUSPECT = {"pd-138387", "sulindac-sulfide"}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def roc(values, positive):
    """ROC points for a score, ties handled by sorting on the score alone."""
    order = np.argsort(-values)
    hits = np.cumsum(positive[order])
    false = np.cumsum(~positive[order])
    return (np.r_[0, false / false[-1]], np.r_[0, hits / hits[-1]])


# ---------------------------------------------------------------- figure 1
def fig_paired_shift(runs, delta, path):
    """Per inhibitor, mean interface frustration in COX-1 -> COX-2.

    A dumbbell rather than two box plots: the measurement is paired (same molecule,
    two targets), and unpaired summaries throw that away.
    """
    wide = runs.pivot(index="ligand", columns="target", values="mean_frustration")
    cls = delta.set_index("ligand")["selectivity_class"]
    # Ordered by the quantity actually plotted (the COX2-COX1 shift), not by either
    # endpoint -- sorting on an endpoint interleaves the classes and hides the very
    # pattern the panel exists to show.
    wide = wide.join(cls).dropna()
    wide = wide.assign(shift=wide["cox2"] - wide["cox1"]).sort_values("shift")
    y = np.arange(len(wide))

    fig, ax = plt.subplots(figsize=(8.5, 11))
    for k, (lig, r) in enumerate(wide.iterrows()):
        c = SEL if r["selectivity_class"] == "Selective" else NON
        ax.plot([r["cox1"], r["cox2"]], [k, k], color=c, lw=2, alpha=0.55, zorder=2)
        ax.scatter(r["cox1"], k, s=26, facecolor="white", edgecolor=c, lw=1.6, zorder=3)
        ax.scatter(r["cox2"], k, s=40, color=c, zorder=3)
    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.set_yticks(y, [f"{l}*" if l in SUSPECT else l for l in wide.index], fontsize=7.5)
    ax.set_xlabel("mean frustration index over the ligand interface")
    ax.set_ylim(-1, len(wide))

    # Legend by hand: the shape carries target, the colour carries class, so a
    # default legend would merge two encodings into one confusing set of entries.
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="COX-1 (hollow)"),
        plt.Line2D([], [], marker="o", ls="", color=INK, label="COX-2 (filled)"),
        plt.Line2D([], [], color=SEL, lw=3, label="Selective"),
        plt.Line2D([], [], color=NON, lw=3, label="Non-Selective"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title("Interface frustration shifts right (less frustrated) in COX-2\n"
                 "for selective compounds; * = chemistry differs by stereo only",
                 loc="left", fontsize=11, color=INK)
    tidy(ax); ax.grid(axis="y", lw=0)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 2
def fig_delta_by_class(delta, path):
    """Distribution of the COX2-COX1 difference, by class, for four descriptors."""
    cols = [("d_n_minimally", "minimally frustrated contacts gained"),
            ("d_sum_frustration", "summed interface frustration"),
            ("d_mean_frustration", "mean interface frustration"),
            ("d_n_ligand_contacts", "ligand contacts gained")]
    pos = (delta["selectivity_class"] == "Selective").to_numpy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, cols):
        v = delta[col].to_numpy()
        for k, (mask, c, name) in enumerate(((pos, SEL, "Selective"),
                                             (~pos, NON, "Non-Selective"))):
            d = v[mask]
            ax.boxplot(d, positions=[k], widths=0.5, showfliers=False,
                       medianprops=dict(color=INK, lw=1.6),
                       boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED),
                       capprops=dict(color=MUTED))
            ax.scatter(k + rng.uniform(-0.13, 0.13, len(d)), d, s=20, color=c,
                       alpha=0.75, zorder=3, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.set_xticks([0, 1], ["Selective", "Non-Sel."], fontsize=9)
        ax.set_title(f"{label}\nAUC = {auc(v, pos):.3f}", fontsize=9.5, color=INK)
        tidy(ax)
    axes[0].set_ylabel("COX-2 minus COX-1")
    fig.suptitle("Every descriptor separates the two classes -- before any control",
                 fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 3
def fig_roc(delta, path):
    """ROC for the docking baseline and the best frustration descriptor.

    Second panel is the same descriptor after the docking score and the contact
    count are regressed out. The gap between the panels is the finding.
    """
    pos = (delta["selectivity_class"] == "Selective").to_numpy()
    dock = -delta["dock_ddG"].to_numpy()        # more negative ddG = more selective
    cont = delta["d_n_ligand_contacts"].to_numpy()
    best = delta["d_n_minimally"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    panels = [
        (axes[0], "As measured",
         [(dock, "docking ddG (baseline)", MUTED),
          (best, "d_n_minimally", STAGE[0])]),
        (axes[1], "After removing docking ddG and contact count",
         [(residualise(best, np.column_stack([dock])), "d_n_minimally - dock", STAGE[1]),
          (residualise(best, np.column_stack([dock, cont])),
           "d_n_minimally - dock - contacts", STAGE[2])]),
    ]
    for ax, title, series in panels:
        ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls="--", zorder=1)
        for v, label, c in series:
            x, y = roc(v, pos)
            ax.plot(x, y, lw=2, color=c, label=f"{label}   AUC {auc(v, pos):.3f}", zorder=3)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        tidy(ax); ax.grid(color="0.92", lw=0.6)
    axes[0].set_ylabel("true positive rate")
    fig.suptitle("The frustration signal is not independent of what we already knew",
                 fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 4
def fig_confound(delta, path):
    """Why it collapses: the frustration descriptor tracks the contact count."""
    pos = (delta["selectivity_class"] == "Selective").to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    pairs = [(axes[0], "d_n_ligand_contacts", "d_sum_frustration",
              "ligand contacts gained in COX-2", "summed interface frustration"),
             (axes[1], "dock_ddG", "d_sum_frustration",
              "docking ddG (COX2 - COX1, kcal/mol)", "summed interface frustration")]
    for ax, xc, yc, xl, yl in pairs:
        x, y = delta[xc].to_numpy(), delta[yc].to_numpy()
        for mask, c, name in ((pos, SEL, "Selective"), (~pos, NON, "Non-Selective")):
            ax.scatter(x[mask], y[mask], s=44, color=c, alpha=0.8, label=name,
                       edgecolor="white", linewidth=0.6, zorder=3)
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, k * xs + b, color=INK, lw=1.2, ls="--", alpha=0.7, zorder=2)
        # Rank correlation, so a couple of extreme ligands cannot manufacture it.
        r = np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"Spearman = {r:+.3f}", fontsize=10.5, color=INK, loc="left")
        ax.axhline(0, color="0.8", lw=0.7); ax.axvline(0, color="0.8", lw=0.7)
        ax.legend(fontsize=8.5, framealpha=0.95); tidy(ax)
    fig.suptitle("Interface frustration is largely a restatement of how many contacts "
                 "the ligand makes", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 5
def fig_auc_ladder(delta, path):
    """The headline: AUC of each descriptor, raw and after each control."""
    pos = (delta["selectivity_class"] == "Selective").to_numpy()
    dock = -delta["dock_ddG"].to_numpy()
    cont = delta["d_n_ligand_contacts"].to_numpy()
    descriptors = [c for c in delta.columns if c.startswith("d_")]

    rows = []
    for name in descriptors:
        v = delta[name].to_numpy()
        # An AUC below 0.5 just means the descriptor runs the other way; folding it
        # up makes the three bars comparable as discrimination, which is the point.
        a0 = auc(v, pos)
        a1 = auc(residualise(v, np.column_stack([dock])), pos)
        a2 = (np.nan if name == "d_n_ligand_contacts"
              else auc(residualise(v, np.column_stack([dock, cont])), pos))
        rows.append((name, max(a0, 1 - a0), max(a1, 1 - a1),
                     np.nan if np.isnan(a2) else max(a2, 1 - a2)))
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(rows)); h = 0.26
    cols = [("raw", STAGE[0], +h), ("minus docking ddG", STAGE[1], 0.0),
            ("minus docking + contact count", STAGE[2], -h)]
    for k, (label, c, off) in enumerate(cols):
        vals = [r[k + 1] for r in rows]
        ax.barh(y + off, [v - 0.5 if not np.isnan(v) else 0 for v in vals], height=h,
                left=0.5, color=c, label=label, zorder=3)
        for yy, v in zip(y + off, vals):
            if not np.isnan(v):
                ax.text(v + 0.006, yy, f"{v:.2f}", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(0.5, color=INK, lw=1)
    ax.text(0.502, -0.75, "0.5 = no discrimination", fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.5, 0.95); ax.set_xlabel("AUC (folded: 0.5 = chance)")
    ax.invert_yaxis()
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.set_title("Nothing survives both controls\n"
                 "contact count alone (2nd bar) is as good as any frustration descriptor",
                 loc="left", fontsize=11, color=INK)
    tidy(ax); ax.grid(axis="x", color="0.92", lw=0.6); ax.grid(axis="y", lw=0)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


def main():
    base = pathlib.Path("results/cox_selectivity")
    delta = pd.read_csv(base / "selectivity_delta.csv").dropna(subset=["selectivity_class"])
    runs = pd.read_csv(base / "per_run_interface.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"n = {len(delta)} inhibitors")
    fig_paired_shift(runs, delta, OUT / "paired_shift.png")
    fig_delta_by_class(delta, OUT / "delta_by_class.png")
    fig_roc(delta, OUT / "roc.png")
    fig_confound(delta, OUT / "confound.png")
    fig_auc_ladder(delta, OUT / "auc_ladder.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
