"""Figures for the paralog pilot comparison.

Reads data/kinome/analysis/pilot_positions*.csv and pilot_comparison*.csv
(scripts/pilot_compare.py) and draws two views of the one question: for a ligand bound to
two closest-relative kinases, does the frustration difference sit at the pocket positions
where the two kinases differ?

    .venv/bin/python scripts/plot_pilot.py --tag _smoke

positions.png is the claim: every matched pocket position, the difference in raw
frustration index between the tighter- and the weaker-binding kinase, with the divergent
positions marked and the decoy noise floor shaded. summary.png is the decision: the
per-pair statistic against that floor, and what a larger decoy ensemble would buy.

Colours are slots 1 and 2 of the reference categorical palette (blue / orange), in fixed
order, matching scripts/plot_cox_selectivity.py. They are a pre-validated adjacent pair;
no custom hues are invented here, and node is unavailable to re-run the validator.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless, same as the other plot scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIV, CONS = "#2a78d6", "#eb6834"      # categorical slots 1, 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "0.90"


def _axes(ax):
    """Recessive grid and spines, as in the other figures."""
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    ax.tick_params(colors=MUTED, labelsize=9)


def positions_figure(pos: pd.DataFrame, summary: pd.DataFrame, path: pathlib.Path):
    """One panel per pair: dF at every matched pocket position.

    Small multiples rather than one crowded axis, because the pairs do not share a
    position set and the y scale is the quantity being compared.
    """
    order = summary.sort_values(["role", "delta_pKd"], ascending=[True, False])
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.25 * n), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, row in zip(axes, order.itertuples()):
        d = pos[pos["pair"] == row.pair]
        _axes(ax)
        # what the decoy count can resolve: anything inside this band is noise
        ax.axhspan(-row.noise_floor_dF, row.noise_floor_dF, color=GRID, zorder=1,
                   label="decoy noise floor")
        ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=2)

        cons = d[~d["divergent"]]
        div = d[d["divergent"]]
        ax.vlines(cons["position"], 0, cons["dF"], color=CONS, lw=2, alpha=0.5, zorder=3)
        ax.scatter(cons["position"], cons["dF"], s=26, color=CONS, zorder=4,
                   label="conserved position")
        ax.vlines(div["position"], 0, div["dF"], color=DIV, lw=2, zorder=5)
        ax.scatter(div["position"], div["dF"], s=70, color=DIV, zorder=6,
                   edgecolor="white", linewidth=1.2, label="divergent position")
        # selective direct labels: only the divergent positions, which are the claim
        for r in div.itertuples():
            ax.annotate(r.klifs_position if hasattr(r, "klifs_position") else
                        str(r.position),
                        (r.position, r.dF), textcoords="offset points",
                        xytext=(0, 9 if r.dF >= 0 else -16), ha="center",
                        fontsize=8, color=INK)

        p = "n/a" if row.p_value is None or np.isnan(row.p_value) else f"{row.p_value:.3f}"
        tag = "negative control" if row.role == "negative_control" else "sharp"
        ax.set_title(f"{row.pair}   {row.ligand_group}   ΔpKd {row.delta_pKd:.2f}   "
                     f"[{tag}]   mean ΔF divergent {row.mean_dF_divergent:+.2f} vs "
                     f"conserved {row.mean_dF_conserved:+.2f}   p {p}",
                     loc="left", fontsize=10, color=INK)
        ax.set_ylabel("ΔF  (tighter − weaker)", fontsize=9, color=MUTED)

    axes[-1].set_xlabel("KLIFS pocket position (1–85)", fontsize=10, color=MUTED)
    axes[-1].set_xlim(0, 86)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Frustration difference per pocket position, tighter-binding kinase "
                 "minus weaker\nabove zero = less frustrated in the tighter binder "
                 "(the hypothesis); inside the grey band = unresolved at this decoy count",
                 fontsize=11, color=INK, ha="left", x=0.045, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_figure(summary: pd.DataFrame, path: pathlib.Path):
    """Left: the per-pair statistic against the noise floor. Right: what more decoys buy."""
    order = summary.sort_values(["role", "delta_pKd"], ascending=[True, False])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(order))[::-1]

    _axes(ax1)
    ax1.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax1.grid(axis="y", visible=False)
    for yy, row in zip(y, order.itertuples()):
        colour = DIV if row.role != "negative_control" else CONS
        ax1.barh(yy, row.statistic, height=0.5, color=colour, zorder=3,
                 edgecolor="white", linewidth=2)
        ax1.errorbar(row.statistic, yy, xerr=row.noise_floor_dF, color=MUTED, lw=1.4,
                     capsize=3, zorder=4)
        ax1.annotate(f"p {row.p_value:.3f}", (row.statistic, yy),
                     textcoords="offset points",
                     xytext=(8 if row.statistic >= 0 else -8, 0),
                     ha="left" if row.statistic >= 0 else "right", va="center",
                     fontsize=9, color=INK)
    ax1.axvline(0, color="0.55", lw=0.8, ls="--", zorder=2)
    ax1.set_yticks(y, [f"{r.pair}\n{r.ligand_group}  ΔpKd {r.delta_pKd:.2f}"
                       for r in order.itertuples()], fontsize=9)
    ax1.set_xlabel("mean ΔF divergent − conserved  (bars: noise floor)", fontsize=10,
                   color=MUTED)
    ax1.set_title("The statistic, per pair", loc="left", fontsize=11, color=INK)

    _axes(ax2)
    n_now = int(order["n_decoys"].iloc[0])
    floor_now = order["noise_floor_dF"].to_numpy()
    # the floor scales as 1/sqrt(n) in its dominant term, so project the same effect
    projected = floor_now * np.sqrt(n_now / 1000.0)
    effect = order["mean_abs_dF"].to_numpy()
    ax2.barh(y + 0.18, effect / floor_now, height=0.34, color=CONS, zorder=3,
             edgecolor="white", linewidth=2, label=f"{n_now} decoys (now)")
    ax2.barh(y - 0.18, effect / projected, height=0.34, color=DIV, zorder=3,
             edgecolor="white", linewidth=2, label="1000 decoys (projected)")
    ax2.axvline(1, color="0.55", lw=0.8, ls="--", zorder=2)
    ax2.set_yticks(y, [r.pair for r in order.itertuples()], fontsize=9)
    ax2.set_xlabel("mean |ΔF| ÷ noise floor   (1 = effect equals noise)", fontsize=10,
                   color=MUTED)
    ax2.set_title("What a bigger decoy ensemble buys", loc="left", fontsize=11, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ana = pathlib.Path("data/kinome/analysis")
    ap.add_argument("--analysis", type=pathlib.Path, default=ana)
    ap.add_argument("--tag", default="", help="suffix of the input files, e.g. _smoke")
    a = ap.parse_args(argv)

    pos = pd.read_csv(a.analysis / f"pilot_positions{a.tag}.csv")
    summary = pd.read_csv(a.analysis / f"pilot_comparison{a.tag}.csv")

    # the KLIFS label per position, for the direct labels on the divergent points
    pmap = pd.read_csv(a.analysis / "pilot_pocket_map.csv", dtype=str)
    labels = dict(zip(pmap["pocket_position"].astype(int), pmap["klifs_position"]))
    pos["klifs_position"] = pos["position"].map(labels)

    p1 = a.analysis / f"pilot_positions{a.tag}.png"
    p2 = a.analysis / f"pilot_summary{a.tag}.png"
    positions_figure(pos, summary, p1)
    summary_figure(summary, p2)
    print(f"wrote {p1}\nwrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
