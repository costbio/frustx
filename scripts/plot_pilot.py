"""Figures for the paralog pilot comparison.

Reads data/kinome/analysis/pilot_positions*.csv and pilot_comparison*.csv
(scripts/pilot_compare.py) and draws two views of the one question: for a ligand bound to
two closest-relative kinases, does the frustration difference sit at the pocket positions
where the two kinases differ?

    .venv/bin/python scripts/plot_pilot.py --tag _smoke

**The figure set is built to show the negative result, not to hide it.** One pair
(ABL1/ABL2) looks significant at p 0.02 and would make a convincing single panel. It is
one contact carrying no interaction energy, so `confound.png` and `sweep.png` are the
load-bearing figures here -- the first two are the claim, the last two are why it does
not hold and why the 1000-decoy run was needed to know that.

  positions.png  the claim: dF at every matched pocket position, divergent marked
  summary.png    the same claim as one number per pair, with its noise floor
  confound.png   why: the extreme dF values sit at contacts with no energy
  sweep.png      and so: the test collapses as soon as an energy floor is applied
  decoys.png     the 50- vs 1000-decoy statistic, where one apparent signal flipped sign

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
    """The headline per pair: the statistic, its noise floor, and the p-value.

    A dot-and-whisker rather than bars: the quantity is a signed difference of means with
    an uncertainty, not a magnitude from zero. The p label is placed beyond the WHISKER
    end, not the point -- anchoring it to the point put the text on top of the whisker
    whenever the noise floor was wide, which is what the 50-decoy figure did.
    """
    order = summary.sort_values(["role", "delta_pKd"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(8.4, 0.72 * len(order) + 2.0))
    y = np.arange(len(order))[::-1]
    _axes(ax)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.grid(axis="y", visible=False)

    for yy, row in zip(y, order.itertuples()):
        colour = CONS if row.role == "negative_control" else DIV
        ax.errorbar(row.statistic, yy, xerr=row.noise_floor_dF, color=MUTED, lw=1.4,
                    capsize=4, zorder=3)
        ax.scatter(row.statistic, yy, s=110, color=colour, zorder=4,
                   edgecolor="white", linewidth=1.8)
        side = 1 if row.statistic >= 0 else -1
        ax.annotate(f"p {row.p_value:.3f}",
                    (row.statistic + side * row.noise_floor_dF, yy),
                    textcoords="offset points", xytext=(side * 11, 0),
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=9, color=INK)
    ax.axvline(0, color="0.55", lw=0.8, ls="--", zorder=2)
    ax.set_yticks(y, [f"{r.pair}\n{r.ligand_group}  ΔpKd {r.delta_pKd:.2f}"
                      + ("  [neg. control]" if r.role == "negative_control" else "")
                      for r in order.itertuples()], fontsize=9)
    # room for the labels at both ends
    span = float(np.max(np.abs(order["statistic"]) + order["noise_floor_dF"]))
    ax.set_xlim(-span * 1.55, span * 1.55)
    # At 1000 decoys the floor is ~0.05, smaller than the marker, so the whiskers are
    # drawn but invisible. Stating the number keeps that from reading as "no uncertainty
    # was shown" -- at 50 decoys the same whiskers are plainly visible.
    floor = float(order["noise_floor_dF"].median())
    ax.set_xlabel(f"mean ΔF divergent − conserved   (whiskers: decoy noise floor, "
                  f"±{floor:.2f}" + (", smaller than the marker)" if floor < 0.1 else ")"),
                  fontsize=10, color=MUTED)
    ax.set_title("The statistic per pair, before any energy floor", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confound_figure(pos: pd.DataFrame, path: pathlib.Path):
    """Why the apparent result is not one: |dF| against the energy the contact carries.

    The index is a z-score, so it is scale-free -- a contact with no interaction energy
    divides a tiny difference by a tiny sigma and lands anywhere. This is the load-bearing
    panel: the largest |dF| in the whole pilot belongs to a contact of -0.07 kcal/mol.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    _axes(ax1)
    for mask, colour, label, size in ((~pos["divergent"], CONS, "conserved position", 26),
                                      (pos["divergent"], DIV, "divergent position", 70)):
        d = pos[mask]
        ax1.scatter(d["min_abs_native"], d["dF"].abs(), s=size, color=colour,
                    alpha=0.8, zorder=3, label=label,
                    edgecolor="white" if size > 30 else "none", linewidth=1.2)
    ax1.axvline(0.02, color="0.55", lw=1.0, ls="--", zorder=2)
    ax1.annotate("0.02 kcal/mol: config.py's\nenergetically dead threshold", (0.02, 0),
                 textcoords="offset points", xytext=(-6, 30), fontsize=8.5, color=MUTED,
                 ha="right")
    worst = pos.loc[pos["dF"].abs().idxmax()]
    ax1.annotate(f"{worst['pair']}  {worst.get('klifs_position', '')}\n"
                 f"{worst['min_abs_native']:.3f} kcal/mol → |ΔF| {abs(worst['dF']):.1f}",
                 (worst["min_abs_native"], abs(worst["dF"])),
                 textcoords="offset points", xytext=(26, -4), fontsize=9, color=INK,
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    # symlog: the energies are non-negative but include exact zeros, so a plain log axis
    # would drop them; the limit keeps the negative half of the symlog off the canvas
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_xlim(left=-0.002)
    ax1.set_xlabel("interaction energy of the contact, min of the pair  (|kcal/mol|)",
                   fontsize=10, color=MUTED)
    ax1.set_ylabel("|ΔF|", fontsize=10, color=MUTED)
    ax1.set_title("A z-score of nothing is still a large z-score", loc="left",
                  fontsize=11, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="center right")

    _axes(ax2)
    counts = [((pos["min_abs_native"] < f).sum(), f) for f in (0.02, 0.1, 0.3, 0.5)]
    ax2.barh(range(len(counts)), [c for c, _ in counts], height=0.55, color=CONS,
             zorder=3, edgecolor="white", linewidth=2)
    ax2.set_yticks(range(len(counts)), [f"< {f} kcal/mol" for _, f in counts], fontsize=9)
    for k, (c, _) in enumerate(counts):
        ax2.annotate(f"{c} of {len(pos)}", (c, k), textcoords="offset points",
                     xytext=(6, 0), va="center", fontsize=9, color=INK)
    ax2.set_xlabel("matched positions below the floor", fontsize=10, color=MUTED)
    ax2.set_title("How much of the interface is energetically empty", loc="left",
                  fontsize=11, color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sweep_figure(sweeps: pd.DataFrame, path: pathlib.Path):
    """The test at every energy floor, frustration beside raw energy.

    Small multiples rather than five series on one axis: each panel then carries only two
    series, which is the validated colour pair, and no hue has to be invented.
    """
    pairs = sorted(sweeps["pair"].unique(),
                   key=lambda p: (sweeps.loc[sweeps["pair"] == p, "role"].iloc[0] ==
                                  "negative_control", p))
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.0 * len(pairs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pair in zip(axes, pairs):
        _axes(ax)
        d = sweeps[sweeps["pair"] == pair]
        for q, colour, label in (("dF", DIV, "frustration (ΔF)"),
                                 ("dE", CONS, "raw energy (ΔE)")):
            dd = d[d["quantity"] == q].sort_values("energy_floor")
            ax.plot(dd["energy_floor"], dd["p_value"], color=colour, lw=2, marker="o",
                    ms=7, zorder=4, label=label)
        ax.axhline(0.05, color="0.55", lw=1.0, ls="--", zorder=2)
        ax.set_ylim(-0.03, 1.0)
        role = d["role"].iloc[0]
        ax.set_title(f"{pair}\nΔpKd {d['delta_pKd'].iloc[0]:.2f}"
                     + ("  [neg. control]" if role == "negative_control" else ""),
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("energy floor (kcal/mol)", fontsize=9, color=MUTED)
    axes[0].set_ylabel("permutation p-value", fontsize=10, color=MUTED)
    axes[0].annotate("α = 0.05", (0.0, 0.05), textcoords="offset points", xytext=(2, 6),
                     fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The one significant result survives no energy floor, and the cheap "
                 "comparator never had it either", fontsize=11, color=INK, ha="left",
                 x=0.012, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def decoys_figure(now: pd.DataFrame, smoke: pd.DataFrame, path: pathlib.Path):
    """50 vs 1000 decoys: which pairs held and which were sampling noise."""
    m = smoke.merge(now, on="pair", suffixes=("_50", "_1000"))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _axes(ax)
    ax.grid(axis="x", visible=False)
    for r in m.itertuples():
        colour = CONS if r.role_1000 == "negative_control" else DIV
        ax.plot([0, 1], [r.statistic_50, r.statistic_1000], color=colour, lw=2,
                alpha=0.85, zorder=3)
        ax.scatter([0, 1], [r.statistic_50, r.statistic_1000], s=60, color=colour,
                   zorder=4, edgecolor="white", linewidth=1.5)
        flipped = np.sign(r.statistic_50) != np.sign(r.statistic_1000)
        ax.annotate(f"{r.pair}{'  ← sign flipped' if flipped else ''}",
                    (1, r.statistic_1000), textcoords="offset points", xytext=(10, 0),
                    va="center", fontsize=9, color=INK if flipped else MUTED)
    ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=2)
    ax.set_xticks([0, 1], ["50 decoys", "1000 decoys"], fontsize=10)
    ax.set_xlim(-0.15, 1.75)
    ax.set_ylabel("mean ΔF divergent − conserved", fontsize=10, color=MUTED)
    ax.set_title("What twenty times more sampling changed", loc="left", fontsize=11,
                 color=INK)
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

    sweeps = pd.read_csv(a.analysis / f"pilot_sweep{a.tag}.csv")
    out = []
    p1 = a.analysis / f"pilot_positions{a.tag}.png"
    positions_figure(pos, summary, p1); out.append(p1)
    p1b = a.analysis / f"pilot_summary{a.tag}.png"
    summary_figure(summary, p1b); out.append(p1b)
    p2 = a.analysis / f"pilot_confound{a.tag}.png"
    confound_figure(pos, p2); out.append(p2)
    p3 = a.analysis / f"pilot_sweep{a.tag}.png"
    sweep_figure(sweeps, p3); out.append(p3)
    smoke = a.analysis / "pilot_comparison_smoke.csv"
    if smoke.exists() and not a.tag:
        p4 = a.analysis / "pilot_decoys.png"
        decoys_figure(summary, pd.read_csv(smoke), p4); out.append(p4)
    print("\n".join(f"wrote {x}" for x in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
