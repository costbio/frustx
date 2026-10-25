"""Figures for the KLIFS-first complex set analysis (scripts/complex_set_compare.py).

    .venv/bin/python scripts/plot_complex_set.py

The question: across one ligand's co-crystal complexes with many kinases, does the
frustration of its contacts follow the measured affinity? The pre-registered verdict is
NO for every frustration descriptor, while Rosetta's raw interaction energy -- which needs
no decoys -- does track pKd. The figure set is built to show both, and why:

  fig1_classes.png      what the data are: each complex's ligand contacts by frustration
                        class, kinases ordered by pKd within each ligand
  fig2_meanF_vs_pKd.png the primary descriptor against pKd, one panel per ligand
  fig3_interaction_vs_pKd.png  the decoy-free control, same layout -- the contrast
  fig4_rho_heatmap.png  per-ligand Spearman rho for every descriptor and control (Kd)
  fig5_pooled.png       pooled within-ligand statistics at all four energy floors, raw
                        and partial, with and without STU, Kd and IC50
  fig6_stu_pocket.png   STU (35 kinases): frustration of the contact at each KLIFS pocket
                        position, kinases ordered by pKd (exploratory, position level)
  fig7_floor.png        why the floor sweep exists: index vs native energy for every
                        ligand contact

Colours: diverging blue (minimally / positive) <-> gray <-> red (highly / negative) for
frustration and rho, the reference palette's diverging pair; categorical slots 1 and 2
(blue / orange) where two groups are compared, in fixed order, as in plot_pilot.py.
Ink, muted and grid tones are the reference chrome. No hue is invented here.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"            # categorical slots 1, 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list(
    "frustration", ["#b8322f", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
CLASS_COLOURS = {"minimally": "#2a78d6", "neutral": "#c3c2b7", "highly": "#e34948"}
FLOOR_COLOURS = {0.0: "#86b6ef", 0.1: "#3987e5", 0.3: "#1c5cab", 0.5: "#0d366b"}
DESC_LABEL = {"mean_F": "mean F", "frac_minimally": "fraction minimally",
              "frac_highly": "fraction highly", "n_contacts": "ligand contacts",
              "interaction": "interaction energy"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})


def _axes(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _ligand_order(desc):
    k = desc[desc["in_Kd_set"] & (desc["floor"] == 0)]
    return k.groupby("ligand_code").size().sort_values(ascending=False).index.tolist()


def _label(r):
    drug = {"STU": "staurosporine", "1N1": "dasatinib", "DB8": "bosutinib",
            "STI": "imatinib", "B96": "doramapimod", "0LI": "ponatinib",
            "3NG": "silmitasertib", "VX6": "tozasertib"}
    return f"{r} ({drug.get(r, r)})"


def fig_classes(pos, desc, path):
    """Stacked class composition per complex. STU gets the tall left column."""
    k = pos[pos["in_Kd_set"] & pos["frustration_index"].notna()]
    comp = (k.groupby(["ligand_code", "kinase_name", "pKd"])["frustration_class"]
            .value_counts(normalize=True).unstack(fill_value=0).reset_index())
    ligs = _ligand_order(desc)
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.55)
    slots = [gs[:, 0]] + [gs[i, j] for i in range(4) for j in (1, 2)]
    for slot, lig in zip(slots, ligs):
        ax = fig.add_subplot(slot)
        d = comp[comp["ligand_code"] == lig].sort_values("pKd")
        y = np.arange(len(d))
        left = np.zeros(len(d))
        for cls in ("minimally", "neutral", "highly"):
            v = d.get(cls, pd.Series(0, index=d.index)).to_numpy()
            ax.barh(y, v, left=left, color=CLASS_COLOURS[cls], height=0.78,
                    edgecolor="white", linewidth=1, label=cls)
            left += v
        ax.set_yticks(y, [f"{n}  {p:.1f}" for n, p in zip(d["kinase_name"], d["pKd"])],
                      fontsize=7 if len(d) > 15 else 8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        _axes(ax, grid=None)
        ax.set_title(_label(lig), loc="left", fontsize=9)
    ax_leg = fig.add_subplot(slots[len(ligs)]) if len(ligs) < len(slots) else None
    if ax_leg:
        ax_leg.axis("off")
        for cls in ("minimally", "neutral", "highly"):
            ax_leg.barh(0, 0, color=CLASS_COLOURS[cls], label=f"{cls} frustrated")
        ax_leg.legend(loc="center left", frameon=False, fontsize=9,
                      title="ligand contacts", title_fontsize=9)
    fig.suptitle("Ligand-contact frustration per complex (kinase, pKd; tightest binder "
                 "on top)", x=0.06, ha="left", fontsize=11)
    fig.text(0.06, 0.035, "Share of each complex's ligand contacts in each frustration "
             "class (frustx cut points 0.78 / -1.0), floor 0, Kd set.", color=MUTED,
             fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def theil_sen(x, y):
    """Robust straight-line fit: slope = median of all pairwise slopes, intercept =
    median of y - slope * x. Chosen over least squares because the panels report
    Spearman rho (a rank statistic) on 5-35 points, where one outlying complex would pull
    a least-squares line -- and could make it disagree with the rho beside it."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return slope, float(np.median(y - slope * x))


def fig_scatter(desc, per_lig, col, ylabel, title, path, floor=0.0):
    """One panel per ligand: descriptor vs pKd, with that ligand's rho and p, and a
    Theil-Sen trend line."""
    k = desc[desc["in_Kd_set"] & (desc["floor"] == floor)]
    ligs = _ligand_order(desc)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for ax, lig in zip(axes.flat, ligs):
        d = k[k["ligand_code"] == lig]
        _axes(ax, grid="both")
        slope, icpt = theil_sen(d["pKd"], d[col])
        xs = np.array([d["pKd"].min(), d["pKd"].max()])
        ax.plot(xs, icpt + slope * xs, color="#184f95", lw=1.6, zorder=2)
        ax.scatter(d["pKd"], d[col], s=34, color=BLUE, edgecolor="white", linewidth=1,
                   zorder=3)
        if len(d) <= 10:              # selective labels: only where they fit
            for r in d.itertuples():
                ax.annotate(r.kinase_name, (r.pKd, getattr(r, col)), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color=MUTED)
        s = per_lig[(per_lig["measure"] == "Kd") & (per_lig["floor"] == floor)
                    & (per_lig["ligand_code"] == lig) & (per_lig["descriptor"] == col)].iloc[0]
        ax.set_title(f"{_label(lig)}\nrho {s.rho:+.2f}   p {s.p:.3f}   n {int(s.n)}",
                     loc="left", fontsize=8.5)
        ax.set_xlabel("pKd", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.06, ha="left", fontsize=11)
    fig.text(0.06, -0.01, "Line: Theil-Sen robust fit (median of pairwise slopes), shown "
             "for direction only. rho and p are Spearman with a permutation test; each "
             "panel has its own y scale.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rho_heatmap(per_lig, desc, path, floor=0.0):
    """Ligand x descriptor rho (Kd), diverging, with p in each cell."""
    cols = ["mean_F", "frac_minimally", "frac_highly", "n_contacts", "interaction"]
    ligs = _ligand_order(desc)
    d = per_lig[(per_lig["measure"] == "Kd") & (per_lig["floor"] == floor)]
    rho = d.pivot(index="ligand_code", columns="descriptor", values="rho").loc[ligs, cols]
    pv = d.pivot(index="ligand_code", columns="descriptor", values="p").loc[ligs, cols]
    n = d.groupby("ligand_code")["n"].max().loc[ligs]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(rho.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1),
                   aspect="auto")
    for i in range(len(ligs)):
        for j in range(len(cols)):
            r, p = rho.iat[i, j], pv.iat[i, j]
            ax.text(j, i, f"{r:+.2f}\np {p:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(r) > 0.65 else INK,
                    fontweight="bold" if p < 0.05 else "normal")
    ax.set_xticks(range(len(cols)), [DESC_LABEL[c] for c in cols], rotation=20,
                  ha="right")
    ax.set_yticks(range(len(ligs)), [f"{_label(l)}  n={n[l]}" for l in ligs])
    ax.axvline(2.5, color="white", lw=3)          # frustration | controls
    ax.tick_params(length=0, labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman rho with pKd", color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("Per-ligand rank correlation with pKd (floor 0; bold: p < 0.05)\n"
                 "frustration descriptors  |  controls", loc="left", fontsize=10)
    fig.text(0.01, -0.02, "Hypothesis direction: + for mean F, fraction minimally and "
             "interaction energy; - for fraction highly (more highly frustrated contacts, "
             "weaker binding).", color=MUTED, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pooled(pooled, path):
    """Pooled within-ligand statistic for every descriptor/test at all four floors.
    Filled marker: p < 0.05; hollow: not significant."""
    rows = [("mean_F", "pooled"), ("mean_F", "partial"),
            ("frac_minimally", "pooled"), ("frac_minimally", "partial"),
            ("frac_highly", "pooled"), ("frac_highly", "partial"),
            ("n_contacts", "pooled"), ("interaction", "pooled")]
    labels = [f"{DESC_LABEL[d]}" + ("  | controls removed" if t == "partial" else "")
              for d, t in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    for (i, m), (j, scope) in [((i, m), (j, s)) for i, m in enumerate(("Kd", "IC50"))
                               for j, s in enumerate(("all", "without_STU"))]:
        ax = axes[i, j]
        _axes(ax, grid="x")
        ax.axvline(0, color="#898781", lw=0.9, ls="--")
        ax.axhspan(5.5, 7.5, color="#f0efec", zorder=0)     # the two controls
        d = pooled[(pooled["measure"] == m) & (pooled["scope"] == scope)]
        for y, (desc, test) in enumerate(rows):
            for k, f in enumerate(sorted(FLOOR_COLOURS)):
                r = d[(d["descriptor"] == desc) & (d["test"] == test) & (d["floor"] == f)]
                if r.empty:
                    continue
                r = r.iloc[0]
                sig = r.p < 0.05
                ax.scatter(r.stat, y + (k - 1.5) * 0.16, s=46, zorder=3,
                           color=FLOOR_COLOURS[f] if sig else "white",
                           edgecolor=FLOOR_COLOURS[f], linewidth=1.6,
                           label=f"floor {f:g} kcal/mol" if (y == 0) else None)
        n = int(d["n"].max())
        ax.set_title(f"{m}, {'all ligands' if scope == 'all' else 'without STU'} "
                     f"(n = {n})", loc="left", fontsize=9.5)
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("pooled within-ligand rank correlation with affinity", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Pooled tests at every energy floor (filled = p < 0.05; shaded rows "
                 "are the controls)", x=0.06, ha="left", fontsize=11, y=1.0)
    fig.legend(h, l, loc="upper left", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.06, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def fig_stu_pocket(pos, path, min_kinases=10, dead=0.1):
    """STU: frustration of its contact at each KLIFS pocket position, per kinase.
    Positions contacted in >= min_kinases complexes. Dots mark contacts with
    |e_ij| < dead (the ones an energy floor would remove). Top: rho(pKd, F) per
    position, Benjamini-Hochberg q -- exploratory, not part of the pre-registered test."""
    s = pos[(pos["ligand_code"] == "STU") & pos["in_Kd_set"]
            & pos["pocket_position"].notna() & pos["frustration_index"].notna()]
    cnt = s.groupby("pocket_position")["kinase_name"].nunique()
    keep = sorted(cnt[cnt >= min_kinases].index)
    s = s[s["pocket_position"].isin(keep)]
    kin = s.groupby("kinase_name")["pKd"].first().sort_values(ascending=False)
    F = s.pivot_table(index="kinase_name", columns="pocket_position",
                      values="frustration_index").reindex(index=kin.index, columns=keep)
    E = s.pivot_table(index="kinase_name", columns="pocket_position",
                      values="native_energy").reindex(index=kin.index, columns=keep)
    labels = s.groupby("pocket_position")["klifs_position"].first().loc[keep]

    rhos, ps = [], []
    for c in keep:
        x = pd.concat([kin, F[c]], axis=1).dropna()
        r = x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank())
        # permutation p, 5000 draws
        rng = np.random.default_rng(int(c))
        a, b = x.iloc[:, 0].rank().to_numpy(), x.iloc[:, 1].rank().to_numpy()
        null = [abs(np.corrcoef(rng.permutation(a), b)[0, 1]) for _ in range(5000)]
        rhos.append(r)
        ps.append(float(np.mean(np.array(null) >= abs(r) - 1e-12)))
    q = _bh(ps)

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 6], width_ratios=[30, 1.2],
                          hspace=0.05, wspace=0.03)
    ax_top, ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(keep))
    ax_top.bar(x, rhos, color=[BLUE if r > 0 else "#e34948" for r in rhos], width=0.7)
    for xi, (r, qq) in enumerate(zip(rhos, q)):
        if qq < 0.1:
            ax_top.text(xi, r + (0.05 if r > 0 else -0.12), "*", ha="center", fontsize=11)
    ax_top.axhline(0, color="#898781", lw=0.8)
    ax_top.set_xlim(-0.5, len(keep) - 0.5)
    ax_top.set_xticks([])
    ax_top.set_ylim(-1, 1)
    ax_top.set_ylabel("rho(pKd, F)", fontsize=8)
    _axes(ax_top, grid="y")
    ax_top.set_title("Staurosporine: frustration of the ligand contact at each KLIFS "
                     "pocket position (kinases by pKd, tightest on top)\n"
                     "top: per-position rho with pKd (* BH q < 0.1, exploratory); "
                     "dot: |e_ij| < 0.1 kcal/mol; blank: no contact", loc="left",
                     fontsize=9.5)
    im = ax.imshow(F.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(0, -3, 3),
                   aspect="auto", interpolation="none")
    yy, xx = np.where(E.abs().to_numpy() < dead)
    ax.scatter(xx, yy, s=5, color=INK, zorder=3)
    ax.set_xticks(x, labels.tolist(), rotation=90, fontsize=7)
    ax.set_yticks(range(len(kin)), [f"{k}  {p:.1f}" for k, p in kin.items()], fontsize=7)
    ax.tick_params(length=0, colors=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("frustration index (clipped at +/-3)", color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame({"pocket_position": keep, "klifs_position": labels.to_numpy(),
                         "n_kinases": cnt.loc[keep].to_numpy(), "rho": rhos, "p": ps,
                         "q": q})


def fig_floor(pos, path):
    """Every ligand contact: index vs native energy. The z-score's artefact is the cloud
    of large |F| at near-zero energy; the dashed lines are the four floors."""
    d = pos[pos["frustration_index"].notna()]
    tiny = d["decoy_std"] <= 0.02
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _axes(ax, grid="both")
    ax.scatter(d.loc[~tiny, "native_energy"], d.loc[~tiny, "frustration_index"], s=10,
               color=BLUE, alpha=0.45, linewidth=0, label="decoy sigma > 0.02")
    ax.scatter(d.loc[tiny, "native_energy"], d.loc[tiny, "frustration_index"], s=14,
               color=ORANGE, alpha=0.85, linewidth=0, label="decoy sigma <= 0.02")
    for f in (0.1, 0.3, 0.5):          # the floor is on |e_ij|: both sides
        ax.axvline(-f, color="#898781", lw=0.8, ls="--")
        ax.axvline(f, color="#898781", lw=0.8, ls="--")
    ax.text(-0.6, ax.get_ylim()[1] * 0.95, "floors |e_ij| = 0.1 / 0.3 / 0.5", fontsize=7.5,
            color=MUTED, ha="right")
    ax.set_xlabel("native interaction energy of the contact e_ij (kcal/mol)")
    ax.set_ylabel("frustration index F")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"All {len(d)} ligand contacts: the index is a z-score, so near-zero "
                 "contacts can score high", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--analysis", type=pathlib.Path,
                   default=pathlib.Path("results/complex_set_analysis"))
    a = p.parse_args(argv)
    d = a.analysis
    desc = pd.read_csv(d / "descriptors.csv")
    per_lig = pd.read_csv(d / "per_ligand.csv")
    pooled = pd.read_csv(d / "pooled.csv")
    pos = pd.read_csv(d / "positions.csv")

    fig_classes(pos, desc, d / "fig1_classes.png")
    fig_scatter(desc, per_lig, "mean_F", "mean frustration index of ligand contacts",
                "Primary descriptor: mean ligand-contact frustration vs pKd (floor 0)",
                d / "fig2_meanF_vs_pKd.png")
    fig_scatter(desc, per_lig, "interaction",
                "interaction energy, -sum e_ij (kcal/mol)",
                "Control: Rosetta interaction energy at the ligand interface vs pKd "
                "(no decoys; floor 0)", d / "fig3_interaction_vs_pKd.png")
    fig_rho_heatmap(per_lig, desc, d / "fig4_rho_heatmap.png")
    fig_pooled(pooled, d / "fig5_pooled.png")
    stu = fig_stu_pocket(pos, d / "fig6_stu_pocket.png")
    stu.to_csv(d / "stu_positions.csv", index=False)
    fig_floor(pos, d / "fig7_floor.png")
    print(f"wrote 7 figures to {d}")


if __name__ == "__main__":
    main()
