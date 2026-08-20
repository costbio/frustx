"""Fig. 2 right-column style: per-residue frustration profile, GDP vs GTP.

Green = minimally frustrated contacts in the 5 A vicinity, red = highly frustrated,
matching the published figure's colour convention. Absolute counts are NOT comparable to
the paper (thresholds and radius unstated there); profile shape is.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REGIONS = {"P-loop": (12, 20), "switch I": (32, 41), "switch II": (63, 79)}


def main():
    m = pd.read_csv("results/fig2/rheb_profile_comparison.csv")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for ax, tag, title in ((axes[0], "gdp", "1XTQ  Rheb-GDP (inactive)"),
                           (axes[1], "gtp", "1XTS  Rheb-GTP (active)")):
        ax.fill_between(m.Res, 0, m[f"nMinimallyFrst_{tag}"], color="tab:green",
                        alpha=0.75, lw=0, label="minimally frustrated")
        ax.fill_between(m.Res, 0, -m[f"nHighlyFrst_{tag}"], color="tab:red",
                        alpha=0.75, lw=0, label="highly frustrated")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("contacts within 5 A")
        ax.set_title(title, loc="left", fontsize=10)

    d_min = m["nMinimallyFrst_gtp"] - m["nMinimallyFrst_gdp"]
    d_high = m["nHighlyFrst_gtp"] - m["nHighlyFrst_gdp"]
    axes[2].bar(m.Res, d_min, color="tab:green", alpha=0.8, label="d minimally")
    axes[2].bar(m.Res, -d_high, color="tab:red", alpha=0.8, label="d highly (plotted down)")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_ylabel("GTP - GDP")
    axes[2].set_xlabel("residue")
    axes[2].set_title("change on activation; functional elements shaded", loc="left", fontsize=10)

    for ax in axes:
        for name, (a, b) in REGIONS.items():
            ax.axvspan(a, b, color="0.55", alpha=0.22, lw=0, zorder=0)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    for name, (a, b) in REGIONS.items():
        axes[0].text((a + b) / 2, axes[0].get_ylim()[1] * 0.92, name,
                     ha="center", fontsize=8, color="0.25")

    fig.suptitle("FrustX per-residue frustration profile, Rheb GDP vs GTP "
                 "(REF2015, w=0, 500 decoys, XAdens 5 A vicinity)", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/fig2/rheb_profile.png", dpi=150)
    print("wrote results/fig2/rheb_profile.png")


if __name__ == "__main__":
    main()
