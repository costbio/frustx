"""The paralog pilot's comparison: is the affinity difference carried by the divergent
pocket positions?

One ligand, two closest-relative kinases, a known affinity difference. For every KLIFS
pocket position that the ligand touches in BOTH structures, this takes the difference in
frustration between the two complexes and asks whether that difference sits at the
positions where the two kinases differ in sequence.

    .venv/bin/python scripts/pilot_compare.py --results results/pilot

The design was fixed BEFORE any result was looked at (see docs/method.md, "The kinome
dataset"), and is implemented here unchanged:

  quantity   dF(p) = F_high(p) - F_low(p), the difference in the RAW frustration index of
             the ligand's contact at pocket position p, high- minus low-affinity kinase.
             High positive = minimally frustrated (the repo's sign convention), so the
             hypothesis predicts dF > 0 at divergent positions: the tighter-binding
             kinase should be less frustrated there.
  test       mean dF at divergent positions vs at conserved positions, with significance
             from permuting the divergent/conserved labels within each pair (the number
             of divergent positions is held fixed).
  covariate  the ligand's contact count per complex, reported beside every result,
             because the COX experiment's entire signal turned out to be contact count.
  control    BMX/BTK, where the ligand binds both kinases equally (delta pKd 0.00) and a
             sound index must show no enrichment.

Why the RAW index and not `frustration_index_specific`: the specific component is an OLS
residual, so it sums to exactly zero over one residue's contacts, and the ligand is one
residue (verified on these runs: -1.8e-12 over 33 contacts). Divergent and conserved
positions are a partition of the ligand's contacts, so in the specific component one half
is positive whenever the other is negative, by algebra rather than by biology. Any
subset-mean of `specific` over a ligand interface is that identity, not a result. It is
reported per position for interpretation and never aggregated.

Also reported, because it decides whether a bigger decoy ensemble is worth the compute:
the per-contact noise floor. The index is a z-score against an ensemble of n decoys, so
its standard error is about sqrt(1/n + F^2/(2(n-1))) -- at n=50 roughly 0.14 and at
n=1000 roughly 0.03, i.e. a 1000-decoy run resolves differences about 4.5x smaller.
`effect_vs_noise` is the observed mean |dF| over that floor.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05
N_PERMUTATIONS = 20000
SEED = 20261011


# --- pure logic --------------------------------------------------------------------

def ligand_contacts(contacts: pd.DataFrame, code: str) -> pd.DataFrame:
    """The ligand's contacts, with the protein partner's residue number as `resnum`.

    The ligand is appended last so in practice it is always the j side, but both are
    tested so that a future reordering fails loudly instead of returning an empty
    interface (the lesson from scripts/cox_selectivity.py)."""
    i_is_lig = contacts["resname_i"] == code
    j_is_lig = contacts["resname_j"] == code
    lig = contacts[i_is_lig ^ j_is_lig].copy()
    lig["resnum"] = np.where(lig.loc[:, "resname_i"] == code,
                             lig["resnum_j"], lig["resnum_i"])
    lig["partner"] = np.where(lig.loc[:, "resname_i"] == code,
                              lig["resname_j"], lig["resname_i"])
    return lig[["resnum", "partner", "frustration_index", "frustration_index_onebody",
                "frustration_index_specific", "frustration_class", "decoy_std",
                "native_energy", "decoy_mean"]]


def index_standard_error(index: pd.Series, n_decoys: int) -> pd.Series:
    """Standard error of a frustration index estimated from `n_decoys` decoys.

    F = (mean_decoy - native) / sd_decoy. The mean carries sd/sqrt(n), which is 1 unit of
    F over sqrt(n); the sd itself has relative error 1/sqrt(2(n-1)), which scales F. The
    two combine (delta method, independent) to sqrt(1/n + F^2/(2(n-1))).
    """
    return np.sqrt(1.0 / n_decoys + index ** 2 / (2 * (n_decoys - 1)))


def matched_positions(hi: pd.DataFrame, lo: pd.DataFrame, pocket_hi: dict,
                      pocket_lo: dict, divergent: set[int]) -> pd.DataFrame:
    """One row per pocket position the ligand touches in BOTH complexes.

    `pocket_*` map a structure's residue number to its KLIFS pocket position, which is
    what makes two different kinases comparable at all. A position touched in only one
    of the two is dropped and counted separately: it is a real difference, but not one
    this quantity can speak about, since dF needs both sides.
    """
    h = hi.assign(position=hi["resnum"].map(pocket_hi)).dropna(subset=["position"])
    l = lo.assign(position=lo["resnum"].map(pocket_lo)).dropna(subset=["position"])
    h, l = h.astype({"position": int}), l.astype({"position": int})
    # a position can be touched by two atoms of one residue only once in contacts.csv,
    # but guard anyway: keep the strongest (most minimally frustrated) contact
    h = h.sort_values("frustration_index", ascending=False).drop_duplicates("position")
    l = l.sort_values("frustration_index", ascending=False).drop_duplicates("position")
    m = h.merge(l, on="position", suffixes=("_hi", "_lo"))
    m["dF"] = m["frustration_index_hi"] - m["frustration_index_lo"]
    m["dF_specific"] = (m["frustration_index_specific_hi"]
                        - m["frustration_index_specific_lo"])
    m["divergent"] = m["position"].isin(divergent)
    return m.sort_values("position")


def enrichment_test(matched: pd.DataFrame, rng: np.random.Generator,
                    n_perm: int = N_PERMUTATIONS) -> dict:
    """mean dF at divergent positions minus mean dF at conserved positions, with a
    permutation p-value from shuffling the divergent labels within this pair.

    Reports the signed statistic (the hypothesis is directional: the tighter-binding
    kinase should be LESS frustrated, i.e. dF > 0, at the divergent positions) and the
    smallest attainable p, which with a handful of positions can exceed alpha -- in which
    case the pair cannot be significant whatever the data does.
    """
    d = matched["divergent"].to_numpy()
    f = matched["dF"].to_numpy()
    n_div = int(d.sum())
    if n_div == 0 or n_div == len(d):
        return {"n_matched": len(d), "n_divergent": n_div, "statistic": None,
                "p_value": None, "min_attainable_p": None}
    obs = f[d].mean() - f[~d].mean()
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(d)
        null[k] = f[p].mean() - f[~p].mean()
    # two-sided, as in design_a_feasibility: the direction is the hypothesis, but a
    # strong effect the other way is also a result and must not be hidden
    p_value = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    n_arrangements = math.comb(len(d), n_div)
    return {"n_matched": len(d), "n_divergent": n_div,
            "mean_dF_divergent": round(float(f[d].mean()), 3),
            "mean_dF_conserved": round(float(f[~d].mean()), 3),
            "statistic": round(float(obs), 3),
            "p_value": round(p_value, 4),
            # 2 of C(n, k) arrangements are the extremes, so this is the floor
            "min_attainable_p": round(2.0 / n_arrangements, 4),
            "arrangements": n_arrangements}


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ana = Path("data/kinome/analysis")
    p.add_argument("--results", type=Path, default=Path("results/pilot"))
    p.add_argument("--pairs", type=Path, default=ana / "pilot_pairs.csv")
    p.add_argument("--pocket-map", type=Path, default=ana / "pilot_pocket_map.csv")
    p.add_argument("--out", type=Path, default=ana)
    p.add_argument("--tag", default="", help="suffix for the output files, e.g. _smoke")
    a = p.parse_args(argv)

    pairs = pd.read_csv(a.pairs, keep_default_na=False)
    pmap = pd.read_csv(a.pocket_map, keep_default_na=False, dtype=str)
    pmap = pmap[pmap["resolved"] == "True"]
    rng = np.random.default_rng(SEED)

    settings, per_contact, per_pair = {}, [], []
    for r in pairs.itertuples():
        lig = r.ligand_group
        # hi = the kinase that binds this ligand more tightly (pilot_pairs is built that way)
        runs = {}
        for role, kin in (("hi", r.kinase_hi), ("lo", r.kinase_lo)):
            rj = a.results / f"{lig}_{kin}" / "run.json"
            cf = a.results / f"{lig}_{kin}" / "contacts.csv"
            if not (rj.exists() and cf.exists()):
                runs = None
                break
            cfg = json.loads(rj.read_text())
            settings[f"{lig}_{kin}"] = cfg
            runs[role] = {"kinase": kin, "n_decoys": cfg["n_decoys"],
                          "contacts": ligand_contacts(pd.read_csv(cf), lig)}
        if runs is None:
            print(f"skipping {r.pair}: missing a run", file=sys.stderr)
            continue

        pocket = {}
        for role, kin in (("hi", r.kinase_hi), ("lo", r.kinase_lo)):
            q = pmap[(pmap["ligand_group"] == lig) & (pmap["kinase_name"] == kin)]
            pocket[role] = dict(zip(q["xray_residue"].astype(int),
                                    q["pocket_position"].astype(int)))
        divergent = {int(x) for x in str(r.divergent_positions).split(";") if x}

        m = matched_positions(runs["hi"]["contacts"], runs["lo"]["contacts"],
                              pocket["hi"], pocket["lo"], divergent)
        n_dec = runs["hi"]["n_decoys"]
        m["dF_se"] = np.sqrt(index_standard_error(m["frustration_index_hi"], n_dec) ** 2
                             + index_standard_error(m["frustration_index_lo"], n_dec) ** 2)
        m.insert(0, "pair", r.pair)
        m.insert(1, "ligand_group", lig)
        m.insert(2, "role", r.role)
        per_contact.append(m)

        test = enrichment_test(m, rng)
        noise = float(m["dF_se"].median())
        per_pair.append({
            "pair": r.pair, "ligand_group": lig, "role": r.role,
            "delta_pKd": r.delta_pKd, "n_divergent_in_pocket": int(len(divergent)),
            "contacts_hi": len(runs["hi"]["contacts"]),
            "contacts_lo": len(runs["lo"]["contacts"]),
            "contact_count_difference": len(runs["hi"]["contacts"])
                                        - len(runs["lo"]["contacts"]),
            "touched_in_both": len(m),
            "mean_abs_dF": round(float(m["dF"].abs().mean()), 3),
            "noise_floor_dF": round(noise, 3),
            "effect_vs_noise": round(float(m["dF"].abs().mean()) / noise, 2),
            "n_decoys": n_dec, **test})

    contacts = pd.concat(per_contact, ignore_index=True)
    summary = pd.DataFrame(per_pair)
    contacts.to_csv(a.out / f"pilot_positions{a.tag}.csv", index=False)
    summary.to_csv(a.out / f"pilot_comparison{a.tag}.csv", index=False)

    comparable = {k: v for k, v in zip(
        ("protocol", "n_decoys", "background_weight", "readout", "cutoff",
         "ligand_cutoff", "contact_atom", "min_seq_sep"),
        (None,) * 8)}
    for key in comparable:
        values = {repr(cfg.get(key)) for cfg in settings.values()}
        comparable[key] = sorted(values)[0] if len(values) == 1 else sorted(values)
    disagree = {k: v for k, v in comparable.items() if isinstance(v, list)}

    (a.out / f"pilot_comparison{a.tag}.json").write_text(json.dumps({
        "results": str(a.results),
        "design": {
            "quantity": "dF(p) = raw frustration_index(high-affinity) - (low-affinity) "
                        "at matched KLIFS pocket positions",
            "hypothesis": "dF > 0 at divergent positions (high positive = minimally "
                          "frustrated)",
            "test": f"mean dF divergent - conserved, {N_PERMUTATIONS} label permutations "
                    f"within each pair, alpha {ALPHA}",
            "why_not_specific": "frustration_index_specific sums to zero over one "
                                "residue's contacts; the ligand is one residue, so any "
                                "subset mean is that identity, not a result",
            "noise_floor": "SE(F) = sqrt(1/n + F^2/(2(n-1))) per contact, propagated to dF",
            "seed": SEED},
        "settings": comparable,
        "settings_disagree": disagree,
        "pairs": per_pair,
    }, indent=2, default=str) + "\n")

    pd.set_option("display.width", 250)
    if disagree:
        print("REFUSING to interpret: runs disagree on", disagree)
        return 1
    print(f"\nsettings: {comparable}\n")
    print(summary[["role", "pair", "ligand_group", "delta_pKd", "contacts_hi",
                   "contacts_lo", "touched_in_both", "n_divergent", "mean_dF_divergent",
                   "mean_dF_conserved", "statistic", "p_value", "min_attainable_p"]]
          .to_string(index=False))
    print("\nnoise: is the effect above what this decoy count can resolve?")
    print(summary[["pair", "n_decoys", "mean_abs_dF", "noise_floor_dF",
                   "effect_vs_noise"]].to_string(index=False))
    print(f"\nwrote {a.out}/pilot_comparison{a.tag}.csv, pilot_positions{a.tag}.csv, "
          f"pilot_comparison{a.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
