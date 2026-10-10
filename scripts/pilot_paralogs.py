"""Define the paralog pilot: which complexes to run, and in what coordinate system.

The COX-1/COX-2 experiment (docs/method.md) was a negative result about frustration
AGGREGATED over a ligand interface: the whole signal reduced to contact count, and two of
its descriptors were degenerate by construction (`frustration_index_specific` sums to
exactly zero over one residue's contacts, and the ligand is one residue). The open
question it leaves is residue-resolved, and the paralog pairs are the sharpest form of it:

    one ligand, two closest-relative kinases, a known affinity difference -- is that
    difference carried by the few pocket positions where the two kinases differ?

This design controls the confound that killed COX structurally rather than statistically:
the two pockets are nearly identical, so the ligand's contact count is nearly constant
across the pair. And KLIFS gives a common frame: its 85-position pocket numbering is
shared by every kinase, and `interactions_match_residues` maps those positions onto each
structure's own residue numbers, so per-contact frustration is comparable position by
position with no alignment work.

This script only DEFINES the pilot. It downloads no structure and computes no
frustration; it selects the pairs, resolves the pocket coordinates and says what the
download step has to handle.

    .venv/bin/python scripts/pilot_paralogs.py

Input (all already on disk):
    data/kinome/analysis/design_a_pairs.csv   paralog pairs with their delta pKd
    data/kinome/klifs/klifs_manifest.csv      the chosen structure per (kinase, ligand)
    data/kinome/klifs/klifs_kinases.csv       the canonical 85-residue pocket per kinase
    data/kinome/klifs/klifs_raw.csv           the pocket as it is in each structure

Output, in data/kinome/analysis/:
    pilot_pairs.csv       one row per pair: kinases, delta pKd, divergent positions
    pilot_complexes.csv   one row per complex: what to download and run
    pilot_pocket_map.csv  (complex, KLIFS position, residue number) -- the common frame
    pilot.json            selection rule, QC findings, what the download step must do
    pocket_map_cache.json KLIFS responses

Selection is a rule, not a list of names: the sharpest pairs by pocket divergence, plus
the sharpest pair whose affinity difference is ~0 as a negative control (a sound index
must find NO ordering there). See select_pilot().

Why pocket divergence decides sharpness: the 19 same-family pairs range from 4 to 50
divergent positions out of 85 (median 31). Past ~20 the two pockets are not variants of
each other any more and "which position carries the difference" has no answer. Note that
delta pKd does NOT track divergence -- ABL1/ABL2 differ at 4 positions with delta 0.96,
BMX/BTK at 22 with delta 0.00 -- which is what leaves something for a structure-based
index to explain, and what makes BMX/BTK the negative control.
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KLIFS_API = "https://klifs.net/api_v2"
POCKET_LEN = 85
GAP_CHARS = "-_"
# A pair counts as a negative control when the two kinases bind the ligand this equally
# (log units). Nothing in the data sits between 0.00 and 0.16, so the cut is not delicate.
NEG_CONTROL_MAX_DELTA = 0.1


# --- pure logic --------------------------------------------------------------------

def pocket_divergence(a: str, b: str) -> dict:
    """Compare two canonical 85-character KLIFS pockets.

    Returns the 1-based positions where the residue identity differs, and the positions
    that are a gap in either string. Gaps are counted apart: a gap means KLIFS could not
    place a residue, which is not evidence of a difference.
    """
    if len(a) != len(b):
        raise ValueError(f"pocket lengths differ: {len(a)} vs {len(b)}")
    divergent, gaps = [], []
    for i, (x, y) in enumerate(zip(a, b), start=1):
        if x in GAP_CHARS or y in GAP_CHARS:
            gaps.append(i)
        elif x != y:
            divergent.append(i)
    return {"n_positions": len(a), "divergent": divergent, "gaps": gaps,
            "n_divergent": len(divergent), "n_gaps": len(gaps),
            "identity": round((len(a) - len(divergent) - len(gaps)) / len(a), 3)}


def pocket_qc(structure_pocket: str, canonical_pocket: str,
              divergent: list[int]) -> dict:
    """Does the crystal's pocket match the kinase's canonical pocket?

    Separates the two ways it can fail, because they mean different things:
      gaps           residues KLIFS could not place (missing density) -- the position
                     carries no contact and no frustration value
      substitutions  a different residue in the crystal than the canonical sequence --
                     an engineered mutation or an isoform, i.e. a potential confound

    `*_at_divergent` is the part that actually threatens this design: a gap or a
    substitution AT one of the positions that distinguishes the two paralogs kills that
    specific comparison, while one elsewhere only costs a contact.
    """
    if len(structure_pocket) != len(canonical_pocket):
        return {"length_mismatch": True, "n_gaps": None, "n_substitutions": None}
    gaps, subs = [], []
    for i, (s, c) in enumerate(zip(structure_pocket, canonical_pocket), start=1):
        if s in GAP_CHARS:
            gaps.append(i)
        elif s != c:
            subs.append(i)
    div = set(divergent)
    return {"length_mismatch": False,
            "n_gaps": len(gaps), "gaps": gaps,
            "n_substitutions": len(subs), "substitutions": subs,
            "gaps_at_divergent": sorted(div & set(gaps)),
            "substitutions_at_divergent": sorted(div & set(subs)),
            "clean": not gaps and not subs}


def select_pilot(pairs: pd.DataFrame, n_sharp: int, n_negative: int,
                 neg_max_delta: float = NEG_CONTROL_MAX_DELTA) -> pd.DataFrame:
    """Pick the pilot pairs by rule. `pairs` needs n_divergent and delta_pKd.

    Positives: the `n_sharp` pairs with the fewest divergent pocket positions among those
    whose affinity actually differs (|delta| > neg_max_delta) -- the sharpest cases, where
    a difference of 1-2 log units has only a handful of positions to be carried by.

    Negative controls: the `n_negative` sharpest pairs whose affinity difference is
    within neg_max_delta of zero. The kinases are equipotent for that ligand, so a sound
    index must find no ordering; these are kept apart as `role`.

    Ties break on delta_pKd (larger first for positives: more signal to explain) and then
    the pair label, so the choice is deterministic.
    """
    p = pairs.copy()
    p["abs_delta"] = p["delta_pKd"].abs()
    is_neg = p["abs_delta"] <= neg_max_delta
    pos = (p[~is_neg].sort_values(["n_divergent", "abs_delta", "pair"],
                                  ascending=[True, False, True]).head(n_sharp)
           .assign(role="sharp"))
    neg = (p[is_neg].sort_values(["n_divergent", "pair"]).head(n_negative)
           .assign(role="negative_control"))
    out = pd.concat([pos, neg], ignore_index=True)
    return out.drop(columns="abs_delta")


def conformation_match(pair: str, rows: pd.DataFrame) -> dict:
    """Are the two structures of a pair in the same conformational state?

    KLIFS's `dfg` and `ac_helix` calls. This outranks the sequence question: if one
    kinase is DFG-out and the other DFG-in for the same ligand, the frustration
    difference is dominated by the activation-loop rearrangement, not by the handful of
    divergent pocket positions, and the pair cannot answer what it was selected for.
    Reported, never repaired -- the fix would be choosing a different structure, which is
    a manifest-level decision.
    """
    dfg, ac = sorted(set(rows["dfg"])), sorted(set(rows["ac_helix"]))
    return {"pair": pair, "dfg": "/".join(dfg), "ac_helix": "/".join(ac),
            "dfg_match": len(dfg) == 1, "ac_helix_match": len(ac) == 1}


def pair_complexes(pilot: pd.DataFrame) -> pd.DataFrame:
    """The distinct (ligand, kinase) complexes the pilot pairs need. A ligand can appear
    in two pairs (STI in ABL1/ABL2 and KIT/PDGFRa), so complexes are deduplicated and
    `in_pairs` records which pairs need each one."""
    rows = []
    for r in pilot.itertuples():
        for kin in (r.kinase_hi, r.kinase_lo):
            rows.append({"ligand_group": r.ligand_group, "kinase_name": kin,
                         "pair": r.pair, "role": r.role})
    d = pd.DataFrame(rows)
    return (d.groupby(["ligand_group", "kinase_name"], as_index=False)
            .agg(pairs=("pair", lambda s: ";".join(sorted(set(s)))),
                 roles=("role", lambda s: ";".join(sorted(set(s)))),
                 n_pairs=("pair", "nunique")))


# --- network -----------------------------------------------------------------------

def fetch_pocket_maps(structure_ids: list[str], cache_path: Path,
                      refresh: bool = False) -> dict:
    """{structure_id: [{index, KLIFS_position, Xray_position}]} from
    `interactions_match_residues`: KLIFS's 85 pocket positions mapped onto THIS
    structure's own residue numbers, with KLIFS's structural labels (I.1, g.l.4, ...).
    One call per structure; only ids missing from the cache go to the network."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not refresh \
        else {"maps": {}, "fetched_at": None}
    need = sorted(set(map(str, structure_ids)) - set(cache["maps"]))
    for sid in need:
        with urllib.request.urlopen(
                f"{KLIFS_API}/interactions_match_residues?structure_ID={sid}",
                timeout=120) as r:
            cache["maps"][sid] = json.load(r)
    if need:
        cache["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return {str(s): cache["maps"][str(s)] for s in structure_ids}


def pocket_map_table(maps: dict, complexes: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Long table: one row per (complex, pocket position). `Xray_position` is "-1" or
    empty where KLIFS placed no residue -- those positions are unusable in that structure
    and are kept with `resolved = False` rather than dropped.

    Also returns the position labels, after checking that every structure agrees on them
    (KLIFS's 85-position scheme is fixed; disagreement would mean the frame is not common
    and the whole comparison is invalid).
    """
    rows, labels, problems = [], {}, []
    for c in complexes.itertuples():
        for e in maps[str(c.klifs_structure_id)]:
            idx, lab = int(e["index"]), e["KLIFS_position"]
            if labels.setdefault(idx, lab) != lab:
                problems.append({"position": idx, "expected": labels[idx], "got": lab,
                                 "structure": c.klifs_structure_id})
            xray = str(e.get("Xray_position", "")).strip()
            # KLIFS marks an unplaced position with "-1" or with its gap character
            # ("_"), which is what the pocket STRINGS use -- both appear in practice
            # (BMX 3SXR has four "_" positions, BTK 3OCT two).
            resolved = xray not in ("", "-1", "0") and not set(xray) <= set(GAP_CHARS)
            rows.append({"ligand_group": c.ligand_group, "kinase_name": c.kinase_name,
                         "pdb": c.pdb, "chain": c.chain,
                         "klifs_structure_id": c.klifs_structure_id,
                         "pocket_position": idx, "klifs_position": lab,
                         "xray_residue": xray, "resolved": resolved})
    return pd.DataFrame(rows), problems


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kin, ana = Path("data/kinome/klifs"), Path("data/kinome/analysis")
    p.add_argument("--pairs", type=Path, default=ana / "design_a_pairs.csv")
    p.add_argument("--manifest", type=Path, default=kin / "klifs_manifest.csv")
    p.add_argument("--klifs-kinases", type=Path, default=kin / "klifs_kinases.csv")
    p.add_argument("--klifs-raw", type=Path, default=kin / "klifs_raw.csv")
    p.add_argument("--out", type=Path, default=ana)
    p.add_argument("--n-sharp", type=int, default=4, help="sharpest pairs (default 4)")
    p.add_argument("--n-negative", type=int, default=1,
                   help="zero-delta pairs as negative controls (default 1)")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    rd = lambda f: pd.read_csv(f, keep_default_na=False, dtype=str)  # noqa: E731
    pairs = pd.read_csv(a.pairs, keep_default_na=False)
    manifest, kinases, raw = rd(a.manifest), rd(a.klifs_kinases), rd(a.klifs_raw)
    canon = kinases[kinases["species"] == "Human"].set_index("name")["pocket"].to_dict()
    struct_pocket = raw.set_index("structure_ID")["pocket"].to_dict()

    # same-family pairs only: the question needs two variants of one pocket
    fam = pairs[pairs["relation"] == "family"].copy()
    fam["pair"] = fam["kinase_hi"] + "/" + fam["kinase_lo"]
    rows, skipped = [], []
    for r in fam.itertuples():
        pa, pb = canon.get(r.kinase_hi, ""), canon.get(r.kinase_lo, "")
        if len(pa) != POCKET_LEN or len(pb) != POCKET_LEN:
            skipped.append({"pair": r.pair, "why": "canonical pocket missing or not 85"})
            continue
        div = pocket_divergence(pa, pb)
        rows.append({"ligand_group": r.ligand_group, "pair": r.pair,
                     "kinase_hi": r.kinase_hi, "kinase_lo": r.kinase_lo,
                     "kinase_family": r.kinase_family, "kinase_group": r.kinase_group,
                     "pKd_hi": r.pKd_hi, "pKd_lo": r.pKd_lo, "delta_pKd": r.delta_pKd,
                     "n_divergent": div["n_divergent"], "n_gaps": div["n_gaps"],
                     "pocket_identity": div["identity"],
                     "divergent_positions": ";".join(map(str, div["divergent"]))})
    scored = pd.DataFrame(rows)
    pilot = select_pilot(scored, a.n_sharp, a.n_negative)

    # the complexes those pairs need, joined to the manifest row that will be run
    cx = pair_complexes(pilot)
    man = manifest.rename(columns={"ligand_code": "ligand_group"})
    cx = cx.merge(man[["ligand_group", "kinase_name", "klifs_kinase_id", "uniprot",
                       "klifs_structure_id", "pdb", "chain", "altloc", "resolution",
                       "quality_score", "missing_residues", "missing_atoms", "dfg",
                       "ac_helix", "ligand_chain_differs", "multi_copy_ligand",
                       "altloc_ligand_conflict"]],
                  on=["ligand_group", "kinase_name"], how="left")
    missing = cx[cx["klifs_structure_id"].isna()]
    if len(missing):
        raise SystemExit("not in the manifest: "
                         + ", ".join(f"{r.ligand_group}/{r.kinase_name}"
                                     for r in missing.itertuples()))

    # QC: does each crystal's pocket match its kinase's canonical pocket?
    div_of = dict(zip(pilot["pair"], pilot["divergent_positions"]))
    qc = []
    for c in cx.itertuples():
        div = [int(x) for x in div_of[c.pairs.split(";")[0]].split(";") if x]
        q = pocket_qc(struct_pocket.get(c.klifs_structure_id, ""),
                      canon.get(c.kinase_name, ""), div)
        qc.append({"ligand_group": c.ligand_group, "kinase_name": c.kinase_name,
                   "pdb": c.pdb, **{k: v for k, v in q.items()
                                    if k in ("clean", "n_gaps", "n_substitutions",
                                             "length_mismatch")},
                   "gaps": ";".join(map(str, q.get("gaps") or [])),
                   "substitutions": ";".join(map(str, q.get("substitutions") or [])),
                   "gaps_at_divergent": ";".join(map(str, q.get("gaps_at_divergent") or [])),
                   "substitutions_at_divergent":
                       ";".join(map(str, q.get("substitutions_at_divergent") or []))})
    qc = pd.DataFrame(qc)
    cx = cx.merge(qc[["ligand_group", "kinase_name", "clean", "n_gaps",
                      "n_substitutions", "gaps_at_divergent",
                      "substitutions_at_divergent"]],
                  on=["ligand_group", "kinase_name"])

    # conformational state must agree within a pair (see conformation_match)
    conf = pd.DataFrame([conformation_match(pair, d)
                         for pair, d in cx.groupby("pairs")])
    pilot = pilot.merge(conf, on="pair", how="left")

    maps = fetch_pocket_maps(cx["klifs_structure_id"].tolist(),
                             a.out / "pocket_map_cache.json", a.refresh)
    pmap, label_problems = pocket_map_table(maps, cx)
    labels = (pmap.drop_duplicates("pocket_position")
              .set_index("pocket_position")["klifs_position"].to_dict())
    pilot["divergent_labels"] = pilot["divergent_positions"].map(
        lambda s: ";".join(labels.get(int(x), f"?{x}") for x in s.split(";") if x))
    unresolved = pmap[~pmap["resolved"]]

    pilot.to_csv(a.out / "pilot_pairs.csv", index=False)
    cx.to_csv(a.out / "pilot_complexes.csv", index=False)
    pmap.to_csv(a.out / "pilot_pocket_map.csv", index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {k: str(v) for k, v in (("pairs", a.pairs), ("manifest", a.manifest),
                                          ("klifs_kinases", a.klifs_kinases),
                                          ("klifs_raw", a.klifs_raw))},
        "selection_rule": {
            "same_family_pairs_scored": len(scored),
            "n_sharp": a.n_sharp, "n_negative": a.n_negative,
            "negative_control_max_abs_delta_pKd": NEG_CONTROL_MAX_DELTA,
            "sharp": "fewest divergent pocket positions among pairs whose affinity "
                     "differs; ties on larger delta_pKd then pair name",
            "negative_control": "fewest divergent positions among pairs with "
                                f"|delta_pKd| <= {NEG_CONTROL_MAX_DELTA}",
            "skipped": skipped},
        "pilot": {"pairs": len(pilot), "complexes": len(cx),
                  "ligands": int(cx["ligand_group"].nunique()),
                  "kinases": int(cx["kinase_name"].nunique()),
                  "divergence_range": [int(pilot["n_divergent"].min()),
                                       int(pilot["n_divergent"].max())]},
        "pocket_frame": {
            "positions": POCKET_LEN,
            "source": f"{KLIFS_API}/interactions_match_residues",
            "label_disagreements": label_problems,
            # keys joined into a string: a (ligand, kinase) tuple is not JSON-serialisable
            "unresolved_positions_per_complex": {
                f"{lig}/{kin}": int(n) for (lig, kin), n in
                unresolved.groupby(["ligand_group", "kinase_name"]).size().items()},
            "unresolved_positions": unresolved[["ligand_group", "kinase_name",
                                                "pocket_position", "klifs_position"]]
                                    .to_dict("records")},
        "qc": {
            "clean_complexes": int(qc["clean"].sum()),
            "conformation": conf.to_dict("records"),
            "pairs_with_mismatched_conformation":
                conf.loc[~conf["dfg_match"] | ~conf["ac_helix_match"],
                         "pair"].tolist(),
            "complexes_with_gaps": qc.loc[qc["n_gaps"] > 0,
                                          ["ligand_group", "kinase_name", "pdb",
                                           "n_gaps", "gaps"]].to_dict("records"),
            "complexes_with_substitutions":
                qc.loc[qc["n_substitutions"] > 0,
                       ["ligand_group", "kinase_name", "pdb", "n_substitutions",
                        "substitutions"]].to_dict("records"),
            "problems_at_divergent_positions":
                qc.loc[(qc["gaps_at_divergent"] != "")
                       | (qc["substitutions_at_divergent"] != ""),
                       ["ligand_group", "kinase_name", "pdb", "gaps_at_divergent",
                        "substitutions_at_divergent"]].to_dict("records")},
        "download_step_must_handle": {
            "mmcif_required (3-char ligand chain)":
                cx.loc[cx["ligand_chain_differs"] == "True",
                       ["pdb", "ligand_group"]].to_dict("records"),
            "pick_the_pocket_copy (several ligand copies)":
                cx.loc[cx["multi_copy_ligand"] == "True",
                       ["pdb", "ligand_group"]].to_dict("records"),
            "altloc_choice (altlocs carry different ligands)":
                cx.loc[cx["altloc_ligand_conflict"] == "True",
                       ["pdb", "ligand_group"]].to_dict("records"),
            "keep_altloc_A": cx.loc[cx["altloc"] == "A", "pdb"].tolist(),
            "ligand_name3": sorted(cx["ligand_group"].unique())},
    }
    (a.out / "pilot.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    pd.set_option("display.width", 220)
    print(f"\nsame-family pairs scored: {len(scored)} | pilot: {len(pilot)} pairs, "
          f"{len(cx)} complexes, {cx['ligand_group'].nunique()} ligands")
    print("\npilot pairs:")
    print(pilot[["role", "ligand_group", "pair", "kinase_family", "delta_pKd",
                 "n_divergent", "pocket_identity", "dfg", "dfg_match", "ac_helix",
                 "divergent_labels"]].to_string(index=False))
    print("\npilot complexes:")
    print(cx[["ligand_group", "kinase_name", "pdb", "chain", "altloc", "resolution",
              "quality_score", "missing_residues", "clean", "n_gaps",
              "n_substitutions", "multi_copy_ligand", "n_pairs"]].to_string(index=False))
    if report["qc"]["problems_at_divergent_positions"]:
        print("\nPROBLEM at positions that distinguish the pair:")
        for r in report["qc"]["problems_at_divergent_positions"]:
            print("   ", r)
    else:
        print("\nno gap or substitution at any divergent position")
    bad_conf = report["qc"]["pairs_with_mismatched_conformation"]
    print("conformational state within each pair:",
          "all matched" if not bad_conf else f"*** MISMATCHED: {bad_conf} ***")
    if label_problems:
        print("\nPROBLEM: structures disagree on the pocket labels:", label_problems[:5])
    print(f"\nwrote {a.out}/pilot_pairs.csv, pilot_complexes.csv, "
          f"pilot_pocket_map.csv, pilot.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
