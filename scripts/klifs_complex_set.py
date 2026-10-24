"""Drug-kinase complexes from KLIFS, with wild-type binding affinities from ChEMBL.

The question this table serves: for a ligand co-crystallised with MANY human kinases,
does a per-complex frustration index follow its measured affinity across those kinases
(selectivity)? So a ligand qualifies only if it has a structure AND a value on at least
--min-kinases kinases, and the affinity measures are never mixed: one analysis reads
pKd, another pIC50, never both.

    .venv/bin/python scripts/klifs_complex_set.py

Input:
    data/kinome/klifs/klifs_raw.csv       every KLIFS structure (scripts/klifs_manifest.py)
    data/kinome/klifs/klifs_kinases.csv   KLIFS kinase_information (multi-domain proteins)

Output:
    data/kinome/DRUG_KINASE_COMPLEXES.csv   THE table: one row per (ligand, kinase domain),
                                            best structure + pKd + pIC50 + in_*_set flags
    data/kinome/complex_set/ligands.csv     one row per ligand: counts per measure
    data/kinome/complex_set/activities.csv  every ChEMBL row fetched, with the reason it
                                            was used or not (audit trail)
    data/kinome/complex_set/affinities.csv  per (ligand, uniprot, measure) wild-type value
                                            on EVERY human KLIFS kinase, structure or not:
                                            the targets a redocking step would need
    data/kinome/complex_set/funnel.json     thresholds, funnel counts, provenance
    data/kinome/complex_set/cache/          raw API responses, reused on re-runs

Where each piece comes from, and why:

  - Structures: KLIFS (klifs_raw.csv), filtered with klifs_manifest.py's own functions
    so "resolution", "quality" and "altloc" mean exactly what they mean there. Default
    resolution <= 3.5 and quality >= 4: chosen from the filter-sensitivity table (Kd set
    11 ligands / 106 complexes, against 11 / 108 unfiltered and 6 / 61 at 2.5 / 6).
    Missing ATOMS are tolerable (frustx repacks side chains); missing pocket RESIDUES are
    backbone gaps whose contacts cannot be scored -- kept, and carried as a column.
  - Ligand -> ChEMBL compound: KLIFS's bioactivity_list_id (its chembl_id field).
  - Affinity values: ChEMBL's activity API, NOT KLIFS's bioactivity list. KLIFS's list is
    a ChEMBL extract (checked on dasatinib: 545 KLIFS rows vs 993 ChEMBL rows with a
    pchembl value, 318 of 496 target/measure pairs larger in ChEMBL) and it carries no
    variant, assay-type or document fields -- so mutants and cell-based IC50s cannot be
    removed from it. The KLIFS row counts are kept in funnel.json as a cross-check.
  - Queried by PARENT molecule: KLIFS pools salt forms (dasatinib/ABL1: molecule_chembl_id
    gives 2 Kd, parent_molecule_chembl_id gives 25, KLIFS 23).

Rules applied to ChEMBL rows, in order (each row gets the first reason that applies):

  1. target is a human SINGLE PROTEIN whose accession is a human KLIFS kinase (any of
     them, not only those with a structure of this ligand). Drops cell lines (K562),
     fusions (Bcr/Abl is CHIMERIC PROTEIN), complexes and non-kinases.
  2. assay_type == "B" (binding). An IC50 from a cell assay is not a binding constant.
  3. data_validity_comment empty (ChEMBL's own "outside typical range", "potential
     transcription error"...).
  4. potential_duplicate == 0: ChEMBL's flag for a value re-cited from an earlier paper;
     counting it again would inflate n.
  5. assay_variant_mutation empty, i.e. wild type. Mutants are counted (n_mutant_*) and
     never pooled into the wild-type value: dasatinib/ABL1 alone has 112 mutant rows,
     T315I at > 10 uM next to wild type at ~0.1 nM. Caveat: ChEMBL's variant annotation
     is incomplete for older papers, so a mutant described only in free text can slip
     through; `inconsistent_*` (spread > 1 log) is the backstop.
  6. pchembl_value present: an exact (`=`) value in molar-convertible units. Censored
     values (">10000") are counted (n_censored_*) and never become numbers.

Ki is fetched by nobody here: excluded from the analysis by decision (2 ligands reach
--min-kinases with Ki). EC50 is a functional readout and never was a candidate.

A ligand's identity is its KLIFS (PDB) code. One drug crystallised under two codes would
count as two ligands; none of the qualifying ligands is affected (checked against KLIFS
drug_list, whose multi-code cells are 0WM/0WN, 1E8/8E8, YY3/Q6K, TIV/M97).

PDB entries are NOT validated against RCSB here (obsolete ids, ligand missing): that
happens when the structures are downloaded, the next step.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from klifs_manifest import (DEFAULT_EXCLUDE, apply_filters, deduplicate,  # noqa: E402
                            multi_domain_uniprots, normalise)

KLIFS_API = "https://klifs.net/api_v2"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

MEASURES = ("Kd", "IC50")       # Ki excluded by decision; see module docstring
MIN_KINASES = 5
MAX_RESOLUTION = 3.5
MIN_QUALITY = 4.0
INCONSISTENT_STD = 1.0          # log units; same threshold as bindingdb_activity.py

ACTIVITY_FIELDS = [
    "activity_id", "molecule_chembl_id", "parent_molecule_chembl_id", "target_chembl_id",
    "standard_type", "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "assay_chembl_id", "assay_type", "assay_variant_mutation",
    "assay_variant_accession", "document_chembl_id", "document_year",
    "data_validity_comment", "potential_duplicate",
]


# --- network (cached; not tested) ----------------------------------------------------

def _get_json(url: str, cache: Path):
    """GET url as JSON, through a file cache: a re-run makes no requests."""
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = json.load(r)
            break
        except OSError as e:
            # a 4xx is an answer, not a transient failure: do not retry it
            if attempt == 2 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                raise
            time.sleep(5 * (attempt + 1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def klifs_bioactivity(ligand_id: int, cache: Path) -> list[dict]:
    try:
        return _get_json(f"{KLIFS_API}/bioactivity_list_id?ligand_ID={ligand_id}",
                         cache / "klifs_bioactivity" / f"{ligand_id}.json")
    except urllib.error.HTTPError as e:   # KLIFS answers 400 for "no bioactivities"
        if e.code in (400, 404):
            return []
        raise


def chembl_parent(molecule_id: str, cache: Path) -> str:
    m = _get_json(f"{CHEMBL_API}/molecule/{molecule_id}.json",
                  cache / "chembl_molecule" / f"{molecule_id}.json")
    return (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or molecule_id


def chembl_activities(parent_id: str, cache: Path) -> list[dict]:
    """All human Kd/IC50 activities of a parent molecule, every page."""
    q = urllib.parse.urlencode({
        "parent_molecule_chembl_id": parent_id, "standard_type__in": ",".join(MEASURES),
        "target_organism": "Homo sapiens", "only": ",".join(ACTIVITY_FIELDS),
        "limit": 1000})
    url, page, out = f"{CHEMBL_API}/activity.json?{q}", 0, []
    while url:
        d = _get_json(url, cache / "chembl_activity" / f"{parent_id}_{page}.json")
        out += d["activities"]
        nxt = d["page_meta"]["next"]
        url, page = (f"https://www.ebi.ac.uk{nxt}" if nxt else None), page + 1
    return out


def chembl_targets(target_ids: list[str], cache: Path) -> dict[str, dict]:
    """target_chembl_id -> {target_type, organism, accessions}."""
    out = {}
    for i in range(0, len(target_ids), 200):
        chunk = sorted(target_ids)[i:i + 200]
        q = urllib.parse.urlencode({"target_chembl_id__in": ",".join(chunk), "limit": 1000})
        d = _get_json(f"{CHEMBL_API}/target.json?{q}",
                      cache / "chembl_target" / f"{chunk[0]}_{len(chunk)}.json")
        for t in d["targets"]:
            out[t["target_chembl_id"]] = {
                "target_type": t["target_type"], "organism": t["organism"],
                "accessions": [c["accession"] for c in t["target_components"]]}
    return out


def klifs_drug_names(cache: Path) -> dict[str, str]:
    """PDB ligand code -> drug INN, from KLIFS's drug list (multi-code cells split)."""
    drugs = _get_json(f"{KLIFS_API}/drug_list", cache / "klifs_drug_list.json")
    return {code.strip(): d["INN"] for d in drugs
            for code in (d.get("PDB") or "").split(",") if code.strip()}


# --- pure logic (tested offline in tests/test_klifs_complex_set.py) -------------------

def select_structures(raw: pd.DataFrame, *, max_resolution: float, min_quality: float,
                      species: str = "Human") -> tuple[pd.DataFrame, list[dict]]:
    """Best structure per (kinase domain, ligand), with how many structures competed."""
    filtered, funnel = apply_filters(normalise(raw), species=species,
                                     exclude=DEFAULT_EXCLUDE,
                                     max_resolution=max_resolution,
                                     min_quality=min_quality)
    n = (filtered.groupby(["klifs_kinase_id", "ligand_code"])["pdb"].nunique()
         .rename("n_structures"))
    best = deduplicate(filtered).join(n, on=["klifs_kinase_id", "ligand_code"])
    # normalise() does not carry KLIFS's ligand_ID, which the bioactivity API is keyed on
    lig_id = dict(zip(raw["structure_ID"].astype(int), raw["ligand_ID"].astype(int)))
    best["klifs_ligand_id"] = best["klifs_structure_id"].map(lig_id)
    return best, funnel


def candidate_ligands(structures: pd.DataFrame, min_kinases: int) -> list[str]:
    """Ligands with structures on >= min_kinases kinase domains. Only these can qualify,
    so only these are sent to the APIs."""
    n = structures.groupby("ligand_code")["klifs_kinase_id"].nunique()
    return sorted(n[n >= min_kinases].index)


def classify_activities(acts: pd.DataFrame, targets: dict[str, dict],
                        klifs_uniprots: set[str]) -> pd.DataFrame:
    """Add `uniprot` and `status` to every ChEMBL row. status is "wild_type" for a row
    that enters a median, otherwise the first rule (module docstring) that removed it."""
    acts = acts.copy()
    t = acts["target_chembl_id"].map(lambda i: targets.get(i, {}))
    single = t.map(lambda x: x.get("target_type") == "SINGLE PROTEIN"
                   and x.get("organism") == "Homo sapiens"
                   and len(x.get("accessions", [])) == 1)
    acts["uniprot"] = t.map(lambda x: x["accessions"][0]
                            if len(x.get("accessions", [])) == 1 else None)

    def blank(col):
        return acts[col].isna() | acts[col].astype(str).str.strip().isin(["", "None"])

    rules = [
        ("not_single_klifs_protein", ~(single & acts["uniprot"].isin(klifs_uniprots))),
        ("not_binding_assay", acts["assay_type"] != "B"),
        ("validity_flagged", ~blank("data_validity_comment")),
        ("potential_duplicate", pd.to_numeric(acts["potential_duplicate"]).fillna(0) != 0),
        ("mutant", ~blank("assay_variant_mutation")),
        ("censored", blank("pchembl_value") & acts["standard_relation"].isin([">", "<",
                                                                             ">=", "<="])),
        ("no_value", blank("pchembl_value")),
    ]
    acts["status"] = "wild_type"
    for name, mask in reversed(rules):      # reversed: the FIRST matching rule wins
        acts.loc[mask, "status"] = name
    return acts


def aggregate(acts: pd.DataFrame) -> pd.DataFrame:
    """Per (ligand_code, uniprot, measure): median pX over wild-type rows, plus the
    counts of what was kept out. A pair with only mutant/censored rows still gets a row
    (pX NaN), so the table can say WHY a kinase has no value."""
    acts = acts[acts["uniprot"].notna()].copy()
    acts["p"] = pd.to_numeric(acts["pchembl_value"], errors="coerce")
    wt = acts["status"] == "wild_type"
    key = ["ligand_code", "uniprot", "standard_type"]
    g_all = acts.groupby(key)
    out = pd.DataFrame({
        "n_mutant": g_all["status"].apply(lambda s: int((s == "mutant").sum())),
        "n_censored": g_all["status"].apply(lambda s: int((s == "censored").sum())),
    })
    g = acts[wt].groupby(key)
    out = out.join(pd.DataFrame({
        "pX": g["p"].median(), "n": g["p"].size(), "std": g["p"].std(),
        "n_documents": g["document_chembl_id"].nunique()}))
    out["n"] = out["n"].fillna(0).astype(int)
    out["n_documents"] = out["n_documents"].fillna(0).astype(int)
    out["inconsistent"] = out["std"] > INCONSISTENT_STD     # NaN (n = 1) -> False
    return out.reset_index().rename(columns={"standard_type": "measure"})


def assign_domains(structures: pd.DataFrame, multi_domain: set[str]) -> pd.Series:
    """ChEMBL measures a PROTEIN; KLIFS structures a DOMAIN. For JAK1-3/TYK2 (JH1/JH2)
    and RSK/MSK (N/C) the value cannot be attributed to a domain from ChEMBL alone.

    single_domain  the protein has one kinase domain: no question.
    structure      multi-domain protein, but this ligand is crystallised on only one of
                   its domains: the value is attributed there (the same fallback rule as
                   bindingdb_activity.py, weaker than a construct range).
    ambiguous      the ligand is crystallised on two domains of one protein: excluded.
    """
    n_dom = structures.groupby(["ligand_code", "uniprot"])["klifs_kinase_id"].transform(
        "nunique")
    return pd.Series(
        ["single_domain" if u not in multi_domain else
         ("structure" if k == 1 else "ambiguous")
         for u, k in zip(structures["uniprot"], n_dom)], index=structures.index)


def build_table(structures: pd.DataFrame, agg: pd.DataFrame, multi_domain: set[str],
                min_kinases: int) -> pd.DataFrame:
    """One row per (ligand, kinase domain). in_<m>_set is the per-row answer to "use
    this complex in the <m> analysis": the row has a consistent wild-type value, its
    domain is attributable, and its ligand has >= min_kinases such rows."""
    t = structures.copy()
    t["domain_assignment"] = assign_domains(t, multi_domain)
    for m in MEASURES:
        a = agg[agg["measure"] == m].set_index(["ligand_code", "uniprot"])
        a = a[["pX", "n", "std", "n_documents", "inconsistent", "n_mutant", "n_censored"]]
        a.columns = [f"p{m}" if c == "pX" else f"{c}_{m}" for c in a.columns]
        t = t.join(a, on=["ligand_code", "uniprot"])
        for c in (f"n_{m}", f"n_documents_{m}", f"n_mutant_{m}", f"n_censored_{m}"):
            t[c] = t[c].fillna(0).astype(int)
        t[f"inconsistent_{m}"] = t[f"inconsistent_{m}"].fillna(False).astype(bool)
        usable = (t[f"p{m}"].notna() & ~t[f"inconsistent_{m}"]
                  & (t["domain_assignment"] != "ambiguous"))
        n_lig = usable.groupby(t["ligand_code"]).transform("sum")
        t[f"in_{m}_set"] = usable & (n_lig >= min_kinases)
    return t


TABLE_COLUMNS = [
    "ligand_code", "drug_name", "ligand_name", "chembl_parent",
    "kinase_name", "klifs_kinase_id", "uniprot", "kinase_family", "kinase_group",
    "domain_assignment",
    "pdb", "chain", "altloc", "klifs_structure_id", "resolution", "quality_score",
    "missing_residues", "missing_atoms", "dfg", "ac_helix", "n_structures",
    *[c for m in MEASURES for c in (
        f"p{m}", f"n_{m}", f"std_{m}", f"n_documents_{m}", f"inconsistent_{m}",
        f"n_mutant_{m}", f"n_censored_{m}")],
    *[f"in_{m}_set" for m in MEASURES],
]


def ligand_summary(table: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby("ligand_code")
    out = pd.DataFrame({
        "drug_name": g["drug_name"].first(), "chembl_parent": g["chembl_parent"].first(),
        "n_kinases_structure": g["klifs_kinase_id"].nunique()})
    for m in MEASURES:
        out[f"n_kinases_{m}"] = g[f"p{m}"].apply(lambda s: int(s.notna().sum()))
        out[f"in_{m}_set"] = g[f"in_{m}_set"].sum().astype(int)
        s = table[table[f"in_{m}_set"]].groupby("ligand_code")[f"p{m}"]
        out[f"range_p{m}"] = (s.max() - s.min()).round(2)
        # consistent wild-type values on ANY human KLIFS kinase: how far redocking
        # could extend this ligand's panel beyond its crystal structures
        a = agg[(agg["measure"] == m) & agg["pX"].notna() & ~agg["inconsistent"]]
        out[f"n_kinases_{m}_any_structure_or_not"] = a.groupby("ligand_code").size()
        out[f"n_kinases_{m}_any_structure_or_not"] = (
            out[f"n_kinases_{m}_any_structure_or_not"].fillna(0).astype(int))
    return out.reset_index().sort_values(
        [f"in_{MEASURES[0]}_set", "n_kinases_structure"], ascending=False)


# --- main ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--klifs", type=Path, default=Path("data/kinome/klifs"))
    p.add_argument("--table", type=Path, default=Path("data/kinome/DRUG_KINASE_COMPLEXES.csv"))
    p.add_argument("--out", type=Path, default=Path("data/kinome/complex_set"))
    p.add_argument("--max-resolution", type=float, default=MAX_RESOLUTION)
    p.add_argument("--min-quality", type=float, default=MIN_QUALITY)
    p.add_argument("--min-kinases", type=int, default=MIN_KINASES)
    a = p.parse_args(argv)
    cache = a.out / "cache"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    raw = pd.read_csv(a.klifs / "klifs_raw.csv", keep_default_na=False, dtype=str)
    kinases = pd.read_csv(a.klifs / "klifs_kinases.csv")
    structures, funnel = select_structures(raw, max_resolution=a.max_resolution,
                                           min_quality=a.min_quality)
    ligands = candidate_ligands(structures, a.min_kinases)
    structures = structures[structures["ligand_code"].isin(ligands)].copy()
    print(f"{len(ligands)} ligands with structures on >= {a.min_kinases} kinase domains",
          file=sys.stderr)

    # ligand -> ChEMBL parent, through KLIFS's own bioactivity link
    lig_ids = structures.groupby("ligand_code")["klifs_ligand_id"].first()
    link, klifs_rows = {}, {}
    for code, lid in lig_ids.items():
        bio = klifs_bioactivity(int(lid), cache)
        ids = sorted({b["chembl_id"] for b in bio if b.get("chembl_id")})
        link[code] = sorted({chembl_parent(i, cache) for i in ids})
        klifs_rows[code] = sum(1 for b in bio if b.get("standard_type") in MEASURES
                               and b.get("organism") == "Homo sapiens"
                               and str(b.get("pchembl_value") or "").strip())

    rows = []
    for code, parents in link.items():
        for par in parents:
            rows += [{**r, "ligand_code": code} for r in chembl_activities(par, cache)]
    acts = pd.DataFrame(rows, columns=ACTIVITY_FIELDS + ["ligand_code"])
    targets = chembl_targets(acts["target_chembl_id"].dropna().unique().tolist(), cache)
    human = kinases[(kinases["species"] == "Human") & kinases["uniprot"].notna()]
    acts = classify_activities(acts, targets, set(human["uniprot"]))
    agg = aggregate(acts)

    multi = multi_domain_uniprots(kinases, "Human")
    table = build_table(structures, agg, multi, a.min_kinases)
    drugs = klifs_drug_names(cache)
    table["drug_name"] = table["ligand_code"].map(drugs)
    table["chembl_parent"] = table["ligand_code"].map(lambda c: ";".join(link[c]))
    sort_p = f"p{MEASURES[0]}"
    table = table.sort_values(["ligand_code", sort_p], ascending=[True, False])[TABLE_COLUMNS]

    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.table, index=False)
    summary = ligand_summary(table, agg)
    summary.to_csv(a.out / "ligands.csv", index=False)
    acts.to_csv(a.out / "activities.csv", index=False)
    agg.sort_values(["ligand_code", "measure", "pX"], ascending=[True, True, False]).to_csv(
        a.out / "affinities.csv", index=False)

    status = acts.groupby(["standard_type", "status"]).size()
    used = acts[acts["status"] == "wild_type"]
    (a.out / "funnel.json").write_text(json.dumps({
        "started_at": started,
        "sources": {"klifs_raw": str(a.klifs / "klifs_raw.csv"), "klifs_api": KLIFS_API,
                    "chembl_api": CHEMBL_API,
                    "chembl_version": _get_json(f"{CHEMBL_API}/status.json",
                                                cache / "chembl_status.json")
                    .get("chembl_db_version")},
        "thresholds": {"max_resolution": a.max_resolution, "min_quality": a.min_quality,
                       "min_kinases": a.min_kinases, "measures": list(MEASURES),
                       "inconsistent_std_log_units": INCONSISTENT_STD,
                       "exclude_ligands": DEFAULT_EXCLUDE},
        "structure_funnel": funnel,
        "candidate_ligands": ligands,
        "chembl_parent": link,
        "activity_status": {f"{m}/{s}": int(n) for (m, s), n in status.items()},
        "klifs_vs_chembl_rows_with_pchembl": {
            c: {"klifs": klifs_rows[c],
                "chembl": int(((acts["ligand_code"] == c)
                               & acts["pchembl_value"].notna()).sum())} for c in ligands},
        "top_documents_in_used_values": {
            m: used[used["standard_type"] == m]["document_chembl_id"].value_counts()
            .head(10).to_dict() for m in MEASURES},
        "sets": {m: {"ligands": int((summary[f"in_{m}_set"] > 0).sum()),
                     "complexes": int(table[f"in_{m}_set"].sum())} for m in MEASURES},
    }, indent=2) + "\n")

    print(summary.to_string(index=False), file=sys.stderr)
    for m in MEASURES:
        print(f"{m} set: {int((summary[f'in_{m}_set'] > 0).sum())} ligands, "
              f"{int(table[f'in_{m}_set'].sum())} complexes", file=sys.stderr)


if __name__ == "__main__":
    main()
