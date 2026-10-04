"""Manifest of human kinase–inhibitor co-crystal structures from KLIFS.

Produces the list of (PDB, chain, altloc, ligand code) that a later step will download
and run as

    frustx complex.pdb -o out/ --ligand lig.sdf --ligand-name3 <ligand_code>

This step downloads NO structures; it queries KLIFS and RCSB metadata only and writes:

    klifs_raw.csv       every KLIFS structure, all species, unfiltered (reproducibility)
    klifs_manifest.csv  one structure per (kinase UniProt, ligand code)
    klifs_funnel.json   counts after each filter, thresholds, fetch dates, API version
    rcsb_cache.json     RCSB metadata responses, reused on the next run (--refresh-rcsb)

    .venv/bin/python scripts/klifs_manifest.py                  # -> data/kinome/klifs/

Data source: the KLIFS REST API v2 (https://klifs.net/api_v2), called directly.
opencadd's `setup_remote()` is a client for this same API, but opencadd is not installed
here (and its bravado/biopandas stack is not a safe bet on Python 3.14); calling the API
with urllib adds no dependency at all.

Facts about the KLIFS fields that the filters below depend on, each checked against the
live API (2026-10-04) rather than assumed:

  - `ligand` / `allosteric_ligand` are the PDB chemical component id (= HETATM residue
    name). "No ligand" is the integer 0, not "" or "-".
  - `resolution` is "0" for NMR entries, so `resolution <= 2.5` alone would let every
    NMR model through. Those rows also carry the model number in `alt` ("1".."20").
  - `resolution` and `quality_score` arrive as strings.
  - The orthosteric `ligand` field contains no ions or buffers at all; its non-inhibitors
    are nucleotides and nucleotide analogues, which is what DEFAULT_EXCLUDE lists.
  - Ligand names (from `ligands_list`) are HTML-escaped ("&apos;").
  - KLIFS still lists PDB entries that RCSB has made obsolete, and occasionally a ligand
    code the entry does not contain. Hence the RCSB check (rcsb_annotate). Obsolete rows
    are remapped for the report, but never enter the manifest (see KEEP_STATUSES).

Pipeline order (build_manifest): KLIFS filters -> RCSB check and its drops ->
deduplicate. The RCSB drops MUST precede deduplication: otherwise a structure that RCSB
rejects can win its (uniprot, ligand) pair and take the whole pair down with it, even
though a valid runner-up existed.

All filtering/annotation/deduplication logic is pure (DataFrames and dicts in, DataFrames
out) and lives apart from the network code, so tests/test_klifs_manifest.py can exercise
it offline.
"""

import argparse
import html
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KLIFS_API = "https://klifs.net/api_v2"
KLIFS_SWAGGER = "https://klifs.net/swagger_v2/swagger.json"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_REMOVED = "https://data.rcsb.org/rest/v1/holdings/removed/{}"

# Kinase IDs per request. 200 returns ~2.6 MB in ~1 s; KLIFS rejects the whole request
# (HTTP 400) if any single ID is unknown, so IDs only ever come from kinase_information.
CHUNK = 200
# PDB ids per GraphQL request. 300 ran cleanly for the ~3,200-id probe.
RCSB_BATCH = 300
# How many supersessions to follow (A obsoleted by B, B obsoleted by C, ...).
MAX_REMAP_DEPTH = 5

# Nucleotides / nucleotide analogues seen as the orthosteric ligand in human KLIFS
# structures (counts at 2026-10-04): ANP 462, ADP 287, ATP 230, ACP 138, AGS 40, AMP 39,
# ADN 24, 3AM 6, 3GU 2, AP2 2, M33 2, 112 2, NBS 1. MG and MN never occur in this field;
# they are listed because the task asked for them, and cost nothing.
# NBS (N6-benzyl-ADP, a substrate for analogue-sensitive kinase mutants) and 3GU
# (N6-cyclopentyladenosine) were kept in the first version as borderline; they are
# nucleotide/nucleoside analogues, not inhibitors, so they are excluded too.
DEFAULT_EXCLUDE = ["ATP", "ADP", "ANP", "ACP", "AGS", "AMP", "ADN", "3AM", "AP2",
                   "M33", "112", "NBS", "3GU", "MG", "MN"]

MANIFEST_COLUMNS = [
    "kinase_name", "kinase_family", "kinase_group", "uniprot", "klifs_structure_id",
    "pdb_klifs", "pdb", "chain", "altloc", "ligand_code", "ligand_name",
    "allosteric_ligand_code", "resolution", "quality_score", "missing_residues",
    "missing_atoms", "dfg", "ac_helix", "rcsb_status", "ligand_chain_differs",
    "multi_copy_ligand", "altloc_ligand_conflict",
]

# Only "ok" stays in the manifest; the rest are dropped, one funnel step each, in
# DROP_STATUSES order. obsolete_remapped is dropped too: the row's quality, resolution
# and missing-residue values are KLIFS's annotation of the OLD deposit, and do not
# describe the replacement file a download would fetch (6MX8, which replaced 5J7H:
# quality 6.8 with 3 missing residues, against 8 and 0 for the old entry). Where KLIFS
# has annotated the replacement itself, that row competes in deduplication on its own.
KEEP_STATUSES = ["ok"]
DROP_STATUSES = ["obsolete_no_replacement", "ligand_missing", "non_xray",
                 "obsolete_remapped"]


# --- network -----------------------------------------------------------------------

def _get(url: str):
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.load(r)


def _chunked(endpoint: str, kinase_ids: list[int]) -> list[dict]:
    rows = []
    for i in range(0, len(kinase_ids), CHUNK):
        ids = ",".join(map(str, kinase_ids[i:i + CHUNK]))
        rows += _get(f"{KLIFS_API}/{endpoint}?kinase_ID={ids}")
    return rows


def fetch_raw() -> tuple[pd.DataFrame, str]:
    """Every KLIFS structure (all species), with kinase family/group/UniProt and the
    ligand's name joined on. Column names are KLIFS's own, untouched.

    Returns (table, KLIFS API version string).
    """
    api_version = _get(KLIFS_SWAGGER)["info"]["version"]

    # No species parameter: fetch everything, so the species filter is a real, counted
    # step in the funnel rather than something done invisibly at query time.
    kinases = pd.DataFrame(_get(f"{KLIFS_API}/kinase_information"))
    ids = kinases["kinase_ID"].tolist()

    structures = pd.DataFrame(_chunked("structures_list", ids))

    # ligands_list has no global form; ask per kinase and collapse on PDB code. Checked:
    # no code maps to two different names.
    ligands = pd.DataFrame(_chunked("ligands_list", ids))
    names = (ligands.drop_duplicates("PDB-code")
             .set_index("PDB-code")["Name"].map(html.unescape))

    raw = structures.merge(
        kinases[["kinase_ID", "family", "group", "uniprot"]], on="kinase_ID", how="left")
    raw["ligand_name"] = raw["ligand"].map(names)
    return raw, api_version


# Polymer fields are not used by rcsb_annotate; they are cached so that a ligand code
# found only inside a polymer (a covalent / peptide-linked ligand) can be diagnosed
# later without another round of requests.
_RCSB_QUERY = """query($ids: [String!]!) { entries(entry_ids: $ids) {
  rcsb_id
  exptl { method }
  nonpolymer_entities { nonpolymer_entity_instances {
    rcsb_nonpolymer_entity_instance_container_identifiers { auth_asym_id comp_id } } }
  polymer_entities { rcsb_polymer_entity_container_identifiers {
    auth_asym_ids chem_comp_nstd_monomers } }
} }"""


def _graphql_entries(ids: list[str]) -> dict:
    """{id: entry}. RCSB silently OMITS ids that are not current entries (obsolete or
    nonexistent) -- no null, no error -- so absence from the result is the signal."""
    out = {}
    for i in range(0, len(ids), RCSB_BATCH):
        body = json.dumps({"query": _RCSB_QUERY,
                           "variables": {"ids": ids[i:i + RCSB_BATCH]}}).encode()
        req = urllib.request.Request(RCSB_GRAPHQL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        if d.get("errors"):
            raise RuntimeError(f"RCSB GraphQL error: {d['errors'][:2]}")
        for e in d["data"]["entries"] or []:
            if e:
                out[e["rcsb_id"]] = e
    return out


def _replaced_by(pdb: str) -> list[str] | None:
    """Ids that superseded an obsolete entry; [] if removed without replacement. None if
    RCSB has no removal record at all (404) -- i.e. it never existed under this id."""
    try:
        d = _get(RCSB_REMOVED.format(pdb))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return d["rcsb_repository_holdings_removed"].get("id_codes_replaced_by") or []


def fetch_rcsb(pdb_ids: list[str], cache_path: Path, refresh: bool = False) -> dict:
    """RCSB metadata for `pdb_ids` and, transitively, for whatever superseded them.

    Returns {"entries": {id: entry or None}, "removed": {id: [ids] or None},
    "fetched_at": ...}. entries[id] is None when the id is not a current entry. Only ids
    not already in the cache go to the network, so a re-run with the same manifest makes
    no requests; refresh=True starts from an empty cache.
    """
    cache = {"entries": {}, "removed": {}, "fetched_at": None}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text())
    fetched = False

    todo = {p.upper() for p in pdb_ids}
    for _ in range(MAX_REMAP_DEPTH + 1):
        need = sorted(p for p in todo if p not in cache["entries"])
        if need:
            got = _graphql_entries(need)
            cache["entries"].update({p: got.get(p) for p in need})
            fetched = True
        gone = sorted(p for p in todo
                      if cache["entries"][p] is None and p not in cache["removed"])
        for p in gone:
            cache["removed"][p] = _replaced_by(p)
            fetched = True
        # next round: the replacements of whatever was obsolete
        todo = {r for p in todo if cache["entries"][p] is None
                for r in (cache["removed"].get(p) or [])}
        if not todo:
            break

    if fetched:
        cache["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return cache


# --- pure logic (tested offline) ---------------------------------------------------

def _code(value) -> str:
    """KLIFS ligand field -> PDB component id, or "" for none.

    KLIFS uses integer 0 for "no ligand"; it becomes the string "0" after a CSV round
    trip. "" and "-" are accepted too, since that is what other KLIFS clients emit.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().upper()
    return "" if s in ("", "-", "0") else s


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """KLIFS column names/types -> manifest names/types. No rows are dropped."""
    df = pd.DataFrame({
        "species": raw["species"],
        "kinase_name": raw["kinase"],
        "kinase_family": raw["family"],
        "kinase_group": raw["group"],
        "uniprot": raw["uniprot"].fillna("").astype(str),
        "klifs_structure_id": raw["structure_ID"].astype(int),
        # pdb_klifs is KLIFS's id exactly as given (lowercase). `pdb` is upper-cased here
        # (RCSB's convention; ids are case-insensitive) and becomes the current RCSB id
        # once rcsb_annotate has followed any supersession.
        "pdb_klifs": raw["pdb"].astype(str),
        "pdb": raw["pdb"].astype(str).str.upper(),
        "chain": raw["chain"].astype(str),
        "altloc": raw["alt"].fillna("").astype(str).str.strip(),
        "ligand_code": raw["ligand"].map(_code),
        "ligand_name": raw["ligand_name"],
        "allosteric_ligand_code": raw["allosteric_ligand"].map(_code),
        "resolution": pd.to_numeric(raw["resolution"], errors="coerce"),
        "quality_score": pd.to_numeric(raw["quality_score"], errors="coerce"),
        "missing_residues": pd.to_numeric(raw["missing_residues"], errors="coerce"),
        "missing_atoms": pd.to_numeric(raw["missing_atoms"], errors="coerce"),
        "dfg": raw["DFG"],
        "ac_helix": raw["aC_helix"],
    })
    # resolution 0 means "not applicable" (NMR), not "infinitely good". NaN makes the
    # resolution filter reject it instead of passing it as 0 <= max.
    df.loc[df["resolution"] <= 0, "resolution"] = float("nan")
    return df


def _counts(step: str, df: pd.DataFrame) -> dict:
    return {
        "step": step,
        "structures": len(df),
        "pdb_entries": int(df["pdb"].nunique()),
        "kinases": int(df["uniprot"].nunique()),
        "ligands": int(df.loc[df["ligand_code"] != "", "ligand_code"].nunique()),
    }


def apply_filters(df: pd.DataFrame, *, species: str, exclude: list[str],
                  max_resolution: float, min_quality: float
                  ) -> tuple[pd.DataFrame, list[dict]]:
    """Filters 1-6, in order. Returns (surviving rows, funnel counts after each step)."""
    excl = {e.strip().upper() for e in exclude}
    steps = [
        ("raw", lambda d: d),
        (f"species == {species}", lambda d: d[d["species"] == species]),
        ("orthosteric ligand present", lambda d: d[d["ligand_code"] != ""]),
        ("ligand not in exclude list", lambda d: d[~d["ligand_code"].isin(excl)]),
        # notna() is explicit even though NaN <= x is already False: it documents that
        # a missing resolution (incl. NMR) fails, rather than relying on NaN semantics.
        (f"resolution <= {max_resolution}",
         lambda d: d[d["resolution"].notna() & (d["resolution"] <= max_resolution)]),
        (f"quality_score >= {min_quality}",
         lambda d: d[d["quality_score"].notna() & (d["quality_score"] >= min_quality)]),
        ("altloc in {'A', ''}", lambda d: d[d["altloc"].isin(["A", ""])]),
    ]
    funnel = []
    for name, f in steps:
        df = f(df)
        funnel.append(_counts(name, df))
    return df, funnel


def altloc_ligand_conflicts(norm_raw: pd.DataFrame) -> set[tuple[str, str]]:
    """(pdb_klifs, chain) pairs where KLIFS's rows for DIFFERENT altlocs of the same
    chain name DIFFERENT orthosteric ligands -- e.g. 6HOP chain A, whose altlocs A-D
    carry four different compounds, or CDK2 1H00/1H01/1H07/1H08, where two enantiomers
    were modelled as altlocs.

    Must be given the normalised RAW table, not the filtered one: the altloc filter has
    already discarded the B/C/D rows that reveal the conflict.

    Only non-empty ligand codes count. An altloc with a ligand next to one without is
    not a conflict -- that is two altlocs, one ligand, not two ligands. (Counting "no
    ligand" as a value would add 45 more pairs, at 2026-10-04.)
    """
    d = norm_raw[norm_raw["ligand_code"] != ""]
    g = d.groupby(["pdb_klifs", "chain"]).agg(
        n_alt=("altloc", "nunique"), n_lig=("ligand_code", "nunique"))
    return set(g[(g["n_alt"] > 1) & (g["n_lig"] > 1)].index)


def _resolve(pdb: str, rcsb: dict, depth: int = MAX_REMAP_DEPTH) -> list[str]:
    """Current RCSB ids standing for `pdb`: [pdb] if it is current; otherwise whatever
    superseded it, followed through chains of supersession; [] if nothing did."""
    if rcsb["entries"].get(pdb) is not None:
        return [pdb]
    if depth == 0:
        return []
    out = []
    for r in sorted(rcsb["removed"].get(pdb) or []):
        out += _resolve(r, rcsb, depth - 1)
    return out


def _ligand_chains(entry: dict, code: str) -> list[str]:
    """auth_asym_id of every non-polymer copy of `code` in `entry`, one per copy."""
    return [i["auth_asym_id"]
            for ne in entry.get("nonpolymer_entities") or []
            for x in ne["nonpolymer_entity_instances"]
            for i in [x["rcsb_nonpolymer_entity_instance_container_identifiers"]]
            if i["comp_id"] == code]


def rcsb_annotate(df: pd.DataFrame, rcsb: dict,
                  conflicts: set[tuple[str, str]]) -> pd.DataFrame:
    """Add `pdb` (current RCSB id), `rcsb_status` and three flag columns. No rows dropped.

    rcsb_status, first match wins (also the funnel's drop order):
      obsolete_no_replacement  pdb_klifs is not a current entry and nothing superseded it
      ligand_missing           ligand_code is not a non-polymer component of the current
                               entry (after remapping, of the replacement). A code found
                               only inside a polymer (covalently linked) also counts here.
      non_xray                 no experimental method is X-RAY DIFFRACTION (e.g. cryo-EM).
                               A joint X-ray/neutron entry still counts as X-ray.
      obsolete_remapped        superseded, and the replacement has the ligand (and is X-ray).
                               Dropped as well: see KEEP_STATUSES.
      ok                       current, has the ligand, X-ray
    If several entries superseded one, the first (by id) that contains the ligand is used.

    Flags (False whenever the ligand is not in the entry at all):
      ligand_chain_differs     the ligand is in the entry but no copy carries the KLIFS
                               protein chain id -- typically 3-character ligand chains
                               ("AAA", "BBB") that do not fit the PDB format, so the
                               download step must use mmCIF for these.
      multi_copy_ligand        more than one copy of ligand_code carries the KLIFS protein
                               chain id; the download step must pick the one in the pocket.
      altloc_ligand_conflict   (pdb_klifs, chain) is in `conflicts`, see
                               altloc_ligand_conflicts(). Keyed on KLIFS's id, since the
                               conflict is a property of KLIFS's rows.
    The chain id is always KLIFS's. For an obsolete_remapped row the flags are computed
    against the replacement entry under that chain id, which is NOT verified to name the
    same chain there -- harmless, since remapped rows never reach the manifest; the remap
    only feeds the report (`pdb`, remap_collisions) and the cache.
    """
    pdb, status, differs, multi = [], [], [], []
    for p, chain, code in zip(df["pdb"], df["chain"], df["ligand_code"]):
        candidates = _resolve(p, rcsb)
        with_lig = [c for c in candidates if _ligand_chains(rcsb["entries"][c], code)]
        current = (with_lig or candidates or [p])[0]
        chains = _ligand_chains(rcsb["entries"][current], code) if candidates else []
        methods = ({m["method"] for m in rcsb["entries"][current].get("exptl") or []}
                   if candidates else set())

        if not candidates:
            s = "obsolete_no_replacement"
        elif not chains:
            s = "ligand_missing"
        elif "X-RAY DIFFRACTION" not in methods:
            s = "non_xray"
        else:
            s = "ok" if current == p else "obsolete_remapped"

        pdb.append(current)
        status.append(s)
        differs.append(bool(chains) and chain not in chains)
        multi.append(chains.count(chain) > 1)

    out = df.copy()
    out["pdb"] = pdb
    out["rcsb_status"] = status
    out["ligand_chain_differs"] = differs
    out["multi_copy_ligand"] = multi
    out["altloc_ligand_conflict"] = [(p, c) in conflicts
                                     for p, c in zip(df["pdb_klifs"], df["chain"])]
    return out


def apply_rcsb_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop each DROP_STATUSES value in turn, one funnel step per status."""
    funnel = []
    for s in DROP_STATUSES:
        df = df[df["rcsb_status"] != s]
        funnel.append(_counts(f"rcsb: drop {s}", df))
    return df, funnel


def remap_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that land on the same (pdb, chain, altloc) as another row once obsolete ids
    are remapped -- e.g. KLIFS lists both an obsolete entry and its replacement. Only
    groups containing at least one remapped row; reported, not resolved."""
    dup = df[df.duplicated(["pdb", "chain", "altloc"], keep=False)]
    remapped = dup.groupby(["pdb", "chain", "altloc"])["rcsb_status"].transform(
        lambda s: (s == "obsolete_remapped").any())
    return dup[remapped].sort_values(["pdb", "chain", "altloc", "pdb_klifs"])


# Tie-break order for choosing one structure per (uniprot, ligand_code). The task fixes
# the first five keys; chain, altloc and structure id are appended only so that two
# chains of the same PDB entry (which tie on everything else) resolve deterministically.
SORT_KEYS = [
    ("quality_score", False), ("resolution", True), ("missing_residues", True),
    ("missing_atoms", True), ("pdb", True), ("chain", True), ("altloc", True),
    ("klifs_structure_id", True),
]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (uniprot, ligand_code): the first under SORT_KEYS."""
    if (df["uniprot"] == "").any():
        # An empty UniProt would silently merge unrelated kinases into one group.
        bad = df.loc[df["uniprot"] == "", "kinase_name"].unique().tolist()
        raise ValueError(f"kinases without a UniProt accession: {bad}")
    cols, asc = zip(*SORT_KEYS)
    ordered = df.sort_values(list(cols), ascending=list(asc), kind="mergesort")
    return ordered.drop_duplicates(["uniprot", "ligand_code"], keep="first")


def build_manifest(norm: pd.DataFrame, get_rcsb, *, species: str, exclude: list[str],
                   max_resolution: float, min_quality: float):
    """The whole selection, in its required order. `get_rcsb(pdb_ids) -> rcsb dict` is
    a callable so tests can pass a fixture; it is asked only about rows that survive the
    KLIFS filters.

    Returns (manifest, annotated rows before the RCSB drops, funnel).
    """
    kept, funnel = apply_filters(norm, species=species, exclude=exclude,
                                 max_resolution=max_resolution, min_quality=min_quality)
    rcsb = get_rcsb(sorted(kept["pdb"].unique()))
    annotated = rcsb_annotate(kept, rcsb, altloc_ligand_conflicts(norm))
    kept, rcsb_funnel = apply_rcsb_filters(annotated)
    funnel += rcsb_funnel
    # Deduplicate LAST, over RCSB-validated rows only (see module docstring).
    manifest = deduplicate(kept)
    funnel.append(_counts("deduplicate (uniprot, ligand_code)", manifest))
    return manifest, annotated, funnel


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("data/kinome/klifs"))
    p.add_argument("--species", default="Human")
    p.add_argument("--exclude-ligands", nargs="*", default=DEFAULT_EXCLUDE,
                   metavar="CODE", help="orthosteric ligand codes to drop "
                   f"(default: {' '.join(DEFAULT_EXCLUDE)})")
    p.add_argument("--max-resolution", type=float, default=2.5)
    p.add_argument("--min-quality", type=float, default=6.0)
    p.add_argument("--refresh-rcsb", action="store_true",
                   help="ignore rcsb_cache.json and re-query RCSB for every id")
    a = p.parse_args(argv)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw, api_version = fetch_raw()
    a.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(a.out / "klifs_raw.csv", index=False)

    cache_path = a.out / "rcsb_cache.json"
    rcsb = {}

    def get_rcsb(ids):
        rcsb.update(fetch_rcsb(ids, cache_path, refresh=a.refresh_rcsb))
        return rcsb

    manifest, annotated, funnel = build_manifest(
        normalise(raw), get_rcsb, species=a.species, exclude=a.exclude_ligands,
        max_resolution=a.max_resolution, min_quality=a.min_quality)

    manifest = manifest.sort_values(["kinase_name", "ligand_code"])[MANIFEST_COLUMNS]
    manifest.to_csv(a.out / "klifs_manifest.csv", index=False)

    # Before the obsolete_remapped drop: shows which remapped rows had a KLIFS row for
    # the replacement entry to fall back on.
    collisions = remap_collisions(
        annotated[annotated["rcsb_status"].isin(["ok", "obsolete_remapped"])])
    flags = ["ligand_chain_differs", "multi_copy_ligand", "altloc_ligand_conflict"]

    (a.out / "klifs_funnel.json").write_text(json.dumps({
        "fetched_at": fetched_at,
        "source": {"api": KLIFS_API, "klifs_api_version": api_version,
                   "client": "urllib (direct REST)", "opencadd_version": None,
                   "rcsb_graphql": RCSB_GRAPHQL, "rcsb_cache": str(cache_path),
                   "rcsb_fetched_at": rcsb.get("fetched_at")},
        "thresholds": {"species": a.species, "exclude_ligands": a.exclude_ligands,
                       "max_resolution": a.max_resolution, "min_quality": a.min_quality,
                       "altloc": ["A", ""], "rcsb_drop": DROP_STATUSES,
                       "dedup_key": ["uniprot", "ligand_code"],
                       "dedup_order": [f"{c} {'asc' if s else 'desc'}"
                                       for c, s in SORT_KEYS]},
        "funnel": funnel,
        "rcsb_status_counts_before_drop":
            annotated["rcsb_status"].value_counts().to_dict(),
        "flag_counts_in_manifest": {f: int(manifest[f].sum()) for f in flags},
        "remap_collisions": collisions[["pdb", "pdb_klifs", "chain", "altloc",
                                        "ligand_code", "klifs_structure_id",
                                        "rcsb_status"]].to_dict("records"),
    }, indent=2) + "\n")

    for f in funnel:
        print(f"{f['step']:40} {f['structures']:6} structures  {f['pdb_entries']:6} pdb  "
              f"{f['kinases']:4} kinases  {f['ligands']:5} ligands", file=sys.stderr)
    if len(collisions):
        print(f"WARNING: {len(collisions)} rows share (pdb, chain, altloc) after "
              "remapping; listed under remap_collisions in klifs_funnel.json",
              file=sys.stderr)


if __name__ == "__main__":
    main()
