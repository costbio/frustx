"""BindingDB activities of the KLIFS-manifest ligands against human kinases.

Input (data/kinome/klifs/, written by scripts/klifs_manifest.py):
    klifs_manifest.csv   ligands, and which (kinase domain, ligand) pairs have a structure
    klifs_kinases.csv    KLIFS kinase_information: ALL human kinases (542 UniProt), the
                         target list -- off-target measurements matter for selectivity
    klifs_raw.csv        which kinases have any KLIFS structure (has_klifs_structure)

Output, under data/kinome/bindingdb/:

    raw/BindingDB_All_<YYYYMM>_tsv.zip   the release, verified against its published md5
    chemcomp_cache.json                   RCSB chem_comp InChIKey/SMILES per ligand code
    uniprot_cache.json                    UniProt kinase-domain spans + sequences of the
                                          proteins KLIFS splits into two kinase domains
    ligand_ids.csv                        ligand_code -> group, InChIKey, match level, HET check
    activities.csv                        per (uniprot, InChIKey, measure): median pX, n, std...
                                          match levels `full` and `bindingdb_no_stereo`
    activities_skeleton.csv               the remaining skeleton-only matches (kept apart)
    bindingdb_funnel.json                 funnel, provenance, coverage report, issues

    .venv/bin/python scripts/bindingdb_activity.py

Downloads no structures and selects no subset: the coverage report is there for the
subset decision to be made by hand.

Facts about the BindingDB_All TSV (release 202610) this code depends on, each checked
against the file rather than assumed:

  - The download page links the zip through SDFdownload.jsp, which returns an HTML page
    holding the real link; an md5 file sits next to the zip. find_release() follows both
    rather than building URLs.
  - One TSV member, 9.0 GB unpacked, 3,243,660 rows. The header has 640 fields: 40 fixed
    columns, then 50 blocks of 12 per target chain. Rows are padded to 640, so a
    multichain target shows up as more NON-EMPTY chain blocks, not as a longer row. Any
    row that disagrees with its declared chain count is counted in `problems`, not used.
  - Affinities are strings: "12.5", ">10000", "<0.03", occasionally "> 20000".
  - The organism column says "Homo sapiens" (2.48M rows), "Human" (0.21M) or
    "Homo sapiens (Human)"; all three mean the same organism (--organisms).
  - The SwissProt primary-id cell can hold several space-separated accessions.
  - Some rows name two or more KLIFS kinases among their chains (e.g. a "PI3K alpha"
    record that also lists ABL1). A value cannot be attributed to one kinase there, so
    they are dropped as a funnel step of their own.
  - BindingDB often stores a chiral or E/Z compound's InChIKey WITHOUT its stereo layer
    (second block "UHFFFAOYSA"): ruxolitinib, staurosporine, axitinib, lestaurtinib all
    sit there, while RCSB's keys carry stereo. BindingDB's own HET-id column confirms
    these are the same compounds. Hence the `bindingdb_no_stereo` match level.
  - Target names often carry the construct's residue range ("JAK2 [808-1132]", "TYK2
    [556-888]", "(aa658-end)", several ranges for deletion mutants). For proteins KLIFS
    splits into two kinase domains (JAK1/2/3 and TYK2 JH1/JH2, RSK/MSK N/C) that range
    says which domain was measured; see assign_domains().

Structure follows scripts/klifs_manifest.py: network code apart from pure functions,
which tests/test_bindingdb_activity.py exercises offline.
"""

import argparse
import hashlib
import io
import json
import math
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BINDINGDB = "https://www.bindingdb.org"
DOWNLOAD_PAGE = f"{BINDINGDB}/rwd/bind/chemsearch/marvin/Download.jsp"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
CHEMCOMP_BATCH = 500

DEFAULT_ORGANISMS = ["Homo sapiens", "Human", "Homo sapiens (Human)"]
# EC50 is deliberately absent: a functional readout, not a binding constant.
MEASURES = {"Ki": "Ki (nM)", "Kd": "Kd (nM)", "IC50": "IC50 (nM)"}
BINDING_CONSTANTS = ["Kd", "Ki"]    # never pooled with IC50, nor with each other
INCONSISTENT_STD = 1.0              # log units
NO_STEREO_BLOCK = "UHFFFAOYSA"      # InChIKey 2nd block of a structure with no stereo layer
MAIN_LEVELS = ["full", "bindingdb_no_stereo"]   # what activities.csv holds
# domain_source values whose kinase domain is known from the measurement itself (one
# KLIFS kinase on the protein, or the construct's residue range), not inferred from
# where the ligand happens to have been crystallised.
PRIMARY_SOURCES = ["single_domain", "construct"]

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/{}.json?fields=ft_domain,sequence"
# A construct belongs to a kinase domain if its residue range(s) cover >= 80% of that
# domain and of no other one (task specification).
MIN_CONSTRUCT_COVERAGE = 0.8
# Locating a KLIFS pocket in the UniProt sequence: only exact, unique substrings of at
# least this many residues count, and at least this fraction of the pocket must be found.
MIN_POCKET_SEGMENT = 5
MIN_POCKET_LOCATED = 0.5

COLUMNS = {
    "reactant_set_id": "BindingDB Reactant_set_id",
    "inchikey": "Ligand InChI Key",
    "bindingdb_name": "BindingDB Ligand Name",
    "target_name": "Target Name",
    "organism": "Target Source Organism According to Curator or DataSource",
    "het": "Ligand HET ID in PDB",
    "source": "Curation/DataSource",
    "pmid": "PMID",
    "doi": "Article DOI",
    "n_chains": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    **MEASURES,
}
CHAIN1_FIRST = "BindingDB Target Chain Sequence 1"
CHAIN2_FIRST = "BindingDB Target Chain Sequence 2"
CHAIN1_UNIPROT = "UniProt (SwissProt) Primary ID of Target Chain 1"


# --- network -----------------------------------------------------------------------

def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read().decode("utf-8", errors="replace")


def find_release() -> dict:
    """The current BindingDB_All TSV, as linked from the download page.

    The page's link goes to SDFdownload.jsp, which answers with an HTML page whose
    "Requested file ... is ready" link is the zip itself.
    """
    page = _get_text(DOWNLOAD_PAGE)
    m = re.search(r'href="([^"]*download_file=[^"]*?(BindingDB_All_(\d{6})_tsv\.zip))"',
                  page)
    md5 = re.search(r'href="([^"]*BindingDB_All_\d{6}_tsv\.md5)"', page)
    if not m or not md5:
        raise RuntimeError(f"no BindingDB_All_*_tsv.zip link on {DOWNLOAD_PAGE}")
    landing = _get_text(BINDINGDB + m.group(1))
    ready = re.search(r'Requested file <a href="([^"]+)"', landing)
    if not ready:
        raise RuntimeError("SDFdownload.jsp did not return the 'is ready' link")
    return {"name": m.group(2), "release": m.group(3),
            "zip_url": BINDINGDB + ready.group(1), "md5_url": BINDINGDB + md5.group(1)}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_download(release: dict, raw_dir: Path) -> tuple[Path, dict]:
    """Download the release zip unless an identical copy (by published md5) is already
    there. A sidecar .provenance.json keeps the original download date across re-runs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / release["name"]
    sidecar = dest.with_suffix(".provenance.json")
    published = _get_text(release["md5_url"]).split()[0].lower()

    if dest.exists() and _md5(dest) == published:
        if sidecar.exists():
            return dest, json.loads(sidecar.read_text())
        # file fetched before this script existed: its mtime is the best date there is
        prov = {"downloaded_at": datetime.fromtimestamp(
                    dest.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "downloaded_at_source": "file mtime"}
    else:
        part = dest.with_suffix(".part")
        h = hashlib.md5()
        with urllib.request.urlopen(release["zip_url"], timeout=600) as r, \
                open(part, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                h.update(block)
                f.write(block)
        if h.hexdigest() != published:
            raise RuntimeError(f"md5 mismatch for {part}: {h.hexdigest()} != {published}")
        part.rename(dest)
        prov = {"downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "downloaded_at_source": "download"}

    prov.update({"file": release["name"], "release": release["release"],
                 "url": release["zip_url"], "md5": published, "md5_published": published})
    sidecar.write_text(json.dumps(prov, indent=2) + "\n")
    return dest, prov


_CHEMCOMP_QUERY = """query($ids: [String!]!) { chem_comps(comp_ids: $ids) {
  rcsb_id rcsb_chem_comp_descriptor { InChIKey SMILES SMILES_stereo } } }"""


def fetch_chemcomps(codes: list[str], cache_path: Path, refresh: bool = False) -> dict:
    """{"comps": {code: {"inchikey", "smiles"} or None}, "fetched_at"}. RCSB silently
    omits codes it does not know (e.g. the withdrawn DRG); those are cached as None.
    Only codes missing from the cache go to the network."""
    cache = {"comps": {}, "fetched_at": None}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text())
    need = sorted(c for c in set(codes) if c not in cache["comps"])
    for i in range(0, len(need), CHEMCOMP_BATCH):
        ids = need[i:i + CHEMCOMP_BATCH]
        req = urllib.request.Request(
            RCSB_GRAPHQL, headers={"Content-Type": "application/json"},
            data=json.dumps({"query": _CHEMCOMP_QUERY, "variables": {"ids": ids}}).encode())
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        if d.get("errors"):
            raise RuntimeError(f"RCSB GraphQL error: {d['errors'][:2]}")
        got = {c["rcsb_id"]: c for c in d["data"]["chem_comps"] or [] if c}
        for code in ids:
            desc = (got.get(code) or {}).get("rcsb_chem_comp_descriptor") or {}
            cache["comps"][code] = ({"inchikey": desc.get("InChIKey") or "",
                                     # stereo SMILES where RCSB has one
                                     "smiles": desc.get("SMILES_stereo")
                                     or desc.get("SMILES") or ""}
                                    if code in got else None)
    if need:
        cache["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return cache


def fetch_uniprot(accessions: list[str], cache_path: Path, refresh: bool = False) -> dict:
    """{"entries": {acc: {"sequence", "crc64", "kinase_domains": [{description, start,
    end}]}}, "fetched_at"}: UniProt's "Protein kinase" Domain features (1-based,
    inclusive) and the canonical sequence. Only accessions missing from the cache go to
    the network."""
    cache = {"entries": {}, "fetched_at": None}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text())
    need = sorted(a for a in set(accessions) if a not in cache["entries"])
    for acc in need:
        with urllib.request.urlopen(UNIPROT_REST.format(acc), timeout=120) as r:
            d = json.load(r)
        cache["entries"][acc] = {
            "sequence": d["sequence"]["value"], "crc64": d["sequence"].get("crc64"),
            "kinase_domains": [
                {"description": f.get("description", ""),
                 "start": f["location"]["start"]["value"],
                 "end": f["location"]["end"]["value"]}
                for f in d.get("features", [])
                if f["type"] == "Domain"
                and f.get("description", "").startswith("Protein kinase")]}
    if need:
        cache["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return cache


def tsv_lines(zip_path: Path):
    """Lines of the single TSV member, decompressed as a stream (never written out)."""
    z = zipfile.ZipFile(zip_path)
    members = [i.filename for i in z.infolist() if i.filename.endswith(".tsv")]
    if len(members) != 1:
        raise RuntimeError(f"expected one .tsv in {zip_path}, found {members}")
    with z.open(members[0]) as f:
        yield from io.TextIOWrapper(f, encoding="utf-8", newline="")


# --- pure logic: reading -----------------------------------------------------------

def column_layout(header: list[str]) -> dict:
    """Field positions, taken from the header by name. Chain blocks repeat with a fixed
    stride; the stride is measured from the header, not hard-coded."""
    idx = {h: i for i, h in enumerate(header)}
    missing = [c for c in [*COLUMNS.values(), CHAIN1_FIRST, CHAIN1_UNIPROT] if c not in idx]
    if missing:
        raise ValueError(f"BindingDB header lacks {missing}")
    start = idx[CHAIN1_FIRST]
    stride = (idx[CHAIN2_FIRST] if CHAIN2_FIRST in idx else len(header)) - start
    return {"n_fields": len(header), "cols": {k: idx[c] for k, c in COLUMNS.items()},
            "chain_start": start, "chain_stride": stride,
            "uniprot_offset": idx[CHAIN1_UNIPROT] - start}


class _Stage:
    """Rows, distinct ligands (InChIKey) and distinct KLIFS kinases seen at one step."""

    def __init__(self, step):
        self.step, self.rows, self.ligands, self.kinases = step, 0, set(), set()

    def add(self, inchikey, kinases):
        self.rows += 1
        if inchikey:
            self.ligands.add(inchikey)
        self.kinases.update(kinases)

    def counts(self):
        return {"step": self.step, "rows": self.rows, "ligands": len(self.ligands),
                "kinases": len(self.kinases)}


def scan(lines, kinases: set[str], organisms: list[str], het_codes: set[str],
         skeletons: set[str]) -> dict:
    """One pass over the TSV lines (header first).

    Keeps, as `records`, only rows with an accepted organism and at least one chain whose
    SwissProt primary id is in `kinases`. Everything else only feeds counters, so memory
    stays at the size of the human-kinase subset.

    Also collects, from ALL rows (any organism/target), what the ligand-identity checks
    need: InChIKeys whose 14-character skeleton is in `skeletons`, and (HET id, InChIKey)
    pairs where the HET id is a manifest code or the skeleton matches.
    """
    it = iter(lines)
    layout = column_layout(next(it).rstrip("\r\n").split("\t"))
    c, start, stride, off = (layout["cols"], layout["chain_start"],
                             layout["chain_stride"], layout["uniprot_offset"])
    orgs = set(organisms)
    problems = Counter()
    stages = [_Stage("all rows"), _Stage(f"organism in {organisms}"),
              _Stage("a chain is a KLIFS human kinase")]
    kinase_rows_by_organism = Counter()
    records, het_pairs, seen = [], [], set()

    for line in it:
        r = line.rstrip("\r\n").split("\t")
        if len(r) != layout["n_fields"]:
            problems["field count differs from header (still parsed)"] += 1
        try:
            n = int(r[c["n_chains"]])
        except (ValueError, IndexError):
            problems["unreadable: chain count not an integer"] += 1
            continue
        if n < 1 or len(r) < start + n * stride:
            problems["unreadable: fewer fields than declared chains"] += 1
            continue
        if any(f.strip() for f in r[start + n * stride:]):
            problems["unreadable: data beyond declared chains"] += 1
            continue

        ik = r[c["inchikey"]].strip()
        if not ik:
            problems["no InChIKey (kept; cannot match a ligand)"] += 1
        hits = sorted({u for k in range(n) for u in r[start + k * stride + off].split()}
                      & kinases)
        stages[0].add(ik, hits)

        het = r[c["het"]].strip()
        if ik[:14] in skeletons:
            seen.add(ik)
        if het in het_codes or (ik and ik[:14] in skeletons):
            het_pairs.append((het, ik))

        organism = r[c["organism"]].strip()
        if hits:
            kinase_rows_by_organism[organism] += 1
        if organism not in orgs:
            continue
        stages[1].add(ik, hits)
        if not hits:
            continue
        stages[2].add(ik, hits)
        records.append({
            "reactant_set_id": r[c["reactant_set_id"]], "inchikey": ik,
            "bindingdb_name": r[c["bindingdb_name"]], "target_name": r[c["target_name"]],
            "organism": organism, "source": r[c["source"]],
            "pmid": r[c["pmid"]].strip(), "doi": r[c["doi"]].strip(), "het": het,
            "n_chains": n,
            "kinase_uniprots": ";".join(hits),
            **{m: r[c[m]] for m in MEASURES}})

    return {"records": pd.DataFrame(records, columns=[
                "reactant_set_id", "inchikey", "bindingdb_name", "target_name",
                "organism", "source", "pmid", "doi", "het", "n_chains", "kinase_uniprots",
                *MEASURES]),
            "funnel": [s.counts() for s in stages], "problems": dict(problems),
            "kinase_rows_by_organism": dict(kinase_rows_by_organism),
            "het_pairs": pd.DataFrame(het_pairs, columns=["het", "inchikey"]),
            "seen_inchikeys": seen}


def parse_affinity(s) -> tuple[str, float]:
    """'12.5' -> ('', 12.5); '>10000' / '> 10000' -> ('>', 10000.0); '<0.03' -> ('<', 0.03).
    Unparseable -> ('?', nan). Empty -> ('', nan)."""
    s = str(s).strip()
    if not s:
        return "", math.nan
    q = s[0] if s[0] in "<>" else ""
    try:
        return q, float(s[len(q):].strip())
    except ValueError:
        return "?", math.nan


def to_px(nM: float) -> float:
    """pX = 9 - log10(value in nM), i.e. -log10(molar). nan for non-positive values."""
    return 9.0 - math.log10(nM) if nM > 0 else math.nan


def _counts(step: str, df: pd.DataFrame) -> dict:
    out = {"step": step, "rows": len(df), "ligands": int(df["inchikey"].nunique()),
           "kinases": int(df["uniprot"].nunique())}
    if "ligand_group" in df:
        out["ligand_groups"] = int(df["ligand_group"].nunique())
    return out


def single_kinase(records: pd.DataFrame) -> pd.DataFrame:
    """Rows whose target names exactly one KLIFS kinase among its chains, with that
    kinase as `uniprot`. multichain_target: the target has other (non-kinase, or
    non-KLIFS) chains too, e.g. CDK2/cyclin A."""
    out = records[~records["kinase_uniprots"].str.contains(";")].copy()
    out["uniprot"] = out["kinase_uniprots"]
    out["multichain_target"] = out["n_chains"] > 1
    return out


# --- pure logic: ligand identity ---------------------------------------------------

def ligand_groups(ligands: pd.DataFrame) -> pd.DataFrame:
    """Add `ligand_group`: PDB codes that share one InChIKey are one compound (38Z/F3Z,
    XGK/XGQ) and are counted once, under their codes joined with "/". A code without an
    InChIKey is its own group."""
    has = ligands["inchikey"] != ""
    names = ligands[has].groupby("inchikey")["ligand_code"].agg(
        lambda s: "/".join(sorted(s)))
    out = ligands.copy()
    out["ligand_group"] = out["inchikey"].map(names).where(has, out["ligand_code"])
    return out


def match_ligands(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Attach manifest ligand groups by InChIKey (block1-block2-block3 = 14-10-1 chars):

      full                 all 27 characters equal
      bindingdb_no_stereo  BindingDB's key has NO stereo layer (block2 UHFFFAOYSA) while
                           the ligand's has one; skeleton and protonation (block3) equal;
                           and that skeleton belongs to exactly ONE manifest group. The
                           record is the ligand with its stereo left unstated.
      skeleton             same skeleton, anything else. match_note says why:
                             different_stereo_or_isotope   both keys state stereo
                             protonation_only              block2 equal, block3 differs
                             no_stereo_and_protonation     no stereo, and block3 differs
                             no_stereo_several_ligands     no stereo, but the skeleton fits
                                                           >1 manifest group (enantiomers
                                                           with own codes): not attributable

    HET tie-break: a no_stereo_several_ligands row whose BindingDB HET id ('Ligand HET ID
    in PDB', column `het` if `df` has it) is one of the codes of the group it is being
    matched to becomes bindingdb_no_stereo with match_note "het_tiebreak" for that group
    -- e.g. stereo-less ruxolitinib records with HET RXT go to RXT, not to its enantiomer
    RG4. An empty HET, or one naming a code outside the candidate groups, decides
    nothing: the row stays skeleton.

    A row appears once per group it matches. Groups without an InChIKey never match.
    """
    g = groups.loc[groups["inchikey"] != "", ["ligand_group", "inchikey"]].drop_duplicates()
    full = df.merge(g, on="inchikey")
    full["match_level"], full["match_note"] = "full", ""

    g2 = g.rename(columns={"inchikey": "_lig"}).assign(_sk=lambda d: d["_lig"].str[:14])
    per_skeleton = g2.groupby("_sk").size()
    sk = df.assign(_sk=df["inchikey"].str[:14]).merge(g2, on="_sk")
    sk = sk[sk["inchikey"] != sk["_lig"]].copy()
    bdb2, lig2 = sk["inchikey"].str[15:25], sk["_lig"].str[15:25]
    same_proton = sk["inchikey"].str[26:] == sk["_lig"].str[26:]
    no_stereo = (bdb2 == NO_STEREO_BLOCK) & (lig2 != NO_STEREO_BLOCK)
    unique = sk["_sk"].map(per_skeleton) == 1
    sk["match_note"] = "different_stereo_or_isotope"
    sk.loc[bdb2 == lig2, "match_note"] = "protonation_only"
    sk.loc[no_stereo & ~same_proton, "match_note"] = "no_stereo_and_protonation"
    sk.loc[no_stereo & same_proton & ~unique, "match_note"] = "no_stereo_several_ligands"
    ok = no_stereo & same_proton & unique
    sk["match_level"] = "skeleton"
    sk.loc[ok, ["match_level", "match_note"]] = ["bindingdb_no_stereo", ""]
    if "het" in sk:
        het_in_group = pd.Series(
            [h != "" and h in grp.split("/") for h, grp in zip(sk["het"], sk["ligand_group"])],
            index=sk.index, dtype=bool)
        tie = (sk["match_note"] == "no_stereo_several_ligands") & het_in_group
        sk.loc[tie, ["match_level", "match_note"]] = ["bindingdb_no_stereo", "het_tiebreak"]
    return pd.concat([full, sk.drop(columns=["_sk", "_lig"])], ignore_index=True)


LEVEL_RANK = {"full": 0, "bindingdb_no_stereo": 1, "skeleton": 2}


def ligand_table(codes: list[str], groups: pd.DataFrame, chem: dict, seen: set[str],
                 het_pairs: pd.DataFrame) -> pd.DataFrame:
    """ligand_code -> group, InChIKey/SMILES, whether BindingDB has it anywhere (any
    target, any organism) and at its best level, and the cross-check against
    BindingDB's own 'Ligand HET ID in PDB' column:
      het_rows                      rows whose HET id is this code
      het_rows_inchikey_full        ... whose InChIKey is this ligand's
      het_rows_inchikey_no_stereo   ... whose InChIKey is this ligand's without stereo
                                    (corroborates the bindingdb_no_stereo level)
      het_rows_inchikey_other       ... whose InChIKey is neither (disagreement)
      inchikey_rows_other_het       rows with this ligand's InChIKey but another HET id
                                    (disagreement; rows with no HET id are not counted)
    """
    # (HET, InChIKey) pairs as well as bare keys, so the HET tie-break applies here too
    probe = pd.concat([het_pairs, pd.DataFrame({"het": "", "inchikey": sorted(seen)})])
    seen_levels = match_ligands(probe.drop_duplicates(), groups)
    best = (seen_levels.assign(r=seen_levels["match_level"].map(LEVEL_RANK))
            .groupby("ligand_group")["r"].min()
            .map({v: k for k, v in LEVEL_RANK.items()}))
    out = groups[groups["ligand_code"].isin(codes)].copy()
    out["smiles"] = out["ligand_code"].map(lambda c: (chem.get(c) or {}).get("smiles", ""))
    out["match_level"] = out["ligand_group"].map(best).fillna("none")
    out.loc[out["inchikey"] == "", "match_level"] = "no_inchikey"
    out.loc[out["ligand_code"].map(lambda c: chem.get(c) is None), "match_level"] = \
        "no_chem_comp"
    out["in_bindingdb"] = out["match_level"].isin(LEVEL_RANK)

    by_het = het_pairs.merge(out[["ligand_code", "inchikey"]], left_on="het",
                             right_on="ligand_code", suffixes=("", "_lig"))
    full = by_het["inchikey"] == by_het["inchikey_lig"]
    nost = (~full & (by_het["inchikey"].str[:14] == by_het["inchikey_lig"].str[:14])
            & (by_het["inchikey"].str[15:25] == NO_STEREO_BLOCK))
    by_ik = het_pairs[het_pairs["het"] != ""].merge(
        out.loc[out["inchikey"] != "", ["ligand_code", "inchikey"]], on="inchikey")
    other_het = by_ik[by_ik["het"] != by_ik["ligand_code"]]
    count = lambda d: out["ligand_code"].map(d.groupby("ligand_code").size())  # noqa: E731
    out["het_rows"] = count(by_het)
    out["het_rows_inchikey_full"] = count(by_het[full])
    out["het_rows_inchikey_no_stereo"] = count(by_het[nost])
    out["het_rows_inchikey_other"] = count(by_het[~full & ~nost])
    out["inchikey_rows_other_het"] = count(other_het)
    het_cols = [c for c in out.columns if c.startswith(("het_rows", "inchikey_rows"))]
    out[het_cols] = out[het_cols].fillna(0).astype(int)
    return out[["ligand_code", "ligand_group", "inchikey", "smiles", "in_bindingdb",
                "match_level", *het_cols]]


# --- pure logic: values and aggregation --------------------------------------------

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (BindingDB row, measure) with a value. Kd, Ki and IC50 stay separate
    rows with their own `measure`; nothing downstream pools across measures."""
    keep = [c for c in df.columns if c not in MEASURES]
    long = df.melt(id_vars=keep, value_vars=list(MEASURES), var_name="measure",
                   value_name="raw_value")
    long = long[long["raw_value"].astype(str).str.strip() != ""].copy()
    parsed = long["raw_value"].map(parse_affinity)
    long["qualifier"] = [q for q, _ in parsed]
    long["value_nM"] = [v for _, v in parsed]
    long["pX"] = long["value_nM"].map(to_px)
    long["censored"] = long["qualifier"].isin(["<", ">"])
    long["valid"] = long["pX"].notna() & (long["qualifier"] != "?")
    return long


DUP_KEY = ["uniprot", "inchikey", "ligand_group", "match_level", "measure", "qualifier",
           "value_nM"]


def drop_source_duplicates(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One copy of a value reported twice: same (uniprot, InChIKey, measure, qualifier,
    value) AND the same PMID, or the same DOI -- BindingDB imports the same paper via
    several routes (its own curation, ChEMBL, ...). Rows sharing neither a PMID nor a DOI
    are never merged: nothing says they are the same report. PMID first, then DOI, so a
    pair matching on either is caught.
    """
    dropped = {}
    for col in ("pmid", "doi"):
        dup = (long[col] != "") & long.duplicated(DUP_KEY + [col], keep="first")
        dropped[col] = int(dup.sum())
        long = long[~dup]
    return long, dropped


# --- pure logic: kinase domains ----------------------------------------------------

_RANGE = re.compile(r"^(?:aa)?(\d+)\s*[-\u2013]\s*(\d+|end)$", re.IGNORECASE)


def parse_ranges(target_name: str) -> list[tuple[int, float]]:
    """Residue ranges of the construct named in a BindingDB target name, as written in
    the 202610 release: "[808-1132]", "[536-812,V617F]", "(536-812)", "(aa658-end)",
    "[1-746,750-1210,A750P]" (several ranges: a deletion construct), "[1-775,'YVMA',
    776-1255]" (an insertion between ranges). Mutations, insertions and labels inside the
    brackets are ignored; "end" is open (inf). No range -> [] (the full protein)."""
    out = []
    for group in re.findall(r"[\[(]([^\[\]()]*)[\])]", target_name):
        for token in group.split(","):
            m = _RANGE.match(token.strip())
            if m:
                a, b = int(m.group(1)), m.group(2)
                out.append((a, math.inf if b.lower() == "end" else int(b)))
    return out


def domain_coverage(ranges: list[tuple[int, float]], start: int, end: int) -> float:
    """Fraction of residues start..end (inclusive) covered by the union of `ranges`."""
    covered = set()
    for a, b in ranges:
        covered.update(range(max(a, start), int(min(b, end)) + 1))
    return len(covered) / (end - start + 1)


def construct_domain(ranges, spans: dict[int, tuple[int, int]]) -> int | None:
    """The one KLIFS kinase whose UniProt domain the construct covers by at least
    MIN_CONSTRUCT_COVERAGE; None if it covers several, or none."""
    hits = [k for k, (a, b) in spans.items()
            if domain_coverage(ranges, a, b) >= MIN_CONSTRUCT_COVERAGE]
    return hits[0] if len(hits) == 1 else None


def locate_pocket(pocket: str, sequence: str) -> list[int]:
    """1-based sequence positions of the KLIFS pocket residues that can be placed
    without guessing. The 85-residue pocket is a concatenation of sequence segments whose
    boundaries KLIFS does not mark ('-'/'_' only mark missing residues), so it is walked
    left to right taking the LONGEST exact substring found in the sequence; a stretch
    counts only if it is >= MIN_POCKET_SEGMENT long and occurs exactly once in the
    sequence. Everything else is left unplaced."""
    pocket = pocket.replace("_", "-")
    pos, i = [], 0
    while i < len(pocket):
        k = 0
        while (i + k < len(pocket) and pocket[i + k] != "-"
               and pocket[i:i + k + 1] in sequence):
            k += 1
        seg = pocket[i:i + k]
        if k >= MIN_POCKET_SEGMENT and sequence.count(seg) == 1:
            j = sequence.index(seg)
            pos.extend(range(j + 1, j + k + 1))
            i += k
        else:
            i += max(k, 1)
    return pos


def map_klifs_domains(kinases: pd.DataFrame, uniprot: dict) -> tuple[dict, list[dict]]:
    """KLIFS kinase id -> (start, end) of its UniProt "Protein kinase" domain.

    `kinases`: kinase_ID, name, uniprot, pocket (multi-domain proteins). A kinase is
    mapped when >= MIN_POCKET_LOCATED of its pocket is placed in the UniProt sequence and
    EVERY placed residue lies inside one and the same UniProt kinase domain, and no other
    KLIFS kinase of that protein maps to that domain. Anything else is returned in
    `unmapped` with the numbers, never resolved by a guess."""
    spans, unmapped, claimed = {}, [], {}
    for r in kinases.itertuples(index=False):
        entry = uniprot.get(r.uniprot)
        if not entry or not entry["kinase_domains"]:
            unmapped.append({"kinase_ID": r.kinase_ID, "name": r.name,
                             "reason": "no UniProt kinase domain features"})
            continue
        n_res = len(r.pocket.replace("-", "").replace("_", ""))
        placed = locate_pocket(r.pocket, entry["sequence"])
        inside = [d for d in entry["kinase_domains"]
                  if all(d["start"] <= x <= d["end"] for x in placed)]
        frac = len(placed) / n_res if n_res else 0.0
        if frac < MIN_POCKET_LOCATED or len(inside) != 1:
            unmapped.append({"kinase_ID": r.kinase_ID, "name": r.name,
                             "reason": "pocket not placed in exactly one domain",
                             "placed_fraction": round(frac, 2),
                             "domains_containing_all": [d["description"] for d in inside]})
            continue
        d = inside[0]
        spans[r.kinase_ID] = (d["start"], d["end"])
        claimed.setdefault((r.uniprot, d["start"]), []).append(r.kinase_ID)
    for (_, _), ids in claimed.items():
        if len(ids) > 1:     # two KLIFS kinases on one UniProt domain: trust neither
            for k in ids:
                unmapped.append({"kinase_ID": k, "name": "", "reason":
                                 f"shares a UniProt domain with {sorted(set(ids) - {k})}"})
                spans.pop(k, None)
    return spans, unmapped


def assign_domains(long: pd.DataFrame, manifest: pd.DataFrame,
                   domains: dict[str, list[int]],
                   spans: dict[int, tuple[int, int]]) -> pd.DataFrame:
    """Per MEASUREMENT (two measurements of one ligand on one protein may come from
    different constructs): klifs_kinase_id, domain_source, multi_domain_uniprot,
    domain_ambiguous.

    BindingDB measures a protein (or a construct of it); KLIFS's unit is a kinase domain.

      single_domain        the UniProt has one KLIFS kinase
    For a UniProt with several (JAK1/2/3, TYK2: JH1 + JH2; RSK, MSK: N- + C-terminal),
    first the construct, if the target name gives residue ranges (single-chain targets
    only -- in "Cyclin-A2 [171-432]/CDK2" the range is the cyclin's):
      construct            ranges cover >= 80% of exactly one domain
      construct_ambiguous  ranges cover both domains, or neither
    then, only for rows WITHOUT ranges (the full protein), where the ligand's crystal is:
      structure            ligand has a manifest structure in exactly one domain
      structure_ambiguous  ... in several
      uniprot_level        ... in none: the row stays at protein level, no id
    The construct rule needs every domain of the protein mapped (map_klifs_domains);
    if one is not, ranged rows fall through to the structure rule.

    Also kept, for reporting what the construct rule changed: structure_rule_kinase_id
    and structure_rule_ambiguous -- what the structure rule alone would have said.
    """
    structured = manifest.groupby(["uniprot", "ligand_group"])["klifs_kinase_id"].agg(
        lambda s: sorted(set(s)))
    ranges_of = {t: parse_ranges(t) for t in long["target_name"].unique()}
    def one(u, g, t, n):
        """-> (kinase id, domain_source, structure-rule id, structure-rule ambiguous)"""
        ids = domains.get(u, [])
        if len(ids) == 1:
            return ids[0], "single_domain", ids[0], False
        st = structured.get((u, g), [])
        s_kid = st[0] if len(st) == 1 else None
        s_src = ("structure" if len(st) == 1 else
                 "structure_ambiguous" if len(st) > 1 else "uniprot_level")
        rng = ranges_of[t] if n == 1 else []
        if rng and ids and all(i in spans for i in ids):
            k = construct_domain(rng, {i: spans[i] for i in ids})
            return k, "construct" if k is not None else "construct_ambiguous", s_kid, len(st) > 1
        return s_kid, s_src, s_kid, len(st) > 1

    res = [one(*row) for row in zip(long["uniprot"], long["ligand_group"],
                                     long["target_name"], long["n_chains"])]
    kid, src, old_kid, old_amb = (list(x) for x in zip(*res)) if res else ([], [], [], [])
    out = long.copy()
    out["klifs_kinase_id"] = pd.array(kid, dtype="Int64")
    out["domain_source"] = src
    out["multi_domain_uniprot"] = out["domain_source"] != "single_domain"
    out["domain_ambiguous"] = out["domain_source"].str.endswith("_ambiguous")
    out["structure_rule_kinase_id"] = pd.array(old_kid, dtype="Int64")
    out["structure_rule_ambiguous"] = old_amb
    return out


def domain_changes(long: pd.DataFrame) -> dict:
    """What the construct rule changed, against the structure rule alone, over the
    multi-domain measurements (valid values, main match levels)."""
    m = long[long["multi_domain_uniprot"]]
    new_ok, old_ok = m["klifs_kinase_id"].notna(), m["structure_rule_kinase_id"].notna()
    kinds = {
        "rescued_from_ambiguous": m["structure_rule_ambiguous"] & new_ok,
        "assigned_from_uniprot_level": ~old_ok & ~m["structure_rule_ambiguous"] & new_ok,
        "reassigned_other_domain": old_ok & new_ok
                                   & (m["klifs_kinase_id"] != m["structure_rule_kinase_id"]),
        "now_ambiguous": ~m["structure_rule_ambiguous"] & m["domain_ambiguous"],
    }
    out = {}
    for name, mask in kinds.items():
        d = m[mask.fillna(False)]
        out[name] = {"values": len(d), "proteins": int(d["uniprot"].nunique()),
                     "ligand_groups": int(d["ligand_group"].nunique()),
                     "by_protein": d.groupby("uniprot").size().to_dict(),
                     "ligand_groups_list": sorted(d["ligand_group"].unique())[:50]}
    return out


def primary_eligible(domain_source: pd.Series) -> pd.Series:
    """True where the measured domain is known from the measurement (PRIMARY_SOURCES);
    False for structure (inferred from the crystal), uniprot_level and *_ambiguous."""
    return domain_source.isin(PRIMARY_SOURCES)


AGG_KEYS = ["uniprot", "klifs_kinase_id", "domain_source", "inchikey", "measure",
            "ligand_group", "match_level", "match_note"]


def aggregate(long: pd.DataFrame) -> pd.DataFrame:
    """Per (uniprot, kinase domain, domain_source, InChIKey, measure): median pX and
    sample std over UNCENSORED values only, n = their count, n_censored = '<'/'>' values
    (counted, never in the median). inconsistent: std > INCONSISTENT_STD log units (needs
    n >= 2). domain_source is part of the key, so a value measured on an isolated
    construct is never pooled with one measured on the full protein."""
    v = long[long["valid"]]
    by = lambda d: d.groupby(AGG_KEYS, dropna=False)  # noqa: E731  (unassigned id = NA)
    unc = by(v[~v["censored"]])["pX"]
    out = pd.DataFrame({
        "multichain_target": by(v)["multichain_target"].any(),
        "n": unc.size(), "median_pX": unc.median(), "std": unc.std(ddof=1),
        "n_censored": by(v[v["censored"]]).size()})
    out["n"] = out["n"].fillna(0).astype(int)
    out["n_censored"] = out["n_censored"].fillna(0).astype(int)
    out["inconsistent"] = out["std"] > INCONSISTENT_STD
    out = out.reset_index()
    out["multi_domain_uniprot"] = out["domain_source"] != "single_domain"
    out["domain_ambiguous"] = out["domain_source"].str.endswith("_ambiguous")
    return out


# --- pure logic: coverage ----------------------------------------------------------

def coverage(act: pd.DataFrame, manifest: pd.DataFrame, names: dict) -> dict:
    """The numbers the subset decision rests on, counted per ligand GROUP.

    `act`: the activity rows to count (a set of match levels); `manifest` carries
    klifs_kinase_id, uniprot, ligand_group. "Measured" means at least one uncensored
    value (n >= 1); censored-only pairs are counted separately. Kinase-domain-level
    questions (pairs, comparable set) use klifs_kinase_id, so domain_ambiguous rows never
    count there; "kinases measured" counts proteins (UniProt), as BindingDB measures them.
    """
    m = act[act["n"] >= 1]
    kdki = m[m["measure"].isin(BINDING_CONSTANTS)]
    assigned = lambda d: d[d["klifs_kinase_id"].notna()]  # noqa: E731
    key = lambda d: set(zip(d["klifs_kinase_id"].astype(int), d["ligand_group"]))  # noqa: E731
    pairs = manifest[["klifs_kinase_id", "uniprot", "ligand_group"]].drop_duplicates()
    p_all = key(pairs)
    p_kdki, p_any = key(assigned(kdki)), key(assigned(m))
    p_cens = key(assigned(act[act["n_censored"] > 0])) - p_any
    amb_up = set(zip(m.loc[m["domain_ambiguous"], "uniprot"],
                     m.loc[m["domain_ambiguous"], "ligand_group"]))
    p_amb = {(k, g) for k, u, g in pairs.itertuples(index=False) if (u, g) in amb_up}

    structured = manifest.groupby("ligand_group")["klifs_kinase_id"].apply(set)
    per = pd.DataFrame({
        "kinases_any": m.groupby("ligand_group")["uniprot"].nunique(),
        "kinases_kd_ki": kdki.groupby("ligand_group")["uniprot"].nunique()}).fillna(0)
    per = per.astype(int)
    kd_ids = assigned(kdki).groupby("ligand_group")["klifs_kinase_id"].apply(
        lambda s: set(s.astype(int)))
    per["structured_kinases"] = [len(structured.get(g, ())) for g in per.index]
    per["structured_with_kd_ki"] = [len(structured.get(g, set()) & kd_ids.get(g, set()))
                                    for g in per.index]

    # The comparable set: a co-crystal structure of this ligand in the manifest AND a Kd
    # or Ki assigned to the same kinase domain, for >= 2 DIFFERENT PROTEINS (UniProt).
    # Two domains of one protein (JAK2 JH1 / JAK2-b JH2) are not selectivity; such pairs
    # are listed apart as intra_protein_domain_pairs, whether or not the ligand is also
    # comparable across proteins.
    # same_measure_max: the most of those proteins sharing one measure (all Kd, or all Ki).
    label = lambda d: "/".join(sorted(set(d["measure"]))) + ":" + "/".join(  # noqa: E731
        sorted(set(d["match_level"]), key=LEVEL_RANK.get))
    comparable, intra = [], []
    for grp, kins in structured.items():
        have = assigned(kdki)
        have = have[(have["ligand_group"] == grp) & have["klifs_kinase_id"].isin(kins)]
        for u, d in have.groupby("uniprot"):
            if d["klifs_kinase_id"].nunique() >= 2:
                intra.append({"ligand_group": grp, "uniprot": u,
                              "domains": sorted(f"{names.get(int(k), k)}({label(x)})"
                                                for k, x in d.groupby("klifs_kinase_id"))})
        if have["uniprot"].nunique() < 2:
            continue
        comparable.append({
            "ligand_group": grp, "n_proteins": int(have["uniprot"].nunique()),
            "n_domains": int(have["klifs_kinase_id"].nunique()),
            "same_measure_max": int(have.groupby("measure")["uniprot"].nunique().max()),
            "match_levels": sorted(set(have["match_level"]), key=LEVEL_RANK.get),
            "kinases": sorted(f"{names.get(int(k), k)}({label(d)})"
                              for k, d in have.groupby("klifs_kinase_id"))})
    comparable.sort(key=lambda d: (-d["n_proteins"], d["ligand_group"]))

    top = per.sort_values(["kinases_any", "kinases_kd_ki"], ascending=False).head(20)
    return {
        "manifest_pairs": len(p_all),
        "pairs_with_kd_or_ki": len(p_all & p_kdki),
        "pairs_with_ic50_only": len((p_all & p_any) - p_kdki),
        "pairs_censored_only": len(p_all & p_cens),
        "pairs_domain_ambiguous_only": len(p_amb - p_any - p_cens),
        "pairs_without_data": len(p_all - p_any - p_cens - p_amb),
        "ligand_groups_measured": {
            measure: {f">={k}": int((per[col] >= k).sum()) for k in (1, 2, 5, 10, 50)}
            for measure, col in (("any", "kinases_any"), ("kd_or_ki", "kinases_kd_ki"))},
        "comparable_ligands": len(comparable),
        "comparable_ligands_same_measure": sum(d["same_measure_max"] >= 2
                                               for d in comparable),
        "comparable": comparable,
        "intra_protein_domain_pairs": intra,
        "top20_by_kinases_measured": top.reset_index(names="ligand_group").to_dict("records"),
    }


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    klifs = Path("data/kinome/klifs")
    p.add_argument("--manifest", type=Path, default=klifs / "klifs_manifest.csv")
    p.add_argument("--klifs-kinases", type=Path, default=klifs / "klifs_kinases.csv")
    p.add_argument("--klifs-raw", type=Path, default=klifs / "klifs_raw.csv")
    p.add_argument("--out", type=Path, default=Path("data/kinome/bindingdb"))
    p.add_argument("--organisms", nargs="+", default=DEFAULT_ORGANISMS)
    p.add_argument("--refresh-chemcomp", action="store_true")
    p.add_argument("--refresh-uniprot", action="store_true")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rd = lambda f: pd.read_csv(f, keep_default_na=False, dtype=str)  # noqa: E731
    manifest, kin_tbl, raw = rd(a.manifest), rd(a.klifs_kinases), rd(a.klifs_raw)
    manifest["klifs_kinase_id"] = manifest["klifs_kinase_id"].astype(int)
    human = kin_tbl[(kin_tbl["species"] == "Human") & (kin_tbl["uniprot"] != "")].copy()
    human["kinase_ID"] = human["kinase_ID"].astype(int)
    kinases = set(human["uniprot"])
    domains = human.groupby("uniprot")["kinase_ID"].agg(lambda s: sorted(s)).to_dict()
    names = human.set_index("kinase_ID")["name"].to_dict()
    uniprot_names = human.groupby("uniprot")["name"].agg(lambda s: "/".join(sorted(s)))
    with_structure = set(raw.loc[raw["species"] == "Human", "uniprot"])

    release = find_release()
    zip_path, prov = ensure_download(release, a.out / "raw")

    codes = sorted(manifest["ligand_code"].unique())
    chem = fetch_chemcomps(codes, a.out / "chemcomp_cache.json", a.refresh_chemcomp)
    groups = ligand_groups(pd.DataFrame(
        [{"ligand_code": c, "inchikey": (chem["comps"].get(c) or {}).get("inchikey", "")}
         for c in codes]))
    manifest["ligand_group"] = manifest["ligand_code"].map(
        groups.set_index("ligand_code")["ligand_group"])

    multi = sorted(u for u, ids in domains.items() if len(ids) > 1)
    uni = fetch_uniprot(multi, a.out / "uniprot_cache.json", a.refresh_uniprot)
    spans, unmapped = map_klifs_domains(
        human.loc[human["uniprot"].isin(multi), ["kinase_ID", "name", "uniprot", "pocket"]],
        uni["entries"])

    s = scan(tsv_lines(zip_path), kinases, a.organisms, set(codes),
             {k[:14] for k in groups["inchikey"] if k})
    funnel = s["funnel"]

    one = single_kinase(s["records"])
    funnel.append(_counts("exactly one KLIFS kinase chain", one))
    has_value = one[one[list(MEASURES)].apply(lambda c: c.str.strip() != "").any(axis=1)]
    funnel.append(_counts("has Ki, Kd or IC50", has_value))
    matched = match_ligands(has_value, groups)
    level = lambda d, lv: d[d["match_level"] == lv]  # noqa: E731
    funnel.append(_counts("ligand match: full InChIKey", level(matched, "full")))
    funnel.append(_counts("ligand match: bindingdb_no_stereo",
                          level(matched, "bindingdb_no_stereo")))
    funnel.append(_counts("[side branch] skeleton only", level(matched, "skeleton")))

    long = to_long(matched)
    main_long = long[long["match_level"].isin(MAIN_LEVELS)]
    funnel.append(_counts("values (full + no_stereo, one row per measure)", main_long))
    long, dropped = drop_source_duplicates(long)
    main_long = long[long["match_level"].isin(MAIN_LEVELS)]
    funnel.append({**_counts("after source-duplicate removal (same value + PMID/DOI)",
                             main_long), "dropped_all_levels": dropped})
    value_stats = {lv: {"values": int((long["match_level"] == lv).sum()),
                        "censored": int((long["censored"] & (long["match_level"] == lv)).sum()),
                        "invalid": int((~long["valid"] & (long["match_level"] == lv)).sum())}
                   for lv in LEVEL_RANK}

    long = assign_domains(long, manifest, domains, spans)
    act = aggregate(long)
    act["kinase_name"] = [names[int(k)] if pd.notna(k) else uniprot_names.get(u, u)
                          for k, u in zip(act["klifs_kinase_id"], act["uniprot"])]
    # has_structure: at domain level when the row has a domain, else at protein level
    dom_pairs = set(zip(manifest["klifs_kinase_id"], manifest["ligand_group"]))
    prot_pairs = set(zip(manifest["uniprot"], manifest["ligand_group"]))
    act["has_structure"] = [(int(k), g) in dom_pairs if pd.notna(k) else (u, g) in prot_pairs
                            for k, u, g in zip(act["klifs_kinase_id"], act["uniprot"],
                                               act["ligand_group"])]
    act["has_klifs_structure"] = act["uniprot"].isin(with_structure)
    act["primary_eligible"] = primary_eligible(act["domain_source"])
    cols = ["ligand_group", "inchikey", "uniprot", "klifs_kinase_id", "kinase_name",
            "measure", "median_pX", "n", "std", "n_censored", "inconsistent",
            "multichain_target", "multi_domain_uniprot", "domain_source", "domain_ambiguous",
            "primary_eligible",
            "has_structure", "has_klifs_structure", "match_level", "match_note"]
    act = act.sort_values(["ligand_group", "uniprot", "measure", "match_level"])[cols]
    main_act = act[act["match_level"].isin(MAIN_LEVELS)]
    main_act.to_csv(a.out / "activities.csv", index=False)
    act[act["match_level"] == "skeleton"].to_csv(a.out / "activities_skeleton.csv",
                                                  index=False)

    lig_ids = ligand_table(codes, groups, chem["comps"], s["seen_inchikeys"], s["het_pairs"])
    lig_ids.to_csv(a.out / "ligand_ids.csv", index=False)

    amb = main_act[main_act["domain_ambiguous"]]
    main_vals = long[long["match_level"].isin(MAIN_LEVELS) & long["valid"]]
    md = main_vals[main_vals["multi_domain_uniprot"]]
    jh_no_range = md[md["target_name"].str.contains(r"\bJH[12]\b")
                     & md["target_name"].map(lambda t: not parse_ranges(t))]
    checks = {}
    for acc, constructs in (("O60674", ["[808-1132]", "[536-812]"]),
                            ("P29597", ["[556-888]", "[871-1187]"])):
        for c in constructs:
            k = construct_domain(parse_ranges(c), {i: spans[i] for i in domains[acc]
                                                   if i in spans})
            checks[f"{uniprot_names[acc]} {c}"] = names.get(k, None) if k else None
    report = {
        "started_at": started,
        "provenance": {"bindingdb": prov, "download_page": DOWNLOAD_PAGE,
                       "chemcomp_fetched_at": chem.get("fetched_at"),
                       "rcsb_graphql": RCSB_GRAPHQL,
                       "inputs": {"manifest": str(a.manifest),
                                  "klifs_kinases": str(a.klifs_kinases),
                                  "klifs_raw": str(a.klifs_raw)}},
        "settings": {"organisms": a.organisms, "measures": list(MEASURES),
                     "kinase_list": f"KLIFS kinase_information, Human ({len(kinases)} "
                                    f"UniProt, {len(human)} kinase domains)",
                     "main_match_levels": MAIN_LEVELS,
                     "inconsistent_std_log_units": INCONSISTENT_STD},
        "problems": s["problems"],
        "funnel": funnel,
        "value_stats": value_stats,
        "activity_rows": {
            lv: {"rows": int((act["match_level"] == lv).sum()),
                 "inconsistent": int(act.loc[act["match_level"] == lv, "inconsistent"].sum())}
            for lv in LEVEL_RANK},
        "skeleton_notes": act.loc[act["match_level"] == "skeleton", "match_note"]
                             .value_counts().to_dict(),
        "ligand_match_levels": lig_ids["match_level"].value_counts().to_dict(),
        "kinase_rows_by_organism_label": s["kinase_rows_by_organism"],
        "kinase_domains": {
            "uniprot_fetched_at": uni.get("fetched_at"),
            "mapped": {names[k]: {"kinase_ID": k, "uniprot_span": list(v)}
                       for k, v in sorted(spans.items())},
            "unmapped": unmapped,
            "construct_checks": checks,
            "domain_source_values": main_vals["domain_source"].value_counts().to_dict(),
            "domain_source_activity_rows": main_act["domain_source"].value_counts().to_dict(),
            "changes_vs_structure_rule": domain_changes(main_vals),
            "jh_label_without_range": jh_no_range["target_name"].value_counts().to_dict(),
            "multichain_ranged_rows_not_parsed": int(
                ((md["n_chains"] > 1) & md["target_name"].map(lambda t: bool(parse_ranges(t))))
                .sum())},
        "issues": {
            "ligand_groups_with_several_codes": sorted(
                g for g in groups["ligand_group"].unique() if "/" in g),
            "het_disagreements": lig_ids[(lig_ids["het_rows_inchikey_other"] > 0)
                                         | (lig_ids["inchikey_rows_other_het"] > 0)]
                                 [["ligand_code", "het_rows", "het_rows_inchikey_full",
                                   "het_rows_inchikey_no_stereo", "het_rows_inchikey_other",
                                   "inchikey_rows_other_het"]].to_dict("records"),
            "domain_ambiguous": [
                {"ligand_group": g, "uniprot": u, "kinase": uniprot_names.get(u, u),
                 "measures": sorted(set(d["measure"])),
                 "structured_domains": sorted(
                     names[k] for k in set(manifest.loc[(manifest["uniprot"] == u)
                                                        & (manifest["ligand_group"] == g),
                                                        "klifs_kinase_id"]))}
                for (g, u), d in amb.groupby(["ligand_group", "uniprot"])]},
        # match levels (full / full + no_stereo) x rows (all / primary_eligible only)
        "coverage": {
            f"{lv}{'_primary' if prim else ''}": coverage(
                d[d["primary_eligible"]] if prim else d, manifest, names)
            for lv, d in (("full", main_act[main_act["match_level"] == "full"]),
                          ("full_and_no_stereo", main_act))
            for prim in (False, True)},
    }
    (a.out / "bindingdb_funnel.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    for f in funnel:
        print(f"{f['step']:56} {f['rows']:9} rows  {f['ligands']:8} ligands  "
              f"{f['kinases']:4} kinases", file=sys.stderr)


if __name__ == "__main__":
    main()
