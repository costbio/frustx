"""Davis et al. 2011 (Nat Biotechnol 29:1046, KINOMEscan) Kd matrix, mapped onto the
KLIFS manifest: the PRIMARY source for selectivity (one assay, one measure, Kd).
BindingDB (scripts/bindingdb_activity.py) stays the source for coverage and sensitivity.

    .venv/bin/python scripts/davis_activity.py

Source: the paper's own supplementary files, located from the article page
(https://www.nature.com/articles/nbt.1990), NOT processed redistributions (TDC/DeepDTA
turn "no binding" into a fixed pKd and drop the kinase variants):
    Supplementary Table 1   kinase list (accession, Entrez symbol, DiscoverX name, Mutant)
    Supplementary Table 3   the 72 compounds, with alternative (generic) names
    Supplementary Table 4   Kd (nM), 442 kinase constructs x 72 compounds

Output, under data/kinome/davis/:
    raw/                    the .xls files + provenance.json (URL, MD5, date)
    kinase_map.csv          442 DiscoverX names -> variant class, UniProt, KLIFS kinase id
    compound_map.csv        72 compounds -> PubChem CID, InChIKey, manifest ligand group
    activities.csv          one row per (kinase construct, compound); BindingDB-compatible
                            columns plus source = "davis2011"
    davis_funnel.json       provenance, mapping problems, coverage, comparable set,
                            overlap with BindingDB, Davis-vs-BindingDB consistency
    idmapping_cache.json, pubchem_cache.json, uniprot_cache.json, pubmed_cache.json
                            network caches

Facts about the files this code depends on, read from them rather than assumed
(checked 2026-10-05):
  - Table 4 is one sheet: Accession Number, Entrez Gene Symbol, Kinase (DiscoverX name),
    then one column per compound. Every Kd is stored as TEXT ("43", "0.016" .. "9900").
  - Blank cells: "combinations that were tested, but for which binding was weak
    (Kd > 10 uM), or not detected in a 10 uM primary screen" (legend of Supplementary
    Table 4 on the article page). So blank = censored, Kd > 10,000 nM, pKd < 5 -- a bound,
    never a number.
  - Table 4's header and Table 3's rows list the compounds in the same order; one name
    differs in spelling (Table 4 "INCB18424", Table 3 "INCB018424"). Columns are paired
    by position and every spelling difference is reported.
  - 23 of the 2011 Entrez symbols are outdated (FRAP1 = MTOR, ZAK, PCTK1, ...), so kinases
    are mapped by accession (RefSeq protein, a few GenBank CDS ids) through UniProt's
    ID mapping, versions stripped (old RefSeq versions do not map).
  - PubChem resolves aliases of one drug to different structures (CHIR-258 -> CID
    135431668, TKI-258 / dovitinib -> CID 135398510, different skeletons), so a compound
    is accepted only when every alias resolves to the same single CID.

Structure follows scripts/bindingdb_activity.py, whose helpers are reused (ligand groups,
match levels, domain spans, coverage, the BindingDB pipeline for the comparison).
Network code apart from pure functions; tests/test_davis_activity.py runs offline.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bindingdb_activity as bdb  # noqa: E402

ARTICLE = "https://www.nature.com/articles/nbt.1990"
DAVIS_PMID, DAVIS_DOI = "22037378", "10.1038/nbt.1990"
SUPPLEMENTS = {"kinases": "Supplementary Table 1", "compounds": "Supplementary Table 3",
               "kd": "Supplementary Table 4"}
UNIPROT_IDMAP = "https://rest.uniprot.org/idmapping"
PUBCHEM_NAME = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/"
                "InChIKey,IsomericSMILES/JSON")
# Legend of Supplementary Table 4: blank = Kd > 10 uM or not detected in a 10 uM screen.
CENSOR_NM = 10_000.0
SOURCE = "davis2011"
PUBMED_SUMMARY = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                  "?db=pubmed&retmode=json&id={}")
# Consistency with BindingDB: a reference whose values equal Davis's for at least this
# share of the pairs both cover is treated as another copy of the Davis data (2026-10-05:
# PMID 18183025 1053/1229, no reference 984/1157, PMID 19654408 542/570) and left out.
DEFAULT_COPY_THRESHOLD = 0.8
# Constructs used as wild type although their name states a phosphorylation state, by
# explicit decision: Davis has no unqualified ABL1, and the non-phosphorylated construct
# is the closest to it. ABL1-phosphorylated stays phospho_or_other.
WT_OVERRIDES = {"ABL1-nonphosphorylated": "abl1_nonphos_as_wt"}
# Compounds whose aliases PubChem resolves to different CIDs that are salt forms of one
# parent, resolved by explicit decision to the free base: CHIR-258 -> CID 135431668 is
# dovitinib lactate, TKI-258 / Dovitinib -> 135398510 the free base (manifest group 38O);
# R406 -> 11984591 is the besylate, Tamatinib -> 11213558 the free base (group 585).
# PTK-787 (vatalanib) and CI-1033 (canertinib) have the same salt pattern but no manifest
# group, so they are left unresolved.
SALT_RESOLUTIONS = {"CHIR-258/TKI-258": 135398510, "R406": 11213558}
MAIN_LEVELS = bdb.MAIN_LEVELS


# --- network -----------------------------------------------------------------------

def _get(url: str, data: bytes | None = None, timeout: int = 120):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def ensure_supplements(raw_dir: Path) -> dict:
    """Find Supplementary Tables 1, 3 and 4 by their captions on the article page,
    download any missing, and record URL, MD5 and date in raw/provenance.json."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    prov_path = raw_dir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    if all(k in prov and (raw_dir / prov[k]["file"]).exists() for k in SUPPLEMENTS):
        return prov
    page = _get(ARTICLE).decode("utf-8", errors="replace")
    for key, caption in SUPPLEMENTS.items():
        m = re.search(r'href="([^"]+\.xls)"[^>]*>\s*' + re.escape(caption) + r"\b", page)
        if not m:
            raise RuntimeError(f"'{caption}' .xls link not found on {ARTICLE}")
        url, name = m.group(1), m.group(1).rsplit("/", 1)[-1]
        dest = raw_dir / name
        if dest.exists():
            when = datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc)
            source = "file mtime"
        else:
            dest.write_bytes(_get(url))
            when, source = datetime.now(timezone.utc), "download"
        prov[key] = {"caption": caption, "file": name, "url": url, "md5": _md5(dest),
                     "downloaded_at": when.isoformat(timespec="seconds"),
                     "downloaded_at_source": source}
    prov["article"] = ARTICLE
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    return prov


def _cache(path: Path, refresh: bool) -> dict:
    return json.loads(path.read_text()) if path.exists() and not refresh else {}


def uniprot_idmap(ids: list[str], from_db: str, cache_path: Path, refresh=False) -> dict:
    """{id: [{accession, reviewed, organism, gene}]} for UniProt ID mapping from `from_db`
    (e.g. RefSeq_Protein) to UniProtKB. [] = no hit. Cached per (from_db, id)."""
    cache = _cache(cache_path, refresh)
    done = cache.setdefault(from_db, {})
    need = sorted(set(ids) - set(done))
    if need:
        body = urllib.parse.urlencode({"from": from_db, "to": "UniProtKB",
                                       "ids": ",".join(need)}).encode()
        job = json.loads(_get(f"{UNIPROT_IDMAP}/run", data=body))["jobId"]
        for _ in range(120):
            st = json.loads(_get(f"{UNIPROT_IDMAP}/status/{job}"))
            if st.get("jobStatus") not in ("NEW", "RUNNING"):
                break
            time.sleep(1)
        results = []
        try:
            fields = "accession,reviewed,organism_name,gene_primary"
            res = json.loads(_get(f"{UNIPROT_IDMAP}/uniprotkb/results/{job}"
                                  f"?fields={fields}&format=json&size=500"))
            results = res.get("results", [])
        except urllib.error.HTTPError as e:
            if e.code != 404:         # 404: the job mapped nothing at all
                raise
        for i in need:
            done[i] = []
        for r in results:
            t = r["to"]
            done[r["from"]].append({
                "accession": t["primaryAccession"],
                "reviewed": "Swiss-Prot" in t.get("entryType", ""),
                "organism": t.get("organism", {}).get("scientificName", ""),
                "gene": ((t.get("genes") or [{}])[0].get("geneName") or {}).get("value", "")})
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return {i: done[i] for i in ids}


def pubchem_names(names: list[str], cache_path: Path, refresh=False) -> dict:
    """{name: [{cid, inchikey, smiles}]} from PubChem's name lookup; [] = not found.
    Every CID PubChem returns is kept: choosing among several is not done here."""
    cache = _cache(cache_path, refresh)
    for n in sorted(set(names) - set(cache)):
        try:
            d = json.loads(_get(PUBCHEM_NAME.format(urllib.parse.quote(n, safe=""))))
            cache[n] = [{"cid": p["CID"], "inchikey": p.get("InChIKey", ""),
                         "smiles": p.get("IsomericSMILES") or p.get("SMILES", "")}
                        for p in d["PropertyTable"]["Properties"]]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            cache[n] = []
        time.sleep(0.25)          # PubChem asks for <= 5 requests per second
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return {n: cache[n] for n in names}


def pubmed_summaries(pmids: list[str], cache_path: Path, refresh=False) -> dict:
    """{pmid: {title, first_author, n_authors, year, journal}} from PubMed esummary."""
    cache = _cache(cache_path, refresh)
    need = sorted(set(pmids) - set(cache))
    for i in range(0, len(need), 100):
        res = json.loads(_get(PUBMED_SUMMARY.format(",".join(need[i:i + 100]))))["result"]
        for pmid in need[i:i + 100]:
            r = res.get(pmid, {})
            authors = [x["name"] for x in r.get("authors", [])]
            cache[pmid] = {"title": r.get("title", ""),
                           "first_author": authors[0] if authors else "",
                           "n_authors": len(authors),
                           "year": (r.get("pubdate", "") or "")[:4],
                           "journal": r.get("source", "")}
        time.sleep(0.4)            # NCBI: <= 3 requests per second without an API key
    if need:
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return {p: cache[p] for p in pmids}


def read_sheet(path: Path) -> list[list[str]]:
    """All cells of the first sheet as stripped strings. xlrd (the only reader of the
    2011 .xls format) is imported here so that the offline tests do not need it."""
    import xlrd
    s = xlrd.open_workbook(str(path)).sheet_by_index(0)
    return [[str(s.cell_value(r, c)).strip() for c in range(s.ncols)] for r in range(s.nrows)]


# --- pure logic: DiscoverX kinase names --------------------------------------------

_NAME = re.compile(r"(?P<base>[^()]+?)(?:\((?P<inner>[^)]*)\))?"
                   r"(?P<suffix>-(?:nonphosphorylated|phosphorylated|cyclin[A-Z0-9]+))?")
_DOMAIN = re.compile(r"^(?P<label>JH(?P<jh>[12])domain-\w+|Kin\.Dom\.(?P<kd>[12])"
                     r"(?:-[NC]-terminal)?)$")
_ORGANISM = re.compile(r"^[A-Z]\.[a-z]+$")                       # P.falciparum
_MUTATION = re.compile(r"^([A-Z]\d+[A-Z*]|[A-Z]\d+-[A-Z]\d+del|ITD|[A-Z]?ins|Sins)$")
# UniProt's accession format: a few Table 1/4 "Accession Number" cells are UniProt
# accessions themselves (WEE2 P0C1S8, RIPK5 Q6XUX3.1, SgK110 P0C264), not RefSeq/GenBank.
UNIPROT_ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")


def parse_discoverx(name: str) -> dict:
    """A DiscoverX construct name, e.g. 'ABL1(E255K)-phosphorylated',
    'JAK1(JH2domain-pseudokinase)', 'GCN2(Kin.Dom.2,S808G)', 'CDK4-cyclinD1',
    'PFPK5(P.falciparum)', into its parts.

    variant_class:
      mutant            any point mutation, deletion, insertion or ITD
      phospho_or_other  no mutation, but a stated phosphorylation state
      wild_type         neither (a domain construct or a cyclin complex is still wild type),
                        or listed in WT_OVERRIDES -- then variant_note says so
    domain_ordinal: 1 = the N-terminal of two kinase domains, 2 = the C-terminal one.
    JAK-family JH2 (pseudokinase) is N-terminal of JH1 (catalytic), so JH2 -> 1, JH1 -> 2.
    Tokens matching none of the patterns land in `unparsed` (reported, never dropped).
    """
    m = _NAME.fullmatch(name.strip())
    base, inner, suffix = m.group("base"), m.group("inner") or "", m.group("suffix") or ""
    out = {"base": base, "mutations": [], "domain_label": "", "domain_ordinal": None,
           "organism": "", "phospho_state": "", "complex_partner": "", "unparsed": [],
           "variant_note": ""}
    for tok in (t.strip() for t in inner.split(",") if t.strip()):
        if d := _DOMAIN.match(tok):
            out["domain_label"] = d.group("label")
            out["domain_ordinal"] = {"1": 2, "2": 1}[d.group("jh")] if d.group("jh") \
                else int(d.group("kd"))
        elif _ORGANISM.match(tok):
            out["organism"] = tok
        elif _MUTATION.match(tok):
            out["mutations"].append(tok)
        else:
            out["unparsed"].append(tok)
    if suffix.endswith("phosphorylated"):
        out["phospho_state"] = suffix[1:]
    elif suffix:
        out["complex_partner"] = suffix[1:]
    out["variant_class"] = ("mutant" if out["mutations"] else
                            "phospho_or_other" if out["phospho_state"] else "wild_type")
    if name.strip() in WT_OVERRIDES:
        out["variant_class"], out["variant_note"] = "wild_type", WT_OVERRIDES[name.strip()]
    return out


def choose_uniprot(hits: list[dict]) -> tuple[str, str]:
    """(accession, status) from UniProt ID-mapping hits: exactly one reviewed human entry
    is accepted; none -> 'no_reviewed_human'; several -> 'several_reviewed_human'."""
    good = sorted({h["accession"] for h in hits
                   if h["reviewed"] and h["organism"] == "Homo sapiens"})
    if len(good) == 1:
        return good[0], "ok"
    return "", "several_reviewed_human" if good else "no_reviewed_human"


def map_kinases(kin: pd.DataFrame, idmap: dict, klifs_human: pd.DataFrame,
                spans: dict[int, tuple[int, int]]) -> pd.DataFrame:
    """Each DiscoverX construct -> UniProt -> KLIFS kinase id, with a `status`:
      ok                 mapped
      non_human          the name states another organism
      no_uniprot         no single reviewed human UniProt from accession or symbol
      not_in_klifs       a human protein KLIFS does not list (lipid/atypical kinases)
      domain_unresolved  the protein has two KLIFS kinase domains and the construct
                         names none, or its domains could not both be placed

    `kin`: accession, gene, name. `idmap`: accession (unversioned) -> UniProt hits.
    UniProt comes from the accession -- used as is when it already is a UniProt
    accession, otherwise through ID mapping; when that gives nothing, from the Entrez
    symbol equal to a KLIFS human gene name (method recorded). Domain constructs go to the KLIFS
    kinase whose UniProt span is the N- (ordinal 1) or C-terminal (2) of the two.
    """
    by_gene = (klifs_human.groupby("gene_name")["uniprot"].agg(lambda s: sorted(set(s)))
               .to_dict())
    ids_of = klifs_human.groupby("uniprot")["kinase_ID"].agg(lambda s: sorted(s)).to_dict()
    name_of = klifs_human.set_index("kinase_ID")["name"].to_dict()
    rows = []
    for acc, gene, name in kin[["accession", "gene", "name"]].itertuples(index=False):
        p = parse_discoverx(name)
        r = {"davis_kinase": name, "accession": acc, "entrez_symbol": gene,
             "variant_class": p["variant_class"], "variant_note": p["variant_note"],
             "mutations": ",".join(p["mutations"]),
             "phospho_state": p["phospho_state"], "domain_label": p["domain_label"],
             "complex_partner": p["complex_partner"], "organism": p["organism"] or "",
             "unparsed": ",".join(p["unparsed"]), "uniprot": "", "uniprot_method": "",
             "uniprot_note": "",
             "klifs_kinase_id": None, "kinase_name": "", "domain_source": "",
             "status": ""}
        if p["organism"]:
            rows.append({**r, "status": "non_human"})
            continue
        bare = acc.split(".")[0]
        if UNIPROT_ACC.match(bare):
            uni, st, method = bare, "ok", "accession_is_uniprot"
        else:
            uni, st = choose_uniprot(idmap.get(bare, []))
            method = "accession" if uni else ""
        if not uni and len(by_gene.get(gene, [])) == 1:
            uni, method = by_gene[gene][0], "entrez_symbol"
        if not uni:
            rows.append({**r, "status": "no_uniprot", "uniprot_note": st})
            continue
        r.update(uniprot=uni, uniprot_method=method)
        ids = ids_of.get(uni, [])
        if not ids:
            rows.append({**r, "status": "not_in_klifs"})
            continue
        if len(ids) == 1:
            k, src = ids[0], "single_domain"
        elif p["domain_ordinal"] and all(i in spans for i in ids):
            by_position = sorted(ids, key=lambda i: spans[i][0])    # N- then C-terminal
            k, src = by_position[p["domain_ordinal"] - 1], "construct"
        else:
            rows.append({**r, "status": "domain_unresolved"})
            continue
        rows.append({**r, "klifs_kinase_id": k, "kinase_name": name_of[k],
                     "domain_source": src, "status": "ok"})
    out = pd.DataFrame(rows)
    out["klifs_kinase_id"] = out["klifs_kinase_id"].astype("Int64")
    return out


# --- pure logic: compounds ---------------------------------------------------------

def compound_aliases(name: str, alternative: str) -> tuple[list[str], bool]:
    """Every name a compound is listed under ('CHIR-258/TKI-258' + 'Dovitinib'), and
    whether it is marked as a derivative ('BIBF-1120 (derivative)'): a derivative is
    NOT the named compound, so it is never looked up as that name."""
    derivative = "derivative" in (name + alternative).lower()
    names = [a.strip() for part in (name, alternative) for a in part.split("/")]
    return [n for n in dict.fromkeys(names) if n], derivative


def resolve_compound(aliases: list[str], derivative: bool, lookup: dict) -> dict:
    """PubChem CID/InChIKey for a compound, accepted only if unambiguous:
      ok                one CID, and every alias PubChem knows gives that same CID
      multiple_cids     some alias returns several CIDs (no automatic choice)
      aliases_disagree  aliases return different CIDs (e.g. CHIR-258 vs dovitinib)
      not_found         no alias is known to PubChem
      derivative        listed as a derivative of the named compound: not looked up
    `lookup`: alias -> [{cid, inchikey, smiles}] (pubchem_names)."""
    if derivative:
        return {"status": "derivative", "cid": None, "inchikey": "", "smiles": "",
                "detail": ""}
    found = {a: lookup.get(a, []) for a in aliases}
    detail = "; ".join(f"{a}->{','.join(str(h['cid']) for h in hs) or 'none'}"
                       for a, hs in found.items())
    if any(len(hs) > 1 for hs in found.values()):
        status = "multiple_cids"
    else:
        cids = {hs[0]["cid"] for hs in found.values() if hs}
        status = "ok" if len(cids) == 1 else "aliases_disagree" if cids else "not_found"
    if status != "ok":
        return {"status": status, "cid": None, "inchikey": "", "smiles": "", "detail": detail}
    hit = next(hs[0] for hs in found.values() if hs)
    return {"status": "ok", "cid": hit["cid"], "inchikey": hit["inchikey"],
            "smiles": hit["smiles"], "detail": detail}


def apply_salt_resolution(compound: str, resolved: dict, aliases: list[str],
                          lookup: dict) -> dict:
    """Use the free-base CID listed in SALT_RESOLUTIONS for a compound PubChem left
    ambiguous (status 'salt_resolved'). The CID must be one of the candidates its aliases
    returned -- a CID from anywhere else is an error, not a silent choice. Compounds not
    listed, or already resolved, pass through unchanged."""
    cid = SALT_RESOLUTIONS.get(compound)
    if cid is None or resolved["status"] not in ("aliases_disagree", "multiple_cids"):
        return resolved
    hit = next((h for a in aliases for h in lookup.get(a, []) if h["cid"] == cid), None)
    if hit is None:
        raise ValueError(f"{compound}: CID {cid} is not among PubChem's candidates")
    return {"status": "salt_resolved", "cid": cid, "inchikey": hit["inchikey"],
            "smiles": hit["smiles"], "detail": resolved["detail"]}


# --- pure logic: values ------------------------------------------------------------

def kd_value(raw: str) -> dict:
    """One Table 4 cell. A number is Kd in nM -> pKd = 9 - log10(Kd). A blank cell is
    censored: Kd > 10,000 nM, kept only as the bound pKd < 5 (censor_bound_pX), never as
    a value. Anything else is invalid (reported)."""
    s = str(raw).strip()
    if not s:
        return {"kd_nM": math.nan, "pX": math.nan, "censored": True,
                "censor_bound_pX": bdb.to_px(CENSOR_NM), "valid": True}
    try:
        v = float(s)
    except ValueError:
        return {"kd_nM": math.nan, "pX": math.nan, "censored": False,
                "censor_bound_pX": math.nan, "valid": False}
    return {"kd_nM": v, "pX": bdb.to_px(v), "censored": False,
            "censor_bound_pX": math.nan, "valid": v > 0}


def matrix_long(matrix: list[list[str]], compounds: list[str]) -> pd.DataFrame:
    """Table 4 (header row + one row per kinase construct) -> one row per (construct,
    compound). Compound columns are named by `compounds` (Table 3, by position)."""
    rows = []
    for r in matrix[1:]:
        for comp, raw in zip(compounds, r[3:]):
            rows.append({"davis_kinase": r[2], "davis_compound": comp, "kd_raw": raw,
                         **kd_value(raw)})
    return pd.DataFrame(rows)


BDB_COLUMNS = ["ligand_group", "inchikey", "uniprot", "klifs_kinase_id", "kinase_name",
               "measure", "median_pX", "n", "std", "n_censored", "inconsistent",
               "multichain_target", "multi_domain_uniprot", "domain_source",
               "domain_ambiguous", "primary_eligible", "has_structure",
               "has_klifs_structure", "match_level", "match_note"]
DAVIS_COLUMNS = ["source", "davis_kinase", "davis_compound", "variant_class",
                 "variant_note", "kd_raw", "kd_nM", "censor_bound_pX"]


def build_activities(long: pd.DataFrame, kinase_map: pd.DataFrame,
                     compound_map: pd.DataFrame, manifest: pd.DataFrame,
                     with_structure: set[str]) -> pd.DataFrame:
    """Davis values in BindingDB's activities.csv columns (+ DAVIS_COLUMNS). One Davis
    measurement per row, so n is 1 (or 0 when censored), std is empty and nothing is
    pooled -- not CDK4's two cyclin complexes, not a mutant with its wild type.
    Only constructs mapped to a KLIFS kinase are kept (kinase_map has the rest)."""
    km = kinase_map[kinase_map["status"] == "ok"]
    cm = compound_map[["davis_compound", "inchikey", "ligand_group", "match_level",
                       "match_note"]]
    a = long.merge(km, on="davis_kinase").merge(cm, on="davis_compound", how="left")
    a[["inchikey", "ligand_group", "match_level", "match_note"]] = \
        a[["inchikey", "ligand_group", "match_level", "match_note"]].fillna("")
    a.loc[a["match_level"] == "", "match_level"] = "none"
    dom_pairs = set(zip(manifest["klifs_kinase_id"], manifest["ligand_group"]))
    out = pd.DataFrame({
        "ligand_group": a["ligand_group"], "inchikey": a["inchikey"],
        "uniprot": a["uniprot"], "klifs_kinase_id": a["klifs_kinase_id"],
        "kinase_name": a["kinase_name"], "measure": "Kd", "median_pX": a["pX"],
        "n": (~a["censored"] & a["valid"]).astype(int), "std": math.nan,
        "n_censored": a["censored"].astype(int), "inconsistent": False,
        "multichain_target": a["complex_partner"] != "",
        "multi_domain_uniprot": a["domain_source"] == "construct",
        "domain_source": a["domain_source"], "domain_ambiguous": False,
        "primary_eligible": bdb.primary_eligible(a["domain_source"]),
        "has_structure": [(int(k), g) in dom_pairs for k, g in
                          zip(a["klifs_kinase_id"], a["ligand_group"])],
        "has_klifs_structure": a["uniprot"].isin(with_structure),
        "match_level": a["match_level"], "match_note": a["match_note"],
        "source": SOURCE, "davis_kinase": a["davis_kinase"],
        "davis_compound": a["davis_compound"], "variant_class": a["variant_class"],
        "variant_note": a["variant_note"],
        "kd_raw": a["kd_raw"], "kd_nM": a["kd_nM"], "censor_bound_pX": a["censor_bound_pX"]})
    return out[BDB_COLUMNS + DAVIS_COLUMNS]


def primary_subset(act: pd.DataFrame) -> pd.DataFrame:
    """The rows selectivity statistics use: wild type, a KLIFS domain known from the
    measurement, ligand matched to a manifest group (full or no_stereo)."""
    return act[(act["variant_class"] == "wild_type") & act["primary_eligible"]
               & act["match_level"].isin(MAIN_LEVELS)]


def reference_of(df: pd.DataFrame) -> pd.Series:
    """'PMID', else 'doi:<DOI>', else 'none' -- the reference a BindingDB row cites."""
    return df["pmid"].where(df["pmid"] != "", "doi:" + df["doi"]).replace("doi:", "none")


def reference_table(davis: pd.DataFrame, other: pd.DataFrame,
                    tol: float = 0.005) -> pd.DataFrame:
    """Per reference cited by `other` (pmid, doi): of its values that have a Davis value
    for the same (kinase domain, ligand group), how many equal it within `tol` log units.
    A second copy of the Davis data under another citation shows up here as a share near
    1 -- the Davis PMID/DOI filter cannot see it. Index: reference; columns identical,
    compared, share; sorted by identical, descending."""
    key = ["klifs_kinase_id", "ligand_group"]
    d = davis.groupby(key)["pX"].median().rename("davis").reset_index()
    j = other.merge(d, on=key)
    j["ref"] = reference_of(j)
    j["same"] = (j["pX"] - j["davis"]).abs() <= tol
    t = j.groupby("ref")["same"].agg(identical="sum", compared="size")
    t["identical"] = t["identical"].astype(int)
    t["share"] = (t["identical"] / t["compared"]).round(3)
    return t.sort_values(["identical", "compared"], ascending=False)


def copy_references(table: pd.DataFrame, threshold: float) -> set[str]:
    """References whose identical share is >= threshold, 'none' (no PMID/DOI) included."""
    return set(table.index[table["share"] >= threshold])


def skeleton_candidates(aliases: list[str], lookup: dict, groups: pd.DataFrame) -> list[dict]:
    """For a compound PubChem could not resolve unambiguously: every candidate CID any
    alias returned, and the manifest ligand groups sharing its InChIKey skeleton (first
    14 characters), each with the relation of the full keys (identical or skeleton only).
    Nothing is chosen here."""
    sk = groups[groups["inchikey"] != ""].drop_duplicates("ligand_group")
    out = {}
    for a in aliases:
        for h in lookup.get(a, []):
            c = out.setdefault(h["cid"], {"cid": h["cid"], "inchikey": h["inchikey"],
                                          "smiles": h["smiles"], "aliases": []})
            c["aliases"].append(a)
    for c in out.values():
        hit = sk[sk["inchikey"].str[:14] == c["inchikey"][:14]]
        c["manifest_groups"] = [{"ligand_group": g, "group_inchikey": k,
                                 "relation": "identical" if k == c["inchikey"] else "skeleton"}
                                for g, k in zip(hit["ligand_group"], hit["inchikey"])]
    return sorted(out.values(), key=lambda c: c["cid"])


def consistency(davis: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Davis vs another source over shared (klifs_kinase_id, ligand_group) pairs, both
    sides uncensored Kd. `davis`/`other`: klifs_kinase_id, ligand_group, pX (one row per
    value; each side is reduced to its median per pair). delta = other - Davis."""
    key = ["klifs_kinase_id", "ligand_group"]
    d = davis.groupby(key)["pX"].median().rename("davis")
    o = other.groupby(key)["pX"].median().rename("other")
    j = pd.concat([d, o], axis=1, join="inner").dropna()
    if j.empty:
        return {"pairs": 0}
    delta = j["other"] - j["davis"]
    return {"pairs": len(j), "ligand_groups": int(j.index.get_level_values(1).nunique()),
            "median_delta": round(float(delta.median()), 3),
            "median_abs_delta": round(float(delta.abs().median()), 3),
            "frac_abs_delta_gt_1": round(float((delta.abs() > 1).mean()), 3),
            # Spearman as Pearson on ranks: pandas' method="spearman" needs scipy
            "spearman_rho": round(float(j["davis"].rank().corr(j["other"].rank())), 3)}


# --- BindingDB side of the comparison ----------------------------------------------

def bindingdb_kd(zip_path: Path, kinases: set[str], groups: pd.DataFrame,
                 manifest: pd.DataFrame, domains: dict, spans: dict) -> tuple[pd.DataFrame, dict]:
    """BindingDB Kd values per measurement, through the same pipeline as
    bindingdb_activity.py (match levels incl. the HET tie-break, source-duplicate
    removal, domain assignment), restricted to what the comparison uses: Kd, valid,
    uncensored, primary_eligible, main match levels. Each row is flagged `from_davis`
    when its PMID or DOI is the Davis paper's -- BindingDB's own copy of this data."""
    s = bdb.scan(bdb.tsv_lines(zip_path), kinases, bdb.DEFAULT_ORGANISMS,
                 set(groups["ligand_code"]), {k[:14] for k in groups["inchikey"] if k})
    one = bdb.single_kinase(s["records"])
    one = one[one["Kd"].str.strip() != ""]
    long = bdb.to_long(bdb.match_ligands(one, groups))
    long = long[(long["measure"] == "Kd") & long["match_level"].isin(MAIN_LEVELS)]
    long, _ = bdb.drop_source_duplicates(long)
    long = bdb.assign_domains(long, manifest, domains, spans)
    long["from_davis"] = (long["pmid"] == DAVIS_PMID) | (long["doi"].str.lower() == DAVIS_DOI)
    stats = {"kd_rows_matched": len(long), "from_davis_rows": int(long["from_davis"].sum()),
             "from_davis_by_source": long.loc[long["from_davis"], "source"]
                                         .value_counts().to_dict()}
    keep = long[long["valid"] & ~long["censored"] & bdb.primary_eligible(long["domain_source"])
                & long["klifs_kinase_id"].notna()]
    return keep[["klifs_kinase_id", "ligand_group", "pX", "from_davis", "target_name",
                 "pmid", "doi", "source"]], stats


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    klifs, bdir = Path("data/kinome/klifs"), Path("data/kinome/bindingdb")
    p.add_argument("--manifest", type=Path, default=klifs / "klifs_manifest.csv")
    p.add_argument("--klifs-kinases", type=Path, default=klifs / "klifs_kinases.csv")
    p.add_argument("--klifs-raw", type=Path, default=klifs / "klifs_raw.csv")
    p.add_argument("--bindingdb-dir", type=Path, default=bdir)
    p.add_argument("--out", type=Path, default=Path("data/kinome/davis"))
    p.add_argument("--copy-threshold", type=float, default=DEFAULT_COPY_THRESHOLD,
                   help="leave out BindingDB references whose values equal Davis's for at "
                        "least this share of shared pairs (default %(default)s)")
    p.add_argument("--refresh", action="store_true", help="ignore all network caches")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # KLIFS side, as in bindingdb_activity.py
    rd = lambda f: pd.read_csv(f, keep_default_na=False, dtype=str)  # noqa: E731
    manifest, kin_tbl, raw = rd(a.manifest), rd(a.klifs_kinases), rd(a.klifs_raw)
    manifest["klifs_kinase_id"] = manifest["klifs_kinase_id"].astype(int)
    human = kin_tbl[(kin_tbl["species"] == "Human") & (kin_tbl["uniprot"] != "")].copy()
    human["kinase_ID"] = human["kinase_ID"].astype(int)
    domains = human.groupby("uniprot")["kinase_ID"].agg(lambda s: sorted(s)).to_dict()
    names = human.set_index("kinase_ID")["name"].to_dict()
    with_structure = set(raw.loc[raw["species"] == "Human", "uniprot"])
    multi = sorted(u for u, ids in domains.items() if len(ids) > 1)
    uni = bdb.fetch_uniprot(multi, a.out / "uniprot_cache.json", a.refresh)
    spans, unmapped_domains = bdb.map_klifs_domains(
        human.loc[human["uniprot"].isin(multi), ["kinase_ID", "name", "uniprot", "pocket"]],
        uni["entries"])
    codes = sorted(manifest["ligand_code"].unique())
    chem = bdb.fetch_chemcomps(codes, a.bindingdb_dir / "chemcomp_cache.json")
    groups = bdb.ligand_groups(pd.DataFrame(
        [{"ligand_code": c, "inchikey": (chem["comps"].get(c) or {}).get("inchikey", "")}
         for c in codes]))
    manifest["ligand_group"] = manifest["ligand_code"].map(
        groups.set_index("ligand_code")["ligand_group"])

    # Davis files
    prov = ensure_supplements(a.out / "raw")
    t1 = read_sheet(a.out / "raw" / prov["kinases"]["file"])
    t3 = read_sheet(a.out / "raw" / prov["compounds"]["file"])
    t4 = read_sheet(a.out / "raw" / prov["kd"]["file"])
    compounds = [r[0] for r in t3[1:]]
    header = t4[0][3:]
    if len(header) != len(compounds):
        raise RuntimeError(f"Table 4 has {len(header)} compounds, Table 3 {len(compounds)}")
    spelling = [{"table4": h, "table3": c} for h, c in zip(header, compounds) if h != c]
    t1_mutant = {r[2]: r[3] for r in t1[1:]}

    # kinases
    kin = pd.DataFrame([r[:3] for r in t4[1:]], columns=["accession", "gene", "name"])
    refseq = sorted({x.split(".")[0] for x in kin["accession"] if x.startswith("NP_")})
    other = sorted({x.split(".")[0] for x in kin["accession"] if not x.startswith("NP_")
                    and not UNIPROT_ACC.match(x.split(".")[0])})
    idmap = {**uniprot_idmap(refseq, "RefSeq_Protein", a.out / "idmapping_cache.json", a.refresh),
             **uniprot_idmap(other, "EMBL-GenBank-DDBJ_CDS", a.out / "idmapping_cache.json",
                             a.refresh)}
    kmap = map_kinases(kin, idmap, human, spans)
    kmap["table1_mutant"] = kmap["davis_kinase"].map(t1_mutant)
    kmap.to_csv(a.out / "kinase_map.csv", index=False)

    # compounds
    alt = {r[0]: r[1] for r in t3[1:]}
    aliases = {c: compound_aliases(c, alt.get(c, "")) for c in compounds}
    lookup = pubchem_names(sorted({n for al, der in aliases.values() if not der for n in al}),
                           a.out / "pubchem_cache.json", a.refresh)
    crow = []
    for c in compounds:
        al, der = aliases[c]
        crow.append({"davis_compound": c, "aliases": " | ".join(al),
                     **apply_salt_resolution(c, resolve_compound(al, der, lookup), al, lookup)})
    cmap = pd.DataFrame(crow).rename(columns={"cid": "pubchem_cid", "status": "pubchem_status"})
    ok = cmap[cmap["inchikey"] != ""]
    matched = bdb.match_ligands(ok[["davis_compound", "inchikey"]], groups)
    best = (matched.assign(r=matched["match_level"].map(bdb.LEVEL_RANK))
            .sort_values(["davis_compound", "r"]).drop_duplicates("davis_compound"))
    cmap = cmap.merge(best[["davis_compound", "ligand_group", "match_level", "match_note"]],
                      on="davis_compound", how="left").fillna(
        {"ligand_group": "", "match_level": "none", "match_note": ""})
    cmap.loc[cmap["pubchem_status"] == "salt_resolved", "match_note"] = "salt_resolved"
    cmap["in_manifest_main"] = cmap["match_level"].isin(MAIN_LEVELS)
    cmap.to_csv(a.out / "compound_map.csv", index=False)
    # unresolved compounds: does any candidate structure belong to a manifest group?
    ambiguous = {c: skeleton_candidates(aliases[c][0], lookup, groups)
                 for c in cmap.loc[cmap["pubchem_status"].isin(
                     ["aliases_disagree", "multiple_cids"]), "davis_compound"]}

    # values
    long = matrix_long(t4, compounds)
    act = build_activities(long, kmap, cmap, manifest, with_structure)
    act.to_csv(a.out / "activities.csv", index=False)
    prim = primary_subset(act)

    # comparable sets: Davis vs BindingDB (primary)
    cov = {"full": bdb.coverage(prim[prim["match_level"] == "full"], manifest, names),
           "full_and_no_stereo": bdb.coverage(prim, manifest, names)}
    bfun = json.loads((a.bindingdb_dir / "bindingdb_funnel.json").read_text())
    overlap = {}
    for lv in ("full", "full_and_no_stereo"):
        dset = {d["ligand_group"] for d in cov[lv]["comparable"]}
        bset = {d["ligand_group"] for d in bfun["coverage"][f"{lv}_primary"]["comparable"]}
        overlap[lv] = {"davis": len(dset), "bindingdb_primary": len(bset),
                       "both": sorted(dset & bset), "davis_only": sorted(dset - bset),
                       "bindingdb_only": sorted(bset - dset)}

    # consistency with BindingDB, its copy of Davis excluded
    zips = sorted((a.bindingdb_dir / "raw").glob("BindingDB_All_*_tsv.zip"))
    bkd, bstats = bindingdb_kd(zips[-1], set(human["uniprot"]), groups, manifest, domains,
                               spans)
    dv = prim[prim["n"] == 1].rename(columns={"median_pX": "pX"})
    dv = dv[dv["klifs_kinase_id"].notna()]
    others = bkd[~bkd["from_davis"]]
    refs = reference_table(dv, others)
    copies = copy_references(refs, a.copy_threshold)
    meta = pubmed_summaries(sorted(r for r in copies if r.isdigit()),
                            a.out / "pubmed_cache.json", a.refresh)
    refs["excluded_as_copy"] = refs.index.isin(copies)
    kept = others[~reference_of(others).isin(copies)]
    cons = {"bindingdb_file": zips[-1].name, **bstats,
            "copy_threshold": a.copy_threshold,
            "davis_copy_only (sanity: should agree)": consistency(dv, bkd[bkd["from_davis"]]),
            "before_copy_exclusion (Davis PMID/DOI removed only)": consistency(dv, others),
            "after_copy_exclusion": consistency(dv, kept),
            "excluded_references": [
                {"reference": r, **refs.loc[r, ["identical", "compared", "share"]].to_dict(),
                 **meta.get(r, {})} for r in refs.index if r in copies],
            "values_removed_as_copies": int(len(others) - len(kept)),
            "reference_table_before": refs.reset_index().to_dict("records"),
            "reference_table_after": refs[~refs["excluded_as_copy"]].reset_index()
                                         .to_dict("records")}

    report = {
        "started_at": started,
        "provenance": {"supplements": prov, "pmid": DAVIS_PMID, "doi": DAVIS_DOI,
                       "censoring": f"blank = Kd > {CENSOR_NM:g} nM (Supplementary Table 4 "
                                    "legend), kept as pKd < 5",
                       "uniprot_domains_fetched_at": uni.get("fetched_at")},
        "tables": {"kinase_constructs": len(kin), "compounds": len(compounds),
                   "cells": len(long), "censored_cells": int(long["censored"].sum()),
                   "invalid_cells": int((~long["valid"]).sum()),
                   "compound_spelling_table4_vs_table3": spelling},
        "kinases": {
            "status": kmap["status"].value_counts().to_dict(),
            "variant_class (mapped)": kmap.loc[kmap["status"] == "ok", "variant_class"]
                                          .value_counts().to_dict(),
            "variant_overrides": kmap.loc[kmap["variant_note"] != "",
                                          ["davis_kinase", "variant_note"]].values.tolist(),
            "uniprot_method": kmap["uniprot_method"].value_counts().to_dict(),
            "domain_constructs": kmap.loc[kmap["domain_source"] == "construct",
                                          ["davis_kinase", "kinase_name"]].values.tolist(),
            "unmapped": kmap.loc[kmap["status"] != "ok",
                                 ["davis_kinase", "status", "uniprot", "uniprot_note"]]
                            .to_dict("records"),
            "unparsed_tokens": kmap.loc[kmap["unparsed"] != "",
                                        ["davis_kinase", "unparsed"]].to_dict("records"),
            "table1_mutant_disagrees": kmap.loc[
                (kmap["table1_mutant"] == "YES") != (kmap["variant_class"] == "mutant"),
                ["davis_kinase", "table1_mutant", "variant_class"]].to_dict("records"),
            "klifs_domain_unmapped": unmapped_domains},
        "compounds": {"ambiguous_candidates": ambiguous,
                      "pubchem_status": cmap["pubchem_status"].value_counts().to_dict(),
                      "match_level": cmap["match_level"].value_counts().to_dict(),
                      "not_ok": cmap.loc[cmap["pubchem_status"] != "ok",
                                         ["davis_compound", "pubchem_status", "detail"]]
                                    .to_dict("records")},
        "primary_rows": len(prim),
        "coverage": cov,
        "comparable_overlap_with_bindingdb_primary": overlap,
        "consistency_with_bindingdb": cons,
    }
    (a.out / "davis_funnel.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("tables", "primary_rows")}, default=str),
          file=sys.stderr)


if __name__ == "__main__":
    main()
