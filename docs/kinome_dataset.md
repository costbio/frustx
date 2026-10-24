# Kinome dataset — end-to-end summary

This file describes the data chain built for the kinase experiments: what each piece is,
where and how it was fetched, what each step produced, and what exactly is on disk now.
Numbers are taken from the `*_funnel.json` files under `data/kinome/` and from
`data/kinome/analysis/*.json`; each of those records its own generation date and the
thresholds used.

The method rationale lives in `docs/method.md`, sections "The kinome dataset" and
"KLIFS-first complex set". This file is the narrative; that file is the spec.

**Goal.** Test whether a frustration index computed on co-crystal structures of one
inhibitor bound to different kinases follows the measured binding affinity across those
kinases — i.e. whether frustration tracks **selectivity** (**Design A**). Nothing in this
file computes frustration except the paralog pilot (section 6); everything else builds the
input for that test and measures whether the test is statistically possible at all.

**The chain.** Two generations, the second being the one in use:

```
Generation 1 (Davis-gated, sections 1-6)
  KLIFS (structures) -> BindingDB (coverage + activities) -> Davis 2011 (primary Kd)
                                      |
                        Design A feasibility -> Extended set -> Paralog pilot (negative)

Generation 2 (KLIFS-first, section 7)  <- CURRENT
  KLIFS (structures + ligand->ChEMBL link) -> ChEMBL (wild-type Kd / IC50)
                                      |
                     data/kinome/DRUG_KINASE_COMPLEXES.csv
```

---

## 1. KLIFS structure manifest

**Script:** `scripts/klifs_manifest.py` · **Output:** `data/kinome/klifs/`

### Source and method

KLIFS REST API v2 (`https://klifs.net/api_v2`), called directly with stdlib `urllib`.
`opencadd` is not installed and its dependency chain (bravado/biopandas) is unreliable on
Python 3.14; since opencadd is only a client for the same API, no dependency was needed.
**No structure files were downloaded** — this step produces metadata only.

### Four facts read before any filter was written

Each was fetched from the live API and printed first; the filter came after:

1. "No ligand" is not `""` or `"-"` but the **integer 0**.
2. NMR entries carry `resolution = "0"`, so `resolution <= 2.5` alone would **pass every
   NMR model**. These rows carry the model number in the `alt` field ("1".."20").
   `normalise()` turns a zero resolution into NaN so the filter rejects it.
3. `resolution` and `quality_score` arrive as strings; ligand names are HTML-escaped
   (`&apos;`).
4. KLIFS still lists PDB IDs that RCSB has made obsolete.

Also: the orthosteric ligand field contains **no ions or buffers at all**; its only
non-inhibitors are nucleotides and their analogues. The exclusion list was therefore built
from what actually occurs (ATP, ADP, ANP, ACP, AGS, AMP, ADN, 3AM, AP2, M33, 112, NBS, 3GU
+ MG, MN).

### RCSB validation, and why the order matters

**All** ~4,400 rows that survive the KLIFS filters were validated against RCSB GraphQL
(cache: `rcsb_cache.json`; a re-run makes no requests).

> **Order matters:** the RCSB check and the drops that depend on it run **BEFORE**
> deduplication. Otherwise a structure RCSB rejects could win its (kinase, ligand) slot
> and take the whole pair down while a valid runner-up existed.

`rcsb_status` values and what they do:

| status | count | outcome |
|---|---|---|
| `ok` | 4,396 | kept |
| `obsolete_remapped` | 12 | **dropped** (below) |
| `ligand_missing` | 10 | dropped |
| `non_xray` | 2 | dropped |
| `obsolete_no_replacement` | 0 | dropped |

### Funnel (fetched 2026-10-04, KLIFS API 0.3.1)

| step | structures | PDB | kinases | UniProt | ligands |
|---|---|---|---|---|---|
| raw (all species) | 14,068 | 6,780 | 359 | 351 | 4,180 |
| `species == Human` | 13,325 | 6,375 | 318 | 311 | 4,014 |
| orthosteric ligand present | 11,286 | 5,541 | 289 | 286 | 4,014 |
| exclusion list | 10,051 | 4,970 | 265 | 262 | 4,001 |
| resolution ≤ 2.5 Å | 7,404 | 3,586 | 222 | 219 | 2,986 |
| quality score ≥ 6 | 7,180 | 3,478 | 218 | 215 | 2,894 |
| altloc `A` or blank | 4,420 | 3,475 | 218 | 215 | 2,886 |
| RCSB: obsolete, no replacement | 4,420 | 3,464 | 218 | 215 | 2,886 |
| RCSB: ligand not in entry | 4,410 | 3,460 | 218 | 215 | 2,882 |
| RCSB: not X-ray | 4,408 | 3,458 | 217 | 214 | 2,882 |
| RCSB: obsolete → remap | 4,396 | 3,458 | 217 | 214 | 2,882 |
| **deduplication** | **3,196** | **3,190** | **217** | **214** | **2,882** |

Note: the "obsolete → remap" step drops no rows but lowers the PDB count — old IDs fold
into new IDs already in the table. CDK7 leaves entirely at these steps (its only
structures, 7B5O and 7B5Q, are cryo-EM).

### Two design decisions

**The deduplication key is the KLIFS kinase (domain) ID, not UniProt.** JAK1/2/3, TYK2
and RSK/MSK carry two kinase domains under one UniProt (JH1 catalytic, JH2 pseudokinase;
N- and C-terminal for RSK). Keyed on UniProt, the two domains' structures merged into one
pair. Switching to the domain key recovered 3 structures: JAK2/2HB (4P7E), JAK2-b/DQX
(5UT4), JAK2-b/SKE (5USZ). 222 manifest rows carry `multi_domain_uniprot`.

**`obsolete_remapped` rows do not enter the manifest.** KLIFS's quality / resolution /
missing-residue values describe the *old* deposit, not the file a download would fetch.
Example: 6MX8, which replaced 5J7H — KLIFS scored the old deposit at quality 8 with 0
missing residues; 6MX8 itself is 6.8 with 3. 12 rows dropped, **no (kinase, ligand) pair
was lost**: KLIFS already listed the replacement under its own row, and that row won
deduplication.

### Flagged, not dropped

Conditions the download step must know about, but which do not justify discarding data:

| flag | count | meaning |
|---|---|---|
| `ligand_chain_differs` | 20 | ligand on a 3-character chain (`AAA`, `BBB`) — does not fit PDB format, needs mmCIF |
| `multi_copy_ligand` | 75 | several copies of the ligand on the same protein chain — the one in the pocket must be chosen |
| `altloc_ligand_conflict` | 8 | different altlocs of one PDB + chain carry **different ligands** (6HOP, CDK2 1H00/1H01/1H07/1H08, 1RDQ, 4KIO, 5LWN, 8AOA) |
| `multi_domain_uniprot` | 222 | the UniProt has more than one KLIFS kinase domain |

### Output files

- `klifs_manifest.csv` — 3,196 rows, 22 columns. Each row is one (kinase domain, ligand)
  complex: `pdb_klifs` (as KLIFS gives it, lowercase), `pdb` (current RCSB ID), chain,
  altloc, `ligand_code` (= HETATM name), `allosteric_ligand_code`, resolution, quality,
  missing residues/atoms, DFG/αC state, `rcsb_status` and the flags above.
- `klifs_raw.csv` — the unfiltered raw table (reproducibility). **Generation 2 starts from
  this file**, not from the manifest.
- `klifs_kinases.csv` — `kinase_information`, all species (1,127 rows). The human part is
  555 kinases / 542 UniProt. This is the target list for BindingDB.
- `klifs_funnel.json` — per-step counts, thresholds, fetch date, API version,
  `remap_collisions`.
- `rcsb_cache.json` — RCSB responses (3.8 MB).

---

## 2. BindingDB activities

**Script:** `scripts/bindingdb_activity.py` · **Output:** `data/kinome/bindingdb/`

### Source

No URL was guessed: the download page is read, its link followed to the `SDFdownload.jsp`
intermediate page, and from there to the real file. The file is
**`BindingDB_All_202610_tsv.zip`** (567 MB), MD5 `ab7ca3dc152d395b773ff987d8e09f64`,
**verified** against the published `.md5`. The zip is never extracted to disk; its 9.0 GB
TSV is streamed (single pass, ~2.5 min). If the copy on disk matches the MD5 it is not
downloaded again; `.provenance.json` keeps the first download date.

### Structure of the file (read, not assumed)

- 3,243,660 rows, **640 fields**: 40 fixed columns + 50 blocks of 12 columns, one per
  target chain.
- Rows are padded to 640. So a multichain target is not a *longer row* but **more filled
  blocks**. A row whose field count disagrees is written to `problems`.
- **Unreadable rows: 0.** 226 rows have no InChIKey (counted; they cannot match).
- Affinities are strings: `"12.5"`, `">10000"`, `"<0.03"`, sometimes `"> 20000"`.
- The organism column spells the same organism three ways: `Homo sapiens` (2.48 M rows),
  `Human` (0.21 M), `Homo sapiens (Human)`. All three accepted (`--organisms`).
- The SwissProt primary-id cell can hold several space-separated accessions.

### Funnel

| step | rows | ligands | kinases |
|---|---|---|---|
| all rows | 3,243,660 | 1,354,629 | 488 |
| human organism (3 labels) | 2,695,170 | 1,155,171 | 474 |
| a chain is a KLIFS human kinase | 840,917 | 323,974 | 474 |
| exactly **one** kinase chain | 831,780 | 321,016 | 474 |
| has Ki, Kd or IC50 | 806,171 | 310,936 | 474 |
| **ligand match: full InChIKey** | **32,083** | **1,297** | 443 |
| **ligand match: `bindingdb_no_stereo`** | **21,082** | **621** | 432 |
| *(side branch)* skeleton only | 1,071 | 68 | 381 |
| values (full + no_stereo, per measure) | 53,207 | 1,918 | 453 |
| source duplicates removed | 50,747 | 1,918 | 453 |

~6,800 rows name **more than one** KLIFS kinase among their targets (e.g. a "PI3K alpha"
record that also lists ABL1); the value cannot be attributed to one kinase, so they drop
at a funnel step of their own.

### Four problems solved along the way

**1. BindingDB often drops the stereo layer.** The InChIKeys of ruxolitinib,
staurosporine, axitinib, lestaurtinib end in `-UHFFFAOYSA-`; RCSB's keys carry stereo.
293 ligands matched nothing because of it. A third match level was defined:

- `full` — all 27 characters identical.
- **`bindingdb_no_stereo`** — first 14 characters identical, BindingDB key's 2nd block is
  `UHFFFAOYSA`, the ligand's key carries stereo, same protonation, **and** the skeleton
  fits exactly one manifest ligand group.
- `skeleton` — everything else, `match_note` gives the reason
  (`different_stereo_or_isotope`, `protonation_only`, `no_stereo_and_protonation`,
  `no_stereo_several_ligands`). Kept in a separate file, never mixed into the main table.

Cross-checked against BindingDB's own HET-ID column, disagreements fell **293 → 7**; the
missing stereo layer explained almost all of the mismatch.

**2. HET tie-break.** When a stereo-less record fits two enantiomers (RXT/RG4), BindingDB's
HET column says which one: `match_note = "het_tiebreak"` (455 activity rows, 18 ligand
groups). RXT went from 0 to 388 activity rows. If HET is empty or not one of the candidate
codes, it decides nothing and the row stays in skeleton.

**3. Domain assignment — per measurement, from the construct range.** Residue ranges in
target names (`JAK2 [808-1132]`, `TYK2 [556-888]`, `(aa658-end)`, several ranges for
deletion mutants, `'YVMA'` insertions, en-dashes) are parsed. "Protein kinase" domain
boundaries come from UniProt REST (cached), and each KLIFS kinase is mapped to a UniProt
domain by locating KLIFS's 85-residue pocket string in the UniProt sequence: **25 of 26
kinases** (GCN2-b did not map — its pocket does not fall inside UniProt's single "Protein
kinase" domain; not guessed, reported).

Checks come out right: JAK2 [808-1132] → JAK2 (JH1), [536-812] → JAK2-b (JH2),
TYK2 [556-888] → TYK2-b (JH2), [871-1187] → TYK2 (JH1).

Rule order: **construct > structure**. If a range exists and covers ≥ 80% of one domain,
that domain (`domain_source = "construct"`); with no range (full protein), the domain
where the ligand's crystal is (`"structure"`); with neither, the row stays at UniProt
level (`"uniprot_level"`). A construct covering both domains is `domain_ambiguous`.

Putting the construct rule before the structure rule moved 2 measurements to the other
domain (M4G, M57: crystals on JAK2-b, but the values were measured on the JH1 construct)
and rescued 14 from `domain_ambiguous`.

**4. Source duplicates.** Rows with the same (uniprot, ligand, measure, value) and the same
PMID were deduplicated before aggregation: 2,546 rows (+1 via DOI).

### Quality columns

`inconsistent` (spread > 1 log unit), `n` / `n_censored` (censored values **never** enter
the median), `multichain_target`, `multi_domain_uniprot`, `domain_source`,
`domain_ambiguous`, and **`primary_eligible`**: True when the domain is known from the
measurement itself (`single_domain`, `construct`), False when inferred from a crystal or
ambiguous (`structure`, `uniprot_level`, `*_ambiguous`).

### Output and coverage

- `activities.csv` — 25,641 aggregated rows (16,375 `full` + 9,266
  `bindingdb_no_stereo`). Key: (uniprot, kinase domain, domain_source, InChIKey, measure,
  ligand group, level). `median_pX`, `n`, `std`, `n_censored` over uncensored values.
- `activities_skeleton.csv` — 601 rows, the skeleton matches kept apart.
- `ligand_ids.csv` — 2,882 ligand codes → group, InChIKey, SMILES, match level, HET
  cross-check counts.
- `bindingdb_funnel.json`, `chemcomp_cache.json`, `uniprot_cache.json`.

Coverage (over 3,194 manifest pairs; "comparable" = the same ligand has a structure on
**≥ 2 different UniProt** *and* an assigned Kd/Ki on each):

| version | pairs with Kd/Ki | IC50 only | no data | comparable ligands |
|---|---|---|---|---|
| full | 413 | 844 | 1,908 | 29 |
| full, primary only | 382 | 797 | 1,989 | 27 |
| full + no_stereo | 640 | 1,234 | 1,280 | 49 |
| **full + no_stereo, primary only** | **588** | 1,153 | 1,417 | **45** |

Two domains of one protein (JAK2/JAK2-b) do not count as selectivity; they are listed
separately as `intra_protein_domain_pairs` — one case: SKE.

---

## 3. Davis 2011 — primary selectivity source (generation 1)

**Script:** `scripts/davis_activity.py` · **Output:** `data/kinome/davis/`

### Why a separate source

BindingDB is compiled from hundreds of papers and assays; values for the same pair come
from different labs. Davis et al. (*Nat Biotechnol* 29:1046, 2011, PMID 22037378, DOI
10.1038/nbt.1990) is **one assay, one measure**: KINOMEscan, 72 inhibitors × 442 kinase
constructs, Kd. In generation 1 it was **primary** for selectivity statistics; BindingDB
stayed for coverage and sensitivity analysis. Generation 2 no longer gates on Davis
(section 7).

### Source

The supplementary links on the article's Nature page were located **by caption** and
downloaded (no URL pattern guessed): Table 1 (kinase list), Table 3 (72 compounds),
**Table 4 (Kd matrix)**. URL + MD5 + date in `raw/provenance.json`.

> **Processed redistributions were not used.** TDC/DeepDTA convert "no binding" into a
> fixed pKd and lose the kinase variants (mutant, phospho, domain construct). Both are
> exactly the information this work needs.

The 2011 `.xls` format required `xlrd` — installed in the venv and added to the `dev`
extra of `pyproject.toml` (openpyxl does not read this format). It is imported inside the
reading function, so the offline tests do not need it.

### The matrix

442 × 72 = **31,824 cells**, all stored as **text** (0.016–9,900 nM). Unreadable cells:
0. **22,400 cells are blank.**

The legend of Table 4 on the article page: *"Blank fields indicate combinations that were
tested, but for which binding was weak (Kd > 10 uM), or not detected in a 10 uM primary
screen."* So a blank cell is `censored=True` and is kept **only as a bound**
(`censor_bound_pX` = pKd < 5). It is never used as a number and never enters a median.

### Kinase mapping (DiscoverX name → UniProt → KLIFS)

**23** of the 2011 Entrez symbols are stale (FRAP1 = MTOR, ZAK, PCTK1, CDC2L1…), so the
mapping goes **through the accession** (RefSeq protein → UniProt ID mapping, version
number stripped; cache: `idmapping_cache.json`).

**437/442 mapped:** 427 by accession, 3 whose accession is already a UniProt (e.g. WEE2
P0C1S8), 8 by gene symbol (fallback, when the accession gave nothing).

| status | count | which |
|---|---|---|
| `ok` | 437 | — |
| `non_human` | 3 | PFCDPK1, PFPK5 (*P. falciparum*), PKNB (*M. tuberculosis*) |
| `no_uniprot` | 1 | CDC2L1 |
| `domain_unresolved` | 1 | GCN2(Kin.Dom.2,S808G) |

Variant classes: **wild_type 382**, `mutant` 54, `phospho_or_other` 1. 17 domain
constructs were assigned directly to their KLIFS domain
(`JAK1(JH2domain-pseudokinase)` → JAK1-b; JH2 is N-terminal, so it is position 1, JH1 → 2).

**Decision on record:** `ABL1-nonphosphorylated` counts as wild_type
(`variant_note = "abl1_nonphos_as_wt"`) — Davis has no unqualified ABL1 row and the
non-phosphorylated construct is the closest. `ABL1-phosphorylated` stays
`phospho_or_other`. This took the comparable set from 17 to 19 ligands (ABL1 added to
1N1, DB8, STI, VX6; newly in: AXI, NIL).

**Naming trap:** Davis's "RSK1" (RPS6KA1) is KLIFS's **RSK3**, and Davis's "RSK3"
(RPS6KA2) is KLIFS's **RSK1**. Protein identity comes from the accession, so the data is
right; only the labels differ.

### Compound mapping (name → PubChem → InChIKey → manifest group)

Every name and synonym was resolved through PubChem (cache: `pubchem_cache.json`). **No
ambiguity was resolved automatically.** 5 compounds remained; their SMILES showed why:
the aliases resolved to **salt forms** of the same parent.

| compound | situation | decision |
|---|---|---|
| CHIR-258/TKI-258 | aliases give different CIDs | **CID 135398510** (dovitinib, free base) → group **38O** |
| R406 | aliases give different CIDs | **CID 11213558** (tamatinib, free base) → group **585** |
| PTK-787 | 151194 free base / 151193 succinate | no manifest group → **out** |
| CI-1033 | 156414 / 156413 (2HCl) / 67292089 (HCl) | no manifest group → **out** |
| BIBF-1120 (derivative) | a derivative, not the named compound | **out** |

The first two are flagged `match_note = "salt_resolved"`; they are written out in the
`SALT_RESOLUTIONS` dict, and the chosen CID must be one of the candidates PubChem returned
(otherwise an error is raised; nothing is chosen silently).

**37 compounds** matched a manifest ligand group (36 `full` + 1 `bindingdb_no_stereo`).
2 only at skeleton level (flavopiridol → F9Z, PHA-665752 → PFY).

Also, Table 4's header spells `INCB18424` while Table 3 spells `INCB018424`; columns were
matched by position and the spelling difference reported.

### Agreement with BindingDB — and the circularity trap

BindingDB **contains Davis**: 14,497 Kd rows imported via ChEMBL under PMID 22037378.
Even after removing that PMID, **74%** of the remaining values were identical to Davis —
copies under other citations.

The script therefore excludes **references whose values are ≥ 80% identical to Davis**
(`--copy-threshold`, default 0.8) and reports **38 references** with title / author / year
from PubMed. The main ones:

| reference | identical / compared | who |
|---|---|---|
| 18183025 | 1,054 / 1,289 | Karaman MW 2008, *Nat Biotechnol* |
| (no reference) | 985 / 1,219 | — |
| 19654408 | 542 / 572 | Zarrinkar PP 2009, *Blood* |

Result:

| comparison | pairs | median Δ | median \|Δ\| | \|Δ\| > 1 | Spearman ρ |
|---|---|---|---|---|---|
| Davis copy (sanity — must agree) | 3,384 | 0.00 | 0.00 | 0.1% | **0.998** |
| Davis PMID removed only | 1,536 | 0.00 | 0.00 | 2.3% | 0.971 |
| **copy references removed as well** | **433** | **−0.16** | **0.30** | **10.2%** | **0.883** |

The first row confirms the kinase and ligand mappings are right. The third is the real
independent agreement: ρ 0.88, 10% of values differ by more than 1 log unit.

The same applies to ChEMBL, which is where BindingDB got the copy: the Kd values in
generation 2 come from ChEMBL and include the Davis and Karaman screens (section 7).

### Output

- `activities.csv` — 31,464 rows. Columns **compatible** with BindingDB's `activities.csv`
  (so they can be concatenated) + `source="davis2011"`, `davis_kinase`, `davis_compound`,
  `variant_class`, `variant_note`, `kd_raw`, `kd_nM`, `censor_bound_pX`. Each row is
  **one** measurement: `n` = 1 (or 0 if censored), `std` empty, nothing pooled — not
  CDK4's two cyclin complexes, not mutant with wild type.
- `kinase_map.csv` (442 constructs), `compound_map.csv` (72 compounds),
  `davis_funnel.json`, 4 caches (`idmapping`, `pubchem`, `pubmed`, `uniprot`).

**Davis comparable set: 20 ligands** — STU 26 proteins, 1N1 10, DB8 9, B49 6, MI1 5,
STI 5, B96 4, VX6 4, 8X7/BAX/R78/VGH 3, and 9 ligands with 2.

---

## 4. Design A feasibility (generation 1)

**Script:** `scripts/design_a_feasibility.py` · **Output:** `data/kinome/analysis/design_a_*`

Question: can this data carry the test? No frustration computed, no model fitted.

### Scope

The Davis comparable set — per ligand, the kinases with a **manifest structure** of that
ligand and a **wild-type, uncensored** Kd: **20 ligands, 96 complexes, 68 proteins**.

Censored cells were **not** pulled to the 10 µM bound but excluded — otherwise every
range would be inflated by an amount the assay never measured.

### Assumptions (all written in the JSON)

- Model: `frustration_i = ratio · pKd_i + N(0, 1)`, independent per complex.
- **Effect size `ratio = slope / noise_sd`**, both in log units of the affinity scale.
  Only the ratio matters → **the frustration index's own unit never enters.**
- Test: two-sided **permutation test** on Spearman ρ, α = 0.05. **Exact enumeration** for
  n ≤ 7, 5,000 samples above. The observed pKd vector is used on the x axis, so the real
  range and ties are kept.
- Ignored: affinity measurement error (Davis Kd taken as exact), censored cells, one
  structure representing the complex, a shared frustration offset within a ligand.

### Range

Median range **1.40 log** units (0.29 – 4.62). 1N1 is widest (4.62), VGH narrowest (0.29).

### Power — a per-ligand test is mostly impossible

| ligand | n | min attainable p | power (ratio 2.0) | power (ratio 1.0) |
|---|---|---|---|---|
| STU | 26 | 0.0002 | **1.00** | **0.99** |
| 1N1 | 10 | 0.0002 | **1.00** | **0.83** |
| DB8 | 9 | 0.0002 | **1.00** | **0.81** |
| B49 | 6 | 0.0028 | 0.58 | 0.28 |
| MI1 | 5 | 0.0167 | 0.36 | 0.12 |
| STI | 5 | 0.0167 | 0.15 | 0.06 |
| B96, VX6 | 4 | 0.0833 | 0 | 0 |
| R78, BAX, VGH | 3 | 0.3333 | 0 | 0 |
| 11 ligands | 2 | **1.0000** | 0 | 0 |

**11 ligands have n = 2** — the smallest attainable p is 1.0, so no result can be
significant however strong the relationship. Even at n = 4 the smallest p is
2/24 = 0.083 > 0.05. Exact enumeration gives this as a fact, not an estimate.

### Power — pooled model

The fixed effect of `frustration ~ pKd + (1|ligand)`: Pearson on ranks taken and centred
**within** each ligand. Only within-ligand signal counts; a ligand's overall frustration
level cannot contribute. Null: independent permutation inside each ligand.

| ratio | 0.1 | 0.2 | 0.33 | 0.5 | **0.6** | 0.8 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| pooled (96 complexes) | 0.07 | 0.17 | 0.38 | 0.70 | **0.85** | 0.97 | 1.00 | 1.00 |
| ranking only (78) | 0.08 | 0.17 | 0.38 | 0.69 | **0.83** | 0.97 | 1.00 | 1.00 |
| negative control (18) | 0.03 | 0.02 | 0.04 | 0.04 | 0.09 | — | 0.09 | **0.26** |

**Smallest detectable effect ≈ ratio 0.6** (0.5 → 0.70; 0.6 → 0.85). The sentence for the
thesis: *the pooled design detects the relationship if the frustration index moves by at
least ~0.6 of its own residual noise per log unit of affinity; anything weaker cannot be
seen with this data.* Monte-Carlo error at 4,000 simulations is ~±0.6 points, so pinning
the ratio finer than 0.05 is meaningless.

### Roles

- `ranking` (13 ligands / 78 complexes) — range ≥ 1 log.
- `negative_control` (**7 ligands / 18 complexes**: VX6, BAX, VGH, GUI, NIL, LY4, 88Z) —
  range < 1 log. **Not removed**: their kinases are effectively equipotent, so they test
  the **opposite hypothesis** (a sound index should *not* find an ordering there).

> **Caution:** the negative-control set is itself underpowered (0.26 at ratio 2.0). A null
> result there does not distinguish "no ordering, as predicted" from "not enough data to
> see one". That is as far as the thesis can say.

### Paralog pairs

**110 pairs** (19 same KLIFS family, 91 same group only). The closest relatives — the
hardest and most interesting cases for a frustration index, since the two pockets differ
least.

| ligand | pair | ΔpKd |
|---|---|---|
| STI | ABL1 / ABL2 | 0.96 |
| VX6 | AurA / AurC | 0.21 |
| MI1 | JakA family, 6 pairs | 0.44 – 1.48 (JAK3 9.80 > JAK2 9.24 > JAK1 8.80 > TYK2 8.32) |
| B49 | HPK1 / PAK6 | 2.18 |
| DB8 | LOK / MST3 | 1.73 |
| 1N1 | BMX / BTK | **0.00** (exact tie) |

### Removed: the BindingDB cross-check

Originally the script checked whether the Davis ranges held in BindingDB. **63%** of the
96 shared complexes had a BindingDB value identical to Davis — BindingDB's Davis copy via
ChEMBL. Since the comparison mostly compared Davis with itself, the section was removed
entirely; the reasoning is in the script's docstring.

---

## 5. Extended set (generation 1)

**Script:** `scripts/extended_set.py` · **Output:** `data/kinome/analysis/extended_set*`

Question: how much do the BindingDB ligands that Davis never measured add?

### Rules

- A ligand takes **all** its values from **one** source; Davis if Davis measured it.
  Mixing the two sources within a ligand would put the assay/aggregation difference inside
  the very comparison Design A makes.
- On the BindingDB side **IC50 is not used at all**; per ligand Kd *or* Ki (whichever
  covers more proteins, Kd on a tie). 37 ligands had both; 5 chose Ki (3NG, 7KC, N5Q, QS0,
  YAM) — each gave up a Kd that existed for one protein.
- `inconsistent` values dropped (45 rows).
- That no censored value enters a median is verified with an **assert** (a check, not a
  claim).
- `single_measurement` = a BindingDB value aggregated from **one report**. Not dropped but
  flagged; every power figure is computed **twice** (with / without). Davis values are not
  flagged: each is one KINOMEscan Kd by definition and Davis is the primary source, so the
  flag measures "how much the extension leans on thin data".

Checks came back empty: no ligand fed from two sources, no ligand mixing two measure
types, no censored value in a median, no complex with more than one structure.

### Structures, not data, are the bottleneck

On the BindingDB side 3,648 rows → **517 after the structure filter** → **47** after the
≥ 2 proteins condition. Binding data is plentiful; co-crystal structures are what is
missing.

### The set

| | ligands | complexes | proteins |
|---|---|---|---|
| Davis | 20 | 96 | 68 |
| BindingDB (added) | 22 | 47 | — |
| **total** | **42** | **143** | **86** |

**20** of the 22 added ligands have **n = 2** (one has 3, one 4) — none can be tested
alone. The extension is wide but shallow.

### Roles — and `exploratory`

| role | ligands | complexes |
|---|---|---|
| `ranking` | 18 | **88** |
| `exploratory` | 3 | 7 |
| `negative_control` | 21 | 48 |

The `exploratory` rule (general; no ligand names hard-coded): a ligand that qualifies for
ranking with a range ≥ 1 log, but whose **highest and lowest values each rest on a single
report**, is taken out of ranking. Reason: that range could be measurement noise. The rule
caught exactly **8X7, 537, YAM**:

| ligand | complexes | situation |
|---|---|---|
| 537 | JNK3 7.66 / PDK1 5.90 | both ends single-report |
| YAM | FAK 8.03 / PYK2 6.82 | both ends single-report, and Ki was chosen |
| 8X7 | PLK1 8.52 / Wee1 8.19 / MYT1 6.49 | both ends single-report, the middle n = 2 |

**They stay in the set** — frustx will still run those complexes; they just do not count
as evidence for the primary ranking test. The `needs_decision` flag keeps on record whom
the rule caught.

### Power (single_measurement included)

| scope | n | 0.5 | **0.6** | 1.0 | 2.0 | 80% threshold | min attainable p |
|---|---|---|---|---|---|---|---|
| davis_only | 96 | 0.70 | **0.85** | 1.00 | 1.00 | 0.6 | 0.0002 |
| extended_all | 143 | 0.70 | **0.85** | 1.00 | 1.00 | 0.6 | 0.0002 |
| **extended_ranking** | **88** | 0.70 | **0.84** | 1.00 | 1.00 | **0.6** | 0.0002 |
| davis_ranking | 78 | 0.69 | 0.83 | 1.00 | 1.00 | 0.6 | 0.0004 |
| extended_negative_control | 48 | 0.08 | 0.09 | 0.18 | **0.53** | never | 0.0004 |
| bindingdb_ranking | 10 | 0 | 0 | 0 | 0 | **never** | **0.0636** |
| extended_exploratory | 7 | 0 | 0 | 0 | 0 | never | 0.0874 |

Excluding `single_measurement` barely moves the large scopes (extended_ranking 80
complexes, 0.846 at 0.6), but `bindingdb_ranking` **collapses from 17 to 2 complexes**:
almost all of BindingDB's ranking contribution rests on single reports.

### Three findings

1. **The extension does not lower the smallest detectable effect.** Davis alone and the
   extended set both cross 80% at ratio 0.6 (0.845 vs 0.849). Two-point groups carry
   almost no within-ligand information.
2. **The gain is in the negative control:** 18 → 48 complexes, power at ratio 2.0
   **0.26 → 0.53**. Still below 80%, but about twice the information.
3. **BindingDB's own ranking contribution is provably insufficient:** 5 ligands ×
   2 complexes → 2/32 = 0.0625 smallest attainable p (measured 0.0636) **> 0.05**. No
   effect size can make it significant alone.

### Should source be a covariate? No — it is not identifiable

| | n | median pKd | mean | std | min–max | within-ligand median range |
|---|---|---|---|---|---|---|
| Davis | 96 | 7.89 | 7.84 | 1.14 | 5.62 – 10.54 | **1.40** |
| BindingDB | 47 | 7.23 | 7.16 | 1.16 | 4.19 – 9.66 | **0.62** |

There is a ~0.65 log level offset with nearly identical spread. But **no ligand mixes the
two sources, so source is nested inside ligand** — perfectly collinear with the ligand
term of the random-intercept model. The level offset is absorbed by the ligand intercept
and **cannot** bias the within-ligand slope, so adding `source` as a fixed effect is
redundant. What matters is the difference in **within**-ligand behaviour: BindingDB
ligands have half the range of Davis ones and thinner provenance. That calls for a
**separate residual variance** for source (or weighting / stratifying), not a source
intercept.

---

## 6. Paralog pilot — the first frustration result (negative)

**Scripts:** `scripts/pilot_paralogs.py`, `scripts/fetch_pilot.py`, `scripts/run_pilot.sh`,
`scripts/pilot_compare.py`, `scripts/plot_pilot.py` · **Output:** `data/kinome/pilot/`,
`data/kinome/analysis/pilot_*`, `results/pilot/`

The only place in this file where frustration was actually computed. The full reasoning is
in the commit messages of `9b4f3fb` and `0d26623`; this is the summary.

**Design.** Picked by rule: the sharpest same-family pairs by pocket divergence plus the
sharpest zero-ΔpKd pair as a negative control — **5 pairs, 10 complexes, 3 ligands**:
STI ABL1/ABL2 (ΔpKd 0.96), STI KIT/PDGFRa (0.38), VX6 AurA/AurC (0.21), 1N1 EphB4/EphA2
(0.40), and 1N1 BMX/BTK (0.00) as the negative control. KLIFS's 85 pocket positions were resolved onto each structure's own residue numbers
(`interactions_match_residues`), which makes two kinases comparable position by position
with no alignment work. All five pairs are matched on DFG and αC state.

**Structures.** `fetch_pilot.py` strips each entry to one chain + one ligand copy. The
chain comes from the manifest, not the file (2HYY has four imatinibs; KLIFS chose chain C);
where a chain holds several copies (3GVU) the one in the pocket is chosen by contact count.
RCSB's `_ideal.sdf` supplies only the chemistry, the pose comes from the HETATM block.
Modified residues (TPO, SEP, MSE) are kept.

**Comparison** (fixed in advance): ΔF at matched pocket positions, tighter minus weaker
binder, tested for enrichment at the divergent positions by label permutation; the raw
interaction-energy difference run beside it as a comparator that needs no decoys.

**Verdict, 1,000 decoys: no.** ABL1/ABL2 is p 0.019 with no energy floor and
0.76 / 0.84 / 0.83 at floors of 0.1 / 0.3 / 0.5 kcal/mol. No pair survives any floor; the
raw-energy comparator never had the signal (all p ≥ 0.18); the negative control stays
silent throughout. The cause is in the index, not the dataset: F is a z-score, so a
physically absent contact with a tiny σ scores high (ABL2 PRO365–STI: −0.065 kcal/mol,
σ 0.019, F +4.4). 18 of 127 matched positions sit below 0.1 kcal/mol. The 1,000 decoys
also showed that the second-best 50-decoy signal (EphB4/EphA2, p 0.052) was sampling noise
(→ +0.13, p 0.49).

This result is why generation 2 wants **many kinases per ligand** instead of pairs.

---

## 7. KLIFS-first complex set (generation 2, current)

**Script:** `scripts/klifs_complex_set.py` · **Main output:**
**`data/kinome/DRUG_KINASE_COMPLEXES.csv`** · Support: `data/kinome/complex_set/`

Decided 2026-10-24. Davis is no longer the gate. The unit is a ligand co-crystallised with
**≥ 5 human kinases** that also has a binding value on ≥ 5 of them, so that a
selectivity panel exists per ligand. Structures come from KLIFS only; affinities never come
from the Davis table.

### Structures, not affinities, are the ceiling

Counted on `klifs_raw.csv` (human, nucleotides excluded, no other filter): ligands
co-crystallised with ≥ 3 / **≥ 5** / ≥ 10 kinase domains: **69 / 18 / 4**. The four at
≥ 10 are staurosporine STU (38), dasatinib 1N1 (11), bosutinib DB8 (11), silmitasertib
3NG (10). KLIFS's `drug_list` has 654 drugs, 281 with a PDB code, 246 with a human
structure — and 178 of those with exactly one kinase.

Re-fetching from RCSB would not change this. Of the PDB entries holding STU, 1N1, STI and
DB8 that KLIFS lacks, only four add a new *human kinase* (STU: TrkA 9DAI, KIT 8S19, AMPK
8BIK; DB8: ILK 9D5F); the rest are rat/bovine/chicken orthologues, non-kinases (NQO2,
PPARγ, transthyretin) or cryo-EM GRK2.

KLIFS's drug-search page (`klifs.net/result.php?drug_search=…`) lists structures of
**analogues** by fingerprint similarity (e.g. 0.48 for agerafenib, which has no structure
of its own). Those are not complexes of the drug; they could serve as templates for
redocking.

### Quality filter: loose, with the flags carried

| filter | ligands ≥ 5 (structure) | Kd: ligands / complexes | IC50 | Ki |
|---|---|---|---|---|
| none | 18 | 11 / 108 | 7 / 72 | 2 / 10 |
| **res ≤ 3.5, quality ≥ 4** (chosen) | 18 | 11 / 106 | 7 / 72 | 2 / 10 |
| res ≤ 3.0, quality ≥ 6 | 15 | 8 / 87 | 6 / 63 | 2 / 10 |
| res ≤ 2.5, quality ≥ 6 (generation-1 manifest) | 14 | 6 / 61 | 6 / 51 | 1 / 5 |
| res ≤ 2.5, quality ≥ 8 | 12 | 3 / 35 | 5 / 39 | 0 |

(Counted against KLIFS's own bioactivity list, before the ChEMBL cleaning below.) Losses
start only at quality ≥ 6. Missing **atoms** are absorbed by frustx's repacking; missing
pocket **residues** are backbone gaps whose contacts cannot be scored, so
`missing_residues` is carried per complex and the analysis can be re-run without them.

### Where each value comes from

- **Structures:** KLIFS, filtered with `klifs_manifest.py`'s own functions, so
  resolution / quality / altloc mean exactly what they mean there; best structure per
  (ligand, kinase domain) by quality, then resolution.
- **Ligand → ChEMBL compound:** KLIFS's `bioactivity_list_id` (its `chembl_id` field).
- **Values:** ChEMBL_37's activity API, **not** KLIFS's bioactivity list. KLIFS's list is
  a ChEMBL extract with no variant, assay-type or document fields, so mutants and
  cell-based IC50s cannot be removed from it (dasatinib: 545 KLIFS rows vs 993 ChEMBL rows
  with a pchembl value; 318 of 496 target/measure pairs larger in ChEMBL). Queried by
  **parent** molecule, because KLIFS pools salt forms (dasatinib/ABL1 Kd: 2 by molecule id,
  25 by parent, 23 in KLIFS). The KLIFS row counts stay in `funnel.json` as a cross-check.

### Rules on ChEMBL rows (first match wins)

1. Target is a human `SINGLE PROTEIN` that is a human KLIFS kinase — drops cell lines
   (K562), fusions (Bcr/Abl is `CHIMERIC PROTEIN`), complexes, non-kinases.
2. `assay_type == "B"` (binding) — a cell-based IC50 is not a binding constant.
3. No `data_validity_comment` (ChEMBL's own error flags).
4. `potential_duplicate == 0` — a value re-cited from an earlier paper would inflate n.
5. No `assay_variant_mutation` — **wild type only**. ChEMBL has 112 mutant dasatinib rows
   on ABL1 alone (T315I at > 10 µM beside wild type at sub-nM). Mutants are counted
   (`n_mutant_*`), never pooled.
6. `pchembl_value` present — censored values (`>10000`) are counted (`n_censored_*`),
   never numbers.

Per (ligand, UniProt, measure): median pX, n, std, number of documents, and
`inconsistent` when the spread exceeds 1 log unit.

**Kd and IC50 are separate columns and separate sets, never pooled. Ki is excluded by
decision** (only 2 ligands would reach 5 kinases).

### Reading `DRUG_KINASE_COMPLEXES.csv`

One row per (ligand, kinase domain): ligand code, drug name (from KLIFS `drug_list`),
ChEMBL parent; kinase, KLIFS id, UniProt, family, group; the chosen PDB/chain/altloc with
resolution, quality, missing residues/atoms, DFG/αC and how many structures competed; then
for each measure `pKd`/`pIC50`, `n_*`, `std_*`, `n_documents_*`, `inconsistent_*`,
`n_mutant_*`, `n_censored_*`; and finally **`in_Kd_set`** / **`in_IC50_set`**.

`in_Kd_set = True` answers "use this complex in the Kd analysis": the row has a
consistent wild-type Kd, its domain is attributable (`domain_assignment` is not
`ambiguous`), and its ligand has ≥ 5 such rows. One analysis reads only `pKd`, another
only `pIC50`; a row can be in both.

`domain_assignment`: `single_domain`; `structure` (multi-domain protein, but this ligand is
crystallised on only one of its domains, so the value is attributed there); `ambiguous`
(crystallised on two domains of one protein — excluded; SKE on JAK2/JAK2-b).

### Result

| ligand | drug | Kd-set complexes | pKd range | IC50-set complexes |
|---|---|---|---|---|
| STU | staurosporine | 35 | 3.60 | 30 |
| 1N1 | dasatinib | 9 | 3.12 | 6 |
| DB8 | bosutinib | 7 | 2.77 | 0 |
| STI | imatinib | 7 | 3.60 | 7 |
| B96 | doramapimod | 6 | 3.82 | 6 |
| 0LI | ponatinib | 5 | 2.09 | 7 |
| 3NG | silmitasertib | 5 | 3.34 | 8 |
| VX6 | tozasertib | 5 | 1.22 | 5 |
| **total** | | **79** (8 ligands) | | **69** (7 ligands) |

What to keep in mind:

- **The consistency rule, not missing data, removes ligands.** Sunitinib has Kd on 7
  structured kinases but 3 are inconsistent (KDR, PHKg2, MST3); tofacitinib loses TYK2
  and stops at 4. The rule also fires on n = 2 (bosutinib/LOK: two values, spread 1.48).
- **Dasatinib/ABL1 is still inconsistent** (n = 6, std 1.05) after 19 annotated mutants
  were removed: ChEMBL's variant annotation is incomplete for older papers, and the spread
  rule is the backstop.
- **23 of the 79 Kd-set complexes have missing pocket residues.**
- **The two sets are not independent:** 57 complexes are in both. Kd is primary — for an
  ATP-competitive inhibitor IC50 = Ki(1 + [ATP]/Km,ATP), and Km,ATP varies across a
  selectivity panel.
- **Most ChEMBL Kd values come from kinome screens** (Davis 2011, Karaman 2008 and
  similar); the Davis table is not used, but the affinity side is not independent of it.
- **Redocking headroom** (`complex_set/ligands.csv`): consistent wild-type Kd on any
  human KLIFS kinase, crystal or not — STU 341, sunitinib 242, tozasertib 195, bosutinib
  182, dovitinib 169, dasatinib 120 — against 5–35 with a crystal.

### Output files

- **`data/kinome/DRUG_KINASE_COMPLEXES.csv`** — the table to use (157 rows: every
  complex of the 18 candidate ligands; the `in_*_set` flags select the analysis sets).
- `complex_set/ligands.csv` — one row per ligand: structures, values and set sizes per
  measure, redocking headroom.
- `complex_set/affinities.csv` — per (ligand, UniProt, measure) wild-type value on every
  human KLIFS kinase, structure or not: the targets a redocking step would need.
- `complex_set/activities.csv` — every ChEMBL row fetched, with the rule that kept or
  removed it (audit trail).
- `complex_set/funnel.json` — thresholds, structure funnel, activity status counts,
  KLIFS-vs-ChEMBL row counts, top source documents, ChEMBL version.
- `complex_set/cache/` — raw KLIFS and ChEMBL responses (23 MB); a re-run makes no
  requests.

### Download (`scripts/fetch_complex_set.py`)

The union of the two sets — **91 complexes, 91 PDB entries, 8 ligands** — is downloaded
and cleaned with `fetch_pilot.py`'s own code (one chain, the ligand copy in the pocket,
altloc blank/A, modified residues kept, waters and additives dropped and counted), after
an RCSB check: all 91 are current X-ray entries with the ligand on the KLIFS chain, all in
PDB format, all 8 ligand SDFs carry hydrogens. **91 / 91 ready.** Output under
`complex_set/structures/{raw,clean}/`, plus `runs.csv` (one frustx command per complex),
`pocket_map.csv` (KLIFS's 85 positions on each structure's residue numbers — the common
frame across kinases) and `fetch.json`.

Reported, not blocking: two crystallisation additives dropped from inside the pocket
(malonate in GSK3B 7Z1G, PEG in TTK 3HMO); no KLIFS pocket map for RSK3 2Z7R.

**Rosetta silently drops three residue types** (frustx runs with
`-ignore_unrecognized_res`): CME, CXM and OCS — all crystallisation artefacts — plus
residues without a full backbone. Two of those losses are pocket positions: STU/PIM1
CME 161 and DB8/ULK3 ALA 26. Mapping the artefacts back to CYS/MET is an open decision.

---

## 8. Bugs found and reported

Nothing was fixed silently; each was reported and the decision left to the user.

### Statistical error in the pooled test (fixed)

`pooled_power` compared the statistic against the null distribution's **95th
percentile**; it should have computed a permutation **p-value**. With a continuous null
the two agree — the large scopes were unaffected (`pooled_ranking` identical to three
decimals, `pooled_all` ≤ 0.001 apart) — but on a **coarse null** they diverge.

| scope | old (percentile) | new (p-value) |
|---|---|---|
| pooled_all (96), ratio 2.0 | 1.000 | 1.000 |
| pooled_ranking (78), ratio 1.0 | 0.996 | 0.996 |
| **negative control (18), ratio 2.0** | **0.394** | **0.261** |
| a 2-complex scope, any ratio | 1.000 *(absurd)* | **0** |

The diagnostic case: a single two-point ligand always has `|stat|` exactly 1; the
percentile is 1 too, every draw "beats" it and power reads 1.000 — while the real p-value
is 1 and nothing can be significant. The fix is pinned by
`tests/test_design_a_feasibility.py` (11 tests): that the two methods diverge on a coarse
null, agree on a continuous one, and that **both scripts** use the correct rule
(`@parametrize` over `design_a` + `extended_set`). An empty-scope crash (`ValueError`) in
the same function was found and guarded.

**The only thesis number this changes is the negative-control power: 0.26, not 0.39.**
The ranking scopes and the ≈ 0.6 smallest detectable effect did not change.

### Other reports

- 7 obsolete PDB IDs (KLIFS still lists them), reported with their replacements.
- 3 rows whose `ligand_code` is not in the entry (1P4F/DRG, 8AN8/N0U, 7OOV/6ID); 4 with a
  similar but different code (e.g. 5N9N 8QK→KC5) — not remapped, counted as
  `ligand_missing`.
- 1J6's JAK2/TYK2 constructs cover both domains → `domain_ambiguous`.
- GCN2-b's pocket could not be placed in a UniProt domain.
- TYK2 JH2 series: 9,016 BindingDB rows / 2,978 ligands measured on JH2 constructs, but
  only **3** have a manifest structure (ZRU, ZS3, ZSB).
- Full-protein measurements are assigned to JH2 by the structure rule (e.g. a full-length
  TYK2 IC50 for KZJ → TYK2-b). An allosteric JH2 binder does inhibit the full protein,
  but that value is an enzyme activity, not a JH2 binding constant.
- Generation 2: KLIFS's bioactivity list cannot separate mutants (section 7); the
  `not_single_klifs_protein` label first counted every kinase without a structure of the
  candidate ligands as "not a KLIFS protein" — fixed so it means what it says, and those
  values now feed `affinities.csv`.

---

## 9. What we have now

### Code

All on the same pattern: network layer apart, filter / mapping / statistics logic in
**pure functions**, tests offline, each script writes its own `*_funnel.json` with
provenance.

| script | size | tests |
|---|---|---|
| `scripts/klifs_manifest.py` | 28 KB | 36 |
| `scripts/bindingdb_activity.py` | 54 KB | 50 |
| `scripts/davis_activity.py` | 42 KB | 29 |
| `scripts/design_a_feasibility.py` | 24 KB | 11 (shared) |
| `scripts/extended_set.py` | 29 KB | ↑ same file, `@parametrize` |
| `scripts/pilot_paralogs.py` | 23 KB | 23 |
| `scripts/fetch_pilot.py` | 19 KB | 12 |
| `scripts/pilot_compare.py` | 17 KB | — |
| **`scripts/klifs_complex_set.py`** | 22 KB | 11 |
| **`scripts/fetch_complex_set.py`** | 14 KB | 10 |

**240 tests collected** across the eight kinome test files, all offline.

### Data (`data/kinome/`, ~685 MB, not in git, tracked with DVC via `data.dvc`)

| path | size | content |
|---|---|---|
| **`DRUG_KINASE_COMPLEXES.csv`** | 38 KB | **the generation-2 table** (section 7) |
| `complex_set/` | 93 MB | 91 downloaded + cleaned complexes, runs.csv, pocket map, ligand summary, affinities, ChEMBL audit trail, funnels, API caches |
| `klifs/` | 9.3 MB | manifest (3,196 structures), raw table, kinase table, funnel, RCSB cache |
| `bindingdb/` | 571 MB | 567 MB raw zip + provenance, 25,641 activity rows, ligand IDs, funnel, 2 caches |
| `davis/` | 5.8 MB | 3 original `.xls` + provenance, 31,464 activity rows, kinase/compound maps, funnel, 4 caches |
| `analysis/` | 1.1 MB | Design A feasibility, extended set, pilot comparison and figures |
| `pilot/` | 5.1 MB | 10 pilot structures (raw + cleaned), ligand SDFs, run list |

### The state, in one paragraph

Generation 1 established that a pooled Design A can detect an effect of ~0.6 noise units
per log unit, but only STU, 1N1 and DB8 carry a per-ligand test, and the paralog pilot on
five pairs came back negative — for a reason inside the index (z-scores of physically
absent contacts), not the data. Generation 2 starts from KLIFS alone and asks for ≥ 5
kinases per ligand: **8 ligands / 79 complexes with Kd, 7 / 69 with IC50**, wild type
only, measures never mixed. Structures, not affinities, remain the ceiling: the same
ligands have consistent Kd on 35–341 human kinases, so redocking is how the panel would
grow.

---

## 10. Open decisions

Generation 2 (current):

1. **Consistency threshold.** 1 log unit removes sunitinib and tofacitinib from the Kd set
   and fires on n = 2 pairs. Keep it, loosen it, or require n ≥ 3 before applying it?
2. **Complexes with missing pocket residues** (23 of 79 in the Kd set): keep and flag (as
   now), drop, or rebuild the gaps?
3. **Artefact residues Rosetta drops** (CME, CXM, OCS; section 7): map them back to
   CYS/MET before running frustx, or accept the gap (one pocket position in STU/PIM1)?
4. **Count check → frustration vs affinity, or redocking.** If 8 ligands / 79 complexes
   are not enough, the redocking pipeline extends each panel using `affinities.csv` as the
   target list.
5. **The energy-floor problem from the pilot** (section 6) applies to any new run: a
   z-score on a near-zero interaction is not evidence. It needs a decision before the
   generation-2 results are read.

Generation 1 (still open, lower priority now):

6. **8X7, 537, YAM** are `exploratory`. Confirm they stay out of the primary ranking set,
   or accept the single-report ends and return them to `ranking`?
7. **Negative-control power is still low** (0.53 at ratio 2.0). A null result counts as
   evidence only with more narrow-range ligands or an equivalence-test (TOST) framing.
8. **TYK2 JH2 series** could be a sub-study: 2,978 ligands measured, 3 with structures.
9. **Domain assignment of full-protein measurements** (section 8): is assigning
   full-length IC50s of JH2 binders to JH2 right?
