"""Parsing, filtering and aggregation of scripts/bindingdb_activity.py, offline.

The fixture TSV mirrors the real BindingDB_All layout (release 202610): fixed columns,
then 12-column blocks per target chain, every row padded to the header's length, so the
chain stride and UniProt offset are read from the header exactly as on the real file.
RCSB chem_comp responses are mocked as the `chem` dict fetch_chemcomps() returns.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bindingdb_activity import (COLUMNS, aggregate, assign_domains,  # noqa: E402
                                coverage, drop_source_duplicates, ligand_groups,
                                ligand_table, match_ligands, parse_affinity, scan,
                                single_kinase, to_long, to_px)

CHAIN_FIELDS = [
    "BindingDB Target Chain Sequence", "PDB ID(s) of Target Chain",
    "UniProt (SwissProt) Recommended Name of Target Chain",
    "UniProt (SwissProt) Entry Name of Target Chain",
    "UniProt (SwissProt) Primary ID of Target Chain",
    "UniProt (SwissProt) Secondary ID(s) of Target Chain",
    "UniProt (SwissProt) Alternative ID(s) of Target Chain",
    "UniProt (TrEMBL) Submitted Name of Target Chain",
    "UniProt (TrEMBL) Entry Name of Target Chain",
    "UniProt (TrEMBL) Primary ID of Target Chain",
    "UniProt (TrEMBL) Secondary ID(s) of Target Chain",
    "UniProt (TrEMBL) Alternative ID(s) of Target Chain",
]
FIXED = list(dict.fromkeys(list(COLUMNS.values()) + ["EC50 (nM)"]))
HEADER = FIXED + [f"{f} {k}" for k in (1, 2) for f in CHAIN_FIELDS]

ABL1, SRC, CCNA2, JAK2 = "P00519", "P12931", "P20248", "O60674"  # CCNA2: cyclin, no kinase
STI_IK = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"            # imatinib: achiral, no stereo block
STI_ENANT = "KTUFNOKKBVMGRW-ZZZZZZZZSA-N"         # same skeleton, a stated stereo block
OTHER_IK = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
RXT_IK = "HFNKQEVNSGCOJV-OAHLLOKOSA-N"            # ruxolitinib, (R): stereo stated
RXT_FLAT = "HFNKQEVNSGCOJV-UHFFFAOYSA-N"          # ruxolitinib as BindingDB stores it
RXT_S = "HFNKQEVNSGCOJV-HNNXBMFYSA-N"             # the (S) enantiomer
KINASES = {ABL1, SRC, JAK2}


def _line(*, ik=STI_IK, org="Homo sapiens", chains=(ABL1,), ki="", kd="", ic50="",
          ec50="", het="", rid="1", pmid="", doi=""):
    vals = dict.fromkeys(HEADER, "")
    vals.update({COLUMNS["reactant_set_id"]: rid, COLUMNS["inchikey"]: ik,
                 COLUMNS["organism"]: org, COLUMNS["het"]: het, COLUMNS["pmid"]: pmid,
                 COLUMNS["doi"]: doi, COLUMNS["n_chains"]: str(len(chains)),
                 "Ki (nM)": ki, "Kd (nM)": kd, "IC50 (nM)": ic50, "EC50 (nM)": ec50})
    for k, u in enumerate(chains, start=1):
        vals[f"BindingDB Target Chain Sequence {k}"] = "MSEQ"
        vals[f"UniProt (SwissProt) Primary ID of Target Chain {k}"] = u
    return "\t".join(vals[h] for h in HEADER) + "\n"


def _scan(lines, codes=("STI",), skeletons=(STI_IK[:14],)):
    return scan(["\t".join(HEADER) + "\n", *lines], KINASES, ["Homo sapiens", "Human"],
                set(codes), set(skeletons))


def _groups(**codes):
    return ligand_groups(pd.DataFrame({"ligand_code": list(codes),
                                       "inchikey": list(codes.values())}))


GROUPS = _groups(STI=STI_IK)


def _long(lines, groups=GROUPS):
    s = _scan(lines, codes=set(groups["ligand_code"]),
              skeletons={k[:14] for k in groups["inchikey"] if k})
    return to_long(match_ligands(single_kinase(s["records"]), groups))


def _pipeline(lines, groups=GROUPS):
    return aggregate(_long(lines, groups))


# --- qualifiers, censoring, pX -----------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("12.5", ("", 12.5)), (">10000", (">", 10000.0)), ("> 20000", (">", 20000.0)),
    ("<0.03", ("<", 0.03)), (" 7 ", ("", 7.0))])
def test_parse_affinity(raw, expected):
    assert parse_affinity(raw) == expected


def test_parse_affinity_unparseable_and_empty():
    q, v = parse_affinity("n/a")
    assert q == "?" and math.isnan(v)
    q, v = parse_affinity("")
    assert q == "" and math.isnan(v)


@pytest.mark.parametrize("nM, px", [(1.0, 9.0), (1000.0, 6.0), (0.1, 10.0), (1e6, 3.0)])
def test_to_px(nM, px):
    assert to_px(nM) == pytest.approx(px)


def test_to_px_rejects_non_positive():
    assert math.isnan(to_px(0.0)) and math.isnan(to_px(-5.0))


def test_censored_values_are_counted_but_kept_out_of_the_median():
    act = _pipeline([_line(kd="10"), _line(kd="100"), _line(kd=">10000"),
                     _line(kd="<0.1")])
    row = act.iloc[0]
    assert (row["n"], row["n_censored"]) == (2, 2)
    assert row["median_pX"] == pytest.approx(7.5)    # median of pKd 8 and 7 only


def test_censored_only_pair_has_no_median():
    act = _pipeline([_line(ic50=">50000")])
    assert act.iloc[0]["n"] == 0 and act.iloc[0]["n_censored"] == 1
    assert math.isnan(act.iloc[0]["median_pX"])


def test_inconsistent_flag_needs_std_above_one_log_unit():
    spread = _pipeline([_line(ki="1"), _line(ki="1000")]).iloc[0]   # pKi 9 and 6
    tight = _pipeline([_line(ki="10"), _line(ki="20")]).iloc[0]
    single = _pipeline([_line(ki="10")]).iloc[0]
    assert spread["inconsistent"] and not tight["inconsistent"]
    assert not single["inconsistent"] and math.isnan(single["std"])


# --- InChIKey matching -------------------------------------------------------------

def _levels(bdb_keys, groups):
    m = match_ligands(pd.DataFrame({"inchikey": bdb_keys}), groups)
    return sorted(zip(m["inchikey"], m["ligand_group"], m["match_level"], m["match_note"]))


def test_full_versus_skeleton_match():
    assert _levels([STI_IK, STI_ENANT, OTHER_IK], GROUPS) == [
        (STI_IK, "STI", "full", ""),
        (STI_ENANT, "STI", "skeleton", "different_stereo_or_isotope")]


def test_no_stereo_match_when_bindingdb_drops_the_stereo_layer():
    # BindingDB's ruxolitinib carries no stereo block; the (R) ligand's key does.
    assert _levels([RXT_FLAT], _groups(RXT=RXT_IK)) == [
        (RXT_FLAT, "RXT", "bindingdb_no_stereo", "")]


def test_real_stereoisomer_stays_out_of_no_stereo():
    # BindingDB states the (S) configuration: a different compound, not "unstated".
    assert _levels([RXT_S], _groups(RXT=RXT_IK)) == [
        (RXT_S, "RXT", "skeleton", "different_stereo_or_isotope")]


def test_protonation_differences_stay_skeleton():
    achiral_other_proton = STI_IK[:-1] + "O"
    flat_other_proton = RXT_FLAT[:-1] + "O"
    assert _levels([achiral_other_proton], GROUPS)[0][2:] == ("skeleton", "protonation_only")
    assert _levels([flat_other_proton], _groups(RXT=RXT_IK))[0][2:] == (
        "skeleton", "no_stereo_and_protonation")


def test_no_stereo_record_fitting_two_manifest_enantiomers_is_not_attributed():
    # both enantiomers are in the manifest under their own codes: a stereo-less record
    # could be either, so it stays skeleton for both
    got = _levels([RXT_FLAT], _groups(RXT=RXT_IK, RXS=RXT_S))
    assert [g[1:] for g in got] == [("RXS", "skeleton", "no_stereo_several_ligands"),
                                   ("RXT", "skeleton", "no_stereo_several_ligands")]


def test_ligand_without_inchikey_never_matches():
    assert match_ligands(pd.DataFrame({"inchikey": [""]}), _groups(XXX="")).empty


def test_no_stereo_and_full_rows_never_pool():
    act = _pipeline([_line(ik=RXT_IK, kd="10"), _line(ik=RXT_FLAT, kd="1000")],
                    _groups(RXT=RXT_IK))
    by_level = act.set_index("match_level")
    assert by_level.loc["full", "median_pX"] == pytest.approx(8.0)
    assert by_level.loc["bindingdb_no_stereo", "median_pX"] == pytest.approx(6.0)
    assert by_level.loc["bindingdb_no_stereo", "inchikey"] == RXT_FLAT


# --- ligand groups -----------------------------------------------------------------

def test_codes_sharing_an_inchikey_form_one_group():
    g = _groups(F3Z=STI_IK, **{"38Z": STI_IK}, AXI=OTHER_IK, XXX="")
    assert g.set_index("ligand_code")["ligand_group"].to_dict() == {
        "F3Z": "38Z/F3Z", "38Z": "38Z/F3Z", "AXI": "AXI", "XXX": "XXX"}


def test_a_group_gets_each_measurement_once():
    act = _pipeline([_line(kd="10")], _groups(F3Z=STI_IK, **{"38Z": STI_IK}))
    assert act["ligand_group"].tolist() == ["38Z/F3Z"] and act.iloc[0]["n"] == 1


# --- source duplicates -------------------------------------------------------------

def test_same_value_and_same_pmid_or_doi_is_counted_once():
    long = _long([
        _line(rid="1", kd="10", pmid="111"),
        _line(rid="2", kd="10", pmid="111"),            # same paper, re-imported
        _line(rid="3", kd="10", doi="10.1/x"),
        _line(rid="4", kd="10", doi="10.1/x"),          # same DOI
        _line(rid="5", kd="10", pmid="222"),            # same value, another paper: kept
        _line(rid="6", kd="20", pmid="111"),            # same paper, another value: kept
        _line(rid="7", kd="10"), _line(rid="8", kd="10")])  # no PMID/DOI: never merged
    kept, dropped = drop_source_duplicates(long)
    assert dropped == {"pmid": 1, "doi": 1}
    assert sorted(kept["reactant_set_id"]) == ["1", "3", "5", "6", "7", "8"]


def test_duplicates_are_only_within_one_measure():
    long = _long([_line(rid="1", kd="10", pmid="111"), _line(rid="2", ki="10", pmid="111")])
    assert drop_source_duplicates(long)[1] == {"pmid": 0, "doi": 0}


# --- target filters ----------------------------------------------------------------

def test_non_human_and_non_kinase_targets_are_dropped():
    s = _scan([_line(rid="1"),                                  # kept
               _line(rid="2", org="Human"),                     # kept: same organism
               _line(rid="3", org="Rattus norvegicus"),         # not human
               _line(rid="4", org=""),                          # organism unknown
               _line(rid="5", chains=(CCNA2,)),                 # human, not a kinase
               _line(rid="6", chains=("Q99999",))])             # human, not in KLIFS
    assert sorted(s["records"]["reactant_set_id"]) == ["1", "2"]
    assert [f["rows"] for f in s["funnel"]] == [6, 4, 2]


def test_multichain_target_with_one_kinase_chain_is_kept_and_flagged():
    s = _scan([_line(rid="1", chains=(ABL1, CCNA2), kd="10"),
               _line(rid="2", chains=(ABL1,), kd="10")])
    one = single_kinase(s["records"]).set_index("reactant_set_id")
    assert one.loc["1", "uniprot"] == ABL1 and one.loc["1", "multichain_target"]
    assert not one.loc["2", "multichain_target"]
    act = aggregate(to_long(match_ligands(one.reset_index(), GROUPS)))
    assert len(act) == 1 and act.iloc[0]["multichain_target"]   # any row multichain


def test_target_with_two_kinase_chains_is_dropped():
    s = _scan([_line(rid="1", chains=(ABL1, SRC)), _line(rid="2", chains=(SRC,))])
    assert list(single_kinase(s["records"])["reactant_set_id"]) == ["2"]


def test_space_separated_uniprot_cell_is_split():
    s = _scan([_line(chains=(f"Q99999 {ABL1}",))])
    assert s["records"]["kinase_uniprots"].tolist() == [ABL1]


def test_rows_inconsistent_with_chain_count_are_counted_not_used():
    good = _line(rid="1")
    fields = good.rstrip("\n").split("\t")
    fields[HEADER.index(COLUMNS["n_chains"])] = "x"
    bad_n = "\t".join(fields) + "\n"
    fields = good.rstrip("\n").split("\t")
    fields[HEADER.index("UniProt (SwissProt) Primary ID of Target Chain 2")] = SRC
    beyond = "\t".join(fields) + "\n"                # data in chain 2, but n_chains == 1
    s = _scan([good, bad_n, beyond, "short\trow\n"])
    assert s["records"]["reactant_set_id"].tolist() == ["1"]
    assert s["problems"] == {
        "field count differs from header (still parsed)": 1,   # the short row
        "unreadable: chain count not an integer": 2,           # "x", and the short row
        "unreadable: data beyond declared chains": 1}


# --- measures ----------------------------------------------------------------------

def test_kd_ki_ic50_never_pooled():
    # one row carrying all three, plus EC50 which must be ignored altogether
    act = _pipeline([_line(kd="1", ki="10", ic50="100", ec50="1000")])
    got = dict(zip(act["measure"], act["median_pX"]))
    assert got == pytest.approx({"Kd": 9.0, "Ki": 8.0, "IC50": 7.0})


# --- domain assignment -------------------------------------------------------------

DOMAINS = {ABL1: [392], JAK2: [436, 439]}              # JAK2 (JH1) and JAK2-b (JH2)


def _act(*rows):
    return pd.DataFrame(rows, columns=["uniprot", "ligand_group"])


def test_single_domain_protein_gets_its_kinase_id():
    out = assign_domains(_act((ABL1, "L1")), pd.DataFrame(
        columns=["uniprot", "ligand_group", "klifs_kinase_id"]), DOMAINS)
    assert out.iloc[0]["klifs_kinase_id"] == 392
    assert not out.iloc[0]["multi_domain_uniprot"] and not out.iloc[0]["domain_ambiguous"]


def test_multi_domain_activity_goes_to_the_domain_with_the_structure():
    manifest = pd.DataFrame({"uniprot": [JAK2, JAK2, JAK2],
                             "ligand_group": ["JH2BINDER", "BOTH", "BOTH"],
                             "klifs_kinase_id": [439, 436, 439]})
    out = assign_domains(_act((JAK2, "JH2BINDER"), (JAK2, "BOTH"), (JAK2, "NOSTRUCT")),
                         manifest, DOMAINS).set_index("ligand_group")
    assert out.loc["JH2BINDER", "klifs_kinase_id"] == 439
    assert not out.loc["JH2BINDER", "domain_ambiguous"]
    # structures in both domains: ambiguous, no id
    assert out.loc["BOTH", "domain_ambiguous"] and pd.isna(out.loc["BOTH", "klifs_kinase_id"])
    # no structure on this protein: left at UniProt level, not ambiguous
    assert pd.isna(out.loc["NOSTRUCT", "klifs_kinase_id"])
    assert not out.loc["NOSTRUCT", "domain_ambiguous"]
    assert out["multi_domain_uniprot"].all()


# --- ligand table / HET cross-check ------------------------------------------------

def test_ligand_table_levels_and_het_disagreements():
    chem = {"STI": {"inchikey": STI_IK, "smiles": "C"},
            "1QO": {"inchikey": STI_ENANT, "smiles": "C"},
            "RXT": {"inchikey": RXT_IK, "smiles": "C"},
            "NEW": {"inchikey": OTHER_IK, "smiles": "C"},
            "DRG": None}
    groups = _groups(STI=STI_IK, **{"1QO": STI_ENANT}, RXT=RXT_IK, NEW=OTHER_IK, DRG="")
    het = pd.DataFrame({"het": ["STI", "STI", "XYZ", "", "RXT"],
                        "inchikey": [STI_IK, OTHER_IK, STI_IK, STI_IK, RXT_FLAT]})
    seen = {STI_IK, RXT_FLAT}
    t = ligand_table(["STI", "1QO", "RXT", "NEW", "DRG"], groups, chem, seen, het)
    t = t.set_index("ligand_code")
    # 1QO is skeleton via BindingDB's achiral STI key (STI's own key has no stereo block,
    # so it is not a "stereo dropped" case)
    assert t["match_level"].to_dict() == {"STI": "full", "1QO": "skeleton",
                                          "RXT": "bindingdb_no_stereo", "NEW": "none",
                                          "DRG": "no_chem_comp"}
    assert t.loc["STI", ["het_rows", "het_rows_inchikey_full", "het_rows_inchikey_other",
                         "inchikey_rows_other_het"]].tolist() == [2, 1, 1, 1]
    assert t.loc["RXT", ["het_rows_inchikey_no_stereo", "het_rows_inchikey_other"]].tolist() == [1, 0]


# --- coverage ----------------------------------------------------------------------

def _cov_act(rows):
    return pd.DataFrame(rows, columns=["ligand_group", "uniprot", "klifs_kinase_id",
                                       "measure", "n", "n_censored", "match_level",
                                       "domain_ambiguous"]).astype({"klifs_kinase_id": "Int64"})


def test_comparable_set_needs_two_structured_kinases_with_kd_or_ki():
    act = _cov_act([
        # L1: structures with ABL1 and SRC, Kd for ABL1, Ki for SRC -> comparable
        ("L1", ABL1, 392, "Kd", 1, 0, "full", False), ("L1", SRC, 400, "Ki", 1, 0, "full", False),
        # L2: SRC has IC50 only -> not comparable
        ("L2", ABL1, 392, "Kd", 1, 0, "full", False), ("L2", SRC, 400, "IC50", 3, 0, "full", False),
        # L3: SRC censored-only -> not comparable
        ("L3", ABL1, 392, "Kd", 1, 0, "full", False), ("L3", SRC, 400, "Kd", 0, 2, "full", False),
    ])
    manifest = pd.DataFrame([(k, u, g) for g in ("L1", "L2", "L3")
                             for k, u in ((392, ABL1), (400, SRC))],
                            columns=["klifs_kinase_id", "uniprot", "ligand_group"])
    cov = coverage(act, manifest, {392: "ABL1", 400: "SRC"})
    assert [d["ligand_group"] for d in cov["comparable"]] == ["L1"]
    assert cov["comparable"][0]["kinases"] == ["ABL1(Kd:full)", "SRC(Ki:full)"]
    assert cov["comparable_ligands_same_measure"] == 0       # Kd vs Ki: not the same
    assert cov["pairs_with_kd_or_ki"] == 4                   # L1x2, L2/ABL1, L3/ABL1
    assert cov["pairs_with_ic50_only"] == 1                  # L2/SRC
    assert cov["pairs_censored_only"] == 1                   # L3/SRC


def test_domain_ambiguous_rows_never_make_a_ligand_comparable():
    # L1 has structures in both JAK2 domains and ABL1; the JAK2 Kd cannot be assigned
    act = _cov_act([("L1", ABL1, 392, "Kd", 1, 0, "full", False),
                    ("L1", JAK2, None, "Kd", 1, 0, "full", True)])
    manifest = pd.DataFrame([(392, ABL1, "L1"), (436, JAK2, "L1"), (439, JAK2, "L1")],
                            columns=["klifs_kinase_id", "uniprot", "ligand_group"])
    cov = coverage(act, manifest, {392: "ABL1", 436: "JAK2", 439: "JAK2-b"})
    assert cov["comparable_ligands"] == 0
    assert cov["pairs_with_kd_or_ki"] == 1 and cov["pairs_domain_ambiguous_only"] == 2
