"""Davis 2011 parsing and mapping in scripts/davis_activity.py, offline.

Names and cell formats are the ones in the paper's Supplementary Tables 1, 3 and 4;
PubChem and UniProt responses are fixtures shaped like what the live services return.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from davis_activity import (build_activities, choose_uniprot,  # noqa: E402
                            compound_aliases, consistency, copy_references, kd_value,
                            map_kinases, reference_table, skeleton_candidates,
                            matrix_long, parse_discoverx, primary_subset,
                            resolve_compound)
from davis_activity import apply_salt_resolution  # noqa: E402


# --- DiscoverX construct names -----------------------------------------------------

@pytest.mark.parametrize("name, cls, muts, phospho", [
    ("AAK1", "wild_type", [], ""),
    ("IKK-alpha", "wild_type", [], ""),                       # "-alpha" is the name
    ("BRAF(V600E)", "mutant", ["V600E"], ""),
    ("EGFR(L858R,T790M)", "mutant", ["L858R", "T790M"], ""),
    ("EGFR(L747-E749del, A750P)", "mutant", ["L747-E749del", "A750P"], ""),
    ("EGFR(L747-T751del,Sins)", "mutant", ["L747-T751del", "Sins"], ""),
    ("FLT3(ITD)", "mutant", ["ITD"], ""),
    ("ABL1(E255K)-phosphorylated", "mutant", ["E255K"], "phosphorylated"),
    ("ABL1(F317I)-nonphosphorylated", "mutant", ["F317I"], "nonphosphorylated"),
    ("ABL1-phosphorylated", "phospho_or_other", [], "phosphorylated"),
    ("ABL1-nonphosphorylated", "wild_type", [], "nonphosphorylated"),   # WT_OVERRIDES
])
def test_variant_class(name, cls, muts, phospho):
    p = parse_discoverx(name)
    assert (p["variant_class"], p["mutations"], p["phospho_state"]) == (cls, muts, phospho)
    assert p["unparsed"] == []


@pytest.mark.parametrize("name, label, ordinal", [
    ("JAK1(JH1domain-catalytic)", "JH1domain-catalytic", 2),     # JH1 is C-terminal
    ("JAK1(JH2domain-pseudokinase)", "JH2domain-pseudokinase", 1),
    ("RSK1(Kin.Dom.1-N-terminal)", "Kin.Dom.1-N-terminal", 1),
    ("RPS6KA5(Kin.Dom.2-C-terminal)", "Kin.Dom.2-C-terminal", 2),
])
def test_domain_constructs(name, label, ordinal):
    p = parse_discoverx(name)
    assert (p["domain_label"], p["domain_ordinal"], p["variant_class"]) == (label, ordinal,
                                                                            "wild_type")


def test_domain_construct_with_a_mutation():
    p = parse_discoverx("GCN2(Kin.Dom.2,S808G)")
    assert (p["domain_ordinal"], p["mutations"], p["variant_class"]) == (2, ["S808G"], "mutant")


def test_organism_and_complex():
    assert parse_discoverx("PFCDPK1(P.falciparum)")["organism"] == "P.falciparum"
    assert parse_discoverx("PKNB(M.tuberculosis)")["organism"] == "M.tuberculosis"
    p = parse_discoverx("CDK4-cyclinD1")
    assert (p["base"], p["complex_partner"], p["variant_class"]) == ("CDK4", "cyclinD1",
                                                                     "wild_type")


def test_wild_type_override_is_explicit_and_noted():
    assert parse_discoverx("ABL1-nonphosphorylated")["variant_note"] == "abl1_nonphos_as_wt"
    assert parse_discoverx("ABL1-phosphorylated")["variant_note"] == ""
    # the override is for this one name: a mutant non-phosphorylated ABL1 stays a mutant
    p = parse_discoverx("ABL1(T315I)-nonphosphorylated")
    assert (p["variant_class"], p["variant_note"]) == ("mutant", "")


def test_unknown_tokens_are_reported_not_dropped():
    assert parse_discoverx("KIN(weird-thing)")["unparsed"] == ["weird-thing"]


# --- values ------------------------------------------------------------------------

def test_kd_to_pkd():
    assert kd_value("1000")["pX"] == pytest.approx(6.0)
    assert kd_value("0.016")["pX"] == pytest.approx(9 - math.log10(0.016))
    v = kd_value("43")
    assert not v["censored"] and v["valid"] and math.isnan(v["censor_bound_pX"])


def test_blank_cell_is_censored_with_the_10uM_bound():
    v = kd_value("")
    assert v["censored"] and v["valid"]
    assert math.isnan(v["pX"]) and math.isnan(v["kd_nM"])        # never a number
    assert v["censor_bound_pX"] == pytest.approx(5.0)            # pKd < 5


def test_unreadable_cell_is_invalid_not_censored():
    v = kd_value("n.d.")
    assert not v["valid"] and not v["censored"]


# --- compounds ---------------------------------------------------------------------

def _hit(cid, ik="AAAAAAAAAAAAAA-UHFFFAOYSA-N"):
    return {"cid": cid, "inchikey": ik, "smiles": "C"}


def test_compound_aliases():
    assert compound_aliases("CHIR-258/TKI-258", "Dovitinib") == (
        ["CHIR-258", "TKI-258", "Dovitinib"], False)
    assert compound_aliases("BIBF-1120 (derivative)", "")[1] is True


def test_compound_ok_when_all_aliases_agree():
    r = resolve_compound(["INCB018424", "Ruxolitinib"], False,
                         {"INCB018424": [_hit(25126798)], "Ruxolitinib": [_hit(25126798)]})
    assert (r["status"], r["cid"]) == ("ok", 25126798)


def test_alias_unknown_to_pubchem_does_not_block():
    r = resolve_compound(["AST-487", "NVP-AST487"], False,
                         {"AST-487": [_hit(11409972)], "NVP-AST487": []})
    assert (r["status"], r["cid"]) == ("ok", 11409972)


def test_aliases_resolving_to_different_structures_are_reported():
    # PubChem: CHIR-258 -> 135431668, TKI-258 and dovitinib -> 135398510
    r = resolve_compound(["CHIR-258", "TKI-258", "Dovitinib"], False,
                         {"CHIR-258": [_hit(135431668)], "TKI-258": [_hit(135398510)],
                          "Dovitinib": [_hit(135398510)]})
    assert r["status"] == "aliases_disagree" and r["cid"] is None and r["inchikey"] == ""
    assert "CHIR-258->135431668" in r["detail"]


def test_multiple_cids_are_never_chosen_automatically():
    r = resolve_compound(["X-1"], False, {"X-1": [_hit(1), _hit(2)]})
    assert r["status"] == "multiple_cids" and r["cid"] is None
    assert r["detail"] == "X-1->1,2"


DOVITINIB = {"CHIR-258": [_hit(135431668, "ZRHDKBOBHHFLBW-UHFFFAOYSA-N")],      # lactate
             "TKI-258": [_hit(135398510, "PIQCTGMSNWUMAF-UHFFFAOYSA-N")],       # free base
             "Dovitinib": [_hit(135398510, "PIQCTGMSNWUMAF-UHFFFAOYSA-N")]}


def test_salt_resolution_picks_the_listed_free_base():
    al = ["CHIR-258", "TKI-258", "Dovitinib"]
    r = apply_salt_resolution("CHIR-258/TKI-258", resolve_compound(al, False, DOVITINIB),
                              al, DOVITINIB)
    assert (r["status"], r["cid"], r["inchikey"]) == (
        "salt_resolved", 135398510, "PIQCTGMSNWUMAF-UHFFFAOYSA-N")
    assert "CHIR-258->135431668" in r["detail"]          # the disagreement stays on record


def test_salt_resolution_leaves_other_compounds_alone():
    # same salt pattern, not listed: stays unresolved
    lookup = {"PTK-787": [_hit(151193)], "Vatalanib": [_hit(151194)]}
    r = resolve_compound(["PTK-787", "Vatalanib"], False, lookup)
    assert apply_salt_resolution("PTK-787", r, ["PTK-787", "Vatalanib"], lookup) == r
    ok = resolve_compound(["Imatinib"], False, {"Imatinib": [_hit(5291)]})
    assert apply_salt_resolution("Imatinib", ok, ["Imatinib"], {}) == ok


def test_salt_resolution_cid_must_be_a_pubchem_candidate():
    al = ["CHIR-258", "TKI-258"]
    lookup = {k: DOVITINIB[k] for k in al}
    del lookup["TKI-258"]                                 # free base no longer offered
    lookup["TKI-258"] = [_hit(999)]
    with pytest.raises(ValueError, match="135398510"):
        apply_salt_resolution("CHIR-258/TKI-258", resolve_compound(al, False, lookup),
                              al, lookup)


def test_not_found_and_derivative():
    assert resolve_compound(["Q"], False, {"Q": []})["status"] == "not_found"
    assert resolve_compound(["BIBF-1120"], True, {})["status"] == "derivative"


# --- kinase mapping ----------------------------------------------------------------

def _hits(*accs, reviewed=True, organism="Homo sapiens"):
    return [{"accession": a, "reviewed": reviewed, "organism": organism, "gene": ""}
            for a in accs]


def test_choose_uniprot():
    assert choose_uniprot(_hits("P00519") + _hits("Q59FK4", reviewed=False)) == ("P00519", "ok")
    assert choose_uniprot(_hits("Q99999", organism="Mus musculus")) == ("", "no_reviewed_human")
    assert choose_uniprot(_hits("P1", "P2")) == ("", "several_reviewed_human")


KLIFS = pd.DataFrame({"kinase_ID": [392, 435, 438, 99],
                      "name": ["ABL1", "JAK1", "JAK1-b", "MTOR"],
                      "gene_name": ["ABL1", "JAK1", "JAK1", "MTOR"],
                      "uniprot": ["P00519", "P23458", "P23458", "P42345"]})
SPANS = {435: (875, 1153), 438: (583, 855)}           # JAK1 JH1, JAK1-b JH2 (UniProt)


def _kmap(rows, idmap):
    kin = pd.DataFrame(rows, columns=["accession", "gene", "name"])
    return map_kinases(kin, idmap, KLIFS, SPANS).set_index("davis_kinase")


def test_kinase_mapping_classes_and_domains():
    idmap = {"NP_005148": _hits("P00519"), "NP_002218": _hits("P23458"),
             "NP_000001": _hits("P99999")}
    m = _kmap([("NP_005148.2", "ABL1", "ABL1(E255K)-phosphorylated"),
               ("NP_005148.2", "ABL1", "ABL1-nonphosphorylated"),
               ("NP_002218.2", "JAK1", "JAK1(JH1domain-catalytic)"),
               ("NP_002218.2", "JAK1", "JAK1(JH2domain-pseudokinase)"),
               ("NP_002218.2", "JAK1", "JAK1"),                       # no domain named
               ("NP_000001.1", "PIK3XX", "PIK3XX"),                   # human, not in KLIFS
               ("CAA12345.1", "PFPK5", "PFPK5(P.falciparum)")], idmap)
    assert m.loc["ABL1(E255K)-phosphorylated", ["status", "variant_class", "klifs_kinase_id"]
                 ].tolist() == ["ok", "mutant", 392]
    assert m.loc["ABL1-nonphosphorylated", ["variant_class", "variant_note"]].tolist() == [
        "wild_type", "abl1_nonphos_as_wt"]
    assert m.loc["JAK1(JH1domain-catalytic)", ["klifs_kinase_id", "domain_source"]
                 ].tolist() == [435, "construct"]
    assert m.loc["JAK1(JH2domain-pseudokinase)", ["klifs_kinase_id", "kinase_name"]
                 ].tolist() == [438, "JAK1-b"]
    assert m.loc["JAK1", "status"] == "domain_unresolved"
    assert m.loc["PIK3XX", "status"] == "not_in_klifs"
    assert m.loc["PFPK5(P.falciparum)", "status"] == "non_human"


def test_outdated_symbol_mapped_by_accession_and_symbol_only_as_fallback():
    # FRAP1 is the 2011 symbol of MTOR: the accession decides
    m = _kmap([("NP_004949.1", "FRAP1", "FRAP1"), ("XYZ00001.1", "MTOR", "MTOR")],
              {"NP_004949": _hits("P42345"), "XYZ00001": []})
    assert m.loc["FRAP1", ["uniprot", "uniprot_method"]].tolist() == ["P42345", "accession"]
    assert m.loc["MTOR", ["uniprot", "uniprot_method"]].tolist() == ["P42345", "entrez_symbol"]


def test_accession_that_is_already_a_uniprot_id_is_used_directly():
    # WEE2's "Accession Number" in the paper is P0C1S8, a UniProt accession
    klifs = pd.concat([KLIFS, pd.DataFrame({"kinase_ID": [7], "name": ["Wee1B"],
                                            "gene_name": ["WEE2"], "uniprot": ["P0C1S8"]})])
    kin = pd.DataFrame([("P0C1S8", "WEE2", "WEE2"), ("Q6XUX3.1", "RIPK5", "RIPK5")],
                       columns=["accession", "gene", "name"])
    m = map_kinases(kin, {}, klifs, SPANS).set_index("davis_kinase")
    assert m.loc["WEE2", ["uniprot", "uniprot_method", "klifs_kinase_id"]].tolist() == [
        "P0C1S8", "accession_is_uniprot", 7]
    assert m.loc["RIPK5", ["uniprot", "status"]].tolist() == ["Q6XUX3", "not_in_klifs"]


def test_domain_unresolved_when_a_domain_span_is_missing():
    kin = pd.DataFrame([("NP_002218.2", "JAK1", "JAK1(JH1domain-catalytic)")],
                       columns=["accession", "gene", "name"])
    m = map_kinases(kin, {"NP_002218": _hits("P23458")}, KLIFS, {435: SPANS[435]})
    assert m.iloc[0]["status"] == "domain_unresolved"


# --- activities --------------------------------------------------------------------

def test_activities_keep_censored_as_bound_and_variants_apart():
    matrix = [["Accession Number", "Entrez Gene Symbol", "Kinase", "Imatinib"],
              ["NP_005148.2", "ABL1", "ABL1(E255K)-phosphorylated", "100"],
              ["NP_002218.2", "JAK1", "JAK1(JH1domain-catalytic)", ""]]
    long = matrix_long(matrix, ["Imatinib"])
    kmap = map_kinases(pd.DataFrame([r[:3] for r in matrix[1:]],
                                    columns=["accession", "gene", "name"]),
                       {"NP_005148": _hits("P00519"), "NP_002218": _hits("P23458")},
                       KLIFS, SPANS)
    cmap = pd.DataFrame({"davis_compound": ["Imatinib"], "inchikey": ["K"],
                         "ligand_group": ["STI"], "match_level": ["full"], "match_note": [""]})
    manifest = pd.DataFrame({"klifs_kinase_id": [435], "ligand_group": ["STI"]})
    act = build_activities(long, kmap, cmap, manifest, {"P00519"}).set_index("davis_kinase")
    mut, jh1 = act.loc["ABL1(E255K)-phosphorylated"], act.loc["JAK1(JH1domain-catalytic)"]
    assert (mut["median_pX"], mut["n"], mut["variant_class"]) == (pytest.approx(7.0), 1, "mutant")
    assert (jh1["n"], jh1["n_censored"]) == (0, 1) and math.isnan(jh1["median_pX"])
    assert jh1["censor_bound_pX"] == pytest.approx(5.0) and jh1["has_structure"]
    assert set(act["source"]) == {"davis2011"} and set(act["measure"]) == {"Kd"}
    # only wild type with a known domain is primary
    assert list(primary_subset(act.reset_index())["davis_kinase"]) == ["JAK1(JH1domain-catalytic)"]


def _refs():
    d = pd.DataFrame({"klifs_kinase_id": [1, 2, 3], "ligand_group": ["A"] * 3,
                      "pX": [8.0, 7.0, 6.0]})
    o = pd.DataFrame({"klifs_kinase_id": [1, 2, 3, 1, 2, 3, 4],
                      "ligand_group": ["A"] * 7,
                      "pX": [8.0, 7.0, 5.0,      # PMID 111: 2 of 3 identical
                             8.0, 7.0, 6.0,      # no reference: 3 of 3 identical
                             9.0],               # PMID 222: no Davis counterpart
                      "pmid": ["111"] * 3 + [""] * 3 + ["222"],
                      "doi": [""] * 7})
    return d, o


def test_reference_table_counts_identical_values_per_reference():
    t = reference_table(*_refs())
    assert t.loc["none"].tolist() == [3, 3, 1.0]
    assert t.loc["111"].tolist() == [2, 3, 0.667]
    assert "222" not in t.index          # nothing to compare with


@pytest.mark.parametrize("threshold, expected", [(0.8, {"none"}), (0.6, {"none", "111"}),
                                                 (1.01, set())])
def test_copy_threshold_includes_the_no_reference_group(threshold, expected):
    assert copy_references(reference_table(*_refs()), threshold) == expected


GROUPS = pd.DataFrame({"ligand_code": ["STI", "X1", "X2", "DRG"],
                       "inchikey": ["KTUFNOKKBVMGRW-UHFFFAOYSA-N",
                                    "PIQCTGMSNWUMAF-UHFFFAOYSA-N",
                                    "PIQCTGMSNWUMAF-ABCDEFGHSA-N", ""],
                       "ligand_group": ["STI", "X1", "X2", "DRG"]})


def test_skeleton_candidates_list_every_cid_and_its_manifest_groups():
    lookup = {"CHIR-258": [_hit(135431668, "ZRHDKBOBHHFLBW-UHFFFAOYSA-N")],
              "TKI-258": [_hit(135398510, "PIQCTGMSNWUMAF-UHFFFAOYSA-N")],
              "Dovitinib": [_hit(135398510, "PIQCTGMSNWUMAF-UHFFFAOYSA-N")]}
    c = {x["cid"]: x for x in skeleton_candidates(["CHIR-258", "TKI-258", "Dovitinib"],
                                                  lookup, GROUPS)}
    assert c[135431668]["manifest_groups"] == []
    assert c[135398510]["aliases"] == ["TKI-258", "Dovitinib"]
    assert [(g["ligand_group"], g["relation"]) for g in c[135398510]["manifest_groups"]] == [
        ("X1", "identical"), ("X2", "skeleton")]


def test_skeleton_candidates_with_several_cids_for_one_name():
    lookup = {"CI-1033": [_hit(1, "AAAAAAAAAAAAAA-UHFFFAOYSA-N"),
                          _hit(2, "KTUFNOKKBVMGRW-ZZZZZZZZSA-N")]}
    c = skeleton_candidates(["CI-1033"], lookup, GROUPS)
    assert [x["cid"] for x in c] == [1, 2]
    assert c[1]["manifest_groups"][0]["relation"] == "skeleton"     # STI's skeleton


def test_consistency_statistics():
    d = pd.DataFrame({"klifs_kinase_id": [1, 2, 3], "ligand_group": ["A"] * 3,
                      "pX": [8.0, 7.0, 6.0]})
    o = pd.DataFrame({"klifs_kinase_id": [1, 2, 3, 4], "ligand_group": ["A"] * 4,
                      "pX": [8.5, 7.2, 3.0, 9.0]})
    c = consistency(d, o)
    assert c["pairs"] == 3 and c["median_delta"] == pytest.approx(0.2)
    # reported rounded to 3 decimals
    assert c["frac_abs_delta_gt_1"] == pytest.approx(1 / 3, abs=1e-3) and c["spearman_rho"] == 1.0
