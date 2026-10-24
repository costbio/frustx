"""Pure logic of scripts/klifs_complex_set.py, offline.

The activity fixtures use ChEMBL's own field names and value types (pchembl_value as a
string or None, potential_duplicate as int, assay_variant_mutation None for wild type),
as returned by the live activity API.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from klifs_complex_set import (aggregate, assign_domains, build_table,  # noqa: E402
                               candidate_ligands, classify_activities, select_structures)

TARGETS = {
    "T_ABL1": {"target_type": "SINGLE PROTEIN", "organism": "Homo sapiens",
               "accessions": ["P00519"]},
    "T_BCRABL": {"target_type": "CHIMERIC PROTEIN", "organism": "Homo sapiens",
                 "accessions": ["P00519", "P11274"]},
    "T_K562": {"target_type": "CELL-LINE", "organism": "Homo sapiens", "accessions": []},
}


def _act(**kw):
    base = dict(activity_id=1, target_chembl_id="T_ABL1", standard_type="Kd",
                standard_relation="=", standard_value="1.0", pchembl_value="9.00",
                assay_type="B", assay_variant_mutation=None, data_validity_comment=None,
                potential_duplicate=0, document_chembl_id="D1", ligand_code="1N1")
    base.update(kw)
    return base


def _classify(rows):
    return classify_activities(pd.DataFrame(rows), TARGETS, {"P00519"})["status"].tolist()


def test_each_rule_removes_its_row():
    assert _classify([
        _act(),
        _act(target_chembl_id="T_BCRABL"),          # fusion: two accessions
        _act(target_chembl_id="T_K562"),            # cell line
        _act(assay_type="F"),                       # cell-based IC50
        _act(data_validity_comment="Outside typical range"),
        _act(potential_duplicate=1),
        _act(assay_variant_mutation="T315I"),
        _act(standard_relation=">", pchembl_value=None, standard_value="10000"),
        _act(standard_relation=None, pchembl_value=None, standard_value=None),
    ]) == ["wild_type", "not_single_klifs_protein", "not_single_klifs_protein",
           "not_binding_assay", "validity_flagged", "potential_duplicate", "mutant",
           "censored", "no_value"]


def test_first_rule_wins():
    # a censored mutant is reported as a mutant: the variant is the more basic reason
    assert _classify([_act(assay_variant_mutation="T315I", standard_relation=">",
                           pchembl_value=None)]) == ["mutant"]


def test_mutants_never_enter_the_wild_type_median():
    acts = classify_activities(pd.DataFrame([
        _act(pchembl_value="10.0"), _act(pchembl_value="9.0", document_chembl_id="D2"),
        _act(pchembl_value="4.0", assay_variant_mutation="T315I"),
        _act(pchembl_value=None, standard_relation=">"),
    ]), TARGETS, {"P00519"})
    a = aggregate(acts).iloc[0]
    assert (a["pX"], a["n"], a["n_mutant"], a["n_censored"], a["n_documents"]) == \
        (9.5, 2, 1, 1, 2)
    assert not a["inconsistent"]


def test_inconsistent_is_spread_above_one_log_and_false_for_a_single_value():
    acts = classify_activities(pd.DataFrame([
        _act(pchembl_value="9.0"), _act(pchembl_value="6.0"),
        _act(standard_type="IC50", pchembl_value="7.0"),
    ]), TARGETS, {"P00519"})
    a = aggregate(acts).set_index("measure")
    assert a.loc["Kd", "inconsistent"] and not a.loc["IC50", "inconsistent"]


def test_pair_with_only_mutant_rows_keeps_a_row_without_value():
    acts = classify_activities(pd.DataFrame([_act(assay_variant_mutation="T315I")]),
                               TARGETS, {"P00519"})
    a = aggregate(acts).iloc[0]
    assert pd.isna(a["pX"]) and a["n"] == 0 and a["n_mutant"] == 1


def test_domain_assignment():
    s = pd.DataFrame({"ligand_code": ["A", "B", "B", "C"],
                      "uniprot": ["P1", "JAK2", "JAK2", "JAK2"],
                      "klifs_kinase_id": [1, 10, 11, 10]})
    assert assign_domains(s, {"JAK2"}).tolist() == \
        ["single_domain", "ambiguous", "ambiguous", "structure"]


def _structures(n, ligand="L"):
    return pd.DataFrame({"ligand_code": [ligand] * n, "uniprot": [f"P{i}" for i in range(n)],
                         "klifs_kinase_id": list(range(n))})


def _agg(values, measure="Kd", ligand="L", inconsistent=None):
    inc = inconsistent or [False] * len(values)
    return pd.DataFrame({"ligand_code": ligand, "uniprot": [f"P{i}" for i in range(len(values))],
                         "measure": measure, "pX": values, "n": 1, "std": None,
                         "n_documents": 1, "inconsistent": inc, "n_mutant": 0,
                         "n_censored": 0})


def test_set_needs_min_kinases_with_usable_values():
    t = build_table(_structures(5), _agg([9, 8, 7, 6, None]), set(), min_kinases=4)
    assert t["in_Kd_set"].tolist() == [True, True, True, True, False]
    assert not t["in_IC50_set"].any()                      # no IC50 at all
    t = build_table(_structures(5), _agg([9, 8, 7, 6, None]), set(), min_kinases=5)
    assert not t["in_Kd_set"].any()                        # only 4 usable


def test_inconsistent_value_is_kept_but_not_in_set():
    t = build_table(_structures(3), _agg([9, 8, 7], inconsistent=[False, True, False]),
                    set(), min_kinases=2)
    assert t["pKd"].notna().all()
    assert t["in_Kd_set"].tolist() == [True, False, True]


def test_measures_counted_separately():
    agg = pd.concat([_agg([9, 8, 7]), _agg([9, None, None], measure="IC50")])
    t = build_table(_structures(3), agg, set(), min_kinases=3)
    assert t["in_Kd_set"].all() and not t["in_IC50_set"].any()


def _raw(**kw):
    base = dict(structure_ID="1", kinase="ABL1", species="Human", kinase_ID="392",
                pdb="1abc", alt="", chain="A", resolution="2.0", quality_score="8",
                missing_residues="0", missing_atoms="0", ligand="STI", ligand_ID="48",
                allosteric_ligand="0", DFG="in", aC_helix="in", family="Abl", group="TK",
                uniprot="P00519", ligand_name="imatinib")
    base.update(kw)
    return base


def test_select_structures_keeps_best_and_counts_competitors():
    raw = pd.DataFrame([
        _raw(structure_ID="1", pdb="1aaa", quality_score="6"),
        _raw(structure_ID="2", pdb="2bbb", quality_score="9"),
        _raw(structure_ID="3", pdb="3ccc", resolution="3.6"),        # fails resolution
        _raw(structure_ID="4", pdb="4ddd", resolution="0"),          # NMR
        _raw(structure_ID="5", pdb="5eee", kinase_ID="1", kinase="AKT1", uniprot="P31749",
             ligand="ATP", ligand_ID="7"),                            # nucleotide
    ])
    best, _ = select_structures(raw, max_resolution=3.5, min_quality=4)
    assert best[["pdb", "n_structures", "klifs_ligand_id"]].values.tolist() == \
        [["2BBB", 2, 48]]


def test_candidate_ligands_counts_kinase_domains():
    s = pd.concat([_structures(5, "A"), _structures(4, "B")])
    assert candidate_ligands(s, 5) == ["A"]
