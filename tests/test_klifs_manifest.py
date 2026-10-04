"""Filter and deduplication logic of scripts/klifs_manifest.py, offline.

The fixture rows use KLIFS's own column names and value types (integer 0 for "no ligand",
resolution and quality as strings, "0" resolution for NMR), as returned by the live API, so
these tests exercise normalise() as well as the filters.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from klifs_manifest import (DEFAULT_EXCLUDE, _code, apply_filters,  # noqa: E402
                            build_manifest, deduplicate, multi_domain_uniprots, normalise,
                            remap_collisions)


def _row(**kw):
    base = dict(structure_ID=1, kinase="ABL1", species="Human", kinase_ID=392,
                pdb="1abc", alt="", chain="A", resolution="2.0", quality_score="8",
                missing_residues=0, missing_atoms=0, ligand="STI", allosteric_ligand=0,
                DFG="in", aC_helix="in", family="Abl", group="TK", uniprot="P00519",
                ligand_name="imatinib")
    base.update(kw)
    return base


def _filter(rows, **kw):
    args = dict(species="Human", exclude=DEFAULT_EXCLUDE, max_resolution=2.5,
                min_quality=6.0)
    args.update(kw)
    return apply_filters(normalise(pd.DataFrame(rows)), **args)


def _manifest(rows, **kw):
    return deduplicate(_filter(rows, **kw)[0])


def _ids(df):
    return sorted(df["klifs_structure_id"])


# --- ligand codes ------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, "0", "", "-", None, float("nan"), "  "])
def test_code_treats_klifs_absence_markers_as_empty(value):
    assert _code(value) == ""


def test_code_normalises_case_and_whitespace():
    assert _code(" sti ") == "STI"
    assert _code("1N1") == "1N1"


def test_rows_without_orthosteric_ligand_are_dropped():
    rows = [_row(structure_ID=1, ligand=0), _row(structure_ID=2, ligand="-"),
            _row(structure_ID=3, ligand=""), _row(structure_ID=4, ligand="STI")]
    assert _ids(_filter(rows)[0]) == [4]


def test_allosteric_ligand_is_carried_but_never_filters():
    # Allosteric-only structure: no orthosteric ligand, so it goes, despite the
    # allosteric inhibitor. A structure with both keeps the allosteric code.
    rows = [_row(structure_ID=1, ligand=0, allosteric_ligand="3YY"),
            _row(structure_ID=2, ligand="STI", allosteric_ligand="3yy"),
            _row(structure_ID=3, ligand="NIL", allosteric_ligand=0)]
    kept = _filter(rows)[0].set_index("klifs_structure_id")
    assert list(kept.index) == [2, 3]
    assert kept.loc[2, "allosteric_ligand_code"] == "3YY"
    assert kept.loc[3, "allosteric_ligand_code"] == ""


# --- exclusion list ----------------------------------------------------------------

def test_default_exclude_drops_nucleotides():
    rows = [_row(structure_ID=i, ligand=c)
            for i, c in enumerate(["ATP", "ADP", "ANP", "ACP", "AMP", "STI"], start=1)]
    assert _filter(rows)[0]["ligand_code"].tolist() == ["STI"]


def test_exclude_list_is_replaceable_and_case_insensitive():
    rows = [_row(structure_ID=1, ligand="ATP"), _row(structure_ID=2, ligand="STI")]
    kept = _filter(rows, exclude=["sti"])[0]
    assert kept["ligand_code"].tolist() == ["ATP"]


def test_empty_exclude_list_keeps_everything():
    rows = [_row(structure_ID=1, ligand="ATP"), _row(structure_ID=2, ligand="STI")]
    assert _ids(_filter(rows, exclude=[])[0]) == [1, 2]


# --- resolution / quality / species ------------------------------------------------

def test_nmr_resolution_zero_fails_resolution_filter():
    # KLIFS reports NMR entries as resolution "0" with the model number in `alt`;
    # "0" must not pass as 0 <= 2.5. alt="" here isolates the resolution filter.
    rows = [_row(structure_ID=1, resolution="0"), _row(structure_ID=2, resolution="2.5"),
            _row(structure_ID=3, resolution="2.51")]
    assert _ids(_filter(rows)[0]) == [2]


def test_quality_threshold_is_inclusive():
    rows = [_row(structure_ID=1, quality_score="6"),
            _row(structure_ID=2, quality_score="5.9")]
    assert _ids(_filter(rows)[0]) == [1]


def test_species_filter():
    rows = [_row(structure_ID=1), _row(structure_ID=2, species="Mouse")]
    assert _ids(_filter(rows)[0]) == [1]


# --- altloc ------------------------------------------------------------------------

def test_altloc_keeps_only_A_or_blank():
    rows = [_row(structure_ID=1, alt=""), _row(structure_ID=2, alt="A"),
            _row(structure_ID=3, alt="B"), _row(structure_ID=4, alt="5"),
            _row(structure_ID=5, alt=None)]
    assert _ids(_filter(rows)[0]) == [1, 2, 5]


# --- funnel ------------------------------------------------------------------------

def test_funnel_counts_each_step():
    rows = [_row(structure_ID=1, species="Mouse"),
            _row(structure_ID=2, ligand=0),
            _row(structure_ID=3, ligand="ATP"),
            _row(structure_ID=4, resolution="3.0"),
            _row(structure_ID=5, quality_score="4"),
            _row(structure_ID=6, alt="B"),
            _row(structure_ID=7)]
    _, funnel = _filter(rows)
    assert [f["structures"] for f in funnel] == [7, 6, 5, 4, 3, 2, 1]
    # ligands counts distinct non-empty codes: STI and ATP at the start, STI at the end
    assert funnel[0]["ligands"] == 2 and funnel[-1]["ligands"] == 1


# --- deduplication tie-break order --------------------------------------------------

def test_one_row_per_kinase_ligand_pair():
    rows = [_row(structure_ID=1, ligand="STI"), _row(structure_ID=2, ligand="STI", pdb="2abc"),
            _row(structure_ID=3, ligand="NIL"),
            _row(structure_ID=4, ligand="STI", uniprot="P42684", kinase="ABL2", kinase_ID=393)]
    m = _manifest(rows)
    assert sorted(zip(m["uniprot"], m["ligand_code"])) == [
        ("P00519", "NIL"), ("P00519", "STI"), ("P42684", "STI")]


def test_two_domains_of_one_uniprot_keep_one_structure_each():
    # JAK2 JH1 (kinase id 1) and JH2 pseudokinase (JAK2-b, id 2): one UniProt, two KLIFS
    # kinases. Under a UniProt key the better-scored JH1 structure would evict JH2's.
    rows = [_row(structure_ID=1, kinase="JAK2", kinase_ID=1, uniprot="O60674",
                 ligand="L01", quality_score="9"),
            _row(structure_ID=2, kinase="JAK2-b", kinase_ID=2, uniprot="O60674",
                 ligand="L01", quality_score="7", pdb="2abc"),
            _row(structure_ID=3, kinase="JAK2", kinase_ID=1, uniprot="O60674",
                 ligand="L01", quality_score="8", pdb="3abc")]
    m = _manifest(rows)
    assert sorted(zip(m["kinase_name"], m["klifs_structure_id"])) == [("JAK2", 1), ("JAK2-b", 2)]


def test_multi_domain_uniprots_uses_the_full_kinase_list():
    kinases = pd.DataFrame({
        "kinase_ID": [1, 2, 3, 4, 5],
        "uniprot": ["O60674", "O60674", "P00519", "Q99999", None],
        "species": ["Human", "Human", "Human", "Mouse", "Human"]})
    # O60674 counts even if only one of its domains has a structure: this table is
    # kinase_information, not the structure list.
    assert multi_domain_uniprots(kinases, "Human") == {"O60674"}


def test_quality_beats_resolution():
    rows = [_row(structure_ID=1, quality_score="7", resolution="1.2"),
            _row(structure_ID=2, quality_score="9", resolution="2.4")]
    assert _ids(_manifest(rows)) == [2]


def test_resolution_beats_missing_residues():
    rows = [_row(structure_ID=1, resolution="2.2", missing_residues=0),
            _row(structure_ID=2, resolution="1.8", missing_residues=10)]
    assert _ids(_manifest(rows)) == [2]


def test_missing_residues_before_missing_atoms():
    rows = [_row(structure_ID=1, missing_residues=1, missing_atoms=0),
            _row(structure_ID=2, missing_residues=0, missing_atoms=30),
            _row(structure_ID=3, missing_residues=0, missing_atoms=5)]
    assert _ids(_manifest(rows)) == [3]


def test_pdb_id_is_the_last_specified_tiebreak():
    # all else equal; KLIFS lowercase ids are upper-cased, and order is by id not input
    rows = [_row(structure_ID=1, pdb="5xyz"), _row(structure_ID=2, pdb="3abc"),
            _row(structure_ID=3, pdb="4abc")]
    m = _manifest(rows)
    assert m["pdb"].tolist() == ["3ABC"]


def test_same_pdb_two_chains_resolves_deterministically():
    rows = [_row(structure_ID=9, chain="B"), _row(structure_ID=8, chain="A")]
    assert _ids(_manifest(rows)) == [8]
    assert _ids(_manifest(list(reversed(rows)))) == [8]


def test_empty_uniprot_refuses_to_deduplicate():
    rows = [_row(structure_ID=1, uniprot=None, kinase="A6"),
            _row(structure_ID=2, uniprot=None, kinase="A6r")]
    with pytest.raises(ValueError, match="A6"):
        _manifest(rows)


# --- RCSB check (responses mocked in the shape RCSB's GraphQL returns) ----------------

def _entry(pdb, ligands=(("STI", "A"),), method="X-RAY DIFFRACTION"):
    """One GraphQL `entries` item. `ligands`: (comp_id, auth_asym_id) per copy."""
    return {"rcsb_id": pdb, "exptl": [{"method": method}],
            "nonpolymer_entities": [{"nonpolymer_entity_instances": [
                {"rcsb_nonpolymer_entity_instance_container_identifiers":
                 {"auth_asym_id": ch, "comp_id": c}} for c, ch in ligands]}],
            "polymer_entities": []}


def _rcsb(entries, removed=None):
    """entries: {id: entry, or None for a non-current id}; removed: {id: [replacements]}"""
    return {"entries": entries, "removed": removed or {}, "fetched_at": None}


def _build(rows, rcsb):
    asked = []

    def get_rcsb(ids):
        asked.extend(ids)
        return rcsb

    manifest, annotated, funnel = build_manifest(
        normalise(pd.DataFrame(rows)), get_rcsb, species="Human",
        exclude=DEFAULT_EXCLUDE, max_resolution=2.5, min_quality=6.0)
    return manifest, annotated.set_index("klifs_structure_id"), funnel, asked


def test_current_entry_with_ligand_is_ok_and_keeps_klifs_id_verbatim():
    _, ann, _, _ = _build([_row(pdb="1abc")], _rcsb({"1ABC": _entry("1ABC")}))
    assert ann.loc[1, "rcsb_status"] == "ok"
    assert ann.loc[1, "pdb_klifs"] == "1abc" and ann.loc[1, "pdb"] == "1ABC"


def test_obsolete_remapped_when_replacement_has_the_ligand():
    m, ann, _, _ = _build([_row(pdb="5j7h")],
                          _rcsb({"5J7H": None, "6MX8": _entry("6MX8")},
                                {"5J7H": ["6MX8"]}))
    assert ann.loc[1, "rcsb_status"] == "obsolete_remapped"
    assert ann.loc[1, "pdb"] == "6MX8" and ann.loc[1, "pdb_klifs"] == "5j7h"
    assert m.empty   # remapped rows are reported, never kept


def test_remapped_row_is_dropped_and_klifs_row_for_replacement_wins():
    # 5J7H/6MX8 as in KLIFS: the obsolete deposit is annotated better (quality 8, no
    # missing residues) than its replacement (6.8, 3 missing), so it would win dedup --
    # with metadata that does not describe the file 6MX8. It must be dropped instead.
    rows = [_row(structure_ID=6655, pdb="5j7h", alt="A", ligand="6GY",
                 quality_score="8", missing_residues=0),
            _row(structure_ID=10798, pdb="6mx8", alt="A", ligand="6GY",
                 quality_score="6.8", missing_residues=3)]
    rcsb = _rcsb({"5J7H": None, "6MX8": _entry("6MX8", [("6GY", "A")])},
                 {"5J7H": ["6MX8"]})
    m, ann, funnel, _ = _build(rows, rcsb)
    assert ann.loc[6655, "rcsb_status"] == "obsolete_remapped"
    assert _ids(m) == [10798]
    assert m.iloc[0]["pdb_klifs"] == "6mx8" and m.iloc[0]["quality_score"] == 6.8
    by_step = {f["step"]: f["structures"] for f in funnel}
    assert by_step["rcsb: drop obsolete_remapped"] == 1


def test_obsolete_remapped_but_ligand_absent_is_ligand_missing():
    m, ann, _, _ = _build([_row(pdb="5j7h")],
                          _rcsb({"5J7H": None, "6MX8": _entry("6MX8", [("XYZ", "A")])},
                                {"5J7H": ["6MX8"]}))
    assert ann.loc[1, "rcsb_status"] == "ligand_missing"
    assert ann.loc[1, "pdb"] == "6MX8"   # still reports where it looked
    assert m.empty


def test_supersession_chain_is_followed():
    rcsb = _rcsb({"1AAA": None, "2BBB": None, "3CCC": _entry("3CCC")},
                 {"1AAA": ["2BBB"], "2BBB": ["3CCC"]})
    _, ann, _, _ = _build([_row(pdb="1aaa")], rcsb)
    assert (ann.loc[1, "pdb"], ann.loc[1, "rcsb_status"]) == ("3CCC", "obsolete_remapped")


@pytest.mark.parametrize("removed", [{"1ABC": []}, {"1ABC": None}, {}])
def test_obsolete_without_replacement(removed):
    _, ann, _, _ = _build([_row(pdb="1abc")], _rcsb({"1ABC": None}, removed))
    assert ann.loc[1, "rcsb_status"] == "obsolete_no_replacement"


def test_ligand_missing_from_current_entry():
    _, ann, _, _ = _build([_row(pdb="1abc", ligand="DRG")],
                          _rcsb({"1ABC": _entry("1ABC", [("86Q", "A")])}))
    assert ann.loc[1, "rcsb_status"] == "ligand_missing"


def test_non_xray_is_dropped():
    m, ann, _, _ = _build([_row(pdb="7b5o")],
                          _rcsb({"7B5O": _entry("7B5O", method="ELECTRON MICROSCOPY")}))
    assert ann.loc[1, "rcsb_status"] == "non_xray"
    assert m.empty


def test_each_rcsb_drop_is_its_own_funnel_step():
    rows = [_row(structure_ID=1, pdb="1aaa", ligand="L01"),
            _row(structure_ID=2, pdb="2bbb", ligand="L02"),
            _row(structure_ID=3, pdb="3ccc", ligand="L03"),
            _row(structure_ID=4, pdb="4ddd", ligand="L04"),
            _row(structure_ID=5, pdb="5eee", ligand="L05")]
    rcsb = _rcsb({"1AAA": None,
                  "2BBB": _entry("2BBB", [("XXX", "A")]),
                  "3CCC": _entry("3CCC", [("L03", "A")], method="SOLUTION NMR"),
                  "4DDD": _entry("4DDD", [("L04", "A")]),
                  "5EEE": None, "6FFF": _entry("6FFF", [("L05", "A")])},
                 {"5EEE": ["6FFF"]})
    _, _, funnel, _ = _build(rows, rcsb)
    by_step = {f["step"]: f["structures"] for f in funnel}
    assert by_step["altloc in {'A', ''}"] == 5
    assert by_step["rcsb: drop obsolete_no_replacement"] == 4
    assert by_step["rcsb: drop ligand_missing"] == 3
    assert by_step["rcsb: drop non_xray"] == 2
    assert by_step["rcsb: drop obsolete_remapped"] == 1
    assert by_step["deduplicate (klifs_kinase_id, ligand_code)"] == 1


def test_rcsb_drop_happens_before_dedup_so_runner_up_wins():
    # Structure 1 outranks 2 on quality, but RCSB rejects it. Deduplicating first would
    # keep 1 and then lose the (P00519, STI) pair altogether.
    rows = [_row(structure_ID=1, pdb="1aaa", quality_score="9.5"),
            _row(structure_ID=2, pdb="2bbb", quality_score="7.0")]
    rcsb = _rcsb({"1AAA": _entry("1AAA", [("ANP", "A")]), "2BBB": _entry("2BBB")})
    m, _, _, _ = _build(rows, rcsb)
    assert _ids(m) == [2]


def test_rcsb_is_only_asked_about_rows_surviving_klifs_filters():
    rows = [_row(structure_ID=1, pdb="1aaa"), _row(structure_ID=2, pdb="2bbb", alt="B")]
    _, _, _, asked = _build(rows, _rcsb({"1AAA": _entry("1AAA")}))
    assert asked == ["1AAA"]


# --- flags -----------------------------------------------------------------------------

def test_ligand_chain_differs_flag():
    rows = [_row(structure_ID=1, pdb="1aaa"), _row(structure_ID=2, pdb="2bbb")]
    rcsb = _rcsb({"1AAA": _entry("1AAA", [("STI", "AAA")]),           # only elsewhere
                  "2BBB": _entry("2BBB", [("STI", "A"), ("STI", "BBB")])})  # also on A
    _, ann, _, _ = _build(rows, rcsb)
    assert ann.loc[1, "ligand_chain_differs"] and ann.loc[1, "rcsb_status"] == "ok"
    assert not ann.loc[2, "ligand_chain_differs"]


def test_multi_copy_ligand_flag_counts_copies_on_the_protein_chain_only():
    rows = [_row(structure_ID=1, pdb="1aaa"), _row(structure_ID=2, pdb="2bbb")]
    rcsb = _rcsb({"1AAA": _entry("1AAA", [("STI", "A"), ("STI", "A")]),
                  "2BBB": _entry("2BBB", [("STI", "A"), ("STI", "B")])})
    _, ann, _, _ = _build(rows, rcsb)
    assert ann.loc[1, "multi_copy_ligand"]
    assert not ann.loc[2, "multi_copy_ligand"]


def test_altloc_ligand_conflict_is_computed_from_the_raw_table():
    rows = [
        # 6HOP-like: altlocs of one chain carry different ligands. Only the A row
        # survives filtering, but it must still be flagged.
        _row(structure_ID=1, pdb="6hop", chain="A", alt="A", ligand="FER"),
        _row(structure_ID=2, pdb="6hop", chain="A", alt="B", ligand="V55"),
        # same ligand in both altlocs: no conflict
        _row(structure_ID=3, pdb="1aaa", chain="A", alt="A", ligand="STI"),
        _row(structure_ID=4, pdb="1aaa", chain="A", alt="B", ligand="STI"),
        # ligand in one altloc, none in the other: no conflict (one ligand)
        _row(structure_ID=5, pdb="2bbb", chain="A", alt="A", ligand="NIL"),
        _row(structure_ID=6, pdb="2bbb", chain="A", alt="B", ligand=0),
        # different ligands on different CHAINS (not altlocs): no conflict
        _row(structure_ID=7, pdb="3ccc", chain="A", ligand="L01"),
        _row(structure_ID=8, pdb="3ccc", chain="B", ligand="L02"),
    ]
    rcsb = _rcsb({"6HOP": _entry("6HOP", [("FER", "A")]), "1AAA": _entry("1AAA"),
                  "2BBB": _entry("2BBB", [("NIL", "A")]),
                  "3CCC": _entry("3CCC", [("L01", "A"), ("L02", "B")])})
    _, ann, _, _ = _build(rows, rcsb)
    assert ann["altloc_ligand_conflict"].to_dict() == {
        1: True, 3: False, 5: False, 7: False, 8: False}


def test_remap_collision_is_reported():
    # KLIFS lists both the obsolete entry and its replacement, same chain/altloc
    rows = [_row(structure_ID=1, pdb="5j7h"), _row(structure_ID=2, pdb="6mx8")]
    rcsb = _rcsb({"5J7H": None, "6MX8": _entry("6MX8")}, {"5J7H": ["6MX8"]})
    _, ann, _, _ = _build(rows, rcsb)
    assert sorted(remap_collisions(ann.reset_index())["klifs_structure_id"]) == [1, 2]
