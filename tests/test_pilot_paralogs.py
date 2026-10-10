"""Selection and QC logic of scripts/pilot_paralogs.py, offline.

Pockets here are short strings rather than 85 characters: the functions take the length
from the data, and a 12-character fixture is readable while an 85-character one is not.
KLIFS `interactions_match_residues` responses are fixtures in the shape the live endpoint
returns (verified 2026-10-10): `index`, `KLIFS_position`, `Xray_position`, the last being
"-1" where KLIFS placed no residue.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pilot_paralogs import (conformation_match, pair_complexes,  # noqa: E402
                            pocket_divergence, pocket_map_table, pocket_qc, select_pilot)


# --- pocket divergence -------------------------------------------------------------

def test_divergence_counts_substitutions_and_keeps_gaps_apart():
    #            pos 1234567890ab
    d = pocket_divergence("ACDEFGHIKLMN",
                          "ACDQFGHIKLMN")       # position 4 differs
    assert d["divergent"] == [4] and d["gaps"] == []
    assert d["n_positions"] == 12 and d["identity"] == pytest.approx(11 / 12, abs=1e-3)


@pytest.mark.parametrize("gap", ["-", "_"])
def test_a_gap_is_not_a_difference(gap):
    """A gap means KLIFS could not place a residue. Counting it as a difference would
    invent divergence out of missing density."""
    d = pocket_divergence("AC" + gap + "EFGHIKLMN", "ACDEFGHIKLMN")
    assert d["divergent"] == [] and d["gaps"] == [3]
    assert d["n_divergent"] == 0 and d["n_gaps"] == 1


def test_identical_pockets():
    d = pocket_divergence("ACDEFG", "ACDEFG")
    assert (d["n_divergent"], d["n_gaps"], d["identity"]) == (0, 0, 1.0)


def test_length_mismatch_is_an_error_not_a_truncation():
    with pytest.raises(ValueError, match="pocket lengths differ"):
        pocket_divergence("ACDEFG", "ACDEF")


# --- crystal vs canonical pocket QC ------------------------------------------------

def test_qc_separates_gaps_from_substitutions():
    # crystal: position 3 unresolved, position 5 a different residue
    q = pocket_qc("AC-EQGHIKLMN", "ACDEFGHIKLMN", divergent=[])
    assert q["gaps"] == [3] and q["substitutions"] == [5]
    assert not q["clean"] and not q["length_mismatch"]


def test_qc_flags_only_the_problems_that_hit_divergent_positions():
    """A gap or substitution AT a position that distinguishes the pair kills that
    comparison; one elsewhere only costs a contact."""
    q = pocket_qc("AC-EQGHIKLMN", "ACDEFGHIKLMN", divergent=[3, 9])
    assert q["gaps_at_divergent"] == [3]          # position 3 is divergent and missing
    assert q["substitutions_at_divergent"] == []  # position 5 is not divergent


def test_qc_clean_crystal():
    q = pocket_qc("ACDEFG", "ACDEFG", divergent=[1, 2])
    assert q["clean"] and q["n_gaps"] == 0 and q["n_substitutions"] == 0


def test_qc_reports_length_mismatch_without_guessing():
    q = pocket_qc("ACDEF", "ACDEFG", divergent=[])
    assert q["length_mismatch"] and q["n_gaps"] is None


# --- pilot selection ---------------------------------------------------------------

def _pairs(rows):
    """rows: (ligand, pair, delta_pKd, n_divergent)"""
    return pd.DataFrame(rows, columns=["ligand_group", "pair", "delta_pKd", "n_divergent"])


SCORED = _pairs([
    ("STI", "ABL1/ABL2", 0.96, 4),      # sharpest with real signal
    ("VX6", "AurA/AurC", 0.21, 11),
    ("1N1", "EphB4/EphA2", 0.40, 14),
    ("1N1", "BMX/BTK", 0.00, 22),       # equipotent -> negative control
    ("DB8", "LOK/MST3", 1.73, 43),      # pockets too far apart
    ("STU", "PAK4/MST3", 1.28, 50),
])


def test_sharp_pairs_are_the_least_divergent_with_real_signal():
    p = select_pilot(SCORED, n_sharp=3, n_negative=1)
    sharp = p[p["role"] == "sharp"]
    assert sharp["pair"].tolist() == ["ABL1/ABL2", "AurA/AurC", "EphB4/EphA2"]
    # BMX/BTK is the 4th least divergent but has no affinity difference to explain
    assert "BMX/BTK" not in sharp["pair"].tolist()


def test_negative_control_comes_from_the_zero_delta_pairs():
    p = select_pilot(SCORED, n_sharp=3, n_negative=1)
    neg = p[p["role"] == "negative_control"]
    assert neg["pair"].tolist() == ["BMX/BTK"]
    assert float(neg["delta_pKd"].iloc[0]) == 0.0


def test_a_pair_is_never_both_roles():
    p = select_pilot(SCORED, n_sharp=6, n_negative=1)
    assert p["pair"].is_unique


def test_no_negative_control_available_is_not_an_error():
    p = select_pilot(SCORED[SCORED["pair"] != "BMX/BTK"], n_sharp=2, n_negative=1)
    assert (p["role"] == "negative_control").sum() == 0
    assert len(p) == 2


def test_selection_is_deterministic_under_ties():
    """Equal divergence: the larger affinity difference wins, then the name. Shuffling
    the input must not change the pilot."""
    tied = _pairs([("L1", "B/B2", 0.5, 10), ("L2", "A/A2", 1.5, 10),
                   ("L3", "C/C2", 0.5, 10)])
    first = select_pilot(tied, n_sharp=2, n_negative=0)["pair"].tolist()
    shuffled = select_pilot(tied.iloc[::-1], n_sharp=2, n_negative=0)["pair"].tolist()
    assert first == ["A/A2", "B/B2"] == shuffled


def test_negative_control_threshold_is_on_the_absolute_difference():
    signed = _pairs([("L1", "A/A2", -0.05, 30), ("L2", "B/B2", 2.0, 5)])
    p = select_pilot(signed, n_sharp=1, n_negative=1)
    assert dict(zip(p["pair"], p["role"])) == {"B/B2": "sharp", "A/A2": "negative_control"}


# --- complexes ---------------------------------------------------------------------

def test_a_ligand_in_two_pairs_gives_one_complex_per_kinase():
    pilot = pd.DataFrame({
        "ligand_group": ["STI", "STI"], "pair": ["ABL1/ABL2", "KIT/PDGFRa"],
        "kinase_hi": ["ABL1", "KIT"], "kinase_lo": ["ABL2", "PDGFRa"],
        "role": ["sharp", "sharp"]})
    cx = pair_complexes(pilot)
    assert len(cx) == 4
    assert sorted(cx["kinase_name"]) == ["ABL1", "ABL2", "KIT", "PDGFRa"]
    assert set(cx["n_pairs"]) == {1}


def test_a_kinase_shared_by_two_pairs_is_run_once():
    pilot = pd.DataFrame({
        "ligand_group": ["B49", "B49"], "pair": ["HPK1/MST3", "MST3/PAK6"],
        "kinase_hi": ["HPK1", "MST3"], "kinase_lo": ["MST3", "PAK6"],
        "role": ["sharp", "sharp"]})
    cx = pair_complexes(pilot).set_index("kinase_name")
    assert len(cx) == 3                                   # not 4
    assert cx.loc["MST3", "n_pairs"] == 2
    assert cx.loc["MST3", "pairs"] == "HPK1/MST3;MST3/PAK6"


# --- conformation ------------------------------------------------------------------

def _cx(dfg, ac):
    return pd.DataFrame({"dfg": dfg, "ac_helix": ac})


def test_conformation_match():
    m = conformation_match("ABL1/ABL2", _cx(["out", "out"], ["in", "in"]))
    assert m["dfg_match"] and m["ac_helix_match"] and m["dfg"] == "out"


def test_mismatched_dfg_is_reported_not_repaired():
    m = conformation_match("X/Y", _cx(["out", "in"], ["in", "in"]))
    assert not m["dfg_match"] and m["dfg"] == "in/out"     # both states kept, sorted
    assert m["ac_helix_match"]


def test_mismatched_ac_helix():
    m = conformation_match("X/Y", _cx(["in", "in"], ["in", "out"]))
    assert m["dfg_match"] and not m["ac_helix_match"]


# --- the common pocket frame -------------------------------------------------------

def _map(entries):
    return [{"index": i, "KLIFS_position": lab, "Xray_position": x}
            for i, lab, x in entries]


COMPLEXES = pd.DataFrame({
    "ligand_group": ["STI", "STI"], "kinase_name": ["ABL1", "ABL2"],
    "pdb": ["2HYY", "3GVU"], "chain": ["C", "A"], "klifs_structure_id": ["1048", "117"]})


def test_pocket_map_is_long_and_keeps_unresolved_positions():
    maps = {"1048": _map([(1, "I.1", "242"), (2, "I.2", "243"), (3, "I.3", "-1")]),
            "117": _map([(1, "I.1", "290"), (2, "I.2", "291"), (3, "I.3", "292")])}
    t, problems = pocket_map_table(maps, COMPLEXES)
    assert problems == [] and len(t) == 6
    abl1 = t[t["kinase_name"] == "ABL1"].set_index("pocket_position")
    assert abl1.loc[1, "xray_residue"] == "242" and abl1.loc[1, "resolved"]
    # the unresolved position is kept, flagged, not dropped
    assert not abl1.loc[3, "resolved"]
    assert t["resolved"].sum() == 5


@pytest.mark.parametrize("absent", ["-1", "", "0"])
def test_absence_markers(absent):
    maps = {"1048": _map([(1, "I.1", absent)]), "117": _map([(1, "I.1", "290")])}
    t, _ = pocket_map_table(maps, COMPLEXES)
    assert t[t["kinase_name"] == "ABL1"]["resolved"].tolist() == [False]


def test_label_disagreement_between_structures_is_reported():
    """KLIFS's 85-position scheme is fixed. If two structures label one position
    differently the frame is not common and the comparison would be invalid."""
    maps = {"1048": _map([(1, "I.1", "242")]), "117": _map([(1, "g.l.4", "290")])}
    _, problems = pocket_map_table(maps, COMPLEXES)
    assert len(problems) == 1
    assert problems[0]["expected"] == "I.1" and problems[0]["got"] == "g.l.4"


def test_klifs_gap_marker_counts_as_unresolved():
    """KLIFS marks an unplaced position either "-1" or with its pocket-string gap
    character "_". Missing the second marker made six positions (BMX 3SXR four, BTK 3OCT
    two) look resolved, and the download step then reported them as absent from the PDB."""
    maps = {"1048": _map([(1, "I.1", "_"), (2, "I.2", "242")]),
            "117": _map([(1, "I.1", "290"), (2, "I.2", "291")])}
    t, _ = pocket_map_table(maps, COMPLEXES)
    abl1 = t[t["kinase_name"] == "ABL1"].set_index("pocket_position")
    assert not abl1.loc[1, "resolved"] and abl1.loc[2, "resolved"]
