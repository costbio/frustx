"""Selection and per-complex checks of scripts/fetch_complex_set.py, offline.

The cleaning itself is fetch_pilot.py's and is tested in test_fetch_pilot.py; these tests
cover what fetch_complex_set adds: which rows are prepared, file names, and which
problems block a run.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_complex_set import (BLOCKING, complex_name, prepare_complex,  # noqa: E402
                               select_complexes)
from fetch_pilot import parse_atoms  # noqa: E402


def _atom(rec, name, resname, chain, resseq, x, y=0.0, z=0.0, el="C"):
    return (f"{rec:<6}{'1':>5} {name:^4} {resname:>3} {chain}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")


def _protein(resseq, x, chain="A"):
    return [_atom("ATOM", n, "ALA", chain, resseq, x, el="N" if n == "N" else "C")
            for n in ("N", "CA", "C", "O")]


def _sdf(heavy, hydrogens):
    """Minimal V2000 molfile: only the counts line and atom block are read."""
    atoms = [f"{0:10.4f}{0:10.4f}{0:10.4f} C   0  0" for _ in range(heavy)] + \
            [f"{0:10.4f}{0:10.4f}{0:10.4f} H   0  0" for _ in range(hydrogens)]
    return "\n".join(["LIG", "  test", "", f"{heavy + hydrogens:3d}  0  0  0  0  0  0  0  0"
                      "  0999 V2000", *atoms, "M  END"])


def _complex(ligand_resseqs=(600,), ligand_heavy=2, extra=()):
    lines = _protein(1, 0.0) + _protein(2, 3.0)
    for r in ligand_resseqs:
        lines += [_atom("HETATM", f"C{i}", "STI", "A", r, 1.0 + i) for i in range(ligand_heavy)]
    return parse_atoms("\n".join(lines + list(extra)))


def test_select_takes_rows_in_any_set_unless_all():
    t = pd.DataFrame({"in_Kd_set": ["True", "False", "False"],
                      "in_IC50_set": ["False", "True", "False"]})
    assert len(select_complexes(t, all_rows=False)) == 2
    assert len(select_complexes(t, all_rows=True)) == 3


def test_complex_name_keeps_klifs_names_and_neutralises_the_rest():
    assert complex_name("STI", "JAK2-b") == "STI_JAK2-b"
    assert complex_name("STI", "../x y") == "STI____x_y"


def test_clean_complex_passes():
    res = prepare_complex(_complex(), "A", "STI", ["1", "2"], _sdf(2, 3))
    assert res["problems"] == []
    assert res["report"]["ligand_heavy_atoms"] == 2 and res["report"]["pocket_present"] == 2


def test_missing_ligand_blocks_and_writes_nothing():
    res = prepare_complex(_complex(ligand_resseqs=()), "A", "STI", ["1"], _sdf(2, 3))
    assert res["lines"] is None
    assert [k for k, _ in res["problems"]] == ["no_ligand_copy"]


def test_heavy_atom_mismatch_and_missing_hydrogens_block():
    kinds = [k for k, _ in prepare_complex(_complex(), "A", "STI", ["1"],
                                           _sdf(3, 0))["problems"]]
    assert kinds == ["heavy_atom_mismatch", "sdf_without_hydrogens"]
    assert all(k in BLOCKING for k in kinds)


def test_missing_pocket_residue_and_het_near_ligand_are_reported_not_blocking():
    # ligand heavy atoms at x = 1, 2; HEM at x = 5 is 3 A away, inside the 6 A cutoff
    res = prepare_complex(_complex(extra=[_atom("HETATM", "FE", "HEM", "A", 700, 5.0,
                                                el="FE")]),
                          "A", "STI", ["1", "2", "99"], _sdf(2, 3))
    kinds = [k for k, _ in res["problems"]]
    assert kinds == ["pocket_residues_missing", "het_dropped_near_ligand"]
    assert not any(k in BLOCKING for k in kinds)
    assert res["report"]["dropped_het_min_distance_to_ligand"] == {"HEM 700": 3.0}


def test_far_het_is_recorded_with_its_distance_but_not_a_problem():
    res = prepare_complex(_complex(extra=[_atom("HETATM", "C1", "BOG", "A", 900, 30.0)]),
                          "A", "STI", ["1", "2"], _sdf(2, 3))
    assert res["problems"] == []
    assert res["report"]["dropped_het_min_distance_to_ligand"] == {"BOG 900": 28.0}


def test_no_pocket_map_is_reported_not_blocking():
    kinds = [k for k, _ in prepare_complex(_complex(), "A", "STI", [], _sdf(2, 3))
             ["problems"]]
    assert kinds == ["pocket_map_unavailable"] and "pocket_map_unavailable" not in BLOCKING


def test_kept_modified_residue_counts_as_pocket_position():
    mse = [_atom("HETATM", n, "MSE", "A", 3, 6.0) for n in ("N", "CA", "C", "O", "SE")]
    res = prepare_complex(_complex(extra=mse), "A", "STI", ["1", "2", "3"], _sdf(2, 3))
    assert res["report"]["pocket_missing"] == []


def test_copy_in_the_pocket_is_chosen():
    far = [_atom("HETATM", f"C{i}", "STI", "A", 601, 50.0 + i) for i in range(2)]
    res = prepare_complex(_complex(extra=far), "A", "STI", ["1", "2"], _sdf(2, 3))
    assert res["report"]["copy_chosen"] == "600"
    assert res["report"]["dropped_other_ligand_copies"] == {"STI 601": 2}


def _cme(resseq=3, x=6.0):
    return [_atom("HETATM", n, "CME", "A", resseq, x, el=n[0])
            for n in ("N", "CA", "CB", "SG", "SD", "CE", "CZ", "OH", "C", "O")]


def test_artefact_is_written_back_as_its_parent_with_parent_atoms_only():
    res = prepare_complex(_complex(extra=_cme()), "A", "STI", ["1", "2", "3"], _sdf(2, 3))
    cys = [l for l in res["lines"] if l[22:26].strip() == "3"]
    assert {l[:6] for l in cys} == {"ATOM  "} and {l[17:20] for l in cys} == {"CYS"}
    assert sorted(l[12:16].strip() for l in cys) == ["C", "CA", "CB", "N", "O", "SG"]
    assert res["report"]["artefacts_reverted"] == {"CME 3": "CYS"}
    assert res["problems"] == []
    # columns other than record and residue name are untouched (same atom, N)
    n_out = next(l for l in cys if l[12:16].strip() == "N")
    assert n_out[6:17] == _cme()[0][6:17] and n_out[20:] == _cme()[0][20:]


def test_artefacts_can_be_kept_and_are_then_flagged():
    res = prepare_complex(_complex(extra=_cme()), "A", "STI", ["1", "2", "3"], _sdf(2, 3),
                          artefact_parent={})
    assert res["report"]["artefacts_reverted"] == {}
    assert [k for k, _ in res["problems"]] == ["modified_residue_unverified"]


def test_verified_modified_residue_is_not_flagged():
    tpo = [_atom("HETATM", n, "TPO", "A", 3, 6.0) for n in ("N", "CA", "C", "O", "OG1", "P")]
    assert prepare_complex(_complex(extra=tpo), "A", "STI", ["1", "2", "3"],
                           _sdf(2, 3))["problems"] == []


def test_pocket_residue_without_backbone_is_reported():
    stub = [_atom("ATOM", n, "ALA", "A", 3, 6.0) for n in ("C", "O")]   # ULK3 ALA 26
    res = prepare_complex(_complex(extra=stub), "A", "STI", ["1", "2", "3"], _sdf(2, 3))
    assert [k for k, _ in res["problems"]] == ["backbone_incomplete_in_pocket"]
    assert res["report"]["backbone_incomplete"] == ["3"]
    # outside the pocket it is recorded, not a problem
    res = prepare_complex(_complex(extra=stub), "A", "STI", ["1", "2"], _sdf(2, 3))
    assert res["problems"] == [] and res["report"]["backbone_incomplete"] == ["3"]


def test_zero_occupancy_backbone_counts_as_missing():
    # CSK 1BYG GLY 187: all four backbone atoms present, N and CA at occupancy 0.00
    gly = [_atom("ATOM", n, "GLY", "A", 3, 6.0).replace("  1.00  0.00", "  0.00  0.00")
           if n in ("N", "CA") else _atom("ATOM", n, "GLY", "A", 3, 6.0)
           for n in ("N", "CA", "C", "O")]
    res = prepare_complex(_complex(extra=gly), "A", "STI", ["1", "2"], _sdf(2, 3))
    assert res["report"]["backbone_incomplete"] == ["3"]


def test_rename_ligand_touches_only_the_ligand_hetatm_name():
    from fetch_complex_set import ROSETTA_CODE_CLASH, rename_ligand
    lines = [_atom("ATOM", "CA", "ALA", "A", 1, 0.0), _atom("HETATM", "C1", "B96", "A", 600, 1.0)]
    out = rename_ligand(lines, "B96", ROSETTA_CODE_CLASH["B96"])
    assert out[0] == lines[0]
    assert out[1][17:20] == "B9X" and out[1][:17] == lines[1][:17] and out[1][20:] == lines[1][20:]
    assert rename_ligand(lines, "STI", "STI") is lines
