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
