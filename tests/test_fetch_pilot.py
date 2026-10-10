"""Cleaning logic of scripts/fetch_pilot.py, offline (no download, no PDB files)."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path('/home/tugbae/frustx/scripts')))
from fetch_pilot import (choose_copy, clean_complex, is_modified_residue, ligand_copies,
                         parse_atoms, pocket_contacts, sdf_atom_counts, check_pocket)

def _atom(rec, name, resname, chain, resseq, x, y=0.0, z=0.0, altloc=" ", el=""):
    return (f"{rec:<6}{'1':>5} {name:^4}{altloc}{resname:>3} {chain}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")

def _protein(chain="A", resseq=1, x=0.0, resname="ALA"):
    return [_atom("ATOM", n, resname, chain, resseq, x, el="C" if n != "N" else "N")
            for n in ("N", "CA", "C", "O")]

def test_parse_atoms_reads_columns():
    a = parse_atoms(_atom("ATOM", "CA", "ALA", "A", 10, 1.5, 2.5, 3.5, el="C"))[0]
    assert (a["record"], a["name"], a["resname"], a["chain"], a["resseq"]) == (
        "ATOM", "CA", "ALA", "A", "10")
    assert (a["x"], a["y"], a["z"]) == (1.5, 2.5, 3.5)

def test_modified_residue_needs_a_backbone():
    assert is_modified_residue(parse_atoms("\n".join(_protein(resname="TPO"))))
    assert not is_modified_residue(parse_atoms(_atom("HETATM", "O", "HOH", "A", 1, 0)))

def test_modified_residue_is_kept_so_the_chain_has_no_hole():
    text = "\n".join(_protein("A", 1) + [_atom("HETATM", n, "TPO", "A", 2, 5.0, el="C")
                                         for n in ("N", "CA", "C", "O", "P")]
                     + _protein("A", 3, 10.0)
                     + [_atom("HETATM", "C1", "STI", "A", 600, 1.0, el="C")])
    atoms = parse_atoms(text)
    lines, st = clean_complex(atoms, "A", "STI", ("600", ""))
    assert st["modified_residues_kept"] == {"TPO 2": 5}
    assert st["ca_count"] == 3                      # 1, TPO 2, 3
    lines, st = clean_complex(atoms, "A", "STI", ("600", ""), keep_modified=False)
    assert st["modified_residues_kept"] == {} and st["ca_count"] == 2
    assert st["dropped_het_on_kept_chain"] == {"TPO 2": 5}

def test_dropped_atoms_are_categorised():
    text = "\n".join(_protein("A", 1) + _protein("B", 1)
                     + [_atom("HETATM", "O", "HOH", "A", 900, 9.0, el="O"),
                        _atom("HETATM", "C1", "STI", "A", 600, 1.0, el="C"),
                        _atom("HETATM", "C1", "STI", "A", 601, 20.0, el="C"),
                        _atom("HETATM", "FE", "HEM", "A", 700, 3.0, el="FE")])
    lines, st = clean_complex(parse_atoms(text), "A", "STI", ("600", ""))
    assert st["dropped_other_chain_atoms"] == 4            # chain B, expected
    assert st["dropped_solvent"] == {"HOH": 1}
    assert st["dropped_other_ligand_copies"] == {"STI 601": 1}
    assert st["dropped_het_on_kept_chain"] == {"HEM 700": 1}   # the only real signal
    assert st["ligand_atoms"] == 1

def test_altloc_b_is_dropped():
    text = "\n".join(_protein("A", 1)
                     + [_atom("ATOM", "CB", "ALA", "A", 1, 2.0, altloc="B", el="C"),
                        _atom("HETATM", "C1", "STI", "A", 600, 1.0, el="C")])
    _, st = clean_complex(parse_atoms(text), "A", "STI", ("600", ""))
    assert st["altloc_atoms_dropped"] == 1

def test_pocket_contacts_uses_minimum_heavy_atom_distance():
    pocket = parse_atoms("\n".join(_protein("A", 1, 0.0) + _protein("A", 2, 5.0)
                                   + _protein("A", 3, 50.0)))
    lig = parse_atoms(_atom("HETATM", "C1", "STI", "A", 600, 1.0, el="C"))
    assert pocket_contacts(lig, pocket, cutoff=6.0) == 2      # residues 1 and 2
    assert pocket_contacts(lig, pocket, cutoff=0.5) == 0

def test_hydrogens_are_ignored_in_contacts():
    pocket = parse_atoms(_atom("ATOM", "H", "ALA", "A", 1, 1.0, el="H"))
    lig = parse_atoms(_atom("HETATM", "C1", "STI", "A", 600, 1.0, el="C"))
    assert pocket_contacts(lig, pocket, cutoff=6.0) == 0

def test_choose_copy_prefers_the_pocket_and_reports_both():
    text = "\n".join(_protein("A", 1, 0.0)
                     + [_atom("HETATM", "C1", "STI", "A", 1001, 1.0, el="C"),
                        _atom("HETATM", "C1", "STI", "A", 1002, 40.0, el="C")])
    atoms = parse_atoms(text)
    copies = ligand_copies(atoms, "STI", "A")
    pocket = [x for x in atoms if x["record"] == "ATOM"]
    key, scores = choose_copy(copies, pocket, cutoff=6.0)
    assert key == ("1001", "")
    assert [s["pocket_contacts"] for s in scores] == [1, 0]

def test_choose_copy_ties_break_on_residue_number():
    text = "\n".join(_protein("A", 1, 0.0)
                     + [_atom("HETATM", "C1", "STI", "A", 1002, 1.0, el="C"),
                        _atom("HETATM", "C1", "STI", "A", 1001, 1.0, el="C")])
    atoms = parse_atoms(text)
    key, _ = choose_copy(ligand_copies(atoms, "STI", "A"),
                         [x for x in atoms if x["record"] == "ATOM"], cutoff=6.0)
    assert key == ("1001", "")

def test_sdf_atom_counts():
    sdf = "name\n  prog\n\n  3  2  0     0  0            999 V2000\n" \
          "    0.0000    0.0000    0.0000 C   0  0\n" \
          "    1.0000    0.0000    0.0000 H   0  0\n" \
          "    2.0000    0.0000    0.0000 N   0  0\n"
    c = sdf_atom_counts(sdf)
    assert (c["atoms"], c["hydrogens"], c["heavy_atoms"]) == (3, 1, 2)
    assert c["elements"] == {"C": 1, "N": 1}

def test_check_pocket_reports_absent_positions():
    lines = [_atom("ATOM", "CA", "ALA", "A", 10, 0.0, el="C"),
             _atom("ATOM", "CA", "ALA", "A", 11, 0.0, el="C")]
    r = check_pocket(lines, ["10", "11", "12"])
    assert r["pocket_present"] == 2 and r["pocket_missing"] == ["12"]
