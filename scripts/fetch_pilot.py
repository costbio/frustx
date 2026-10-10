"""Download and clean the paralog pilot's 10 complexes, ready for frustx.

Input is `pilot_complexes.csv` and `pilot_pocket_map.csv` from
scripts/pilot_paralogs.py. For each (ligand, kinase) complex this fetches the PDB entry
and the ligand's chemistry, strips the file to exactly one protein chain plus one ligand
copy, and then CHECKS the result instead of trusting it. It runs no frustration
calculation; the last thing it writes is the command line that would.

    .venv/bin/python scripts/fetch_pilot.py

Output, under data/kinome/pilot/:
    raw/<PDB>.pdb, raw/<CODE>_ideal.sdf   as downloaded, with provenance.json
    clean/<LIGAND>_<KINASE>.pdb           one chain + one ligand copy
    clean/<CODE>.sdf                      the ligand's chemistry (copied from raw)
    pilot_runs.csv                        one row per complex: the frustx command
    fetch_pilot.json                      provenance, what was dropped, every check

Decisions, and why:

  - **The ligand SDF is RCSB's `_ideal.sdf`, used with `--ligand`, not
    `--ligand-placed`.** With `--ligand` the SDF supplies only the chemistry and the
    coordinates come from the structure's HETATM block, which is what we want: these are
    real co-crystal poses. README's warning that an `_ideal.sdf` is "a generated
    conformer near the origin" applies to `--ligand-placed`, where the file's coordinates
    ARE the pose. Checked for all three pilot ligands: the ideal SDFs carry their
    hydrogens (STI 31, 1N1 26, VX6 28), which frustx requires and would otherwise refuse.

  - **The chain comes from the manifest, not from the file.** 2HYY has four copies of
    imatinib, one per chain, and KLIFS chose chain C. Taking "the first ATOM chain" --
    which scripts/egfr_ligand_check.py can do because its entries are single-chain --
    would silently score a different copy than the one the manifest and the pocket map
    describe.

  - **Waters and every other heteroatom are dropped**, for the reason this project
    already learned: Rosetta's `is_ligand()` is True for water, so waters would enter the
    pose as residues and become contact nodes. Everything dropped is counted per residue
    name in the report -- a second inhibitor or a pocket ion must not disappear quietly.

  - **Altloc: keep blank or A.** Rosetta keeps only the first conformer, so leaving both
    puts a residue in the pose twice. Six of the ten pilot structures carry altloc A.

  - **Which ligand copy, when a chain has several**, is decided by pocket contact count
    (3GVU has two imatinibs on chain A, at 1001 and 1002). The pocket is the 85 KLIFS
    positions from `pilot_pocket_map.csv`, so the choice is made in the same coordinate
    system the analysis will use, and both copies' counts are reported.

Everything this script cannot resolve is reported and left alone, never patched.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frustx.config import DEFAULT_LIGAND_CUTOFF  # noqa: E402

RCSB_STRUCTURE = "https://files.rcsb.org/download/{}.pdb"
RCSB_LIGAND = "https://files.rcsb.org/ligands/download/{}_ideal.sdf"
# Residue names that are solvent/cryoprotectant rather than chemistry worth keeping.
# Only used to label the report; everything that is not the chosen ligand is dropped.
SOLVENT = {"HOH", "DOD", "GOL", "EDO", "PEG", "PG4", "MPD", "DMS", "SO4", "PO4", "ACT",
           "CL", "NA", "K", "MG", "MN", "ZN", "CA", "IOD", "BR", "NO3", "FMT", "TRS"}


# --- network -----------------------------------------------------------------------

def fetch(url: str, path: Path, refresh: bool = False) -> Path:
    """Download unless already there. Structures are immutable once released, so an
    existing file is not re-fetched; `refresh` forces it."""
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- pure logic: PDB text ----------------------------------------------------------

def parse_atoms(text: str) -> list[dict]:
    """ATOM/HETATM records as dicts, by PDB column positions. Only the fields this
    script needs; coordinates as floats so distances can be computed without numpy
    gymnastics on strings."""
    out = []
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        out.append({
            "record": line[:6].strip(), "name": line[12:16].strip(),
            "altloc": line[16], "resname": line[17:20].strip(), "chain": line[21],
            "resseq": line[22:26].strip(), "icode": line[26].strip(),
            "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            "element": line[76:78].strip() if len(line) >= 78 else "",
            "line": line})
    return out


# A modified amino acid arrives as HETATM but carries a protein backbone, so it is part
# of the chain: MSE (selenomethionine), TPO/SEP/PTR (phospho-Thr/Ser/Tyr) and the like.
# Tested as a rule rather than a name list, and Rosetta reads them natively (checked:
# 3E5A chain A with TPO 288 builds a 265-residue pose, 264 without it).
BACKBONE = {"N", "CA", "C"}


def is_modified_residue(residue_atoms: list[dict]) -> bool:
    """True when a HETATM residue has a protein backbone, i.e. dropping it would break
    the chain rather than remove a bystander."""
    return BACKBONE <= {a["name"] for a in residue_atoms}


def _heavy(atoms: list[dict]) -> list[dict]:
    return [a for a in atoms if a["element"] != "H" and not a["name"].startswith("H")]


def ligand_copies(atoms: list[dict], code: str, chain: str) -> dict:
    """{(resseq, icode): [atoms]} for every copy of `code` on `chain` (altloc blank or A)."""
    copies: dict = {}
    for a in atoms:
        if a["record"] == "HETATM" and a["resname"] == code and a["chain"] == chain \
                and a["altloc"] in " A":
            copies.setdefault((a["resseq"], a["icode"]), []).append(a)
    return copies


def pocket_contacts(copy_atoms: list[dict], pocket_atoms: list[dict],
                    cutoff: float = DEFAULT_LIGAND_CUTOFF) -> int:
    """How many distinct pocket residues have a heavy atom within `cutoff` of this
    ligand copy's heavy atoms -- frustx's own ligand contact rule (minimum heavy-atom
    distance, `DEFAULT_LIGAND_CUTOFF`), so the copy chosen here is the copy frustx would
    find contacts for."""
    lig = _heavy(copy_atoms)
    c2 = cutoff * cutoff
    hit = set()
    for p in _heavy(pocket_atoms):
        for l in lig:
            if ((p["x"] - l["x"]) ** 2 + (p["y"] - l["y"]) ** 2
                    + (p["z"] - l["z"]) ** 2) <= c2:
                hit.add((p["resseq"], p["icode"]))
                break
    return len(hit)


def choose_copy(copies: dict, pocket_atoms: list[dict],
                cutoff: float = DEFAULT_LIGAND_CUTOFF) -> tuple[tuple, list[dict]]:
    """The copy in the pocket: most pocket residues contacted, ties by residue number.
    Returns (chosen key, per-copy counts) -- the counts go in the report so the choice
    is visible rather than implied."""
    scored = [{"resseq": k[0], "icode": k[1], "pocket_contacts":
               pocket_contacts(v, pocket_atoms, cutoff), "heavy_atoms": len(_heavy(v))}
              for k, v in copies.items()]
    scored.sort(key=lambda s: (-s["pocket_contacts"], int(s["resseq"])))
    best = scored[0]
    return (best["resseq"], best["icode"]), scored


def clean_complex(atoms: list[dict], chain: str, code: str, copy_key: tuple,
                  keep_modified: bool = True) -> tuple[list[str], dict]:
    """One protein chain plus one ligand copy, as PDB lines.

    Kept: ATOM records of `chain`; the chosen `code` copy; and, unless `keep_modified` is
    off, HETATM residues on `chain` that carry a protein backbone (see
    is_modified_residue) -- dropping those would leave a hole in the chain. Altloc blank
    or A throughout.

    What leaves is counted in four separate buckets, because they mean different things:
    other chains (expected -- 2HYY has four), solvent (expected), other copies of the
    same ligand (expected -- the copy rule just rejected them), and other heteroatoms on
    the KEPT chain, which is the only bucket that can hide something that matters.
    """
    by_res: dict = {}
    for a in atoms:
        by_res.setdefault((a["record"], a["chain"], a["resseq"], a["icode"],
                           a["resname"]), []).append(a)

    kept, alt_dropped = [], 0
    other_chains, solvent, other_copies, other_het, modified = (Counter(), Counter(),
                                                                Counter(), Counter(), {})
    for (record, ch, resseq, icode, resname), res_atoms in by_res.items():
        usable = [a for a in res_atoms if a["altloc"] in " A"]
        alt_dropped += len(res_atoms) - len(usable)
        if ch != chain:
            other_chains[f"{ch}:{resname}"] += len(res_atoms)
            continue
        if record == "ATOM":
            kept += usable
        elif resname == code and (resseq, icode) == copy_key:
            kept += usable
        elif resname == code:
            other_copies[f"{resname} {resseq}"] += len(res_atoms)
        elif keep_modified and is_modified_residue(res_atoms):
            kept += usable
            modified[f"{resname} {resseq}"] = len(res_atoms)
        elif resname in SOLVENT:
            solvent[resname] += len(res_atoms)
        else:
            other_het[f"{resname} {resseq}"] += len(res_atoms)

    kept.sort(key=lambda a: (int(a["resseq"]), a["icode"], a["name"]))
    lines = [a["line"] for a in kept]
    ligand = [a for a in kept if a["resname"] == code]
    protein = [a for a in kept if a["resname"] != code]
    stats = {
        "protein_atoms": len(protein), "ligand_atoms": len(ligand),
        "ligand_heavy_atoms": len(_heavy(ligand)),
        "ca_count": sum(1 for a in protein if a["name"] == "CA"),
        "altloc_atoms_dropped": alt_dropped,
        "modified_residues_kept": modified,
        "dropped_other_chains": dict(other_chains.most_common(8)),
        "dropped_other_chain_atoms": sum(other_chains.values()),
        "dropped_solvent": dict(solvent),
        "dropped_other_ligand_copies": dict(other_copies),
        "dropped_het_on_kept_chain": dict(other_het),
        "insertion_codes": sorted({a["icode"] for a in protein if a["icode"]}),
    }
    return lines, stats


def sdf_atom_counts(text: str) -> dict:
    """(heavy, hydrogen) atom counts of an MDL molfile. frustx refuses a ligand without
    hydrogens, so this is checked before a run rather than after it fails."""
    lines = text.splitlines()
    n = int(lines[3][:3])
    elements = Counter(lines[4 + i][31:34].strip().upper() for i in range(n))
    h = elements.pop("H", 0)
    return {"atoms": n, "hydrogens": h, "heavy_atoms": n - h,
            "elements": dict(elements)}


def check_pocket(kept_lines: list[str], expected: list[str]) -> dict:
    """Are the pocket positions still in the cleaned file? `expected` is the list of
    X-ray residue numbers from `pilot_pocket_map.csv` that KLIFS resolved. A position
    missing here is a hole in the position-resolved analysis, so it is counted, not
    assumed absent."""
    present = {l[22:26].strip() for l in kept_lines if l.startswith("ATOM  ")}
    missing = [r for r in expected if r not in present]
    return {"pocket_expected": len(expected), "pocket_present": len(expected) - len(missing),
            "pocket_missing": missing}


# --- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ana = Path("data/kinome/analysis")
    p.add_argument("--complexes", type=Path, default=ana / "pilot_complexes.csv")
    p.add_argument("--pocket-map", type=Path, default=ana / "pilot_pocket_map.csv")
    p.add_argument("--out", type=Path, default=Path("data/kinome/pilot"))
    p.add_argument("--ligand-cutoff", type=float, default=DEFAULT_LIGAND_CUTOFF)
    p.add_argument("--drop-modified", action="store_true",
                   help="drop modified amino acids (TPO, SEP, MSE...) instead of keeping "
                        "them; leaves a gap in the chain, so off by default")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    raw_dir, clean_dir = a.out / "raw", a.out / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    cx = pd.read_csv(a.complexes, keep_default_na=False, dtype=str)
    pmap = pd.read_csv(a.pocket_map, keep_default_na=False, dtype=str)

    runs, report_rows, problems = [], [], []
    for c in cx.itertuples():
        code, kin, pdb, chain = c.ligand_group, c.kinase_name, c.pdb, c.chain
        struct = fetch(RCSB_STRUCTURE.format(pdb), raw_dir / f"{pdb}.pdb", a.refresh)
        sdf_raw = fetch(RCSB_LIGAND.format(code), raw_dir / f"{code}_ideal.sdf", a.refresh)
        atoms = parse_atoms(struct.read_text())

        # the pocket residues KLIFS resolved for THIS structure
        pm = pmap[(pmap["ligand_group"] == code) & (pmap["kinase_name"] == kin)
                  & (pmap["resolved"] == "True")]
        pocket_resseq = pm["xray_residue"].tolist()
        pocket_atoms = [x for x in atoms
                        if x["record"] == "ATOM" and x["chain"] == chain
                        and x["resseq"] in set(pocket_resseq)]

        copies = ligand_copies(atoms, code, chain)
        if not copies:
            problems.append({"complex": f"{code}/{kin}", "pdb": pdb,
                             "problem": f"no {code} copy on chain {chain}"})
            continue
        copy_key, copy_scores = choose_copy(copies, pocket_atoms, a.ligand_cutoff)
        lines, stats = clean_complex(atoms, chain, code, copy_key,
                                     keep_modified=not a.drop_modified)
        pocket_check = check_pocket(lines, pocket_resseq)
        sdf_counts = sdf_atom_counts(sdf_raw.read_text())

        out_pdb = clean_dir / f"{code}_{kin}.pdb"
        out_pdb.write_text("\n".join(lines) + "\nEND\n")
        out_sdf = clean_dir / f"{code}.sdf"
        if not out_sdf.exists():
            out_sdf.write_text(sdf_raw.read_text())

        # the heavy-atom count in the crystal must match the chemistry we hand frustx
        heavy_match = stats["ligand_heavy_atoms"] == sdf_counts["heavy_atoms"]
        if not heavy_match:
            problems.append({
                "complex": f"{code}/{kin}", "pdb": pdb,
                "problem": f"ligand heavy atoms {stats['ligand_heavy_atoms']} in the "
                           f"crystal vs {sdf_counts['heavy_atoms']} in the SDF"})
        if sdf_counts["hydrogens"] == 0:
            problems.append({"complex": f"{code}/{kin}", "pdb": pdb,
                             "problem": "SDF has no hydrogens; frustx will refuse it"})
        if pocket_check["pocket_missing"]:
            problems.append({
                "complex": f"{code}/{kin}", "pdb": pdb,
                "problem": f"{len(pocket_check['pocket_missing'])} pocket residues absent "
                           f"from the cleaned chain: {pocket_check['pocket_missing']}"})
        if stats["dropped_het_on_kept_chain"]:
            problems.append({
                "complex": f"{code}/{kin}", "pdb": pdb,
                "problem": "heteroatoms dropped from the KEPT chain: "
                           f"{stats['dropped_het_on_kept_chain']}"})

        runs.append({
            "ligand_group": code, "kinase_name": kin, "pair": c.pairs, "role": c.roles,
            "pdb": pdb, "chain": chain, "ligand_resseq": copy_key[0],
            "structure": str(out_pdb), "ligand_sdf": str(out_sdf),
            "frustx_command":
                f"frustx {out_pdb} -o results/pilot/{code}_{kin} "
                f"--ligand {out_sdf} --ligand-name3 {code}"})
        report_rows.append({
            "complex": f"{code}/{kin}", "pdb": pdb, "chain": chain,
            "ligand_copies_on_chain": len(copies), "copy_chosen": copy_key[0],
            "copy_scores": copy_scores, **stats, **pocket_check,
            "sdf": sdf_counts, "heavy_atoms_match": heavy_match,
            "structure_md5": md5(struct)})

    runs = pd.DataFrame(runs)
    runs.to_csv(a.out / "pilot_runs.csv", index=False)
    rep = pd.DataFrame(report_rows)
    (a.out / "fetch_pilot.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"structure": RCSB_STRUCTURE, "ligand": RCSB_LIGAND},
        "inputs": {"complexes": str(a.complexes), "pocket_map": str(a.pocket_map)},
        "settings": {"ligand_cutoff": a.ligand_cutoff,
                     "altloc_kept": [" ", "A"],
                     "ligand_from": "_ideal.sdf used with --ligand (chemistry only; the "
                                    "pose comes from the structure's HETATM block)",
                     "modified_residues": "dropped" if a.drop_modified else
                                          "kept (they carry a protein backbone)"},
        "complexes": report_rows,
        "problems": problems,
    }, indent=2, default=str) + "\n")

    pd.set_option("display.width", 220)
    print(f"\n{len(runs)} / {len(cx)} complexes prepared -> {clean_dir}")
    print(rep[["complex", "pdb", "chain", "ligand_copies_on_chain", "copy_chosen",
               "ca_count", "ligand_heavy_atoms", "heavy_atoms_match", "pocket_present",
               "pocket_expected", "altloc_atoms_dropped"]].to_string(index=False))
    print("\nligand chemistry:")
    for r in report_rows:
        if r["complex"].split("/")[1] in ("ABL1", "EphB4", "AurA"):   # one per ligand
            print(f"   {r['complex'].split('/')[0]:4} {r['sdf']}")
    print("\nwhat left each file:")
    for r in report_rows:
        print(f"   {r['complex']:14} other chains {r['dropped_other_chain_atoms']:5} at | "
              f"solvent {sum(r['dropped_solvent'].values()):4} at "
              f"{sorted(r['dropped_solvent'])} | "
              f"other ligand copies {r['dropped_other_ligand_copies'] or '-'}"
              + (f" | HET ON KEPT CHAIN {r['dropped_het_on_kept_chain']}"
                 if r["dropped_het_on_kept_chain"] else ""))
    kept_mod = {r["complex"]: r["modified_residues_kept"] for r in report_rows
                if r["modified_residues_kept"]}
    print("\nmodified amino acids kept (HETATM with a backbone):",
          kept_mod or "none")
    if problems:
        print(f"\n{len(problems)} PROBLEM:")
        for q in problems:
            print("   ", q)
    else:
        print("\nno problems: every ligand copy found, heavy-atom counts match the SDFs, "
              "all pocket residues present")
    print(f"\nwrote {a.out}/pilot_runs.csv, fetch_pilot.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
