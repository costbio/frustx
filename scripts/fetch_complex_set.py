"""Download and clean the complexes of DRUG_KINASE_COMPLEXES.csv, ready for frustx.

The generation-2 counterpart of scripts/fetch_pilot.py, for the KLIFS-first complex set
(scripts/klifs_complex_set.py). Every step that decides something is fetch_pilot.py's
own code -- which chain, which ligand copy, what is dropped, what is checked -- so the
pilot's structures and these are prepared the same way. What is new is the input, an
RCSB check in front, and running it over ~100 complexes instead of 10.

    .venv/bin/python scripts/fetch_complex_set.py          # complexes in the Kd or IC50 set
    .venv/bin/python scripts/fetch_complex_set.py --all    # every row of the table

Input:
    data/kinome/DRUG_KINASE_COMPLEXES.csv

Output, under data/kinome/complex_set/:
    structures/raw/<PDB>.pdb, <CODE>_ideal.sdf   as downloaded
    structures/clean/<CODE>_<KINASE>.pdb         one chain + one ligand copy
    structures/clean/<CODE>.sdf                  the ligand's chemistry
    pocket_map.csv     KLIFS's 85 pocket positions on each structure's own residue
                       numbers: the common frame for comparing one ligand's complexes
                       position by position across kinases
    runs.csv           one row per complex: files, checks, `ready`, the frustx command
    fetch.json         settings, per-complex report, every problem

Steps, per complex:

  1. RCSB, BEFORE downloading: klifs_manifest.py's fetch_rcsb + rcsb_annotate, the same
     statuses the manifest uses (obsolete, ligand_missing, non_xray). Only `ok` rows are
     downloaded. klifs_complex_set.py deliberately left this to here.
  2. Download the entry (PDB format) and RCSB's `_ideal.sdf` for the ligand. The SDF is
     used with `--ligand`: chemistry only, the pose is the crystal's HETATM block (see
     fetch_pilot.py for why that is the right mode for co-crystal poses).
  3. Pocket map from KLIFS (interactions_match_residues, pilot_paralogs.py's code).
  4. Clean: the KLIFS chain; the ligand copy with most pocket contacts if the chain
     holds several; altloc blank or A; modified residues with a backbone kept; waters
     and every other heteroatom dropped and counted. Then the crystallisation artefacts
     Rosetta cannot read (CME, OCS -> CYS; CXM -> MET; see ARTEFACT_PARENT) are written
     back as their parent residue, so they do not vanish from the pose.
  5. Check, and say so rather than patch: ligand heavy atoms in the crystal == in the
     SDF; SDF has hydrogens; pocket residues KLIFS resolved are present in the cleaned
     chain; nothing but solvent dropped near the ligand; no pocket residue missing its
     backbone (Rosetta drops it); no kept modified residue outside ROSETTA_READS.

`ready` in runs.csv is False when a check that would make the frustx result wrong or
impossible failed (no ligand copy, heavy-atom mismatch, no hydrogens). Reported but not
blocking: pocket residues missing from the file (already carried as `missing_residues`
in the table); no KLIFS pocket map for the structure (frustx does not need it, a
position-resolved comparison does); and a heteroatom dropped from the kept chain within
the ligand cutoff of the ligand -- a crystallisation additive in the pocket, or worse a
cofactor, whose absence changes the contacts. Every dropped heteroatom's distance to the
ligand is in fetch.json, near or not.
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_pilot import (RCSB_LIGAND, RCSB_STRUCTURE, _heavy,  # noqa: E402
                         check_pocket, choose_copy, clean_complex, fetch, ligand_copies, md5,
                         parse_atoms, sdf_atom_counts)
from klifs_manifest import fetch_rcsb, rcsb_annotate  # noqa: E402
from pilot_paralogs import fetch_pocket_maps, pocket_map_table  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frustx.config import DEFAULT_LIGAND_CUTOFF  # noqa: E402

# problems that make a frustx run wrong or impossible; everything else is reported only
BLOCKING = ("no_ligand_copy", "heavy_atom_mismatch", "sdf_without_hydrogens")

# Crystallisation artefacts that Rosetta does not read: frustx initialises with
# -ignore_unrecognized_res, so they vanish from the pose WITHOUT a message (checked by
# loading the cleaned files with frustx's init flags: PIM1 1YHS CME 161 -- a pocket
# position, alphaE.64 -- BRK 5H2U CXM 184 and CDK2 4ERW OCS 177 were all missing from the
# pose). Each is the parent residue plus atoms added during purification/crystallisation:
#   CME  S,S-(2-hydroxyethyl)thiocysteine  beta-mercaptoethanol adduct on Cys
#   OCS  cysteine sulfonic acid             oxidised Cys
#   CXM  N-carboxymethionine                carbamylated N-terminal Met
# so the parent is recovered by renaming the residue and keeping only the parent's atoms
# (all present in every case seen). Decided 2026-10-24; --keep-artefacts turns it off.
ARTEFACT_PARENT = {"CME": "CYS", "OCS": "CYS", "CXM": "MET"}
PARENT_ATOMS = {"CYS": {"N", "CA", "C", "O", "OXT", "CB", "SG"},
                "MET": {"N", "CA", "C", "O", "OXT", "CB", "CG", "SD", "CE"}}

# Modified residues verified to load under frustx's init flags (same check): TPO, SEP and
# PTR become phosphorylated variants, MSE loads. Any other modified residue that
# clean_complex keeps is flagged, since it may be dropped just as silently.
ROSETTA_READS = {"TPO", "SEP", "PTR", "MSE"}


# --- pure logic (tested offline in tests/test_fetch_complex_set.py) -------------------

def select_complexes(table: pd.DataFrame, all_rows: bool) -> pd.DataFrame:
    """Rows to prepare: by default those used by at least one analysis set."""
    if all_rows:
        return table.copy()
    sets = [c for c in table.columns if c.startswith("in_") and c.endswith("_set")]
    use = table[sets].astype(str).eq("True").any(axis=1)
    return table[use].copy()


def complex_name(ligand_code: str, kinase_name: str) -> str:
    """File stem. KLIFS kinase names are alphanumeric plus '-' today (JAK2-b); anything
    else is replaced so a future name cannot write outside the directory."""
    return re.sub(r"[^A-Za-z0-9-]", "_", f"{ligand_code}_{kinase_name}")


def dropped_het_distances(atoms: list[dict], chain: str, ligand_atoms: list[dict],
                          dropped: dict) -> dict:
    """{"RES resseq": min heavy-atom distance to the kept ligand copy, in A} for every
    heteroatom clean_complex dropped from the KEPT chain. Distance is what separates a
    harmless additive (BOG detergent 27 A away in p38a 3LFA) from one sitting in the
    pocket (malonate 3.4 A from silmitasertib in GSK3B 7Z1G) or a second inhibitor
    (imatinib 22 A from dasatinib in ABL1 7N9G)."""
    lig = [(a["x"], a["y"], a["z"]) for a in _heavy(ligand_atoms)]
    out = {}
    for key in dropped:
        resname, resseq = key.split(" ")
        het = [(a["x"], a["y"], a["z"]) for a in _heavy(atoms)
               if a["record"] == "HETATM" and a["chain"] == chain
               and a["resname"] == resname and a["resseq"] == resseq]
        out[key] = round(min(math.dist(h, l) for h in het for l in lig), 1)
    return out


def revert_artefacts(lines: list[str], parent: dict[str, str]) -> tuple[list[str], dict]:
    """Rewrite each artefact residue (ARTEFACT_PARENT) as its parent amino acid: ATOM
    record, parent residue name, parent atoms only. Returns (lines, {"CME 161": "CYS"}).
    Column layout is fixed-width PDB: record 1-6, residue name 18-20."""
    out, reverted = [], {}
    for line in lines:
        resname = line[17:20].strip()
        if line.startswith("HETATM") and resname in parent:
            p = parent[resname]
            reverted[f"{resname} {line[22:26].strip()}"] = p
            if line[12:16].strip() not in PARENT_ATOMS[p]:
                continue                       # an atom the artefact added
            line = "ATOM  " + line[6:17] + f"{p:>3}" + line[20:]
        out.append(line)
    return out, reverted


def backbone_incomplete(lines: list[str], code: str) -> list[str]:
    """Residue numbers (not the ligand) lacking N, CA or C. Rosetta drops such a residue
    from the pose -- C-terminal stubs (LCK 1QPD GLU 502, PDK1 1OKY ALA 360: only N) are
    harmless, but ULK3 6FDY ALA 26 (only C, O) is a pocket position (g.l.9).

    An atom with occupancy 0.00 counts as absent: Rosetta ignores those by default
    (-ignore_zero_occupancy), which is how CSK 1BYG GLY 187 vanished from the pose with
    all four backbone atoms in the file. Zero-occupancy SIDE-CHAIN atoms are not checked:
    frustx's repack rebuilds them."""
    names: dict = {}
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() != code:
            if float(line[54:60].strip() or 1) == 0.0:
                continue
            names.setdefault(line[22:27].strip(), set()).add(line[12:16].strip())
    return [r for r, n in names.items() if not {"N", "CA", "C"} <= n]


def prepare_complex(atoms: list[dict], chain: str, code: str, pocket_resseq: list[str],
                    sdf_text: str, cutoff: float = DEFAULT_LIGAND_CUTOFF,
                    artefact_parent: dict[str, str] = ARTEFACT_PARENT) -> dict:
    """Clean one complex and check it. Returns {lines, report, problems}; `lines` is
    None when there is no ligand copy to keep. No I/O, so it is testable on text."""
    pocket_atoms = [x for x in atoms if x["record"] == "ATOM" and x["chain"] == chain
                    and x["resseq"] in set(pocket_resseq)]
    copies = ligand_copies(atoms, code, chain)
    if not copies:
        return {"lines": None, "report": {"ligand_copies_on_chain": 0},
                "problems": [("no_ligand_copy", f"no {code} copy on chain {chain}")]}
    copy_key, copy_scores = choose_copy(copies, pocket_atoms, cutoff)
    lines, stats = clean_complex(atoms, chain, code, copy_key)
    lines, reverted = revert_artefacts(lines, artefact_parent)
    unverified = sorted(k for k in stats["modified_residues_kept"]
                        if k not in reverted and k.split(" ")[0] not in ROSETTA_READS)
    no_backbone = backbone_incomplete(lines, code)
    no_backbone_pocket = [r for r in no_backbone if r in set(pocket_resseq)]
    pocket = check_pocket(lines, pocket_resseq, ligand_code=code)
    sdf = sdf_atom_counts(sdf_text)
    het_dist = dropped_het_distances(atoms, chain, copies[copy_key],
                                     stats["dropped_het_on_kept_chain"])
    near = {k: d for k, d in het_dist.items() if d <= cutoff}

    problems = []
    if not pocket_resseq:
        problems.append(("pocket_map_unavailable",
                         "KLIFS placed no pocket position on this structure; fine for "
                         "frustx, missing for a position-resolved comparison"))
    if stats["ligand_heavy_atoms"] != sdf["heavy_atoms"]:
        problems.append(("heavy_atom_mismatch",
                         f"{stats['ligand_heavy_atoms']} heavy atoms in the crystal vs "
                         f"{sdf['heavy_atoms']} in the SDF"))
    if sdf["hydrogens"] == 0:
        problems.append(("sdf_without_hydrogens", "frustx will refuse this ligand"))
    if pocket["pocket_missing"]:
        problems.append(("pocket_residues_missing",
                         f"{len(pocket['pocket_missing'])} KLIFS-resolved pocket residues "
                         f"absent: {pocket['pocket_missing']}"))
    if near:
        problems.append(("het_dropped_near_ligand",
                         f"within {cutoff} A of the ligand: {near}"))
    if no_backbone_pocket:
        problems.append(("backbone_incomplete_in_pocket",
                         f"pocket residues without N/CA/C, Rosetta will drop them: "
                         f"{no_backbone_pocket}"))
    if unverified:
        problems.append(("modified_residue_unverified",
                         f"not known to load under frustx's flags, may be dropped "
                         f"silently: {unverified}"))
    report = {"ligand_copies_on_chain": len(copies), "copy_chosen": copy_key[0],
              "copy_scores": copy_scores, **stats, **pocket, "sdf": sdf,
              "dropped_het_min_distance_to_ligand": het_dist,
              "artefacts_reverted": reverted, "backbone_incomplete": no_backbone}
    return {"lines": lines, "report": report, "problems": problems}


# --- driver ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", type=Path, default=Path("data/kinome/DRUG_KINASE_COMPLEXES.csv"))
    p.add_argument("--out", type=Path, default=Path("data/kinome/complex_set"))
    p.add_argument("--results", type=Path, default=Path("results/complex_set"),
                   help="where the frustx commands written to runs.csv put their output")
    p.add_argument("--all", action="store_true",
                   help="prepare every row of the table, not only the Kd/IC50 sets")
    p.add_argument("--ligand-cutoff", type=float, default=DEFAULT_LIGAND_CUTOFF)
    p.add_argument("--keep-artefacts", action="store_true",
                   help="keep CME/OCS/CXM as they are instead of reverting them to "
                        "CYS/MET (Rosetta will then drop them from the pose)")
    p.add_argument("--refresh", action="store_true", help="re-download structures")
    a = p.parse_args(argv)
    raw_dir, clean_dir = a.out / "structures" / "raw", a.out / "structures" / "clean"
    cache = a.out / "cache"
    clean_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(a.table, keep_default_na=False, dtype=str)
    cx = select_complexes(table, a.all)
    print(f"{len(cx)} complexes, {cx['pdb'].nunique()} PDB entries, "
          f"{cx['ligand_code'].nunique()} ligands", file=sys.stderr)

    # 1. RCSB first: nothing is downloaded for an entry that would be refused anyway
    rcsb = fetch_rcsb(sorted(cx["pdb"].unique()), cache / "rcsb_cache.json")
    cx = rcsb_annotate(cx.assign(pdb_klifs=cx["pdb"]), rcsb, conflicts=set())

    # 3. pocket maps (pilot_paralogs.py's column names)
    maps = fetch_pocket_maps(cx["klifs_structure_id"].tolist(),
                             cache / "pocket_map_cache.json")
    pmap, label_problems = pocket_map_table(
        maps, cx.rename(columns={"ligand_code": "ligand_group"}))
    pmap.to_csv(a.out / "pocket_map.csv", index=False)

    runs, reports = [], []
    for c in cx.itertuples():
        name = complex_name(c.ligand_code, c.kinase_name)
        row = {"complex": name, "ligand_code": c.ligand_code, "kinase_name": c.kinase_name,
               "pdb": c.pdb, "chain": c.chain, "rcsb_status": c.rcsb_status,
               "in_Kd_set": c.in_Kd_set, "in_IC50_set": c.in_IC50_set}
        if c.rcsb_status != "ok":
            runs.append({**row, "ready": False, "problems": f"rcsb_status={c.rcsb_status}"})
            continue

        # 2. download
        struct = fetch(RCSB_STRUCTURE.format(c.pdb), raw_dir / f"{c.pdb}.pdb", a.refresh)
        sdf_raw = fetch(RCSB_LIGAND.format(c.ligand_code),
                        raw_dir / f"{c.ligand_code}_ideal.sdf", a.refresh)

        # 4-5. clean and check
        pm = pmap[(pmap["klifs_structure_id"].astype(str) == str(c.klifs_structure_id))
                  & pmap["resolved"]]
        res = prepare_complex(parse_atoms(struct.read_text()), c.chain, c.ligand_code,
                              pm["xray_residue"].tolist(), sdf_raw.read_text(),
                              a.ligand_cutoff,
                              artefact_parent={} if a.keep_artefacts else ARTEFACT_PARENT)
        kinds = [k for k, _ in res["problems"]]
        ready = res["lines"] is not None and not any(k in BLOCKING for k in kinds)

        out_pdb, out_sdf = clean_dir / f"{name}.pdb", clean_dir / f"{c.ligand_code}.sdf"
        if res["lines"] is not None:
            out_pdb.write_text("\n".join(res["lines"]) + "\nEND\n")
            if not out_sdf.exists():
                out_sdf.write_text(sdf_raw.read_text())
        rep = res["report"]
        runs.append({
            **row, "ready": ready, "problems": "; ".join(kinds),
            "ligand_resseq": rep.get("copy_chosen"),
            "ligand_copies_on_chain": rep.get("ligand_copies_on_chain"),
            "ca_count": rep.get("ca_count"),
            "ligand_heavy_atoms": rep.get("ligand_heavy_atoms"),
            "pocket_present": rep.get("pocket_present"),
            "pocket_expected": rep.get("pocket_expected"),
            "structure": str(out_pdb) if res["lines"] is not None else "",
            "ligand_sdf": str(out_sdf),
            "frustx_command": (f"frustx {out_pdb} -o {a.results / name} "
                               f"--ligand {out_sdf} --ligand-name3 {c.ligand_code}"
                               if ready else "")})
        reports.append({"complex": name, "pdb": c.pdb, "chain": c.chain,
                        "structure_md5": md5(struct),
                        "problems": [{"kind": k, "detail": d} for k, d in res["problems"]],
                        **rep})

    runs = pd.DataFrame(runs)
    runs.to_csv(a.out / "runs.csv", index=False)
    (a.out / "fetch.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"table": str(a.table), "all_rows": a.all},
        "sources": {"structure": RCSB_STRUCTURE, "ligand": RCSB_LIGAND,
                    "rcsb_fetched_at": rcsb.get("fetched_at")},
        "settings": {"ligand_cutoff": a.ligand_cutoff, "altloc_kept": [" ", "A"],
                     "ligand_from": "_ideal.sdf with --ligand (chemistry only; pose from "
                                    "the crystal's HETATM block)",
                     "modified_residues": "kept (they carry a protein backbone)",
                     "artefacts_reverted_to_parent": {} if a.keep_artefacts
                                                     else ARTEFACT_PARENT,
                     "rosetta_reads_modified": sorted(ROSETTA_READS),
                     "blocking_problems": list(BLOCKING)},
        "rcsb_status": cx["rcsb_status"].value_counts().to_dict(),
        "pocket_label_problems": label_problems,
        "complexes": reports,
    }, indent=2, default=str) + "\n")

    pd.set_option("display.width", 200)
    print(runs[["complex", "pdb", "chain", "rcsb_status", "ready", "ligand_copies_on_chain",
                "ca_count", "pocket_present", "pocket_expected", "problems"]]
          .to_string(index=False), file=sys.stderr)
    print(f"\n{int(runs['ready'].sum())} / {len(runs)} ready -> {clean_dir}", file=sys.stderr)
    kinds = runs["problems"].str.split("; ").explode()
    print("problems:", kinds[kinds.astype(bool)].value_counts().to_dict(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
