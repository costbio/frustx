"""End-to-end check of novel-ligand parametrisation on the EGFR inhibitor set.

Four EGFR kinase-domain complexes, each with a drug-like inhibitor that does NOT ship in
Rosetta's fa_standard residue set. None of them can be scored without parametrisation, and
all four go through `core::chemical::sdf` rather than molfile_to_params.

This lives in scripts/ rather than tests/ because it needs ~1.2 MB of PDBs from RCSB,
which CLAUDE.md keeps out of git. Its output goes to results/egfr_ligand/ so that it does
not have to be recomputed from scratch next session -- the scratchpad is wiped, results/
is DVC-tracked.

    .venv/bin/python scripts/egfr_ligand_check.py --out results/egfr_ligand

What it checks, in order of how much it would hurt to get wrong:

 1. The ligand loads with the coordinates the structure actually contains (not rebuilt
    from internal coordinates, not silently truncated).
 2. FrustX's contact count agrees with an independent numpy calculation done straight from
    the PDB text. This is the check that catches a ligand that loaded with missing atoms.
 3. The ligand does not move during decoy generation -- with the freeze off as a control,
    because a displacement of zero proves nothing unless a nonzero one is achievable.
 4. Serial and parallel runs are bit-identical with a ligand present, which is a repo-wide
    invariant and the one most easily broken by a residue type that lives in the pose.
 5. A real frustration calculation completes and produces finite indices for
    protein-ligand contacts.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# The EGFR set. Each inhibitor is a novel ligand as far as Rosetta is concerned; the
# `code` is its PDB chemical component id, which is also the HETATM residue name.
CASES = [
    ("1M17", "AQ4", "erlotinib"),
    ("2ITY", "IRE", "gefitinib"),
    ("1XKK", "FMM", "lapatinib"),
    ("3POZ", "03P", "TAK-285"),
]

RCSB_STRUCTURE = "https://files.rcsb.org/download/{}.pdb"
RCSB_LIGAND = "https://files.rcsb.org/ligands/download/{}_ideal.sdf"


def fetch(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def clean_structure(raw, out, code):
    """Keep one protein chain plus the ligand: drop waters, other heteroatoms, altloc B.

    Waters are dropped because Rosetta types them as ligands (`is_ligand()` is True for
    water -- this project already learned that the hard way) and they would become
    contact nodes. Alternate conformers are dropped to altloc A because Rosetta keeps only
    the first, and a comparison against raw PDB text that pools both altlocs disagrees:
    CYS A 751 in 1M17 sits at 6.017 A in conformer A and 5.267 A in conformer B, straddling
    the 6.0 A ligand cutoff.
    """
    chain = None
    kept = []
    for line in raw.read_text().splitlines():
        if line.startswith("ATOM"):
            if chain is None:
                chain = line[21]
            if line[21] != chain or line[16] not in " A":
                continue
            kept.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() == code:
            if line[16] not in " A":
                continue
            kept.append(line)
    out.write_text("\n".join(kept) + "\nEND\n")
    return out


def independent_contact_count(structure, code, ligand_cutoff):
    """Protein residues within `ligand_cutoff` of any ligand heavy atom, from raw text.

    Deliberately reimplemented in numpy from the PDB file rather than reusing anything in
    frustx, because its whole purpose is to be an independent witness.
    """
    ligand = []
    protein = {}
    for line in structure.read_text().splitlines():
        if len(line) < 78 or line[76:78].strip() == "H":
            continue
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if line.startswith("HETATM") and line[17:20].strip() == code:
            ligand.append(xyz)
        elif line.startswith("ATOM"):
            protein.setdefault((line[21], int(line[22:26])), []).append(xyz)
    ligand = np.asarray(ligand)
    n = 0
    for atoms in protein.values():
        d = np.linalg.norm(np.asarray(atoms)[:, None, :] - ligand[None, :, :], axis=-1)
        if d.min() <= ligand_cutoff:
            n += 1
    return n, len(ligand)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/egfr_ligand"))
    parser.add_argument("--decoys", type=int, default=25,
                        help="decoys for the frustration run; 25 is enough to show finite "
                             "indices, not enough for a publishable number")
    parser.add_argument("--protocol", default="min")
    parser.add_argument("--ligand-cutoff", type=float, default=6.0)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(exist_ok=True)

    import pyrosetta
    from frustx.contacts import (min_heavy_distances, pairwise_distances,
                                 select_pairs_by_kind)
    from frustx.energies import init_rosetta, make_score_function
    from frustx.frustration import (compute_frustration, contact_coords_from_pose,
                                    heavy_atom_coords_from_pose, residue_kinds_from_pose,
                                    residues_from_pose)
    from frustx.ligand_params import load_pose_with_ligands, make_spec
    from frustx.output import contact_table
    from frustx import decoys as decoy_mod

    init_rosetta()
    report = []

    for pdb_id, code, drug in CASES:
        started = time.time()
        row = {"pdb": pdb_id, "ligand": code, "drug": drug}
        warnings = []

        structure = clean_structure(
            fetch(RCSB_STRUCTURE.format(pdb_id), raw_dir / f"{pdb_id}.pdb"),
            args.out / f"{pdb_id}_clean.pdb", code)
        sdf = fetch(RCSB_LIGAND.format(code), raw_dir / f"{code}.sdf")

        # `topology` mode: the ligand is already a HETATM block in the structure, and the
        # SDF supplies only its chemistry. The SDF's atom names (C1, C2, ...) do not match
        # the PDB's (C13, O2, ...) -- geometric remapping is what makes this work.
        spec = make_spec(sdf, mode="topology")
        pose, loaded = load_pose_with_ligands(structure, [spec], warn=warnings.append)
        entry = loaded[0]
        row.update(n_residues=pose.total_residue(), ligand_index=entry["pose_index"],
                   chain=entry.get("chain"), resnum=entry.get("resnum"),
                   n_heavy=entry["n_heavy_atoms"], warnings=warnings)

        # --- 1. coordinates came from the structure, atom for atom
        residue = pose.residue(entry["pose_index"])
        posed = np.array([list(residue.xyz(k))
                          for k in range(1, residue.nheavyatoms() + 1)
                          if not residue.is_virtual(k)])
        het = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                        for l in structure.read_text().splitlines()
                        if l.startswith("HETATM") and l[17:20].strip() == code
                        and l[76:78].strip() != "H"])
        # Sorted per axis: the pose's atom ORDER follows the SDF, the PDB's follows the
        # HETATM block, and remapping matched them by geometry rather than by position.
        row["max_coord_deviation"] = float(
            np.abs(np.sort(posed, axis=0) - np.sort(het, axis=0)).max())

        # --- 2. contacts, against an independent witness
        coords, offsets = heavy_atom_coords_from_pose(pose)
        kinds = residue_kinds_from_pose(pose)
        residues = residues_from_pose(pose)
        pairs = select_pairs_by_kind(
            residues,
            pairwise_distances(contact_coords_from_pose(pose, "CA", missing="nan")),
            min_heavy_distances(coords, offsets), kinds,
            cutoff=10.0, ligand_cutoff=args.ligand_cutoff, min_seq_sep=2)
        n_pl = sum(1 for i, j in pairs if not (kinds[i] and kinds[j]))
        naive, n_lig_atoms = independent_contact_count(structure, code, args.ligand_cutoff)
        row.update(n_contacts=len(pairs), n_protein_ligand=n_pl,
                   n_protein_ligand_independent=naive, contacts_agree=(n_pl == naive),
                   n_ligand_heavy_in_pdb=n_lig_atoms)

        # --- 3. the ligand does not move, with a control that CAN move
        sf_pack = make_score_function(remove_fa_rep=False)
        reference = np.array([list(residue.xyz(k))
                              for k in range(1, residue.natoms() + 1)])

        def displacement(other):
            r = other.residue(entry["pose_index"])
            now = np.array([list(r.xyz(k)) for k in range(1, r.natoms() + 1)])
            return float(np.abs(now - reference).max())

        moved = {}
        for freeze in (True, False):
            d = [displacement(decoy_mod.make_decoy(pose, sf_pack, seed=s,
                                                   protocol=args.protocol, repeats=1,
                                                   freeze_ligand=freeze))
                 for s in range(3)]
            d.append(displacement(decoy_mod.native_reference(
                pose, sf_pack, protocol=args.protocol, repeats=1, freeze_ligand=freeze)))
            moved["frozen" if freeze else "unfrozen"] = max(d)
        row["max_displacement_frozen"] = moved["frozen"]
        row["max_displacement_unfrozen"] = moved["unfrozen"]
        row["n_chi"] = residue.type().nchi()
        # The control is only meaningful if the ligand HAS torsions to move. A ligand with
        # nchi == 0 cannot move whatever the setting, so a zero here is not evidence.
        row["freeze_control_is_meaningful"] = (
            row["n_chi"] > 0 and moved["unfrozen"] > 0.1)

        # --- 4 & 5. a real run, serial and parallel, bit-identical
        common = dict(sf_pack=sf_pack,
                      sf_measure=make_score_function(remove_fa_rep=True),
                      n_decoys=args.decoys, seed=0, protocol=args.protocol, repeats=1,
                      cutoff=10.0, min_seq_sep=2, ligand_cutoff=args.ligand_cutoff,
                      freeze_ligand=True,
                      # Bit-identity is only claimable with the packing RNG pinned. By
                      # default FrustX draws fresh entropy per invocation, so serial and
                      # parallel differ for a legitimate reason and comparing them
                      # unpinned tests nothing at all. tests/test_parallel.py:57 pins it
                      # for exactly this reason.
                      packing_seed=4242)
        serial = compute_frustration(pose, n_jobs=1, **common)
        parallel = compute_frustration(pose, n_jobs=args.jobs, **common)

        # nan_to_num FIRST: NaN != NaN, and the index is NaN wherever sigma == 0, which is
        # a property of the ensemble and not of the parallelism. tests/test_parallel.py
        # does the same for the same reason.
        row["serial_equals_parallel"] = all(
            np.array_equal(np.nan_to_num(getattr(serial, f)),
                           np.nan_to_num(getattr(parallel, f)))
            for f in ("native_energy", "decoy_mean", "decoy_std", "index"))

        table = contact_table(serial)
        is_ligand_row = (table["resname_i"] == code) | (table["resname_j"] == code)
        ligand_rows = table[is_ligand_row]
        finite = np.isfinite(ligand_rows["frustration_index"])
        row.update(n_ligand_rows=int(len(ligand_rows)),
                   n_ligand_rows_finite=int(finite.sum()),
                   ligand_index_min=float(ligand_rows["frustration_index"][finite].min()),
                   ligand_index_max=float(ligand_rows["frustration_index"][finite].max()),
                   elapsed_s=round(time.time() - started, 1))
        table.to_csv(args.out / f"{pdb_id}_contacts.csv", index=False)

        report.append(row)
        print(f"{pdb_id} {code} ({drug}): "
              f"dev={row['max_coord_deviation']:.4f} A  "
              f"contacts {row['n_protein_ligand']}=={row['n_protein_ligand_independent']} "
              f"{'OK' if row['contacts_agree'] else 'MISMATCH'}  "
              f"frozen={row['max_displacement_frozen']:.4f} "
              f"unfrozen={row['max_displacement_unfrozen']:.4f}  "
              f"bit-identical={row['serial_equals_parallel']}  "
              f"[{row['elapsed_s']}s]", flush=True)

    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    ok = all(r["contacts_agree"] and r["max_coord_deviation"] < 0.01
             and r["max_displacement_frozen"] == 0.0 and r["serial_equals_parallel"]
             for r in report)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} -> {args.out}/summary.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
