"""Command-line entry point.

    frustx structure.pdb -o results/

Uses stdlib argparse rather than typer/click: this is a handful of flags and the
dependency would not pay for itself.

Every run writes a run.json alongside the results recording the input, all parameters
and the package version. Frustration indices are only comparable between runs made with
the same decoy count and relaxation protocol, so a results directory that cannot say how
it was produced is not much use.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from frustx import __version__
# Only frustx.config and frustx.progress at module scope: both are stdlib-only by
# design, so `frustx --help` and `--version` stay instant. Everything that touches
# PyRosetta is imported in main().
from frustx.config import (DEFAULT_BACKGROUND_WEIGHT, DEFAULT_CUTOFF,
                          DEFAULT_N_DECOYS, DEFAULT_READOUT, READOUT_SCOPES,
                          CONTACT_ATOMS, DEFAULT_CONTACT_ATOM, RELAX_PROTOCOLS,
                          DEFAULT_LIGAND_CUTOFF)
from frustx.progress import Reporter


def build_parser():
    p = argparse.ArgumentParser(
        prog="frustx",
        description="Local frustration in protein structures at atomistic resolution "
                    "(Chen et al., Nat Commun 11:5944, 2020).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("structure", type=Path, help="input PDB file")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output directory (created if absent)")
    p.add_argument("-n", "--decoys", type=int, default=DEFAULT_N_DECOYS,
                   help="number of shuffled decoys")
    p.add_argument("--seed", type=int, default=0,
                   help="base RNG seed for sequence shuffling; decoy k uses seed+k")
    p.add_argument("--protocol", choices=RELAX_PROTOCOLS, default="relax",
                   help="side-chain relaxation: 'relax' is FastRelax (the paper's "
                        "'short Monte-Carlo relaxation'), 'min' is ~10x faster "
                        "gradient minimisation, 'none' skips it")
    p.add_argument("--repeats", type=int, default=1,
                   help="FastRelax repeats; ignored unless --protocol relax")
    p.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF,
                   help="Ca-Ca contact cutoff in Angstrom")
    p.add_argument("--min-seq-sep", type=int, default=1,
                   help="minimum |i-j| within a chain for a pair to count as a contact; "
                        "1 keeps sequential neighbours, which is the paper's literal reading")
    p.add_argument("--background-weight", type=float, default=DEFAULT_BACKGROUND_WEIGHT,
                   help="weight w on the many-body background in "
                        "E_ij = e_ij + w/2 (R_i + R_j). 0 (default) uses the direct pair "
                        "energy alone; 1 reproduces the paper's Eq. 2, which was measured "
                        "to give a degenerate index -- see docs/method.md")
    p.add_argument("--contact-atom", choices=CONTACT_ATOMS, default=DEFAULT_CONTACT_ATOM,
                   help="atom representing a residue in the contact map. 'CA' (default) "
                        "is the paper's definition; 'CB' (Gly falls back to CA) admits "
                        "far fewer pairs with no atomistic interaction -- see config.py")
    p.add_argument("--readout", choices=READOUT_SCOPES, default=DEFAULT_READOUT,
                   help="energy scope Eq. 1 is applied to: 'pair' (default, the bare "
                        "contact energy) or 'neighbourhood' (that contact plus every "
                        "other one touching i or j, matching what frustratometeR sums). "
                        "Diagnostic -- see docs/method.md before using it")
    p.add_argument("--packing-frustration", action="store_true",
                   help="keep the repulsive fa_rep term when measuring e_ij. This is the "
                        "paper's separate 'packing frustration', which diagnoses structure "
                        "quality rather than functional frustration -- NOT the default index")
    p.add_argument("--ligand", action="append", default=[], metavar="FILE",
                   help="parametrise a ligand from an SDF/MOL or a Rosetta .params file. "
                        "The ligand must already be a HETATM block in the structure; this "
                        "file supplies only its chemistry, so its atom names need not "
                        "match (they are remapped geometrically). Repeat for several "
                        "ligands. Rosetta ships parameters for common cofactors (ATP, ZN, "
                        "SAH...) already, and passing a file that would shadow one is "
                        "refused rather than silently preferred")
    p.add_argument("--ligand-placed", action="append", default=[], metavar="FILE",
                   help="as --ligand, but the ligand is NOT in the structure and is "
                        "appended at the coordinates its own file carries. This is the "
                        "post-docking case, where the pose lives in a separate SDF")
    p.add_argument("--ligand-name", action="append", default=[], metavar="NAME",
                   help="override the residue name a ligand file declares, applied to the "
                        "ligand files in order. Needed when two docking hits are both "
                        "called UNL or LIG, which is what Vina, Glide and Boltz emit")
    p.add_argument("--ligand-name3", action="append", default=[], metavar="XXX",
                   help="override the 3-letter code, applied in order. This is the code "
                        "written into contacts.csv and frustration.pdb, and for --ligand "
                        "it must match the HETATM residue name in the structure")
    p.add_argument("--ligand-cutoff", type=float, default=DEFAULT_LIGAND_CUTOFF,
                   help=f"minimum heavy-atom distance defining a contact that involves a "
                        f"ligand, in A (default {DEFAULT_LIGAND_CUTOFF}). A ligand has no "
                        f"CA, so the protein --cutoff cannot apply to it; this default is "
                        f"Rosetta's fa_max_dis, beyond which the pair energy is zero by "
                        f"construction -- see config.py")
    p.add_argument("--no-freeze-ligand", dest="freeze_ligand", action="store_false",
                   help="let ligands move during decoy generation. NOT recommended and "
                        "not what the current decoy model means: decoys shuffle amino "
                        "acid identity only, so a ligand that relaxes into a shuffled "
                        "pocket is answering a different question. Frozen by default")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="decoy-building processes (default 1, serial). Decoys are "
                        "independent, so this scales nearly linearly up to the physical "
                        "core count; measured ~3.6x at -j 4 on 4 physical cores")
    p.add_argument("--packing-seed", type=int, default=None,
                   help="pin decoy k's packing to PACKING_SEED+k, making the ensemble a "
                        "pure function of the seed and identical at any --jobs. Off by "
                        "default: an unseeded packer is what makes a re-run an "
                        "independent sample. Mainly for verification")
    p.add_argument("--jran-base", type=int, default=None,
                   help="base for the per-worker Rosetta RNG streams. Default draws "
                        "fresh entropy per invocation, so re-running with --jobs grows "
                        "the ensemble as it does serially. Set it to reproduce a "
                        "specific parallel run; the resolved value is in run.json")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    p.add_argument("--version", action="version", version=f"frustx {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.structure.is_file():
        print(f"error: no such file: {args.structure}", file=sys.stderr)
        return 2
    if args.decoys < 2:
        print("error: --decoys must be at least 2 to estimate a standard deviation",
              file=sys.stderr)
        return 2

    # Imported here, not at module scope: PyRosetta takes ~2.5 s to load, and
    # `frustx --help` must not pay for it. Everything below this line is heavy.
    import pyrosetta
    from frustx.energies import init_rosetta, make_score_function
    from frustx.frustration import compute_frustration
    from frustx.output import contact_table, residue_table, write_bfactor_pdb
    # DEFAULT_INIT_FLAGS and DEFAULT_WEIGHTS live in energies, which does `import
    # pyrosetta` at module scope -- so they belong here, not at the top of the file.
    # tests/test_cli.py asserts frustx.cli does not pull PyRosetta in.
    from frustx.energies import DEFAULT_INIT_FLAGS, DEFAULT_WEIGHTS
    from frustx.provenance import file_key, regeneration_key
    from frustx.ligand_params import ligands_setting

    args.out.mkdir(parents=True, exist_ok=True)

    init_rosetta()
    ligand_report = []
    if args.ligand or args.ligand_placed:
        # Imported here for the same reason as everything else in this block: it touches
        # PyRosetta, and `frustx --help` must not pay for it.
        from frustx.ligand_params import (LigandParametrisationError, ligands_setting,
                                          load_pose_with_ligands, make_spec)
        paths = [(f, "topology") for f in args.ligand]
        paths += [(f, "placed") for f in args.ligand_placed]
        # Name overrides are positional over the combined list, so --ligand-name applies
        # to --ligand entries first and then --ligand-placed ones, in the order given.
        names = args.ligand_name + [""] * (len(paths) - len(args.ligand_name))
        names3 = args.ligand_name3 + [""] * (len(paths) - len(args.ligand_name3))
        try:
            specs = [make_spec(path, name=n, name3=n3, mode=mode)
                     for (path, mode), n, n3 in zip(paths, names, names3)]
            pose, ligand_report = load_pose_with_ligands(
                args.structure, specs,
                warn=lambda m: print(f"frustx: warning: {m}", file=sys.stderr))
        except (LigandParametrisationError, RuntimeError) as exc:
            # RuntimeError too: Rosetta's C++ utility_exit surfaces as a bare
            # RuntimeError, so catching only our own exception would let a params file
            # Rosetta rejects escape as an untranslated traceback.
            print(f"frustx: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            for entry in ligand_report:
                print(f"frustx: ligand {entry['name3']} ({entry['name']}) at "
                      f"{entry.get('chain')}{entry.get('resnum')}, "
                      f"{entry['n_heavy_atoms']} heavy atoms, from {entry['path']}",
                      file=sys.stderr)
    else:
        pose = pyrosetta.pose_from_pdb(str(args.structure))

    if not args.quiet:
        print(f"{args.structure.name}: {pose.total_residue()} residues, "
              f"{args.decoys} decoys, protocol={args.protocol}", file=sys.stderr)

    # Constructed here because the ETA counts from the reporter's own clock, and
    # this is where the run begins.
    reporter = Reporter(enabled=not args.quiet)
    result = compute_frustration(
        pose,
        # Structures are BUILT with fa_rep (it is what stops rotamers overlapping) and
        # MEASURED without it, unless the user asked for packing frustration.
        sf_pack=make_score_function(remove_fa_rep=False),
        sf_measure=make_score_function(remove_fa_rep=not args.packing_frustration),
        n_decoys=args.decoys,
        seed=args.seed,
        protocol=args.protocol,
        repeats=args.repeats,
        cutoff=args.cutoff,
        min_seq_sep=args.min_seq_sep,
        background_weight=args.background_weight,
        readout=args.readout,
        contact_atom=args.contact_atom,
        n_jobs=args.jobs,
        packing_seed=args.packing_seed,
        jran_base=args.jran_base,
        ligand_cutoff=args.ligand_cutoff,
        freeze_ligand=args.freeze_ligand,
        progress=reporter,
    )
    elapsed = time.time() - reporter.started
    reporter.finish()

    # A ligand that touches nothing is almost always a mistake, and it is a SILENT one:
    # the run completes and reports a perfectly clean protein-only answer. The usual cause
    # is placed mode fed a file whose coordinates are not the docked pose -- RCSB's
    # `_ideal.sdf`, for instance, carries a generated conformer near the origin, so the
    # ligand lands nowhere near the protein and every contact count is zero. Warned rather
    # than refused: a genuinely non-contacting ligand is possible, and this is the same
    # class of failure that the NaN contact guard exists for.
    if ligand_report:
        touched = set(result.contacts[:, 0]) | set(result.contacts[:, 1])
        for entry in ligand_report:
            if entry["pose_index"] - 1 not in touched:
                print(f"frustx: warning: ligand {entry['name3']} at "
                      f"{entry.get('chain')}{entry.get('resnum')} makes NO contacts with "
                      f"the protein at --ligand-cutoff {args.ligand_cutoff} A. If this was "
                      f"--ligand-placed, its file's coordinates are probably not the pose "
                      f"you meant -- an RCSB _ideal.sdf is a generated conformer near the "
                      f"origin, not the crystallographic or docked position.",
                      file=sys.stderr)

    contacts = contact_table(result)
    # Recorded because it is the single number saying whether this run's index can support
    # a claim about a particular contact, as opposed to about the two residues in it.
    from frustx.additivity import additive_decomposition
    additive_r2 = additive_decomposition(
        contacts["frustration_index"].to_numpy(), result.contacts, len(result.residues)
    ).r2
    residues = residue_table(result)
    contacts.to_csv(args.out / "contacts.csv", index=False)
    residues.to_csv(args.out / "residues.csv", index=False)
    write_bfactor_pdb(pose.clone(), result, args.out / "frustration.pdb")

    # Provenance: without this a results directory cannot say how it was produced.
    run_settings = {
        "structure_key": file_key(args.structure),
        "init_flags": DEFAULT_INIT_FLAGS,
        "weights": DEFAULT_WEIGHTS,
        "protocol": args.protocol,
        "repeats": args.repeats,
        "seed": args.seed,
        "n_decoys": args.decoys,
        "packing_seed": args.packing_seed,
        "freeze_ligand": args.freeze_ligand,
        "ligands": ligands_setting(ligand_report),
        "background_weight": args.background_weight,
        "readout": args.readout,
        "packing_frustration": args.packing_frustration,
        "contact_atom": args.contact_atom,
        "cutoff": args.cutoff,
        "min_seq_sep": args.min_seq_sep,
        "ligand_cutoff": args.ligand_cutoff,
    }
    (args.out / "run.json").write_text(json.dumps({
        "frustx_version": __version__,
        "structure": str(args.structure.resolve()),
        "n_residues": pose.total_residue(),
        "n_contacts": int(len(result.contacts)),
        "n_decoys": args.decoys,
        "seed": args.seed,
        "protocol": args.protocol,
        "repeats": args.repeats,
        "cutoff": args.cutoff,
        "min_seq_sep": args.min_seq_sep,
        "background_weight": args.background_weight,
        "readout": args.readout,
        "contact_atom": args.contact_atom,
        "n_jobs": args.jobs,
        "packing_seed": args.packing_seed,
        "jran_base": result.jran_base,
        "additive_r2": None if additive_r2 != additive_r2 else round(float(additive_r2), 6),
        "packing_frustration": args.packing_frustration,
        # Ligands are recorded by CONTENT HASH, not by path: `--ligand hit.sdf` names a
        # different molecule tomorrow if the screen was re-run, and a record that says
        # only "hit.sdf" cannot tell the two runs apart. The regeneration keys are written
        # alongside so a later run can ask whether this ensemble is still comparable
        # without having to reconstruct the settings dict by hand.
        "ligands": ligand_report,
        "ligand_cutoff": args.ligand_cutoff,
        "freeze_ligand": args.freeze_ligand,
        "regeneration_keys": {
            stage: regeneration_key(run_settings, stage)
            for stage in ("structure", "contacts", "ensemble", "measurement")
        },
        "elapsed_seconds": round(elapsed, 1),
    }, indent=2) + "\n")

    if not args.quiet:
        counts = contacts["frustration_class"].value_counts().to_dict()
        print(f"{len(contacts)} contacts in {elapsed:.0f}s -> {args.out}", file=sys.stderr)
        print(f"  {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
