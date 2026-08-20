"""Strip a PDB down to one protein chain, so the contact map can be built.

Why this exists. Every Fig. 2 candidate in Chen et al. is a nucleotide-binding protein,
and PyRosetta *recognises* MG and GDP/GTP rather than dropping them -- so they enter the
pose as residues, and frustration.ca_coords_from_pose then dies with

    RuntimeError: ResidueType MG does not have an atom CA

Dropping them at the file level rather than filtering the pose keeps pose numbering and
the contact map in the same 1:1 correspondence the rest of the code assumes.

Scope note: this discards the ligand entirely. For Fig. 2 that is correct -- the figure's
quantity is a per-residue count over protein-protein contacts. It would NOT be correct for
any question about the nucleotide site itself; ligand work is out of scope by decision.

Usage:
    .venv/bin/python scripts/prep_structure.py <in.pdb> <out.pdb> [chain]
"""
import sys


def prep(in_path, out_path, chain=None):
    """Keep ATOM records for one chain; drop HETATM, altlocs, hydrogens are left alone.

    Returns (n_ca, first_resseq, last_resseq, gaps) so the caller can assert on the
    result instead of trusting it.
    """
    kept, seen = [], []
    for line in open(in_path):
        if line.startswith(("ATOM  ",)):
            ch = line[21]
            if chain is not None and ch != chain:
                continue
            # altloc: keep only the primary conformer, else a residue enters twice and
            # Rosetta silently builds a pose with duplicated positions.
            if line[16] not in (" ", "A"):
                continue
            kept.append(line)
            if line[12:16] == " CA ":
                seen.append((int(line[22:26]), line[26]))
        elif line.startswith("TER") and kept:
            kept.append(line)
    kept.append("END\n")
    with open(out_path, "w") as fh:
        fh.writelines(kept)

    nums = [n for n, icode in seen]
    icodes = {ic for _, ic in seen if ic != " "}
    # A gap here would break residue-to-residue comparison between the two conformers,
    # so it is reported rather than assumed absent.
    gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b != a + 1]
    return len(seen), nums[0], nums[-1], gaps, icodes


if __name__ == "__main__":
    chain = sys.argv[3] if len(sys.argv) > 3 else None
    n, first, last, gaps, icodes = prep(sys.argv[1], sys.argv[2], chain)
    print(f"{sys.argv[2]}: {n} CA, resseq {first}-{last}, "
          f"gaps={gaps or 'none'}, icodes={sorted(icodes) or 'none'}")
