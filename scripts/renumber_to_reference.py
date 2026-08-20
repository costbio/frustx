"""Renumber a PDB positionally against a reference of identical sequence.

Why this exists. 2RAP (Rap2A-GTP) numbers Glu123 as **222**, and every residue after it is
shifted down by one, so 2RAP 123 is 1KAO 124 and so on. Its sequence is identical to
1KAO's in identical order -- 167 residues, zero positional mismatches -- so the deposited
numbering is simply wrong, not a real insertion.

Joining the two on resseq would silently misalign 44 residues by one position and orphan
residue 222. Nothing would error; the per-residue comparison would just be wrong. Hence
this is a hard failure if the sequences do not match exactly, rather than a best-effort
alignment: a mismatch means the assumption behind positional renumbering is false.

Usage:
    .venv/bin/python scripts/renumber_to_reference.py <in.pdb> <reference.pdb> <out.pdb>
"""
import sys

THREE2ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))


def ca_sequence(path, chain="A"):
    """(resseq, one-letter) per residue, in file order."""
    return [(int(l[22:26]), THREE2ONE.get(l[17:20].strip(), "X"))
            for l in open(path)
            if l.startswith("ATOM") and l[12:16] == " CA " and l[21] == chain
            and l[16] in (" ", "A")]


def renumber(in_path, ref_path, out_path, chain="A"):
    src, ref = ca_sequence(in_path, chain), ca_sequence(ref_path, chain)
    if len(src) != len(ref):
        raise SystemExit(f"length differs: {len(src)} vs {len(ref)} -- cannot renumber positionally")
    bad = [(k, a, b) for k, (a, b) in enumerate(zip(src, ref)) if a[1] != b[1]]
    if bad:
        raise SystemExit(f"{len(bad)} residue-type mismatches, first at position {bad[0][0]} "
                         f"({bad[0][1]} vs {bad[0][2]}) -- sequences are not identical")

    mapping = {a[0]: b[0] for a, b in zip(src, ref)}   # old resseq -> reference resseq
    n = 0
    with open(out_path, "w") as fh:
        for line in open(in_path):
            if line.startswith(("ATOM", "HETATM")) and line[21] == chain:
                old = int(line[22:26])
                if old in mapping:
                    line = f"{line[:22]}{mapping[old]:>4d}{line[26:]}"
                    n += 1
            fh.write(line)
    changed = sum(1 for a, b in zip(src, ref) if a[0] != b[0])
    print(f"{out_path}: {len(src)} residues verified identical; "
          f"{changed} renumbered ({n} atom records rewritten)")


if __name__ == "__main__":
    renumber(sys.argv[1], sys.argv[2], sys.argv[3])
