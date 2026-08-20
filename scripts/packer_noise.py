"""How much of the decoy energy variance is irreducible packer noise?

THE QUESTION THIS GATES. Roughly 28% of the within-contact decoy energy variance is not
explained by the identity of the contacting pair. The pair-local decoy proposal
(scripts/local_decoy.py) is motivated by the hope that this residue is CONTEXT -- the
other 90-odd% of the sequence being shuffled too -- and that freezing the environment
would recover it as signal.

But there is a second candidate that would make the whole idea worthless: the packer.
Rosetta is initialised without -constant_seed (frustx/energies.py:37), so building the
SAME sequence twice gives two different rotamer assignments and two different energies.
If most of the non-pair variance is that, then no decoy-locality scheme recovers anything,
because the variance is not information about context at all -- it is Monte-Carlo noise in
the side-chain packing.

THE MEASUREMENT. Rebuild ONE fixed decoy sequence R times and measure the spread of e_ij
across repeats. That is packer noise with sequence and context both held exactly constant.
Compare it to the total spread across different decoy sequences from the existing tensor.

    packer_var / total_var  near 1  ->  the residue is noise; locality recovers nothing
    packer_var / total_var  near 0  ->  the residue is context; locality has something

Usage:
    .venv/bin/python scripts/packer_noise.py [pdb] [n_repeats] [n_seqs]
"""
import random
import sys

import numpy as np
import pyrosetta

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from frustx.contacts import Residue, contact_pairs
from frustx.decoys import make_decoy, shuffle_sequence
from frustx.energies import init_rosetta, make_score_function
from frustx.frustration import contact_energy_matrix, residues_from_pose
from vicinity_profile import load_atoms


def main(pdb, n_repeats=12, n_seqs=4, npz="results/validation/decoy_samples/1ubq_min200.npz"):
    init_rosetta()
    pose = pyrosetta.pose_from_pdb(pdb)
    sf_pack = make_score_function(remove_fa_rep=False)
    sf_measure = make_score_function(remove_fa_rep=True)

    res = residues_from_pose(pose)
    ca, _ = load_atoms(pdb, atom="CA")
    xyz = np.array([ca[(r.chain, r.resseq)] for r in res])
    P = contact_pairs(res, xyz, cutoff=10.0, min_seq_sep=2)
    i, j = P[:, 0], P[:, 1]

    # Several fixed sequences, each rebuilt n_repeats times. Several rather than one
    # because packer noise could plausibly depend on which sequence was drawn.
    print(f"{n_seqs} fixed sequences x {n_repeats} rebuilds each "
          f"({n_seqs * n_repeats} poses)", flush=True)
    # DELIBERATELY SERIAL, and it must stay that way. This loop measures packer
    # stochasticity by rebuilding one sequence repeatedly, which depends on the packer
    # drawing fresh randomness each time. Parallelising it by copying the pattern in
    # frustx/frustration.py would give forked workers -constant_seed streams, identical
    # rebuilds, and a measured packer noise of exactly 0.0 -- a wrong answer that looks
    # like a clean result.
    within = []
    for s in range(n_seqs):
        # seed=s reproduces decoy s's SEQUENCE exactly; only the packing differs between
        # repeats, because Rosetta's packer RNG is not seeded per call.
        E = []
        for r in range(n_repeats):
            d = make_decoy(pose, sf_pack, seed=s, protocol="min")
            E.append(contact_energy_matrix(d, sf_measure)[i, j])
            print(f"  seq {s} rebuild {r + 1}/{n_repeats}", flush=True)
        E = np.array(E)
        within.append(E.var(axis=0, ddof=1))
        # Guard: the repeats must really share a sequence.
        assert shuffle_sequence(pose.sequence(), random.Random(s)) == \
               shuffle_sequence(pose.sequence(), random.Random(s))
    within = np.mean(within, axis=0)          # mean packer variance per contact

    # Total variance across DIFFERENT decoy sequences, from the run already on disk.
    d = np.load(npz, allow_pickle=True)
    assert d["decoys"].shape[1] == len(res), "tensor is for a different structure"
    total = d["decoys"][:, i, j].var(axis=0, ddof=1)

    ok = total > 1e-6
    frac = within[ok] / total[ok]
    print(f"\n{ok.sum()} contacts with non-zero total variance")
    print(f"packer variance / total variance:")
    for q in [10, 25, 50, 75, 90]:
        print(f"  p{q:<3d} {np.percentile(frac, q):.3f}")
    print(f"  mean {frac.mean():.3f}   "
          f"pooled {within[ok].sum() / total[ok].sum():.3f}")
    np.savez("results/packer_noise.npz", within=within, total=total,
             pairs=P, resseq=np.array([r.resseq for r in res]))
    print("\n-> results/packer_noise.npz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12,
         int(sys.argv[3]) if len(sys.argv) > 3 else 4)
