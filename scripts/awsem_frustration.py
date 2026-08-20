"""Run AWSEM's energy through FrustX's frustration machinery.

The point is to separate two things that were previously confounded when FrustX and
frustratometeR disagreed -- a different energy function and a different protocol:

    FrustX-REF2015  vs  FrustX-AWSEM     same protocol, different force field
    FrustX-AWSEM    vs  frustratometeR   same force field, different protocol

Everything except the energy function is held identical to the REF2015 run: the same
contact set (Ca-Ca <= 10 A, min_seq_sep from run.json), the same whole-sequence shuffle
decoys from frustx.decoys.shuffle_sequence with the same seeds, and the same Eq. 1 with
a per-contact sigma.

    e_ij = water_energy(r_ij, type_i, type_j, rho_i, rho_j)      the AWSEM pair term
    F_ij = ( <e_ij>_decoys - e_ij_native ) / sd(e_ij)_decoys     FrustX sign convention

Two honest differences that cannot be removed:

  * No repacking. AWSEM is coarse-grained and has no side chains, so a shuffled sequence
    is scored directly on the fixed geometry. The REF2015 decoys are repacked and
    relaxed. There is nothing to repack here; this is a property of the force field,
    not a choice.

  * Densities are read from frustratometeR's own output rather than recomputed. Our
    transcription of AWSEM's compute_ro does not reproduce their reported densities
    (see docs/method.md) -- the energy function is verified exact against their output,
    the density function is not. Since rho depends only on geometry, and geometry is
    fixed across the native and every decoy, taking their values is both safe and
    strictly more faithful than using a version we cannot verify.

Usage:
    .venv/bin/python scripts/awsem_frustration.py <pdb> <singleresidue_file> <out.csv> \\
        [n_decoys] [min_seq_sep]
"""
import random
import sys

import numpy as np
import pandas as pd

from frustx import awsem
from frustx.contacts import contact_pairs, load_ca
from frustx.decoys import shuffle_sequence
from frustx.frustration import classify

AWSEM_FILES = "/usr/local/lib/R/site-library/frustratometeR/Scripts/AWSEMFiles"


def main(pdb, sr_file, out_csv, n_decoys, min_seq_sep):
    water_gamma, _ = awsem.load_gammas(AWSEM_FILES)
    resnames, chains, resseqs, coords = awsem.load_cb(pdb)

    # Densities from their output; see the module docstring for why.
    sr = pd.read_csv(sr_file, sep=r"\s+").set_index("Res")
    rho = sr["DensityRes"].reindex(resseqs).values
    if np.isnan(rho).any():
        raise ValueError("singleresidue file does not cover every residue")

    # The contact SET is FrustX's, deliberately: it must match the REF2015 run so the
    # only thing differing between the two is the energy. Note this means the contact
    # map is Ca-based while the AWSEM energy is CB-based, which is intentional.
    residues, ca = load_ca(pdb)
    pairs = contact_pairs(residues, ca, min_seq_sep=min_seq_sep)
    i, j = pairs[:, 0], pairs[:, 1]
    rij = np.linalg.norm(coords[i] - coords[j], axis=1)
    rho_i, rho_j = rho[i], rho[j]

    seq = "".join(awsem.THREE_TO_ONE[r] for r in resnames)
    types = np.array([awsem.AA_INDEX[c] for c in seq])

    def energies(t):
        return awsem.water_energy(rij, t[i], t[j], rho_i, rho_j, water_gamma)

    native = energies(types)

    # Same accumulate-as-you-go form as frustx.frustration.compute_frustration, so the
    # sigma is the population sd over N decoys exactly as the paper specifies.
    total = np.zeros(len(pairs))
    total_sq = np.zeros(len(pairs))
    for k in range(n_decoys):
        shuffled = shuffle_sequence(seq, random.Random(k))
        e = energies(np.array([awsem.AA_INDEX[c] for c in shuffled]))
        total += e
        total_sq += e * e

    mean = total / n_decoys
    std = np.sqrt(np.maximum(total_sq / n_decoys - mean * mean, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(std > 1e-12, (mean - native) / std, np.nan)

    pd.DataFrame({
        "chain_i": chains[i], "resnum_i": resseqs[i], "resname_i": [resnames[k] for k in i],
        "chain_j": chains[j], "resnum_j": resseqs[j], "resname_j": [resnames[k] for k in j],
        "native_energy": native, "decoy_mean": mean, "decoy_std": std,
        "frustration_index": index,
        # Same thresholds as the REF2015 path. They are AWSEM-derived to begin with,
        # so applying them to an AWSEM energy is at least self-consistent here.
        "frustration_class": classify(index),
    }).to_csv(out_csv, index=False)

    ok = ~np.isnan(index)
    print(f"{len(pairs)} contacts, {n_decoys} decoys, {ok.sum()} with a defined index")
    print(f"index: mean {index[ok].mean():+.3f}  sd {index[ok].std(ddof=1):.3f}  "
          f"min {index[ok].min():+.2f}  max {index[ok].max():+.2f}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 4:
        print(__doc__)
        sys.exit(2)
    main(a[1], a[2], a[3],
         int(a[4]) if len(a) > 4 else 500,
         int(a[5]) if len(a) > 5 else 2)
