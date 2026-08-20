"""Sweep the Eq. 2 background weight w from ONE decoy ensemble.

Why this exists. The original sweep in docs/method.md ran a separate 500-decoy job per w
and scored each against frustratometeR's *configurational* mode -- which we now know uses
a single global decoy (mean, sd) for the whole protein, so its index is affine in its own
native energy. The sweep was therefore comparing against a target that carries no
per-contact decoy information, and its conclusion needs rechecking against `mutational`
mode, which does have a genuine per-contact sigma.

The trick that makes the recheck cheap. The contact energy is linear in w:

    E_ij(w) = e_ij + w * B_ij,     B_ij = 1/2 (R_i + R_j)

so over a decoy ensemble the first two moments follow in closed form:

    mean_E(w) = <e> + w <B>
    var_E(w)  = var(e) + 2 w cov(e, B) + w^2 var(B)

Accumulating the five running sums (Se, See, Sb, Sbb, Seb) therefore yields the index at
ANY w from a single pass. One ~5 min ensemble replaces one job per w, and every w shares
the identical decoys, so differences between w values are not confounded by decoy noise
the way independent runs are.

Usage:
    .venv/bin/python scripts/sweep_background.py <pdb> <out.csv> [n_decoys] [protocol]
"""

import sys

import numpy as np
import pandas as pd

import pyrosetta

from frustx.contacts import contact_pairs
from frustx.decoys import make_decoy, native_reference
from frustx.energies import pair_energy_matrix
from frustx.config import DEFAULT_CUTOFF
from frustx.energies import init_rosetta, make_score_function
from frustx.frustration import ca_coords_from_pose, residues_from_pose


def _e_and_b(pose, sf_measure):
    """Return (e_ij, B_ij) for one pose. B is the Eq. 2 background, 1/2 (R_i + R_j)."""
    e = pair_energy_matrix(pose, sf_measure)
    R = e.sum(axis=1)
    b = 0.5 * (R[:, None] + R[None, :])
    np.fill_diagonal(e, 0.0)
    np.fill_diagonal(b, 0.0)
    return e, b


def main(pdb, out_csv, n_decoys=500, protocol="min", seed=0, min_seq_sep=2):
    init_rosetta()
    # Same pair as a real run: fa_rep KEPT for packing, DROPPED for measurement.
    sf_pack = make_score_function(remove_fa_rep=False)
    sf_measure = make_score_function(remove_fa_rep=True)
    pose = pyrosetta.pose_from_pdb(pdb)

    residues = residues_from_pose(pose)
    contacts = contact_pairs(residues, ca_coords_from_pose(pose),
                             cutoff=DEFAULT_CUTOFF, min_seq_sep=min_seq_sep)

    # Native reference goes through the identical repack/relax as the decoys.
    e0, b0 = _e_and_b(native_reference(pose, sf_pack, protocol=protocol), sf_measure)

    n = e0.shape[0]
    Se = np.zeros((n, n)); See = np.zeros((n, n))
    Sb = np.zeros((n, n)); Sbb = np.zeros((n, n)); Seb = np.zeros((n, n))
    for k in range(n_decoys):
        e, b = _e_and_b(make_decoy(pose, sf_pack, seed=seed + k, protocol=protocol),
                        sf_measure)
        Se += e; See += e * e
        Sb += b; Sbb += b * b; Seb += e * b
        if (k + 1) % 50 == 0:
            print(f"  decoy {k+1}/{n_decoys}", flush=True)

    N = float(n_decoys)
    me, mb = Se / N, Sb / N
    ve = np.maximum(See / N - me * me, 0.0)          # population moments, as in Eq. 1
    vb = np.maximum(Sbb / N - mb * mb, 0.0)
    ce = Seb / N - me * mb

    rows = []
    for (i, j) in contacts:
        rec = {"chain_i": residues[i].chain, "resnum_i": residues[i].resseq,
               "chain_j": residues[j].chain, "resnum_j": residues[j].resseq}
        for w in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            var = ve[i, j] + 2 * w * ce[i, j] + w * w * vb[i, j]
            sd = np.sqrt(max(var, 0.0))
            mean = me[i, j] + w * mb[i, j]
            nat = e0[i, j] + w * b0[i, j]
            # Sign flipped relative to the paper, as everywhere in FrustX.
            rec[f"w{w}"] = (mean - nat) / sd if sd > 0 else np.nan
        rows.append(rec)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(rows)} contacts, {n_decoys} decoys, protocol={protocol})")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 500, a[3] if len(a) > 3 else "min")
