"""Per-pose cost of scripts/local_decoy.py -- the missing number in the compute budget.

Every pair-local estimate so far scaled the whole-sequence cost by (shell/n)^1.36. This
measures it instead, across the real range of shell sizes.
"""
import sys, time
import numpy as np, pyrosetta
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from frustx.contacts import Residue, contact_pairs
from frustx.energies import init_rosetta, make_score_function
from frustx.frustration import contact_energy_matrix, residues_from_pose
from local_decoy import local_decoy, shell_of
from vicinity_profile import load_atoms

init_rosetta()
sf_pack = make_score_function(remove_fa_rep=False)
sf_measure = make_score_function(remove_fa_rep=True)
AAS = "ACDEFGHIKLMNPQRSTVWY"

for label, pdb in [("1UBQ", "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb"),
                   ("1XTQ", "results/fig2/1XTQ_A.pdb")]:
    pose = pyrosetta.pose_from_pdb(pdb)
    res = residues_from_pose(pose)
    ca, _ = load_atoms(pdb, atom="CA")
    xyz = np.array([ca[(r.chain, r.resseq)] for r in res])
    P = contact_pairs(res, xyz, cutoff=10.0, min_seq_sep=2)
    shells = np.array([len(shell_of(xyz, i, j)) for i, j in P])
    # Span the shell-size range rather than sampling uniformly.
    order = np.argsort(shells)
    pick = order[np.linspace(0, len(order) - 1, 12).astype(int)]
    rng = np.random.default_rng(0)
    ts, ss = [], []
    for k in pick:
        i, j = P[k]
        t0 = time.perf_counter()
        d = local_decoy(pose, xyz, int(i), int(j),
                        AAS[rng.integers(20)], AAS[rng.integers(20)], sf_pack)
        contact_energy_matrix(d, sf_measure)
        ts.append(time.perf_counter() - t0); ss.append(shells[k])
    ts, ss = np.array(ts), np.array(ss)
    print(f"{label} (n={len(res)}): shell {ss.min()}-{ss.max()} res "
          f"(median {int(np.median(shells))} over all {len(P)} contacts)")
    print(f"   s/pose: min {ts.min():.3f}  median {np.median(ts):.3f}  max {ts.max():.3f}")
    a, b = np.polyfit(ss, ts, 1)
    print(f"   fit: {a:.4f} s per shell residue + {b:.3f} s fixed")
    print(f"   -> full {len(P)} contacts x 200 decoys = "
          f"{len(P)*200*np.median(ts)/3600:.1f} core-h; balanced 224 x 200 = "
          f"{224*200*np.median(ts)/3600:.1f} core-h", flush=True)
