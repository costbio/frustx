"""Pair-local decoys: randomise ONLY the two contacting residues.

An alternative decoy model to the paper's whole-sequence shuffling, kept because it
produced a measurable result (see docs/method.md): it does not improve rank correlation
with frustratometeR, but it nearly doubles the separation between the contacts
frustratometeR calls frustrated and those it calls minimally frustrated.

Rationale. AWSEM's configurational decoy randomises the identities and geometry of the
contacting pair while holding the rest of the protein native. FrustX (following the
paper) shuffles the whole sequence, so a decoy energy for contact (i,j) also absorbs the
effect of every other position changing. That extra variance lands in sigma, which is
Eq. 1's denominator.

Not wired into the main pipeline: the evidence does not yet justify replacing the
paper's protocol, only recording that the alternative discriminates better.
"""

import numpy as np

from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
from pyrosetta.rosetta.utility import vector1_bool

# CA-CA radius defining which side chains may repack around the mutated pair.
SHELL_RADIUS = 10.0


def shell_of(ca_coords, i, j, radius=SHELL_RADIUS):
    """0-based residue indices within `radius` of residue i or j (both included)."""
    d = np.minimum(
        np.linalg.norm(ca_coords - ca_coords[i], axis=1),
        np.linalg.norm(ca_coords - ca_coords[j], axis=1),
    )
    return set(np.nonzero(d <= radius)[0].tolist()) | {i, j}


def local_decoy(native_pose, ca_coords, i, j, aa_i, aa_j, sf_pack, relax=True):
    """Mutate residues i,j (0-based) to aa_i,aa_j and repack only their shell.

    Everything outside the shell is frozen -- side chains included -- so the only thing
    varying between decoys is the contacting pair and its immediate environment.
    Verified on ubiquitin: exactly two positions change identity and no non-shell residue
    moves at all.
    """
    pose = native_pose.clone()
    shell = shell_of(ca_coords, i, j)

    task = TaskFactory.create_packer_task(pose)
    for k in range(1, pose.total_residue() + 1):
        rt = task.nonconst_residue_task(k)
        idx = k - 1
        if idx in (i, j):
            keep = vector1_bool(20)
            keep[int(aa_from_oneletter_code(aa_i if idx == i else aa_j))] = True
            rt.restrict_absent_canonical_aas(keep)
        elif idx in shell:
            rt.restrict_to_repacking()   # may move, may not change identity
        else:
            rt.prevent_repacking()       # frozen entirely
    PackRotamersMover(sf_pack, task).apply(pose)

    if relax:
        mm = MoveMap()
        mm.set_bb(False)
        for idx in shell:
            mm.set_chi(idx + 1, True)    # only shell side chains relax
        MinMover(mm, sf_pack, "lbfgs_armijo_nonmonotone", 1e-2, True).apply(pose)
    return pose

def local_native_reference(native_pose, ca_coords, i, j, sf_pack, relax=True):
    """The native put through the IDENTICAL shell-restricted treatment as a local decoy.

    This is the pair-local analogue of decoys.native_reference, and it is not optional.
    Eq. 1 compares E0 against a decoy ensemble; if E0 comes from an untouched crystal pose
    while the decoys have been repacked and minimised in a 10 A shell, the difference
    between them is partly the preparation, not the sequence. The whole-sequence path has
    made that mistake impossible since decoys.py existed -- this closes the same hole here.

    Implemented by running local_decoy() with the residues' OWN identities: they are still
    repacked (restrict_absent_canonical_aas to a single allowed amino acid re-designs the
    rotamer), the shell still repacks, the same MinMover still runs.
    """
    aa_i = native_pose.residue(i + 1).name1()
    aa_j = native_pose.residue(j + 1).name1()
    return local_decoy(native_pose, ca_coords, i, j, aa_i, aa_j, sf_pack, relax=relax)


def sample_pair(q, rng):
    """Draw (a_i, a_j) from the SAME law the whole-sequence decoys induce.

    q comes from rao_blackwell.exact_pair_weights: under a shuffle of the native sequence,
    positions i and j receive identities (a,b) with probability n_a n_b / (L(L-1)), or
    n_a(n_a-1)/(L(L-1)) on the diagonal.

    Matching this law is what keeps the pair-local index comparable to the production one:
    the two schemes then differ ONLY in whether the rest of the protein is also shuffled,
    which is the single variable the experiment is meant to isolate. Sampling uniformly
    over the 20x20 grid instead would change the estimand as well, confounding the two.
    """
    cells = list(q)
    p = np.array([q[c] for c in cells])
    return cells[rng.choice(len(cells), p=p / p.sum())]

