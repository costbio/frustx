"""Decoy generation: the randomised reference ensemble for Eq. 1.

The paper's protocol, verbatim:

    "we randomly shuffle the protein sequence and then repack the resulting sequence
     onto the backbone that is provided without perturbing the backbone coordinates
     within each protein chain to make sure that only the side chains are re-packed.
     A short Monte-Carlo relaxation is then performed to better eliminate many of the
     possible side-chain clashes with the backbone fixed."

and, critically, for the native reference:

    "The contact energies of native sequence (E0) are obtained in a similar fashion by
     omitting the shuffling step."

That last sentence is not a detail.  The native pose must go through the *same*
repack-and-relax as the decoys, otherwise E0 and the decoy ensemble are measured on
structures prepared differently and the Z-score of Eq. 1 compares two incomparable
things.  `native_reference()` exists solely to make that hard to get wrong.

Why shuffling rather than random mutation: a permutation preserves the native amino-acid
composition exactly, which is what satisfies the paper's "sequence space is randomly
sampled according to the native amino acid frequency distribution".

NOT what frustratometeR does, despite the similar-sounding justification. Its decoys draw
each identity as `get_residue_type(rand() % n)` -- i.i.d. WITH replacement, two independent
draws per contact (AWSEM fix_backbone.cpp:5550-5555). A permutation reproduces the native
composition exactly and makes the two positions weakly dependent; their scheme reproduces
it only in expectation and keeps the positions independent. The difference is small
(indicator correlation -1/(L-1), ~0.6% at L~170) and is not worth engineering around, but
the two nulls are not the same and the comparison should not be described as if they were.
See "What the three modes actually randomise" in docs/method.md.

Two score functions are in play and they are not interchangeable:

  * `sf_pack` KEEPS fa_rep.  It builds the structure, and fa_rep is the only thing
    stopping rotamers from occupying the same space.  Packing without it produces
    physically meaningless overlapping side chains.
  * `sf_measure` DROPS fa_rep.  It is used later, in energies.py, to read e_ij.

Measured on ubiquitin: backbone atoms move by exactly 0.0 A through both design and
relaxation, so "the backbone is fixed" is enforced, not merely intended.
"""

import random

import numpy as np

import pyrosetta
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.utility import vector1_bool

from frustx.config import RELAX_PROTOCOLS  # see there for measured costs


# Two DIFFERENT predicates, deliberately, because non-protein and non-designable are not
# the same set and conflating them is how a modified residue gets silently mutated.
#
#   canonical amino acid      -> designed: the shuffle may change its identity
#   non-canonical PROTEIN     -> identity fixed, side chain still repacks: it is part of
#                                the chain and its packing is real
#   everything else           -> frozen: ligand, ion, water, virtual
#
# is_canonical_aa() rather than is_protein(): a spin label like R1A reports is_protein
# True and name1 'Z', so it would crash apply_sequence exactly as a ligand does, with no
# ligand anywhere in the file.
# is_protein() rather than "not is_ligand()": Rosetta reports is_ligand() True for WATER,
# so is_ligand cannot discriminate. frustration.py:residue_kinds_from_pose chose
# is_protein for the same reason.


def is_designable(residue):
    """Whether the sequence shuffle may change this residue's identity."""
    return residue.type().is_canonical_aa()


def is_frozen(residue):
    """Whether this residue must not move at all -- ligand, ion, water, virtual."""
    return not residue.is_protein()


def _sidechain_only_movemap(pose=None):
    """MoveMap allowing side-chain torsions only. This is what freezes the backbone.

    With a pose, also switches chi off for frozen residues and states set_jump(False)
    explicitly. The jump line is already MoveMap's default; it is written out because a
    ligand pose is the first pose in this project that HAS a jump, and a reader should
    not have to look up whether an unstated default is doing the work.
    """
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(True)
    if pose is not None:
        mm.set_jump(False)
        for i in range(1, pose.total_residue() + 1):
            if is_frozen(pose.residue(i)):
                mm.set_chi(i, False)
    return mm


def _relax_task_factory(pose):
    """TaskFactory that repacks the protein and leaves frozen residues untouched.

    Needed IN ADDITION to the movemap, not instead of it: the MoveMap does not bind
    FastRelax's internal packer. With chi explicitly off for a ligand, FastRelax still
    repacked it. A future refactor that drops this "because the movemap already freezes
    chi" reintroduces the bug, and it is invisible -- the ligand simply ends up in a
    different rotamer and every energy involving it shifts.
    """
    from pyrosetta.rosetta.core.pack.task import TaskFactory as _TF
    from pyrosetta.rosetta.core.pack.task.operation import (
        OperateOnResidueSubset, PreventRepackingRLT, RestrictToRepacking)
    from pyrosetta.rosetta.core.select.residue_selector import (
        NotResidueSelector, ResiduePropertySelector)
    from pyrosetta.rosetta.core.chemical import ResidueProperty

    tf = _TF()
    tf.push_back(RestrictToRepacking())
    tf.push_back(OperateOnResidueSubset(
        PreventRepackingRLT(),
        NotResidueSelector(ResiduePropertySelector(ResidueProperty.PROTEIN))))
    return tf


def frozen_coords(pose):
    """All atom coordinates of every frozen residue, (A, 3), in pose order.

    ALL atoms, not just heavy: a ligand hydroxyl rotating is a real change to the
    energies even though no heavy atom moved. This is the array that must be identical
    between the input pose, the native reference and every decoy -- see the tests. A
    protein-only pose gives shape (0, 3), so the comparison is trivially true rather
    than a special case.
    """
    out = []
    for i in range(1, pose.total_residue() + 1):
        r = pose.residue(i)
        if is_frozen(r):
            out.extend(np.array(r.xyz(k)) for k in range(1, r.natoms() + 1))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def shuffle_sequence(sequence, rng):
    """Return a random permutation of `sequence`.

    Composition is preserved exactly, which is the point -- see module docstring.
    `rng` is an explicit random.Random so decoy generation is reproducible.
    """
    chars = list(sequence)
    rng.shuffle(chars)
    return "".join(chars)


def shuffled_target_sequence(pose, rng):
    """Full-length target sequence with only the DESIGNABLE positions permuted.

    shuffle_sequence itself is unchanged: it still permutes a string, and the paper's
    requirement that composition be preserved exactly is satisfied over the designable
    positions, which are the only ones whose identity can change.

    Without this, pose.sequence() hands the shuffle a ligand's one-letter code -- 'Z' for
    ATP -- and the permutation MOVES it, so the design would try to mutate the ligand into
    an amino acid and some amino acid into ATP. On a ligand-free pose this returns exactly
    what shuffle_sequence(pose.sequence(), rng) returns, verified by test.
    """
    positions = [i for i in range(1, pose.total_residue() + 1)
                 if is_designable(pose.residue(i))]
    permuted = shuffle_sequence(
        "".join(pose.residue(i).name1() for i in positions), rng)

    letters = [pose.residue(i).name1() for i in range(1, pose.total_residue() + 1)]
    for i, aa in zip(positions, permuted):
        letters[i - 1] = aa
    return "".join(letters)


def apply_sequence(pose, sequence, sf_pack, freeze_ligand=True):
    """Design `pose` to `sequence` in place, repacking side chains on a fixed backbone.

    Implemented by restricting each position to exactly one allowed amino acid, which
    turns Rosetta's design machinery into "mutate to this sequence, then pack".
    """
    if len(sequence) != pose.total_residue():
        raise ValueError(
            f"sequence length {len(sequence)} != pose residues {pose.total_residue()}"
        )

    task = TaskFactory.create_packer_task(pose)
    for i, aa in enumerate(sequence, start=1):
        res = pose.residue(i)
        rt = task.nonconst_residue_task(i)

        if is_frozen(res) and freeze_ligand:
            # prevent_repacking, not restrict_to_repacking. A frozen residue must not
            # even change rotamer: its coordinates are an input, not something this
            # calculation is entitled to move.
            rt.prevent_repacking()
        elif not is_designable(res):
            # Non-canonical protein: identity fixed, side chain still packs.
            rt.restrict_to_repacking()
        else:
            idx = int(aa_from_oneletter_code(aa))
            if not 1 <= idx <= 20:
                # restrict_absent_canonical_aas does NOT raise on a non-canonical code --
                # it leaves the position packable with one allowed type, so a wrong index
                # here would design silently rather than crash.
                raise ValueError(
                    f"residue {i} ({res.name3()}) was given target letter {aa!r}, which "
                    f"is not one of the 20 canonical amino acids"
                )
            # A 20-long bool vector, one flag per canonical amino acid.
            keep = vector1_bool(20)
            keep[idx] = True
            rt.restrict_absent_canonical_aas(keep)

    PackRotamersMover(sf_pack, task).apply(pose)
    return pose


def repack(pose, sf_pack, freeze_ligand=True):
    """Repack side chains without changing the sequence, backbone fixed.

    This is the native reference's counterpart to `apply_sequence`: same packing, no
    sequence change.
    """
    task = TaskFactory.create_packer_task(pose)
    task.restrict_to_repacking()
    if freeze_ligand:
        # NOT redundant with restrict_to_repacking, which leaves a ligand packable --
        # measured. Freezing the decoys but not the native reference is the silent
        # version of this bug: E0 shifts and nothing raises.
        for i in range(1, pose.total_residue() + 1):
            if is_frozen(pose.residue(i)):
                task.nonconst_residue_task(i).prevent_repacking()
    PackRotamersMover(sf_pack, task).apply(pose)
    return pose


def relax_sidechains(pose, sf_pack, protocol="relax", repeats=1,
                     freeze_ligand=True):
    """Relieve side-chain clashes with the backbone held fixed.

    See RELAX_PROTOCOLS for what the choice costs and why it is not settled.
    """
    if protocol not in RELAX_PROTOCOLS:
        raise ValueError(f"protocol must be one of {RELAX_PROTOCOLS}, got {protocol!r}")

    if protocol == "none":
        return pose

    mm = _sidechain_only_movemap(pose if freeze_ligand else None)
    if protocol == "min":
        # MinMover has no packer, so the movemap alone is sufficient here.
        MinMover(mm, sf_pack, "lbfgs_armijo_nonmonotone", 1e-2, True).apply(pose)
    else:
        mover = FastRelax(sf_pack, repeats)
        mover.set_movemap(mm)
        if freeze_ligand:
            # FastRelax packs as well as minimises, and its packer does NOT consult the
            # movemap. Without this the ligand is repacked despite chi being off.
            mover.set_task_factory(_relax_task_factory(pose))
        mover.apply(pose)
    return pose


def make_decoy(native_pose, sf_pack, seed, protocol="relax", repeats=1,
               freeze_ligand=True):
    """Build one decoy: shuffle -> design onto fixed backbone -> relax.

    `native_pose` is never modified; a clone is returned.
    """
    rng = random.Random(seed)
    sequence = shuffled_target_sequence(native_pose, rng)

    pose = native_pose.clone()
    apply_sequence(pose, sequence, sf_pack, freeze_ligand=freeze_ligand)
    relax_sidechains(pose, sf_pack, protocol=protocol, repeats=repeats,
                     freeze_ligand=freeze_ligand)
    return pose


def native_reference(native_pose, sf_pack, protocol="relax", repeats=1,
                     freeze_ligand=True):
    """Build the native reference for E0: repack -> relax, no shuffling.

    Deliberately mirrors `make_decoy` step for step. E0 and the decoy energies must come
    off structures prepared identically, or the Z-score is meaningless.
    """
    pose = native_pose.clone()
    repack(pose, sf_pack, freeze_ligand=freeze_ligand)
    relax_sidechains(pose, sf_pack, protocol=protocol, repeats=repeats,
                     freeze_ligand=freeze_ligand)
    return pose


def backbone_coords(pose):
    """N, CA, C coordinates, shape (3 * n_residues, 3).

    Used by the tests to assert the backbone never moves. Cheap enough to call in
    production as a safety check.
    """
    out = []
    for i in range(1, pose.total_residue() + 1):
        r = pose.residue(i)
        # Protein only: a ligand has no N/CA/C, and Rosetta's error names the residue
        # TYPE and not which residue. frozen_coords is the companion covering the rest.
        if not r.is_protein():
            continue
        for atom in ("N", "CA", "C"):
            if not r.has(atom):
                raise ValueError(
                    f"protein residue {i} ({r.name3()}) has no {atom} atom, so its "
                    f"backbone cannot be checked"
                )
            out.append(np.array(r.xyz(atom)))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)
