"""Eq. 1: the local frustration index.

    F_ij = ( <E_decoy> - E_native ) / sigma(E_decoy)

SIGN CONVENTION -- this is the negative of the paper's Eq. 1, deliberately.

The paper writes F = (E0 - <E_U>)/sigma, which makes a *minimally* frustrated contact
come out negative: the native contact energy sits below the decoy mean. Every existing
frustration tool reports the opposite sign, with frustratometeR classifying minimally
frustrated as F > 0.78 and highly frustrated as F < -1. Measured on ubiquitin, the
paper's literal formula gives us a range of -4.05 to +0.55 -- flipped relative to the
convention, as expected. We negate here so that in FrustX output, as everywhere else in
the field:

    high positive F  ->  minimally frustrated (native much better than random)
    near zero        ->  neutral
    negative F       ->  highly frustrated (native no better, or worse, than random)

The contact energy E_ij is E_ij = e_ij + w * 1/2 (R_i + R_j), where w is
DEFAULT_BACKGROUND_WEIGHT in config.py. At the default w = 0 this is just the bare
pairwise energy e_ij; the many-body background is OFF. Eq. 2 as literally written is
w = 1, which simplifies exactly to 1/2 (R_i + R_j) and makes the index degenerate --
87% reducible to a residue-level quantity, with zero frustrated contacts on ubiquitin.
That path is retained only so the sweep can be reproduced. See docs/method.md for the
derivation, the numerical verification, and the w sweep.

KNOWN ARTIFACT -- glycine positions.

Whole-sequence shuffling drops bulky residues onto positions where only glycine fits
sterically. Decoy energies there balloon, and the native glycine looks superb by
comparison, producing a strongly "minimally frustrated" reading. On ubiquitin the
C-terminal LRGG tail scores among the most minimally frustrated contacts
(GLY75-GLY76 at F=+3.66 in our sign) even though that tail is flexible, solvent-exposed
and the functional conjugation site -- it should read as frustrated.

This is a composition-and-sterics effect, not folding frustration. It is inherent to the
paper's shuffling protocol rather than specific to this implementation, and the paper does
not address it. Treat glycine-rich and tail regions with suspicion until the
frustratometeR comparison shows whether the reference implementation behaves the same way.
"""

import os
from dataclasses import dataclass

import numpy as np

from frustx.contacts import (DEFAULT_CUTOFF, Residue, contact_pairs,
                            min_heavy_distances, pairwise_distances,
                            select_pairs_by_kind)
from frustx.decoys import make_decoy, native_reference
from pyrosetta.rosetta.core.scoring import ScoreType

from frustx.energies import (DEFAULT_WEIGHTS, make_fa_rep_score_function,
                            pair_energy_matrix)

from frustx.config import (  # constants live there so the CLI can read
    DEFAULT_BACKGROUND_WEIGHT,  # them without importing PyRosetta
    DEFAULT_N_DECOYS,
    DEFAULT_CONTACT_ATOM,
    DEFAULT_LIGAND_CUTOFF,
    CONTACT_ATOMS,
    DEFAULT_READOUT,
    READOUT_SCOPES,
    HIGHLY_FRUSTRATED,
    MINIMALLY_FRUSTRATED,
)


@dataclass
class FrustrationResult:
    """Everything needed to interpret or re-analyse a run, without recomputing."""

    residues: list          # list[Residue], 0-based, aligned to matrix indices
    contacts: np.ndarray    # (n_contacts, 2) int, i < j
    native_energy: np.ndarray   # (n, n) E_ij of the native
    decoy_mean: np.ndarray      # (n, n) <E_ij> over decoys
    decoy_std: np.ndarray       # (n, n) sigma(E_ij)
    index: np.ndarray           # (n, n) F_ij, conventional sign
    n_decoys: int
    jran_base: int = None       # resolved per-worker RNG base; None if run serially

    # The fa_rep part of the SAME readout quantity, tracked alongside rather than
    # thrown away. sf_measure zeroes fa_rep at BUILD time, so without these the
    # repulsive term is unrecoverable and answering "what would this look like as
    # packing frustration?" costs a second full run of the decoy ensemble.
    fa_rep_native: np.ndarray = None    # (n, n) R_ij of the native
    fa_rep_mean: np.ndarray = None      # (n, n) <R_ij> over decoys
    fa_rep_std: np.ndarray = None       # (n, n) sigma(R_ij)
    # Cov(E, R) over the decoys. Required, not decorative: Var(E + wR) is
    # Var(E) + w^2 Var(R) + 2w Cov(E, R), so without this term the two columns cannot
    # be recombined into a correct sigma and the feature would only LOOK usable.
    fa_rep_cov: np.ndarray = None       # (n, n) Cov(E_ij, R_ij)
    # Whether sf_measure ALREADY included fa_rep, i.e. whether this is a
    # --packing-frustration run. Recorded because index_at_fa_rep cannot otherwise
    # tell which direction is the correcting one, and getting it backwards produces a
    # doubled index that correlates with the true one at r = 0.99 -- the same shape and
    # sign pattern, silently twice the scale. That survives eyeballing.
    fa_rep_in_measure: bool = None

    def index_at_fa_rep(self, weight=None):
        """F_ij recomputed as if the measured energy had been E_ij + weight * R_ij.

        `weight=None` (the default) flips the run to its complement -- the repulsive
        term added back to a normal run, or removed from a --packing-frustration one.
        That is the only interpretation that is right without knowing which run this is,
        which is why the default is not a bare +1.

        An explicit weight is checked against the direction the run can support. Adding
        fa_rep to a measurement that already contains it is not a variant reading, it is
        double counting, and the result looks entirely plausible: on the test peptide it
        correlates with the true index at r = 0.99 with twice the scale.

        Exact, not an approximation: every term is linear in the per-pair energies and
        the variance is reconstructed from the stored covariance.
        """
        if self.fa_rep_native is None:
            raise ValueError("this result carries no fa_rep decomposition")
        if self.fa_rep_in_measure is None:
            raise ValueError(
                "this result does not record whether fa_rep was in the measurement, so "
                "the correcting direction is unknown"
            )
        if weight is None:
            weight = -1.0 if self.fa_rep_in_measure else +1.0
        elif self.fa_rep_in_measure and weight > 0:
            raise ValueError(
                f"sf_measure already included fa_rep, so weight={weight} would count it "
                f"twice; use a negative weight to remove it"
            )
        elif not self.fa_rep_in_measure and weight < 0:
            raise ValueError(
                f"sf_measure excluded fa_rep, so weight={weight} would subtract a term "
                f"that is not in the energy; use a positive weight to add it"
            )
        mean = self.decoy_mean + weight * self.fa_rep_mean
        var = (self.decoy_std ** 2
               + weight ** 2 * self.fa_rep_std ** 2
               + 2.0 * weight * self.fa_rep_cov)
        # Clipped for the same reason the primary variance is: catastrophic
        # cancellation can push a near-zero variance slightly negative.
        std = np.sqrt(np.maximum(var, 0.0))
        e0 = self.native_energy + weight * self.fa_rep_native
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(std > 0, (mean - e0) / std, np.nan)


def residues_from_pose(pose):
    """Build Residue records from a pose, preserving PDB chain/number/icode.

    Deliberately reads the pose rather than re-parsing the PDB with Biopython: the
    energies come from the pose, so taking residue identity from anywhere else risks
    the two numbering schemes drifting apart silently.
    """
    info = pose.pdb_info()
    residues = []
    for i in range(1, pose.total_residue() + 1):
        if info is not None:
            chain, resseq, icode = info.chain(i), info.number(i), info.icode(i)
        else:
            # Poses built from sequence have no PDB info.
            chain, resseq, icode = "A", i, " "
        residues.append(
            Residue(
                index=i - 1,
                chain=chain,
                resseq=resseq,
                icode=icode,
                resname=pose.residue(i).name3(),
            )
        )
    return residues


def ca_coords_from_pose(pose):
    """CA coordinates, shape (n_residues, 3), aligned to pose numbering."""
    return np.array(
        [np.array(pose.residue(i).xyz("CA")) for i in range(1, pose.total_residue() + 1)]
    )


def contact_coords_from_pose(pose, atom=DEFAULT_CONTACT_ATOM, missing="raise"):
    """Contact-representative coordinates, shape (n_residues, 3), aligned to pose numbering.

    "CB" falls back to CA wherever there is no CB -- glycine, and any residue built
    without one. Same convention as contacts.load_ca, and as frustratometeR.
    """
    if atom not in CONTACT_ATOMS:
        raise ValueError(f"atom must be one of {CONTACT_ATOMS}, got {atom!r}")
    out = []
    for i in range(1, pose.total_residue() + 1):
        r = pose.residue(i)
        name = atom if (atom == "CA" or r.has(atom)) else "CA"
        if not r.has(name) and missing == "nan":
            # NaN, deliberately, and ONLY where a caller has said it will apply a
            # different rule to these residues. select_pairs_by_kind never consults the
            # protein branch for a non-protein pair, so the NaN is inert there. Anywhere
            # else a NaN would silently drop pairs, which is why "raise" is the default.
            out.append(np.full(3, np.nan))
            continue
        if not r.has(name):
            # Rosetta does raise here on its own, but its message is
            # "ResidueType HOH does not have an atom CA" -- it names the residue TYPE
            # and not WHICH residue, which in a 500-residue pose is not enough to act
            # on. It also arrives as a bare RuntimeError out of C++, so it reads like a
            # crash rather than an input problem. Say which residue, and why.
            raise ValueError(
                f"pose residue {i} ({r.name3()}) has no {name} atom, so it has no "
                f"contact coordinate. Ligands, waters, ions and other heteroatoms have "
                f"no backbone: either strip them from the input, or give them a contact "
                f"rule of their own -- a ligand needs a heavy-atom-minimum distance, not "
                f"a CA-CA one. See docs/method.md, 'What the EGFR atomfrust branch "
                f"already solved'."
            )
        out.append(np.array(r.xyz(name)))
    return np.array(out)


def heavy_atom_coords_from_pose(pose):
    """All heavy-atom coordinates, packed, plus each residue's slice.

    Returns (coords, offsets): coords is (A, 3) with every residue's heavy atoms
    concatenated in pose order, offsets is (n + 1,) so residue i owns
    coords[offsets[i]:offsets[i+1]]. Ragged rather than padded because atom counts vary
    by an order of magnitude -- glycine has 4 heavy atoms, ATP has 33 -- and padding
    would need a sentinel that every consumer then has to remember to mask.

    Rosetta orders heavy atoms FIRST within a residue, so nheavyatoms() is a prefix count
    and no per-atom hydrogen test is needed.

    But nheavyatoms() is NOT the same as "real atoms", and the difference is not a corner
    case for exactly the residues this function exists to serve: it COUNTS VIRTUAL ATOMS.
    Rosetta's metal params carry a shell of them at the coordination positions -- ZN has
    5 heavy atoms of which 4 are virtual, MG has 7 of which 6. Those are fictitious points
    sitting 1.0-2.2 A off the metal, and feeding them to a minimum-distance rule shrinks
    every distance to that ion and invents contacts that no atom supports. Measured on a
    real MG site: up to 0.99 A of inflation and 8 real contacts at 6 A reported as 10.

    So virtual atoms are filtered explicitly. ATP carries two as well; they happen to sit
    0.00 A from real atoms so they cannot move a minimum, which is precisely why this was
    invisible when the function was first written and tested against a ligand rather than
    an ion.

    This is what a ligand contact rule is built on: a ligand has no CA and no CB, so the
    only geometry it shares with a protein residue is atom positions.
    """
    coords, offsets = [], [0]
    for i in range(1, pose.total_residue() + 1):
        r = pose.residue(i)
        for k in range(1, r.nheavyatoms() + 1):
            if r.is_virtual(k):
                continue
            coords.append(np.array(r.xyz(k)))
        offsets.append(len(coords))
    return np.asarray(coords, dtype=np.float64), np.asarray(offsets, dtype=np.intp)


def residue_kinds_from_pose(pose):
    """Per-residue kind flags, aligned to pose numbering.

    Returns a boolean array, True where the residue is a polymer amino acid. The
    per-kind contact rule needs exactly this: protein-protein pairs keep the paper's
    CA-CA criterion, and any pair involving something else has to fall back to a
    heavy-atom minimum, because the paper's criterion is not merely wrong for a ligand,
    it is undefined.

    is_protein() rather than not is_ligand(): waters, ions and virtual residues are all
    "not protein" but are not ligands either, and lumping them in would silently give a
    water the same treatment as a drug.
    """
    return np.array([pose.residue(i).is_protein()
                     for i in range(1, pose.total_residue() + 1)], dtype=bool)


def contact_energy_matrix(pose, sf_measure, background_weight=DEFAULT_BACKGROUND_WEIGHT):
    """Contact energies for a pose:

        E_ij = e_ij + w * 1/2 (R_i + R_j)

    where e_ij is the direct pair interaction and R_i is residue i's total pairwise
    interaction energy. w = 0 (the default) uses the direct pair energy alone.

    w = 1 APPROXIMATES Eq. 2 of the paper -- it does not equal it. Eq. 2's k!=j / l!=i
    exclusions make the direct term cancel, leaving exactly 1/2(R_i + R_j); this form
    keeps e_ij on top and so overshoots by e_ij (~0.7 REU against a ~10 REU background,
    so no conclusion turns on it). scripts/eq2_literal.py uses the exact parameterisation
    (1-w)*e_ij + w*1/2(R_i+R_j) when the endpoint has to be Eq. 2 verbatim.

    The default departs from the paper deliberately. Eq. 2's background term is ~14x
    larger than e_ij, and at w = 1 it swamps the contact: the index becomes ~75% reducible
    to a residue-additive function and cannot report a frustrated contact at all (10
    frustrated contacts in ubiquitin at w = 0, zero at w >= 0.1). The reason is a one-body
    tautology, not a tuning problem: shuffling a real sequence degrades it globally, so
    1/2(R_i + R_j) makes the numerator positive at 99.6% of contacts regardless of local
    frustration. See "The literal Eq. 2 experiment" in docs/method.md.

    Note what does NOT justify this choice. Correlation with frustratometeR's mutational
    mode RISES with w, +0.32 -> +0.54. But that gain is entirely one-body: residualising
    c + a_i + a_j out of both sides leaves ~0 at every w. Higher w buys agreement only by
    making the index as residue-additive as the reference already is. So frustratometeR
    cannot adjudicate w, and w = 0 is chosen on internal grounds. See
    DEFAULT_BACKGROUND_WEIGHT in config.py and the corrected sweep in docs/method.md.
    """
    e = pair_energy_matrix(pose, sf_measure)
    if background_weight == 0.0:
        E = e.copy()
    else:
        R = e.sum(axis=1)
        E = e + background_weight * 0.5 * (R[:, None] + R[None, :])
    np.fill_diagonal(E, 0.0)
    return E


def contact_mask(n, contacts):
    """Symmetric (n, n) boolean mask that is True at every contacting pair.

    `contacts` is the (n_contacts, 2) i<j array from contacts.contact_pairs. The mask is
    what makes "neighbourhood" a *contact* neighbourhood rather than a sum over the whole
    matrix -- without it, residues that never touch would contribute to the readout.
    """
    M = np.zeros((n, n), dtype=bool)
    M[contacts[:, 0], contacts[:, 1]] = True
    return M | M.T


def apply_readout(E, mask, readout=DEFAULT_READOUT):
    """Map a contact-energy matrix to the quantity Eq. 1 is applied to.

        "pair"           E_ij                                      (unchanged)
        "neighbourhood"  sum_k E_ik + sum_l E_jl - E_ij            (k, l over contacts)

    The neighbourhood form matches the scope frustratometeR sums over -- see
    READOUT_SCOPES in config.py for why that is worth measuring. E_ij is subtracted once
    because it appears in BOTH row sums; the result counts it exactly once, as
    frustratometeR does.

    MUST be applied per-structure BEFORE Eq. 1, never afterwards. The map is linear in E,
    but Eq. 1 is not: sigma of a sum is not the sum of sigmas, so transforming the native
    and each decoy separately (as compute_frustration does) is the only correct order.
    """
    if readout == "pair":
        return E
    if readout != "neighbourhood":
        raise ValueError(f"unknown readout {readout!r}, expected one of {READOUT_SCOPES}")
    # mask * E zeroes non-contacts, so `row` is each residue's summed contact energy.
    row = (E * mask).sum(axis=1)
    # Subtract the masked E so that a non-contacting (i, j) -- which was never in either
    # row sum -- is not wrongly decremented. At real contacts mask is 1 and this is E_ij.
    return row[:, None] + row[None, :] - (E * mask)


# --- Parallel decoy generation ------------------------------------------------------
#
# Module-level, because a fork Pool needs the worker function importable and the heavy
# objects (pose, score functions) inherited rather than pickled. ScoreFunction is NOT
# picklable at all, so fork is mandatory here -- spawn cannot work without rebuilding
# them, and the default start method is not guaranteed to stay fork.
_WORKER = {}


def _worker_init(jran_base, counter):
    """Give this worker its own Rosetta RNG stream.

    MUST call pyrosetta.init directly. energies.init_rosetta() is USELESS here: its
    module-level _INITIALISED guard (frustx/energies.py:39,49) is inherited as True
    across the fork, so the call silently no-ops and every worker keeps the parent's
    RNG state. The symptom is not an error -- it is correlated packings across workers
    and a quietly understated sigma. Verified: forked children without a real re-init
    all return bit-identical decoy energies.

    The id comes from a shared counter rather than os.getpid() so the streams are a
    contiguous, reproducible set, and per WORKER rather than per task -- reseeding per
    task would reset the stream repeatedly and reintroduce correlation in another guise.
    """
    import pyrosetta
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    # DEFAULT_INIT_FLAGS rather than a copy of its value: duplicating the literal would
    # let the parent and the workers silently diverge the moment a flag is added there.
    from frustx.energies import DEFAULT_INIT_FLAGS
    pyrosetta.init(
        f"{DEFAULT_INIT_FLAGS} -constant_seed -jran {jran_base + wid}", silent=True,
    )


def _seed_packer(packing_seed, k):
    """Pin the global RNG so decoy k packs identically wherever it is built.

    Optional and OFF by default. When set, decoy k is a pure function of
    (packing_seed, k): identical across worker counts, arrival order and resume points,
    which is what makes serial and parallel runs comparable bit-for-bit.

    Deliberately NOT inside decoys.make_decoy. scripts/packer_noise.py measures packer
    stochasticity by rebuilding one sequence repeatedly; seeding inside make_decoy would
    force that measurement to read exactly zero and silently invalidate it.
    """
    if packing_seed is None:
        return
    from pyrosetta.rosetta.numeric.random import rg
    rg().set_seed(packing_seed + k)


# Offsets into the packing_seed stream. The native gets 0 and decoy k gets 1 + k, so a
# decoy can never be handed the native's stream.
_NATIVE_OFFSET = 0
_DECOY_OFFSET = 1


def _check_accounting(seen, n_decoys):
    """Every decoy contributed exactly once, or raise.

    Integrity, not statistics. compute_frustration divides by n_decoys regardless of how
    many contributions actually arrived, so one lost decoy of 500 shifts the mean by 0.2%
    and a lost chunk of 8 corrupts sigma -- both invisible in the output. Silently
    renormalising on a short count would turn a bug into a quietly degraded run, so this
    raises instead.
    """
    if sorted(seen) != list(range(n_decoys)):
        raise RuntimeError(
            f"decoy accounting failed: {len(seen)} results, {len(set(seen))} distinct, "
            f"for {n_decoys} decoys"
        )


def _worker_decoy(k):
    """Build decoy k and return its readout matrix. Runs in a forked child."""
    c = _WORKER
    _seed_packer(c["packing_seed"], _DECOY_OFFSET + k)
    decoy = make_decoy(c["pose"], c["sf_pack"], seed=c["seed"] + k,
                       protocol=c["protocol"], repeats=c["repeats"],
                       freeze_ligand=c["freeze_ligand"])
    E = apply_readout(
        contact_energy_matrix(decoy, c["sf_measure"], c["background_weight"]),
        c["mask"], c["readout"])
    R = apply_readout(
        contact_energy_matrix(decoy, c["sf_fa_rep"], c["background_weight"]),
        c["mask"], c["readout"])
    return k, E, R


def compute_frustration(
    native_pose,
    sf_pack,
    sf_measure,
    n_decoys=DEFAULT_N_DECOYS,
    seed=0,
    protocol="relax",
    repeats=1,
    cutoff=DEFAULT_CUTOFF,
    min_seq_sep=1,
    contact_atom=DEFAULT_CONTACT_ATOM,
    background_weight=DEFAULT_BACKGROUND_WEIGHT,
    readout=DEFAULT_READOUT,
    n_jobs=1,
    packing_seed=None,
    jran_base=None,
    decoy_timeout=3600.0,
    weights=DEFAULT_WEIGHTS,
    ligand_cutoff=DEFAULT_LIGAND_CUTOFF,
    freeze_ligand=True,
    progress=None,
):
    """Run the full protocol and return a FrustrationResult.

    Parameters
    ----------
    sf_pack : score function WITH fa_rep -- builds decoy structures.
    sf_measure : score function WITHOUT fa_rep -- measures e_ij. See energies.py.
    protocol : passed to relax_sidechains. Defaults to "relax" (FastRelax), the literal
        reading of the paper's "short Monte-Carlo relaxation". "min" is ~10x faster.
    n_jobs : decoy-building processes. 1 (the default) keeps the plain serial loop, so
        every existing caller and every archived run stays exactly as it was. Contacts
        are unaffected; only decoy construction is split. Measured speedup on 8 logical
        cores (4 physical + SMT) is ~3.6x at n_jobs=4 and ~3.8-4.4x at n_jobs=8 -- the
        gap from linear is SMT and memory contention, not Amdahl: the serial part
        (setup + native reference) is ~5 s against ~2200 s of decoys at N=1000.
    packing_seed : if set, decoy k's PACKING is pinned to packing_seed + k, making the
        whole ensemble a pure function of (packing_seed, k) -- identical for any n_jobs,
        any arrival order, any resume point. Off by default because it changes the
        numbers relative to every archived run, and because an unseeded packer is what
        makes the ensemble an independent sample on a re-run. Its purpose is
        verification: with it set, serial and parallel results are bit-identical.
    jran_base : base for the per-worker Rosetta RNG streams (worker w gets
        jran_base + w). Only used when n_jobs > 1 and packing_seed is None.

        DEFAULTS TO None, MEANING "FRESH PER INVOCATION", and it must. A fixed base
        makes every parallel run of the same command bit-identical, which silently
        destroys the property the serial path has for free: that re-running grows an
        ensemble. With a fixed base, running twice with n_jobs=8 and merging adds ZERO
        information while the apparent standard error falls as if it had. Measured
        before this was fixed: two parallel invocations gave maxdiff exactly 0.0
        against 0.53 for two serial ones. The resolved value is returned on the result
        and written to run.json so a run stays reproducible ON DEMAND without being
        reproducible BY ACCIDENT.
    decoy_timeout : seconds to wait for any single decoy before giving up. Guards
        against a worker dying WITHOUT raising -- a Rosetta hard-exit, a segfault in the
        packer, or the OOM killer taking one of several forked Rosetta processes. Pool
        has no broken-worker detection: the task simply never returns a result and the
        parent blocks forever with the progress line frozen. Generous by default so it
        never fires on a slow FastRelax decoy.
    progress : optional callable(done, total) invoked after each decoy. `done` is a
        monotone count of completions, not a decoy index, so the contract is identical
        in serial and parallel. A 1000-decoy run takes roughly 12 min with
        protocol="min" and around an hour with "relax", so a caller usually wants some
        feedback.

    The native reference goes through the same repack-and-relax as the decoys, which the
    paper requires -- see decoys.native_reference.
    """
    if n_decoys < 2:
        raise ValueError("n_decoys must be at least 2 to estimate a standard deviation")

    residues = residues_from_pose(native_pose)
    kinds = residue_kinds_from_pose(native_pose)
    if kinds.all():
        # All-protein: the original call, untouched. Not merely an optimisation -- it is
        # what guarantees every existing run stays bit-identical, since it cannot reach
        # any of the ligand code below.
        contacts = contact_pairs(
            residues,
            contact_coords_from_pose(native_pose, contact_atom),
            cutoff=cutoff,
            min_seq_sep=min_seq_sep,
        )
    else:
        # Mixed. Protein pairs keep the paper's rule; anything else falls back to a
        # minimum heavy-atom distance. See select_pairs_by_kind and DEFAULT_LIGAND_CUTOFF.
        contacts = select_pairs_by_kind(
            residues,
            pairwise_distances(
                contact_coords_from_pose(native_pose, contact_atom, missing="nan")),
            min_heavy_distances(*heavy_atom_coords_from_pose(native_pose)),
            kinds,
            cutoff=cutoff,
            ligand_cutoff=ligand_cutoff,
            min_seq_sep=min_seq_sep,
        )

    # The readout mask is built ONCE from the native contact map and reused for every
    # decoy. Decoys keep the native backbone, so the contact map cannot drift -- and if
    # it could, letting each decoy define its own neighbourhood would make the decoy
    # ensemble incomparable to the native.
    mask = contact_mask(len(residues), contacts)

    # Pinned FIRST, and computed in the parent exactly once. If a worker ever rebuilt E0
    # the index would become a mixture of incomparable references -- visible only as mild
    # extra scatter that no test would flag.
    _seed_packer(packing_seed, _NATIVE_OFFSET)
    native = native_reference(native_pose, sf_pack, protocol=protocol, repeats=repeats,
                              freeze_ligand=freeze_ligand)
    E0 = apply_readout(
        contact_energy_matrix(native, sf_measure, background_weight), mask, readout
    )
    # Built here, once, and inherited by the fork workers like the other two.
    #
    # `weights` has to be passed rather than assumed: sf_measure is a caller-supplied
    # object and make_score_function takes its own weights argument, so
    # compute_frustration(sf_measure=make_score_function(weights="score12")) is a legal
    # call. score12's fa_rep weight is 0.44 against ref2015's 0.55, and picking the
    # wrong one would make the fa_rep column 25% wrong with nothing raising.
    sf_fa_rep = make_fa_rep_score_function(weights)
    # When fa_rep survives in sf_measure we can actually check the two agree. When it
    # was removed its weight is 0 and there is nothing to compare against -- hence the
    # explicit argument above rather than relying on this.
    measured_fa_rep = sf_measure.weights()[ScoreType.fa_rep]
    if measured_fa_rep and measured_fa_rep != sf_fa_rep.weights()[ScoreType.fa_rep]:
        raise ValueError(
            f"sf_measure carries fa_rep weight {measured_fa_rep} but weights={weights!r} "
            f"gives {sf_fa_rep.weights()[ScoreType.fa_rep]}; pass the weights set "
            f"sf_measure was built from"
        )
    R0 = apply_readout(
        contact_energy_matrix(native, sf_fa_rep, background_weight), mask, readout
    )

    # Accumulate running sums rather than holding 1000 (n x n) matrices: at n=500 that
    # would be 2 GB.
    n = E0.shape[0]
    total = np.zeros((n, n))
    total_sq = np.zeros((n, n))
    # Three more of the same shape for the fa_rep part: sum, sum of squares, and the
    # cross term. ~6 MB extra at n=500 against the ~2 GB this accumulation already
    # exists to avoid, so the running-sum argument is unchanged.
    total_r = np.zeros((n, n))
    total_sq_r = np.zeros((n, n))
    total_cross = np.zeros((n, n))
    # Raise rather than clamp up. `-1` is joblib's "use all cores", and silently running
    # an hour-long job serially because the user typed a familiar flag -- while run.json
    # dutifully records n_jobs=-1 -- is worse than refusing.
    if n_jobs is None or int(n_jobs) < 1:
        raise ValueError(
            f"n_jobs must be >= 1, got {n_jobs!r}. There is no 'use all cores' sentinel; "
            f"pass an explicit count."
        )
    n_jobs = min(int(n_jobs), n_decoys)             # never fork more workers than decoys

    def accumulate(k, E, R):
        nonlocal total, total_sq, total_r, total_sq_r, total_cross
        total += E
        total_sq += E * E
        total_r += R
        total_sq_r += R * R
        total_cross += E * R
        seen.append(k)
        if progress is not None:
            progress(len(seen), n_decoys)

    seen = []
    resolved_jran_base = None
    if n_jobs == 1:
        for k in range(n_decoys):
            _seed_packer(packing_seed, _DECOY_OFFSET + k)
            decoy = make_decoy(
                native_pose, sf_pack, seed=seed + k, protocol=protocol, repeats=repeats,
                freeze_ligand=freeze_ligand,
            )
            E = apply_readout(
                contact_energy_matrix(decoy, sf_measure, background_weight), mask, readout
            )
            R = apply_readout(
                contact_energy_matrix(decoy, sf_fa_rep, background_weight), mask, readout
            )
            accumulate(k, E, R)
    else:
        import multiprocessing as mp

        # fork, explicitly: the workers inherit the pose and both score functions, and
        # ScoreFunction cannot be pickled, so no other start method works without
        # rebuilding them. Verified that inherited objects give bit-identical energies.
        ctx = mp.get_context("fork")
        if jran_base is None:
            # Fresh entropy per invocation -- see the jran_base docstring. os.urandom
            # rather than the `seed` argument, which controls the SEQUENCE shuffle and
            # is deliberately fixed across runs.
            jran_base = int.from_bytes(os.urandom(4), "little") % (2 ** 31 - n_jobs - 1)
        resolved_jran_base = jran_base
        _WORKER.update(pose=native_pose, sf_pack=sf_pack, sf_measure=sf_measure,
                       sf_fa_rep=sf_fa_rep, freeze_ligand=freeze_ligand,
                       mask=mask, seed=seed, protocol=protocol, repeats=repeats,
                       background_weight=background_weight, readout=readout,
                       packing_seed=packing_seed)
        counter = ctx.Value("i", 0)     # from the SAME context as the Pool, or SemLock errors
        try:
            with ctx.Pool(n_jobs, initializer=_worker_init,
                          initargs=(jran_base, counter)) as pool:
                # imap, NOT imap_unordered. Workers still run ahead freely -- only the
                # YIELDING is ordered -- but the parent then accumulates in the same
                # order the serial loop does. Float addition is not associative, so this
                # is what lets serial and parallel agree bit-for-bit rather than to
                # ~1e-12, and it keeps `progress` firing once per decoy in order.
                # Windowed, not one long imap. imap buffers out-of-order results in the
                # PARENT with no bound: one slow task made 299 of 300 results pile up in
                # a probe (+94 MB), and at n=500 / N=1000 that worst case is 2.0 GB --
                # exactly what the running-sum accumulation above exists to avoid.
                # A window of 4*n_jobs bounds it to tens of MB for a negligible number
                # of barriers.
                window = max(4 * n_jobs, 16)
                for start in range(0, n_decoys, window):
                    batch = range(start, min(start + window, n_decoys))
                    it = pool.imap(_worker_decoy, batch, chunksize=1)
                    for _ in batch:
                        try:
                            k, E, R = it.next(timeout=decoy_timeout)
                        except mp.TimeoutError:
                            pool.terminate()
                            raise RuntimeError(
                                f"no decoy completed within {decoy_timeout}s -- a worker "
                                f"probably died without raising (Rosetta hard-exit, "
                                f"segfault, or OOM kill). Pool cannot detect this itself."
                            ) from None
                        accumulate(k, E, R)
        finally:
            _WORKER.clear()

    _check_accounting(seen, n_decoys)


    mean = total / n_decoys
    # Population variance, matching the paper's sigma over the N decoys. Clipped at zero
    # because catastrophic cancellation can push it slightly negative.
    var = np.maximum(total_sq / n_decoys - mean * mean, 0.0)
    std = np.sqrt(var)

    # Sign flipped relative to the paper -- see module docstring.
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(std > 0, (mean - E0) / std, np.nan)

    # Same estimator as above, applied to the fa_rep part. The covariance uses the
    # same 1/N convention as the variance, so the two recombine consistently in
    # FrustrationResult.index_at_fa_rep; mixing 1/N and 1/(N-1) here would be a bias
    # that only shows up at small n_decoys, which is exactly where nobody looks.
    mean_r = total_r / n_decoys
    var_r = np.maximum(total_sq_r / n_decoys - mean_r * mean_r, 0.0)
    cov = total_cross / n_decoys - mean * mean_r

    return FrustrationResult(
        residues=residues,
        contacts=contacts,
        native_energy=E0,
        decoy_mean=mean,
        decoy_std=std,
        index=index,
        n_decoys=n_decoys,
        jran_base=resolved_jran_base,
        fa_rep_native=R0,
        fa_rep_mean=mean_r,
        fa_rep_std=np.sqrt(var_r),
        fa_rep_cov=cov,
        fa_rep_in_measure=bool(measured_fa_rep),
    )


def classify(values):
    """Label frustration indices using the frustratometeR cut points.

    See the caveat on MINIMALLY_FRUSTRATED: these thresholds are AWSEM-calibrated and
    are placeholders until we recalibrate against atomistic output.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, "neutral", dtype=object)
    out[values >= MINIMALLY_FRUSTRATED] = "minimally"
    out[values <= HIGHLY_FRUSTRATED] = "highly"
    out[~np.isfinite(values)] = "undefined"
    return out
