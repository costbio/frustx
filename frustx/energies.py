"""Pairwise residue interaction energies e_ij from the Rosetta REF2015 force field.

This module answers one question: for a given pose, what is the interaction energy
e_ij between every pair of residues?  Eq. 2 of the paper builds its many-body contact
energy out of those numbers, so getting the extraction exactly right matters more than
anything downstream.

Two non-obvious things are required to make the decomposition complete.  Both were
established empirically -- see "Implementation notes" in docs/method.md for the
measurements.

1. Backbone-backbone hydrogen bonds are NOT written to the energy graph by default.
   On ubiquitin that silently hides 41.96 REU of hbond_sr_bb + hbond_lr_bb -- exactly
   the interactions holding secondary structure together.  Fixed by an option on the
   score function (NOT a command-line flag, which does not exist).

2. rama_prepro is not on the energy graph either; it lives in a long-range container as
   strict (i, i+1) pairs.  It cannot be dismissed as constant under sequence shuffling:
   with the backbone held fixed, single mutations move it by 2.6 to 12.5 REU.

With both handled, one-body + graph edges + long-range containers reproduces the REF2015
total to machine precision.  `tests/test_energies.py` asserts exactly that, so a future
Rosetta version that moves a term elsewhere fails loudly instead of quietly biasing
every frustration index.
"""

import numpy as np

import pyrosetta
from pyrosetta.rosetta.core.scoring import (EMapVector, ScoreFunction, ScoreType,
                                            methods)

# REF2015: "we employed the REF2015 version of the rosetta energy function, which has a
# set of well-tested weights for each energy term."
DEFAULT_WEIGHTS = "ref2015"

# Rosetta is chatty and aborts on unrecognised residues; neither is useful here.
#
# -in:file:load_PDB_components false is what makes -ignore_unrecognized_res mean what it
# says. Left at its default, Rosetta silently falls back to its bundled Chemical
# Component Dictionary and BUILDS a residue for any three-letter code it recognises --
# so a HETATM with no .params anywhere is not ignored at all, it is admitted with
# CCD-derived atom types and charges that nobody chose and nothing records. Verified
# here: a bare BNZ HETATM loads as a fifth residue with the flag at its default and is
# correctly dropped with it off. The EGFR atomfrust branch lost a run to exactly this,
# its manifest claiming curated params for a pose that had none -- see docs/method.md.
#
# When ligands land properly this flag stays and the params come from an explicit
# -extra_res_fa, which is the point: parametrisation should be a decision, not a
# fallback.
DEFAULT_INIT_FLAGS = ("-mute all -ignore_unrecognized_res "
                      "-in:file:load_PDB_components false")

_INITIALISED = False


def init_rosetta(extra_flags=""):
    """Initialise PyRosetta once per process. Safe to call repeatedly.

    Rosetta's init is global and expensive, and re-initialising mid-run can silently
    reset options, so we guard it.
    """
    global _INITIALISED
    if _INITIALISED:
        return
    pyrosetta.init(f"{DEFAULT_INIT_FLAGS} {extra_flags}".strip(), silent=True)
    _INITIALISED = True


def make_score_function(remove_fa_rep=True, weights=DEFAULT_WEIGHTS):
    """Build the REF2015 score function configured for frustration analysis.

    Parameters
    ----------
    remove_fa_rep : bool
        Zero the repulsive Lennard-Jones weight, as the paper requires:

            "it is both easy and appropriate to simply remove the harsh rapidly
             varying repulsive force term."

        Their reasoning is that residual clashes survive relaxation and inflate the
        decoy variance, which is the denominator of the frustration index -- an
        inflated sigma washes out real signal.  Setting this False instead yields what
        the authors call "packing frustration", a different (also useful) quantity that
        diagnoses structure quality rather than functional frustration.
    """
    sf = pyrosetta.create_score_function(weights)

    # Push backbone-backbone hbonds onto energy-graph edges.  Without this they are
    # accumulated as whole-structure energy and are invisible to a pairwise read.
    # Verified: the weighted total is identical with this on or off -- it changes where
    # energy is stored, not the physics.
    opts = sf.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sf.set_energy_method_options(opts)

    if remove_fa_rep:
        sf.set_weight(ScoreType.fa_rep, 0.0)

    return sf


def make_fa_rep_score_function(weights=DEFAULT_WEIGHTS):
    """A score function carrying ONLY fa_rep, at its REF2015 weight.

    Feeding this to pair_energy_matrix yields the fa_rep-only part of e_ij, which is
    what lets a run record the repulsive term SEPARATELY instead of destroying it.

    The alternative -- score once with fa_rep present and subtract it at extraction --
    was measured and rejected: it perturbs the surviving terms by ~1 ULP (2.2e-16
    relative), and this project deliberately holds serial and parallel runs to
    BIT-identical agreement rather than ~1e-12 (see the ordered-imap note in
    compute_frustration and docs/method.md). Paying one extra scoring pass keeps every
    existing number exactly as it was. That pass costs ~3.2 ms against a ~700 ms decoy,
    i.e. ~0.5%, which is why it is unconditional rather than behind a flag.

    Built empty and given one weight rather than built from ref2015 and stripped: the
    stripping version has to enumerate every other ScoreType correctly, and a term
    missed there would silently contaminate the "fa_rep" column.

    The weight is READ from the real REF2015 function rather than written as 0.55, so it
    cannot drift from whatever the weights file actually says.
    """
    w = pyrosetta.create_score_function(weights).weights()[ScoreType.fa_rep]
    if w == 0.0:
        # An empty ScoreFunction scores everything as zero, so pair_energy_matrix would
        # return an all-zero matrix and the whole fa_rep column would read as "this
        # structure has no repulsion" rather than as a misconfiguration.
        raise ValueError(
            f"weights set {weights!r} has fa_rep weight 0, so there is no repulsive "
            f"term to isolate"
        )
    sf = ScoreFunction()
    sf.set_weight(ScoreType.fa_rep, w)
    return sf


def pair_energy_matrix(pose, sf):
    """Weighted pairwise interaction energies e_ij.

    Returns a symmetric (n_residues, n_residues) float array in REU with a zero
    diagonal, indexed 0-based (Rosetta numbers residues from 1).

    Each entry is the *weighted* sum over all score terms of the interaction between
    that pair -- i.e. e_ij as Eq. 2 uses it.  One-body terms (fa_dun, ref, p_aa_pp,
    omega, fa_intra_*) are deliberately absent: they are properties of a single residue,
    not interactions between two, and Eq. 2 is built purely from pair terms.  A
    consequence worth remembering: sum(e_ij) does NOT equal the pose total score.
    """
    sf(pose)  # scoring is what populates pose.energies(); without it the graph is stale
    energies = pose.energies()
    weights = sf.weights()
    n = pose.total_residue()

    e = np.zeros((n, n), dtype=np.float64)

    # --- short-range pair terms, from the energy graph ----------------------
    # Iterate the edge list directly rather than probing all n^2 pairs with
    # find_energy_edge: only ~1600 edges exist for a 76-residue protein, and this has to
    # run once per decoy.  EnergyEdge.dot(weights) does the weighted sum in C++.
    graph = energies.energy_graph()
    it, end = graph.const_edge_list_begin(), graph.const_edge_list_end()
    while it.__ne__(end):
        edge = it.dereference()
        i = edge.get_first_node_ind() - 1
        j = edge.get_second_node_ind() - 1
        val = edge.dot(weights)
        e[i, j] += val
        e[j, i] += val
        it.pre_increment()

    # --- long-range pair terms, from their containers -----------------------
    # Swept generically rather than hard-coding ramaprepro_lr: dslf_fa13 appears the same
    # way for disulfide-bonded proteins, and our ubiquitin test case has no disulfides,
    # so hard-coding would have passed the test and been wrong on the next protein.
    for lr_type in methods.LongRangeEnergyType.__members__.values():
        container = energies.long_range_container(lr_type)
        if container is None or container.empty():
            continue
        for res in range(1, n + 1):
            it = container.const_upper_neighbor_iterator_begin(res)
            end = container.const_upper_neighbor_iterator_end(res)
            while it.__ne__(end):
                emap = EMapVector()
                it.retrieve_energy(emap)
                i = res - 1
                j = it.upper_neighbor_id() - 1
                val = emap.dot(weights)
                e[i, j] += val
                e[j, i] += val
                it.pre_increment()

    return e


def onebody_energy_vector(pose, sf):
    """Weighted one-body energy per residue, shape (n_residues,), 0-based.

    Not used by Eq. 2 -- these are single-residue properties, not interactions.  It
    exists so the completeness test can verify that one-body + pairwise accounts for the
    entire score, which is the guarantee that no pair energy is being dropped.
    """
    sf(pose)
    energies = pose.energies()
    weights = sf.weights()
    n = pose.total_residue()
    return np.array(
        [energies.onebody_energies(i).dot(weights) for i in range(1, n + 1)],
        dtype=np.float64,
    )
