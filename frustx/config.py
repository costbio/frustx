"""Tunable defaults, kept free of heavy imports.

This module exists so `frustx --help` does not have to load PyRosetta. The CLI needs
these values to build its argument parser; importing them from decoys.py or
frustration.py would drag in PyRosetta (~2.5 s) just to print usage text.

Nothing here may import pyrosetta, numpy or Bio.
"""

# Ca-Ca contact cutoff in Angstrom.
#   "Protein contacts are defined by the CaCa distances between residues, and a cutoff
#    of 10 A is used."
DEFAULT_CUTOFF = 10.0

# "giving 1000 appropriately distributed decoys for each contact"
DEFAULT_N_DECOYS = 1000

# Side-chain relaxation protocol. The paper says "short Monte-Carlo relaxation" without
# naming a mover, and the readings differ in cost and outcome (measured on a shuffled
# ubiquitin decoy; 1000-decoy cost in brackets):
#
#   "min"    chi-only gradient minimisation   0.35 s -> 413 REU   [~6 min]
#   "relax"  FastRelax, backbone frozen       3.63 s -> 429 REU   [~1 h]
#
# "relax" is the more literal reading -- FastRelax is Monte-Carlo based, minimisation is
# not. Since the relaxation protocol sets the decoy spread, and that spread is the
# DENOMINATOR of Eq. 1, this choice moves every frustration index. Settled empirically
# in validation, not by preference.
RELAX_PROTOCOLS = ("relax", "min", "none")

# Classification thresholds, taken from frustratometeR so output is comparable.
# NOTE: calibrated on the AWSEM coarse-grained energy function, NOT atomistic REF2015.
# Almost certainly the wrong cut points here; recalibrating is part of validation.
MINIMALLY_FRUSTRATED = 0.78
HIGHLY_FRUSTRATED = -1.0

# Weight on Eq. 2's many-body background term in
#
#     E_ij = e_ij + w * 1/2 (R_i + R_j)
#
# w = 1 is Eq. 2 as the paper writes it (its literal form is 1/2(R_i + R_j), which
# differs from w=1 only by the e_ij term -- measured to be indistinguishable, since
# e_ij ~ -0.7 REU against 1/2(R_i+R_j) ~ -10 REU).
#
# DEFAULT IS 0, a deliberate departure from the paper. At w=1 the index is degenerate:
# validated against frustratometeR on 1UBQ it reports ZERO frustrated contacts where
# frustratometeR reports 30, and 86% of its per-contact signal is reducible to a
# residue-level quantity. At w=0 the pair-specificity (0.44), spread (0.92) and glycine
# behaviour all match frustratometeR closely. See docs/method.md for the sweep.
#
# This does NOT fix the correlation with frustratometeR, which sits at rho ~ 0.15
# regardless of w. It fixes the character of the index, not its agreement.
DEFAULT_BACKGROUND_WEIGHT = 0.0


# --- Readout scope -----------------------------------------------------------------
#
# WHAT ENERGY GOES INTO Eq. 1, as opposed to how much background is mixed into it
# (DEFAULT_BACKGROUND_WEIGHT, above). The two are orthogonal knobs and are easy to
# confuse: `background_weight` changes the *definition* of a contact energy E_ij;
# `readout` changes *which set of contact energies* Eq. 1 is applied to.
#
#   "pair"          E_ij alone -- what the paper's Eq. 1 scores, and FrustX's default.
#   "neighbourhood" E_ij plus every other contact energy touching i or j.
#
# "neighbourhood" exists because it is what frustratometeR actually sums. Its mutational
# decoy energy is not a pair energy at all: fix_backbone.cpp:5214-5243 sums
# water(i,j) + burial_i + burial_j + sum_k water(i,k) + sum_k water(j,k). Comparing
# FrustX's bare pair energy against that is a scope mismatch, not just a force-field
# difference, and it is the single largest identified driver of the two tools'
# disagreement -- larger than the whole background-weight sweep.
#
# It is NOT the default, and should not become one on current evidence: the
# neighbourhood readout is ~94% reducible to a residue-additive function, i.e. it buys
# agreement partly by becoming the same kind of degenerate quantity that w=1 does. It is
# provided so the scope axis can be measured separately from the force-field axis.
# See "Readout scope" in docs/method.md.
DEFAULT_READOUT = "pair"
READOUT_SCOPES = ("pair", "neighbourhood")


# --- Ligand contact cutoff ----------------------------------------------------------
#
# A ligand has no CA and no CB, so the paper's Ca-Ca criterion is not merely inaccurate
# for it -- it is UNDEFINED. Any pair involving a non-protein residue therefore falls back
# to a minimum heavy-atom distance, and needs its own cutoff.
#
# 6.0 A is not arbitrary: it is exactly Rosetta's etable interaction radius
# (`score:fa_max_dis` = 6.0), so it is the distance beyond which REF2015 pair terms stop
# firing between heavy atoms. Measured on 1XTQ: every one of the 186 protein-protein pairs
# within 6.0 A minimum-heavy carries a nonzero e_ij (0.0% energetically dead), against
# 2.9% dead at 6.5 A, 7.5% at 7.0 A and 30.9% at 8.0 A.
#
# AMBIGUITY, stated rather than hidden. 6.0 A is NOT calibrated to match the protein
# branch: only 73% of the pairs CA <= 10.0 admits also satisfy heavy-min <= 6.0, and the
# count-matched equivalent of CA <= 10.0 (248 pairs) is about 7.0 A (240 pairs), not 6.0
# (186 pairs). Hydrogens also carry REF2015 past 6.0 A of HEAVY-atom separation --
# protein-protein pairs show nonzero e_ij out to 7.85 A minimum-heavy. So 6.0 A is the
# conservative choice (every admitted pair is energetically live) rather than the
# count-matched one, and it is exposed as a parameter because the paper says nothing
# about ligands at all.
DEFAULT_LIGAND_CUTOFF = 6.0


# --- Contact representative atom ----------------------------------------------------
#
# "CA" is the paper's definition, quoted in contacts.py, and stays the default so runs
# remain comparable with everything measured so far.
#
# MEASURED CASE FOR "CB", recorded so the default can be revisited deliberately rather
# than drifting. A Ca-Ca cutoff admits pairs whose side chains point away from each
# other, for which no REF2015 term fires at all: across the six GTPase runs, 1824 of 7687
# contacts (23.7%) have decoy sigma <= 0.02, and Eq. 1 is then 0/0 or near it. Those
# contacts are genuinely less reliable, not merely quiet -- split-half reliability on
# 1XTQ is 0.897 for them against 0.988 for the rest.
#
# At MATCHED contact counts CB separates the dead pairs better than either alternative
# tested (~4000 contacts kept: 0.4% dead by CB, 1.8% by minimum heavy-atom distance,
# 3.1% by CA). It beats minimum heavy-atom distance because CB encodes side-chain
# DIRECTION -- two residues can have close backbones while their side chains point apart.
#
#   CA <= 10.0 (current) 7687 kept, 23.7% dead
#   CB <=  9.5 (fR's)    5713 kept,  6.0% dead
#   CB <=  9.0           5103 kept,  3.0% dead
#   CB <=  8.5           4532 kept,  1.0% dead
#
# This is also part of why FrustX and frustratometeR disagree on which contacts exist at
# all (475 vs 389 on 1UBQ, only 344 shared). See "Contact definition" in docs/method.md.
DEFAULT_CONTACT_ATOM = "CA"
CONTACT_ATOMS = ("CA", "CB")
