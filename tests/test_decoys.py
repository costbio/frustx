"""Tests for decoy generation.

The property that matters most is that the backbone never moves: the whole method rests
on decoys sharing the native's exact backbone, so any drift silently invalidates every
comparison. It is asserted after every mutating operation here rather than once.

Tests use protocol="min" (chi-only minimisation) because it is ~10x faster than
FastRelax and these tests are checking mechanics, not the scientific choice between them.
"""

import random
from collections import Counter

import numpy as np
import pytest

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.decoys import (  # noqa: E402
    apply_sequence,
    backbone_coords,
    make_decoy,
    native_reference,
    relax_sidechains,
    repack,
    shuffle_sequence,
)
from frustx.energies import init_rosetta, make_score_function  # noqa: E402

# A heterogeneous sequence: shuffling poly-alanine is a no-op and would make the
# composition and sequence-change assertions vacuous.
HELIX_SEQ = "AEALKKLAEELKKG"


@pytest.fixture(scope="module")
def helix():
    """Mixed-sequence alpha helix, built from ideal torsions. No data file."""
    init_rosetta()
    pose = pyrosetta.pose_from_sequence(HELIX_SEQ)
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    return pose


@pytest.fixture(scope="module")
def sf_pack():
    # Packing keeps fa_rep -- it is what stops rotamers overlapping.
    return make_score_function(remove_fa_rep=False)


# --- shuffle_sequence is pure, and needs no Rosetta ------------------------

def test_shuffle_preserves_composition():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 2
    out = shuffle_sequence(seq, random.Random(0))
    assert Counter(out) == Counter(seq)
    assert len(out) == len(seq)


def test_shuffle_is_reproducible_and_seed_dependent():
    seq = "AEALKKLAEELKKG"
    a = shuffle_sequence(seq, random.Random(1))
    b = shuffle_sequence(seq, random.Random(1))
    c = shuffle_sequence(seq, random.Random(2))
    assert a == b, "same seed must give the same shuffle"
    assert a != c, "different seeds should give different shuffles"


# --- everything below mutates a pose --------------------------------------

def test_apply_sequence_realises_sequence_without_moving_backbone(helix, sf_pack):
    target = shuffle_sequence(helix.sequence(), random.Random(3))
    pose = helix.clone()
    before = backbone_coords(pose)

    apply_sequence(pose, target, sf_pack)

    assert pose.sequence() == target
    np.testing.assert_array_equal(backbone_coords(pose), before)


def test_apply_sequence_rejects_wrong_length(helix, sf_pack):
    with pytest.raises(ValueError, match="sequence length"):
        apply_sequence(helix.clone(), "AAA", sf_pack)


def test_repack_keeps_sequence_and_backbone(helix, sf_pack):
    pose = helix.clone()
    before = backbone_coords(pose)
    repack(pose, sf_pack)
    assert pose.sequence() == HELIX_SEQ
    np.testing.assert_array_equal(backbone_coords(pose), before)


@pytest.mark.parametrize("protocol", ["min", "none"])
def test_relax_never_moves_backbone(helix, sf_pack, protocol):
    pose = helix.clone()
    before = backbone_coords(pose)
    relax_sidechains(pose, sf_pack, protocol=protocol)
    np.testing.assert_array_equal(backbone_coords(pose), before)


def test_relax_rejects_unknown_protocol(helix, sf_pack):
    with pytest.raises(ValueError, match="protocol must be one of"):
        relax_sidechains(helix.clone(), sf_pack, protocol="wiggle")


def test_make_decoy_shuffles_but_preserves_backbone_and_composition(helix, sf_pack):
    before = backbone_coords(helix)
    decoy = make_decoy(helix, sf_pack, seed=7, protocol="min")

    assert Counter(decoy.sequence()) == Counter(HELIX_SEQ)
    assert decoy.sequence() != HELIX_SEQ, "seed 7 should actually permute this sequence"
    np.testing.assert_array_equal(backbone_coords(decoy), before)
    # The native pose must be left alone -- callers reuse it for every decoy.
    assert helix.sequence() == HELIX_SEQ


def test_native_reference_does_not_change_sequence(helix, sf_pack):
    before = backbone_coords(helix)
    ref = native_reference(helix, sf_pack, protocol="min")
    assert ref.sequence() == HELIX_SEQ
    np.testing.assert_array_equal(backbone_coords(ref), before)


def test_decoy_sequence_is_reproducible_for_a_given_seed(helix, sf_pack):
    """Sequence choice is seeded and must repeat exactly.

    Note this asserts the SEQUENCE only, not the packed coordinates: Rosetta's packer
    draws from its own global RNG, so identical rotamers are not guaranteed without
    fixing Rosetta's seed at init too.
    """
    a = make_decoy(helix, sf_pack, seed=11, protocol="none")
    b = make_decoy(helix, sf_pack, seed=11, protocol="none")
    assert a.sequence() == b.sequence()
