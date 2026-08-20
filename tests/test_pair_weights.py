"""Exact decoy-draw probabilities under the shuffling law.

These weights are the whole basis of the Rao-Blackwell estimator: they replace sampled
frequencies with known ones. If they are wrong the estimator is silently biased, so they
are checked against brute-force enumeration of every permutation, not just against
themselves.
"""
import itertools
import sys
from collections import Counter

import pytest

sys.path.insert(0, "scripts")
from rao_blackwell import exact_pair_weights


def brute_force(seq):
    """q(a,b) by enumerating every distinct arrangement of the sequence.

    Positions 0 and 1 stand in for the two contacting positions; under a permutation the
    joint law is the same for any distinct pair of positions.
    """
    perms = list(itertools.permutations(seq))
    c = Counter((p[0], p[1]) for p in perms)
    return {k: v / len(perms) for k, v in c.items()}


def test_weights_sum_to_one():
    q = exact_pair_weights("ACDEFGHIKLMNPQRSTVWY" * 3)
    assert sum(q.values()) == pytest.approx(1.0)


def test_tiny_sequence_by_hand():
    """seq = 'AAB': L=3, n_A=2, n_B=1.

        q(A,A) = 2*1 / (3*2) = 1/3
        q(A,B) = 2*1 / (3*2) = 1/3
        q(B,A) = 1*2 / (3*2) = 1/3
        q(B,B) = 1*0 / (3*2) = 0     <- cannot draw B twice, there is only one
    """
    q = exact_pair_weights("AAB")
    assert q[("A", "A")] == pytest.approx(1 / 3)
    assert q[("A", "B")] == pytest.approx(1 / 3)
    assert q[("B", "A")] == pytest.approx(1 / 3)
    assert q[("B", "B")] == pytest.approx(0.0)


@pytest.mark.parametrize("seq", ["AAB", "AABB", "ABCD", "AAABC", "AABBC"])
def test_matches_brute_force_enumeration(seq):
    q = exact_pair_weights(seq)
    bf = brute_force(seq)
    for k, v in q.items():
        assert v == pytest.approx(bf.get(k, 0.0)), f"{seq} {k}"


def test_diagonal_is_without_replacement():
    """The (n_a - 1) is what distinguishes shuffling from i.i.d. sampling.

    frustratometeR draws WITH replacement, where q(a,a) would be (n_a/L)^2. For 'AAB'
    that is (2/3)^2 = 0.444; without replacement it is 1/3. They are not the same null.
    """
    q = exact_pair_weights("AAB")
    assert q[("A", "A")] == pytest.approx(1 / 3)
    assert q[("A", "A")] != pytest.approx((2 / 3) ** 2)
