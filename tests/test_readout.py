"""Readout scope: hand-verifiable toy matrices.

The neighbourhood readout is a sum over each residue's contact partners, so the two
things that can silently go wrong are (a) non-contacting pairs leaking into the sum and
(b) the shared contact E_ij being counted twice or zero times. Both are checked here on
matrices small enough to verify by mental arithmetic.
"""
import numpy as np
import pytest

from frustx.frustration import apply_readout, contact_mask


def toy():
    """4 residues, contacts (0,1) and (1,2) only.

    Non-contacting pairs are given the absurd value 100 so that any leak is obvious:
    every correct answer below is single-digit.

        E[0,1] = 1     contact
        E[1,2] = 2     contact
        everything else = 100, non-contact
    """
    E = np.full((4, 4), 100.0)
    np.fill_diagonal(E, 0.0)
    E[0, 1] = E[1, 0] = 1.0
    E[1, 2] = E[2, 1] = 2.0
    contacts = np.array([[0, 1], [1, 2]])
    return E, contact_mask(4, contacts)


def test_mask_is_symmetric_and_marks_only_contacts():
    _, M = toy()
    assert M[0, 1] and M[1, 0] and M[1, 2] and M[2, 1]
    assert not M[0, 2] and not M[0, 3] and not M[3, 3]
    assert M.sum() == 4          # two contacts, both directions


def test_pair_readout_is_the_identity():
    E, M = toy()
    assert apply_readout(E, M, "pair") is E


def test_neighbourhood_by_hand():
    E, M = toy()
    nb = apply_readout(E, M, "neighbourhood")
    # row sums over contact partners: row = [1, 1+2, 2, 0]
    # nb[0,1] = row0 + row1 - E01 = 1 + 3 - 1 = 3
    # nb[1,2] = row1 + row2 - E12 = 3 + 2 - 2 = 3
    assert nb[0, 1] == pytest.approx(3.0)
    assert nb[1, 2] == pytest.approx(3.0)


def test_noncontact_energies_never_enter_the_sum():
    """The 100s must be invisible: every entry stays single-digit."""
    E, M = toy()
    nb = apply_readout(E, M, "neighbourhood")
    assert nb.max() < 10.0
    # A non-contacting pair is not decremented (it was in neither row sum):
    # nb[0,2] = row0 + row2 - 0 = 1 + 2 = 3
    assert nb[0, 2] == pytest.approx(3.0)


def test_lone_contact_counts_itself_exactly_once():
    """Two residues, one contact: its neighbourhood IS its pair energy.

    This is the direct test of the '- E_ij' term. Counting it twice would give 10,
    dropping it would give 0; only counting once gives 5.
    """
    E = np.array([[0.0, 5.0], [5.0, 0.0]])
    M = contact_mask(2, np.array([[0, 1]]))
    nb = apply_readout(E, M, "neighbourhood")
    assert nb[0, 1] == pytest.approx(5.0)


def test_neighbourhood_stays_symmetric():
    E, M = toy()
    nb = apply_readout(E, M, "neighbourhood")
    assert np.allclose(nb, nb.T)


def test_unknown_readout_raises():
    E, M = toy()
    with pytest.raises(ValueError, match="unknown readout"):
        apply_readout(E, M, "neighborhood")      # US spelling is NOT accepted
