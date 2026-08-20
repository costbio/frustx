"""Tests for the Fig. 2 "vicinity" rule (frustratometeR's XAdens).

Coordinates are placed on a straight line so every expected count is checkable by
mental arithmetic. The point being locked in is the one that is easy to get wrong:
a contact is counted at the MIDPOINT of its two residues, not at either endpoint,
so a residue accumulates contacts it takes no part in.
"""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "scripts")
from vicinity_profile import profile


def _setup(positions):
    """Residues on the x axis at the given positions; CA and CB coincide."""
    ca = {("A", i + 1): np.array([x, 0.0, 0.0]) for i, x in enumerate(positions)}
    return ca, dict(ca)


def _contacts(rows):
    return pd.DataFrame(
        [{"ci": "A", "i": i, "cj": "A", "j": j, "index": v} for i, j, v in rows]
    )


def test_contact_is_counted_at_its_midpoint_not_its_endpoints():
    """Residues 1 and 3 are 20 A apart, so neither is within 5 A of the other.

    Their contact's midpoint is at x=10, which is exactly where residue 2 sits. So the
    contact must be counted for residue 2 -- which takes no part in it -- and for
    NEITHER of the two residues that form it.
    """
    ca, sel = _setup([0.0, 10.0, 20.0])
    prof = profile(_contacts([(1, 3, 2.0)]), ca, sel, radius=5.0)
    got = prof.set_index("Res")["Total"].to_dict()
    assert got == {1: 0, 2: 1, 3: 0}


def test_radius_is_strict():
    """The R code uses `< radius`, not `<=`. A midpoint at exactly 5.0 A is excluded."""
    ca, sel = _setup([0.0, 5.0, 10.0])
    # Contact 1-3 has midpoint x=5.0: distance to residue 1 is exactly 5.0.
    prof = profile(_contacts([(1, 3, 2.0)]), ca, sel, radius=5.0)
    got = prof.set_index("Res")["Total"].to_dict()
    assert got[1] == 0, "midpoint at exactly the radius must be excluded"
    assert got[2] == 1, "midpoint at distance 0 is inside"


def test_classification_cut_points_are_inclusive_on_both_ends():
    """highly is F <= -1.0, minimally is F >= 0.78; the open interval is neutral."""
    ca, sel = _setup([0.0])
    con = _contacts([(1, 1, -1.0), (1, 1, 0.78), (1, 1, -0.999), (1, 1, 0.779)])
    prof = profile(con, ca, sel, radius=5.0)
    row = prof.iloc[0]
    assert (row["Total"], row["nHighlyFrst"], row["nMinimallyFrst"]) == (4, 1, 1)
    assert row["nNeutrallyFrst"] == 2


def test_undefined_indices_are_dropped_not_counted_as_neutral():
    """sigma = 0 contacts carry a NaN index. Counting them as neutral would inflate
    Total and silently bias every relative fraction."""
    ca, sel = _setup([0.0])
    prof = profile(_contacts([(1, 1, np.nan), (1, 1, 2.0)]), ca, sel, radius=5.0)
    row = prof.iloc[0]
    assert row["Total"] == 1 and row["nMinimallyFrst"] == 1


def test_relative_fractions_are_zero_when_nothing_is_nearby():
    """A residue with no contacts in range must not divide by zero."""
    ca, sel = _setup([0.0, 100.0])
    prof = profile(_contacts([(2, 2, 2.0)]), ca, sel, radius=5.0).set_index("Res")
    assert prof.loc[1, "Total"] == 0
    assert prof.loc[1, "relHighlyFrustrated"] == 0.0
    assert prof.loc[1, "relMinimallyFrustrated"] == 0.0
