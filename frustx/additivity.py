"""Splitting a per-contact index into its one-body and contact-specific parts.

WHY THIS IS A FIRST-CLASS OUTPUT AND NOT A CRITICISM.

A contact index F_ij can be written as

    F_ij  =  [ c + a_i + a_j ]  +  [ what is left over ]
             \\_______________/     \\________________/
              one-body / additive     contact-specific

The first bracket is everything a residue carries to ALL of its contacts -- burial,
exposure, local packing density. The second is what is particular to this pair being in
contact with each other.

BOTH ARE REAL QUANTITIES. It is tempting to read a high additive fraction as a defect,
and this document has done so more than once, but that framing is wrong on its own terms:
Ferreiro's single-residue frustration is deliberately a residue-level index, and burial is
genuinely most of what determines whether a contact is well-optimised. A 95%-additive
index is not measuring nothing -- it is measuring a residue property.

What a high additive fraction does mean is narrower and worth stating precisely: such an
index cannot support a claim ABOUT A PARTICULAR CONTACT, because ~all of its variation is
predictable from the two residues' identities alone. Measured here:

    FrustX, bare pair readout, w=0    additive R^2 ~ 0.26
    FrustX, neighbourhood readout     additive R^2 ~ 0.95
    frustratometeR, mutational        additive R^2 ~ 0.95

So rather than choosing between them, both parts are reported per contact. A user asking
"is this residue in a frustrated environment?" wants the additive part; a user asking "is
THIS PAIR frustrated, beyond what its residues would predict?" wants the specific part.
They are different questions and the answer to one is not evidence about the other.

IDENTIFIABILITY -- the one piece of maths that can bite. The design matrix for
c + a_i + a_j is rank deficient: adding t to c while subtracting t/2 from every a_i leaves
every fitted value unchanged (verified: rank 6 of 7 columns on a 6-residue toy, two
coefficient vectors differing by 3.7 giving identical fits). Consequences:

  * `fitted` and `residual` are UNIQUE. Everything reported per contact is well defined.
  * the per-residue COEFFICIENTS are not. numpy's lstsq returns the minimum-norm solution,
    which is deterministic and reproducible, but the absolute level is arbitrary. Only
    DIFFERENCES between residues are meaningful, and only within one run.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class AdditiveDecomposition:
    """Result of splitting an index into one-body and contact-specific parts."""

    fitted: np.ndarray       # (n_contacts,) c + a_i + a_j       -- unique
    residual: np.ndarray     # (n_contacts,) index - fitted      -- unique
    coefficients: np.ndarray  # (n_residues,) a_i                -- only differences mean anything
    intercept: float
    r2: float                # fraction of index variance the additive part explains
    n_used: int              # contacts that entered the fit (NaN indices are excluded)


def additive_decomposition(index, contacts, n_residues):
    """Least-squares fit of index ~ c + a_i + a_j, with one coefficient per POSITION.

    Parameters
    ----------
    index : (n_contacts,) values, one per row of `contacts`. NaN is allowed and excluded
        from the fit; such contacts get NaN for both fitted and residual rather than a
        fabricated number. This matters because sigma=0 contacts are already emitted as
        "undefined" elsewhere, and quietly imputing them here would undo that.
    contacts : (n_contacts, 2) array of 0-based residue indices.
    n_residues : total residues, so coefficients line up with the residue table even for
        residues that happen to have no contacts.

    Note the dummy for a residue is incremented ONCE PER APPEARANCE, so a self-contact
    (i, i) would contribute 2 -- consistent with c + a_i + a_j at i == j. The contact map
    never produces those (min_seq_sep >= 1), but the arithmetic is right if it ever does.
    """
    index = np.asarray(index, dtype=float)
    contacts = np.asarray(contacts)
    ok = np.isfinite(index)

    fitted = np.full(index.shape, np.nan)
    residual = np.full(index.shape, np.nan)
    coefficients = np.zeros(n_residues)

    # A fit needs more contacts than free parameters to mean anything. Below that, report
    # nothing rather than a perfect-looking fit of noise.
    if ok.sum() <= n_residues + 1:
        return AdditiveDecomposition(fitted, residual, coefficients, np.nan, np.nan,
                                     int(ok.sum()))

    X = np.zeros((int(ok.sum()), n_residues + 1))
    X[:, 0] = 1.0
    for row, (i, j) in enumerate(contacts[ok]):
        X[row, 1 + i] += 1.0
        X[row, 1 + j] += 1.0

    y = index[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)   # minimum-norm: deterministic
    fit = X @ beta

    fitted[ok] = fit
    residual[ok] = y - fit
    coefficients[:] = beta[1:]
    var = y.var()
    r2 = float(1.0 - (y - fit).var() / var) if var > 0 else np.nan
    return AdditiveDecomposition(fitted, residual, coefficients, float(beta[0]), r2,
                                 int(ok.sum()))
