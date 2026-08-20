"""Readout scope: does Eq. 1 see a bare pair energy, or the neighbourhood frustratometeR sums?

WHY THIS EXISTS. Every previous FrustX-vs-frustratometeR comparison put our bare e_ij
against their neighbourhood sum. frustratometeR's mutational decoy energy is
water(i,j) + burial_i + burial_j + sum_k water(i,k) + sum_k water(j,k)
(AWSEM fix_backbone.cpp:5214-5243) -- not a pair energy at all. So the comparison
confounded two independent axes: the force field (REF2015 vs AWSEM) and the READOUT
SCOPE. This script isolates the scope axis by holding the decoy ensemble, the energy
function and Eq. 1 completely fixed and changing only what Eq. 1 is applied to.

It needs no PyRosetta: the stored (N, n, n) tensors already contain every pair energy,
so both readouts are recoverable from runs already on disk.

FOUR NUMBERS, pre-specified. A scatterplot cannot distinguish "agrees better" from
"became more residue-additive", and that distinction has overturned three conclusions in
docs/method.md already. So every scheme is reported as:

    rho          Spearman vs frustratometeR mutational
    additive R2  fraction of the index reducible to c + a_i + a_j (per-position dummies)
    resid rho    Spearman AFTER regressing that additive part out of BOTH sides
                 -- the only one of the four that can show contact-specific agreement
    AUC          rank separation of fR's 'minimally' vs 'highly' classes

Usage:
    .venv/bin/python scripts/readout_scope.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from frustx.contacts import Residue, contact_pairs
from frustx.frustration import apply_readout, contact_mask

sys.path.insert(0, "scripts")
from vicinity_profile import load_atoms

# (label, decoy tensor, frustratometeR reference dir)
RUNS = [
    ("1UBQ", "results/validation/decoy_samples/1ubq_min200.npz",
     "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb"),
] + [
    (p, f"results/fig2/decoys/{p}_min500.npz",
     f"results/fig2/frustra_mutational/{p}_A.done/FrustrationData/{p}_A.pdb")
    for p in ["1XTQ", "1XTS", "1KAO", "2RAP", "1OIV", "1OIW"]
]


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def additive_fit(y, ii, jj):
    """Least-squares fit of y ~ c + a_i + a_j with one dummy per POSITION.

    Returns (R2, residuals). This is the one-body control: a_i absorbs everything about
    residue i that is the same at all of its contacts (burial, exposure, local packing),
    so what survives in the residual is contact-specific by construction.
    """
    labs = sorted(set(ii) | set(jj))
    col = {a: k for k, a in enumerate(labs)}
    X = np.zeros((len(y), len(labs) + 1))
    X[:, 0] = 1.0
    for r, (a, b) in enumerate(zip(ii, jj)):
        X[r, 1 + col[a]] += 1.0
        X[r, 1 + col[b]] += 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1.0 - res.var() / y.var(), res


def auc(scores, positive):
    """P(score of a 'minimally' contact > score of a 'highly' one). Rank-based on purpose:
    the class-separation statistic that moved in the old pair-local experiment was a
    SCALE statistic, which any variance-shrinking scheme inflates mechanically."""
    r = pd.Series(scores).rank().to_numpy()
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def index_from(tensor_slice, native_slice):
    """Eq. 1 at w=0, conventional sign. sigma=0 -> NaN, explicitly not neutral."""
    m, s = tensor_slice.mean(0), tensor_slice.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 1e-12, (m - native_slice) / s, np.nan)


def analyse(label, npz_path, fr_pdb):
    d = np.load(npz_path, allow_pickle=True)
    D, N, resseq = d["decoys"], d["native"], d["resseq"]

    # Contact map from the frustratometeR-side PDB so both tools index the same residues.
    ca, _ = load_atoms(fr_pdb, atom="CA")
    res_ca = np.array([ca[(c, int(r))] for c, r in zip(d["chain"], resseq)])
    n = len(resseq)

    residues = [Residue(index=k, chain=str(c), resseq=int(r), icode=" ", resname=str(nm))
                for k, (c, r, nm) in enumerate(zip(d["chain"], resseq, d["resname"]))]
    pairs = contact_pairs(residues, res_ca, cutoff=10.0, min_seq_sep=2)
    mask = contact_mask(n, pairs)

    i, j = pairs[:, 0], pairs[:, 1]
    out = {}
    for scope in ("pair", "neighbourhood"):
        # Applied per-structure BEFORE Eq. 1 -- see apply_readout's docstring.
        Dt = np.stack([apply_readout(D[k], mask, scope) for k in range(D.shape[0])])
        Nt = apply_readout(N, mask, scope)
        out[scope] = index_from(Dt[:, i, j], Nt[i, j])

    fr = pd.read_csv(fr_pdb + "_mutational", sep=r"\s+")
    key = {(min(a, b), max(a, b)): (f, s)
           for a, b, f, s in zip(fr.Res1, fr.Res2, fr.FrstIndex, fr.FrstState)}

    rows = []
    for k in range(len(pairs)):
        kk = (min(resseq[i[k]], resseq[j[k]]), max(resseq[i[k]], resseq[j[k]]))
        if kk in key and np.isfinite(out["pair"][k]) and np.isfinite(out["neighbourhood"][k]):
            rows.append((out["pair"][k], out["neighbourhood"][k],
                         key[kk][0], key[kk][1], int(i[k]), int(j[k])))
    if not rows:
        return None
    A = np.array([[r[0], r[1], r[2]] for r in rows])
    state = np.array([r[3] for r in rows])
    ii = [r[4] for r in rows]
    jj = [r[5] for r in rows]

    _, rf = additive_fit(A[:, 2], ii, jj)
    recs = []
    for c, scope in [(0, "pair"), (1, "neighbourhood")]:
        r2, rs = additive_fit(A[:, c], ii, jj)
        keep = np.isin(state, ["minimally", "highly"])
        recs.append(dict(
            structure=label, scope=scope, n=len(A),
            rho=spearman(A[:, c], A[:, 2]),
            additive_r2=r2,
            resid_rho=spearman(rs, rf),
            auc=auc(A[keep, c], state[keep] == "minimally"),
        ))
    return pd.DataFrame(recs)


def main():
    frames = [f for f in (analyse(*r) for r in RUNS) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("results/readout_scope.csv", index=False)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nMean over the 7 structures:")
    print(df.groupby("scope")[["rho", "additive_r2", "resid_rho", "auc"]]
            .mean().to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\n-> results/readout_scope.csv")


if __name__ == "__main__":
    main()
