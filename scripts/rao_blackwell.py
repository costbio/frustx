"""Rao-Blackwellisation: can exact decoy-draw probabilities replace sampled ones?

THE IDEA. Eq. 1's numerator needs E[e_ij] over the decoy law. The plain estimator is the
sample mean over N decoys, which carries Monte-Carlo error in *which identity pairs
happened to be drawn*. But that law is known exactly: FrustX shuffles the native sequence,
so position i gets amino acid a and position j gets b with probability

    q(a,b) = n_a n_b / (L(L-1))          a != b
    q(a,a) = n_a (n_a - 1) / (L(L-1))

Conditioning on the drawn pair and re-weighting by the exact q instead of the empirical
frequency is textbook post-stratification (Rao-Blackwell): SAME estimand, and in the usual
case lower variance. It costs no new Rosetta compute -- decoy sequences replay exactly
from random.Random(k) (verified against PyRosetta; see docs/method.md).

THE CATCH, stated up front because it is the likely outcome. At N=500 a contact sees ~224
of 400 cells with a median of ~1.1 observations each. Post-stratification with singleton
strata is unstable: the cell "mean" IS the single observation, and re-weighting by q/p_hat
can easily *increase* variance rather than reduce it. Two conditionings are therefore
tested, joint and additive-marginal, and the endpoint is measured, not assumed.

ENDPOINT. Subsample n = 250 of the 500 decoys many times and measure, per contact, the SD
of each estimator across replicates. Lower SD at equal N = a better estimator. Bias is
checked separately against the full-500 plain value.

Usage:
    .venv/bin/python scripts/rao_blackwell.py
"""
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from frustx.contacts import Residue, contact_pairs
from frustx.decoys import shuffle_sequence
from vicinity_profile import load_atoms

THREE2ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))

RUNS = [("1UBQ", "results/validation/decoy_samples/1ubq_min200.npz",
         "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb"),
        ("1XTQ", "results/fig2/decoys/1XTQ_min500.npz", "results/fig2/1XTQ_A.pdb"),
        ("1OIV", "results/fig2/decoys/1OIV_min500.npz", "results/fig2/1OIV_A.pdb")]


def exact_pair_weights(seq):
    """q(a,b) for the permutation law: the chance positions (i,j) get identities (a,b).

    Sampling WITHOUT replacement, hence the (L-1) and the (n_a - 1) on the diagonal.
    This is the whole point of the method -- these numbers are known, not estimated.
    """
    n = Counter(seq)
    L = len(seq)
    aas = sorted(n)
    q = {}
    for a in aas:
        for b in aas:
            q[(a, b)] = (n[a] * (n[a] - 1) if a == b else n[a] * n[b]) / (L * (L - 1))
    return q


def plain(e, e0):
    s = e.std()
    return (e.mean() - e0) / s if s > 1e-12 else np.nan


def rb_joint(e, e0, cells, q):
    """Post-stratify on the joint (a_i, a_j) cell.

    q is renormalised over OBSERVED cells only -- an unobserved cell has no mean to
    weight, so its mass has nowhere to go. That renormalisation is itself a bias source
    and is exactly why this may not work; it is measured, not defended.
    """
    mean_num = var_num = wsum = 0.0
    for cell, idx in cells.items():
        w = q.get(cell, 0.0)
        if w <= 0:
            continue
        wsum += w
        mean_num += w * e[idx].mean()
    if wsum <= 0:
        return np.nan
    m = mean_num / wsum
    # Law of total variance: E[Var(e|cell)] + Var(E[e|cell]).
    for cell, idx in cells.items():
        w = q.get(cell, 0.0)
        if w <= 0:
            continue
        var_num += w * (e[idx].var() + (e[idx].mean() - m) ** 2)
    s = np.sqrt(var_num / wsum)
    return (m - e0) / s if s > 1e-12 else np.nan


def rb_additive(e, e0, ai, aj, qa):
    """Condition on the two identities SEPARATELY via an additive fit.

    20 + 20 levels instead of 400 cells, so ~25 observations per level at N=500 rather
    than ~1. Buys conditioning stability at the cost of assuming no pair interaction --
    which is a real assumption, since pair identity is what governs sigma here.
    """
    labs = sorted(set(ai) | set(aj))
    col = {a: k for k, a in enumerate(labs)}
    X = np.zeros((len(e), 1 + 2 * len(labs)))
    X[:, 0] = 1.0
    for r, (a, b) in enumerate(zip(ai, aj)):
        X[r, 1 + col[a]] = 1.0
        X[r, 1 + len(labs) + col[b]] = 1.0
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    # Predicted mean under the EXACT marginal, not the sampled one.
    pred = beta[0] + sum(qa.get(a, 0) * beta[1 + col[a]] for a in labs) \
                   + sum(qa.get(a, 0) * beta[1 + len(labs) + col[a]] for a in labs)
    s = e.std()
    return (pred - e0) / s if s > 1e-12 else np.nan


def analyse(label, npz_path, pdb_path, n_sub=250, n_rep=25, max_contacts=120, seed=0):
    d = np.load(npz_path, allow_pickle=True)
    D, N0, rs, ch, rn = d["decoys"], d["native"], d["resseq"], d["chain"], d["resname"]
    nd = D.shape[0]
    n_sub = min(n_sub, nd // 2)
    native_seq = "".join(THREE2ONE[r] for r in rn)
    seqs = [shuffle_sequence(native_seq, random.Random(k)) for k in range(nd)]
    S = np.array([list(s) for s in seqs])              # (n_decoys, L)

    q = exact_pair_weights(native_seq)
    cnt = Counter(native_seq)
    L = len(native_seq)
    qa = {a: c / L for a, c in cnt.items()}            # exact marginal

    ca, _ = load_atoms(pdb_path, atom="CA")
    xyz = np.array([ca[(str(c), int(r))] for c, r in zip(ch, rs)])
    res = [Residue(index=k, chain=str(c), resseq=int(r), icode=" ", resname=str(m))
           for k, (c, r, m) in enumerate(zip(ch, rs, rn))]
    P = contact_pairs(res, xyz, cutoff=10.0, min_seq_sep=2)

    rng = np.random.default_rng(seed)
    # Restrict to well-defined contacts: sigma~0 ones are the population already shown to
    # be a contact-definition problem, and would swamp a variance comparison with noise.
    sig = D[:, P[:, 0], P[:, 1]].std(0)
    ok = np.where(sig > 0.02)[0]
    pick = rng.choice(ok, size=min(max_contacts, len(ok)), replace=False)

    rows = []
    for k in pick:
        i, j = P[k]
        e_all, e0 = D[:, i, j], N0[i, j]
        ai_all, aj_all = S[:, i], S[:, j]
        est = {"plain": [], "rb_joint": [], "rb_additive": []}
        for _ in range(n_rep):
            sel = rng.choice(nd, size=n_sub, replace=False)
            e, ai, aj = e_all[sel], ai_all[sel], aj_all[sel]
            cells = {}
            for t, (a, b) in enumerate(zip(ai, aj)):
                cells.setdefault((a, b), []).append(t)
            cells = {c: np.array(v) for c, v in cells.items()}
            est["plain"].append(plain(e, e0))
            est["rb_joint"].append(rb_joint(e, e0, cells, q))
            est["rb_additive"].append(rb_additive(e, e0, ai, aj, qa))
        row = {"structure": label, "contact": f"{rs[i]}-{rs[j]}",
               "full500_plain": plain(e_all, e0),
               "n_cells": len(set(zip(ai_all, aj_all)))}
        for name, v in est.items():
            v = np.array(v, dtype=float)
            row[f"sd_{name}"] = np.nanstd(v)
            row[f"mean_{name}"] = np.nanmean(v)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.concat([analyse(*r) for r in RUNS], ignore_index=True)
    df.to_csv("results/rao_blackwell.csv", index=False)
    print(f"{len(df)} contacts, 25 replicates of n=250 decoys each\n")
    print("Sampling SD of the index at fixed N (lower is better):")
    g = df.groupby("structure")[["sd_plain", "sd_rb_joint", "sd_rb_additive"]].median()
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nRatio to plain (>1 means the RB estimator is WORSE):")
    for c in ["sd_rb_joint", "sd_rb_additive"]:
        print(f"  {c:16s} median ratio = {(df[c] / df.sd_plain).median():.3f}")
    print("\nBias check -- mean estimate vs the full-500 plain value:")
    for c in ["plain", "rb_joint", "rb_additive"]:
        print(f"  {c:12s} median (mean_est - full500) = "
              f"{(df[f'mean_{c}'] - df.full500_plain).median():+.4f}")
    print(f"\nOccupied cells per contact (of 400): median {df.n_cells.median():.0f}")
    print("\n-> results/rao_blackwell.csv")


if __name__ == "__main__":
    main()
