"""Is the decoy energy distribution normal enough for a Z-score threshold to mean anything?

FrustX's index is (decoy_mean - native)/sigma, and the sign convention makes a positive
index mean the native is *better* than the decoy mean. Reading a cut at Z = 0.78 as
"fewer than ~22% of random substitutions do this well" is a normal-distribution claim.
This script tests that claim two ways, on the tensor dumped by dump_decoy_samples.py:

1. Shape. Skewness and excess kurtosis per contact, plus a D'Agostino-Pearson K^2
   omnibus test. K^2 is chi-square(2) under normality, so the p-value needs no scipy.

2. Calibration, which is the question we actually care about. For each contact, the
   *empirical* fraction of decoys at least as good as the native (E_decoy <= E_native)
   is compared against the normal prediction Phi(-index). If a contact scoring +0.78
   really is beaten by ~22% of decoys, the borrowed threshold at least means what it
   says; if the empirical tail is far off, Z thresholds are not interpretable and the
   cut points must come from empirical quantiles instead.

Usage:
    .venv/bin/python scripts/test_decoy_normality.py <samples.npz> [pdb] [min_seq_sep]

min_seq_sep defaults to 2 to match results/validation/frustx_min500/run.json, the
production run whose index distribution raised the threshold question. Passing 1
adds the 75 sequential (i, i+1) pairs back.
"""
import math
import sys

import numpy as np

from frustx.contacts import contact_pairs, load_ca


def norm_cdf(x):
    """Standard normal CDF via erf. Vectorised over a numpy array."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def chi2_sf_2df(x):
    """Survival function of chi-square with 2 df, which is just exp(-x/2)."""
    return np.exp(-x / 2.0)


def dagostino_k2(a):
    """D'Agostino-Pearson K^2 for each row of `a` (n_contacts, n_samples).

    Transforms skewness and kurtosis to approximately standard normal Z1, Z2 and
    returns K^2 = Z1^2 + Z2^2, which is ~chi-square(2) under the null. Formulas are
    the standard ones (D'Agostino, Belanger & D'Agostino 1990); written out rather
    than pulled from scipy because scipy is not a dependency of this repo.
    """
    n = a.shape[1]
    m = a.mean(axis=1, keepdims=True)
    d = a - m
    m2 = (d ** 2).mean(axis=1)
    m3 = (d ** 3).mean(axis=1)
    m4 = (d ** 4).mean(axis=1)
    # Guard against zero-variance contacts; they are filtered out by the caller.
    m2 = np.maximum(m2, 1e-300)

    b1 = m3 / m2 ** 1.5                      # skewness
    b2 = m4 / m2 ** 2                        # kurtosis (3.0 under normality)

    # Skewness -> Z1
    y = b1 * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1.0 + math.sqrt(2.0 * (beta2 - 1.0))
    delta = 1.0 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1.0))
    z1 = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1.0))

    # Kurtosis -> Z2
    e_b2 = 3.0 * (n - 1.0) / (n + 1.0)
    var_b2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
    x = (b2 - e_b2) / math.sqrt(var_b2)
    sqrt_b1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) * math.sqrt(
        6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0)))
    a_ = 6.0 + 8.0 / sqrt_b1 * (2.0 / sqrt_b1 + math.sqrt(1.0 + 4.0 / sqrt_b1 ** 2))
    term = (1.0 - 2.0 / a_) / (1.0 + x * math.sqrt(2.0 / (a_ - 4.0)))
    z2 = (1.0 - 2.0 / (9.0 * a_) - np.cbrt(term)) / math.sqrt(2.0 / (9.0 * a_))

    return b1, b2 - 3.0, z1 ** 2 + z2 ** 2


def main(npz_path, pdb_path, min_seq_sep):
    z = np.load(npz_path)
    decoys, native = z["decoys"], z["native"]          # (K, n, n), (n, n)
    n_decoys = decoys.shape[0]

    # Restrict to real contacts -- the tensor is dense but most entries are not contacts.
    residues, ca = load_ca(pdb_path)
    pairs = contact_pairs(residues, ca, min_seq_sep=min_seq_sep)

    samples = np.array([decoys[:, i, j] for i, j in pairs])   # (n_contacts, K)
    nat = np.array([native[i, j] for i, j in pairs])

    sd = samples.std(axis=1)
    keep = sd > 1e-9                                   # drop the sigma = 0 cutoff-edge contacts
    samples, nat, sd = samples[keep], nat[keep], sd[keep]
    mean = samples.mean(axis=1)
    index = (mean - nat) / sd                          # FrustX sign convention

    print(f"contacts: {len(pairs)} total, {keep.sum()} with sigma > 0, {n_decoys} decoys each\n")

    skew, exkurt, k2 = dagostino_k2(samples)
    p = chi2_sf_2df(k2)
    print("=== 1. shape of the decoy energy distribution ===")
    for name, v in [("skewness", skew), ("excess kurtosis", exkurt)]:
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"  {name:<16} med={q[2]:+.2f}  IQR=[{q[1]:+.2f},{q[3]:+.2f}]  "
              f"p5={q[0]:+.2f} p95={q[4]:+.2f}")
    for alpha in (0.05, 0.01):
        print(f"  normality rejected at p<{alpha}: {(p < alpha).mean():.1%} of contacts")

    print("\n=== 2. does a Z of 0.78 mean what it claims? ===")
    # Empirical: fraction of decoys at least as good as (i.e. energy <= ) the native.
    emp = (samples <= nat[:, None]).mean(axis=1)
    pred = norm_cdf(-index)
    print(f"  {'index band':<16} {'n':>4} {'empirical tail':>15} {'normal predicts':>16}")
    for lo, hi, label in [(-99, -1.0, "<= -1.0 (highly)"), (-1.0, 0.0, "-1.0 .. 0"),
                          (0.0, 0.78, "0 .. 0.78"), (0.78, 1.5, "0.78 .. 1.5 (min)"),
                          (1.5, 99, ">= 1.5")]:
        m = (index > lo) & (index <= hi)
        if m.sum():
            print(f"  {label:<16} {m.sum():>4} {emp[m].mean():>14.1%} {pred[m].mean():>15.1%}")

    # A single headline number: how far off is the normal approximation in the tail
    # region the thresholds actually live in?
    band = (index > 0.5) & (index < 1.5)
    if band.sum():
        print(f"\n  in the threshold region (0.5 < Z < 1.5, n={band.sum()}): "
              f"empirical {emp[band].mean():.1%} vs normal {pred[band].mean():.1%}")

    # The decisive question is not whether the *average* tail matches, but whether Z
    # pins the tail down at all. If contacts sharing a Z have wildly different
    # empirical tails, Z is not a usable classifier no matter where the cut is put.
    print("\n=== 3. how tightly does Z pin down the actual tail? ===")
    print(f"  {'Z band':<14} {'n':>4} {'p10':>7} {'median':>8} {'p90':>7}   normal says")
    for lo, hi in [(0.4, 0.6), (0.7, 0.9), (1.0, 1.2), (1.4, 1.6)]:
        m = (index > lo) & (index <= hi)
        if m.sum() >= 3:
            q = np.percentile(emp[m], [10, 50, 90])
            mid = norm_cdf(np.array([-(lo + hi) / 2]))[0]
            print(f"  {lo:.1f} - {hi:.1f}      {m.sum():>4} {q[0]:>6.1%} {q[1]:>7.1%} "
                  f"{q[2]:>6.1%}   {mid:>6.1%}")

    # What Z would we have to pick to hit a chosen empirical tail? Reported as the
    # median Z among contacts whose empirical tail is near the target.
    print("\n=== 4. Z implied by an empirical-tail definition ===")
    for target in (0.05, 0.10, 0.20, 0.80, 0.90):
        m = np.abs(emp - target) < 0.03
        if m.sum() >= 3:
            q = np.percentile(index[m], [25, 50, 75])
            print(f"  tail = {target:>4.0%}: Z median {q[1]:+.2f}  "
                  f"IQR [{q[0]:+.2f}, {q[2]:+.2f}]  (n={m.sum()})")


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else "results/validation/1ubq.pdb",
         int(sys.argv[3]) if len(sys.argv) > 3 else 2)
