"""Regenerate the classification-threshold tables in docs/method.md.

Compares the shape of the three index distributions on 1UBQ and the class fractions
each produces under the borrowed AWSEM cut points (0.78 / -1.0), then reports what
cut points percentile-matching to the reference would demand.

This decides nothing -- see docs/method.md, "Classification thresholds are borrowed,
and measurably wrong", for why percentile matching is rejected. It exists so the
numbers quoted there can be reproduced rather than trusted.

Usage:
    .venv/bin/python scripts/threshold_report.py
"""
import csv
import numpy as np

MIN_T, HIGH_T = 0.78, -1.0


def load_frustx(path):
    """Load the index column, dropping contacts with no defined index.

    17 of 475 contacts on 1UBQ sit at the 10 A cutoff edge, where no decoy
    changes the energy at all, so sigma = 0 and the index is undefined
    (docs/method.md:846). They carry class "undefined" and must not be counted
    in a threshold calibration -- including them as if they were neutral would
    dilute every class fraction by ~3.6%.
    """
    with open(path) as fh:
        return np.array([float(r["frustration_index"]) for r in csv.DictReader(fh)
                         if r["frustration_index"] != ""])


def load_fr(path):
    """frustratometeR outputs are whitespace-delimited with a header line."""
    with open(path) as fh:
        rows = [ln.split() for ln in fh.read().splitlines() if ln.strip()]
    hdr = rows[0]
    k = hdr.index("FrstIndex")
    return np.array([float(r[k]) for r in rows[1:]])


def describe(name, v):
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    # Class fractions under the borrowed AWSEM cut points.
    f_min = float((v >= MIN_T).mean())
    f_high = float((v <= HIGH_T).mean())
    print(f"{name:<28} n={len(v):>4}  mean={v.mean():+.3f}  sd={v.std(ddof=1):.3f}")
    print(f"{'':28}  min={q[0]:+.2f} p5={q[1]:+.2f} p25={q[2]:+.2f} "
          f"med={q[3]:+.2f} p75={q[4]:+.2f} p95={q[5]:+.2f} max={q[6]:+.2f}")
    print(f"{'':28}  under 0.78/-1.0: minimally {f_min:6.1%}  "
          f"neutral {1-f_min-f_high:6.1%}  highly {f_high:6.1%}")
    return f_min, f_high


B = "results/validation"
fx = load_frustx(f"{B}/frustx_min500/contacts.csv")
cf = load_fr(f"{B}/frustra/1ubq_A.done/FrustrationData/1ubq_A.pdb_configurational")
mu = load_fr(f"{B}/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb_mutational")

print("=== index distributions, 1UBQ ===")
fx_min, fx_high = describe("FrustX (REF2015, min500)", fx)
cf_min, cf_high = describe("frustratometeR config.", cf)
mu_min, mu_high = describe("frustratometeR mutational", mu)

# If we instead demanded that FrustX call the *same fraction* of contacts
# minimally / highly frustrated as the reference does, where would the cut
# points have to sit? (Percentile matching -- one candidate rule, not a decision.)
print("\n=== percentile-matched FrustX cut points ===")
for name, fmin, fhigh in [("vs configurational", cf_min, cf_high),
                          ("vs mutational", mu_min, mu_high)]:
    t_min = float(np.quantile(fx, 1.0 - fmin))
    t_high = float(np.quantile(fx, fhigh))
    print(f"{name:<22} minimally >= {t_min:+.3f}   highly <= {t_high:+.3f}   "
          f"(targets {fmin:.1%} / {fhigh:.1%})")
