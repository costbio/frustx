"""Compare the pair-local index against the whole-sequence one, on identical contacts.

ENDPOINTS ARE PRE-REGISTERED -- this file was written while the run was still in flight,
before any pair-local index had been seen. Four numbers per scheme, never a scatterplot
alone, because a scatterplot cannot distinguish "measures pair frustration better" from
"became more residue-additive", and the readout-scope experiment showed those look
identical (docs/method.md).

    rho          Spearman vs frustratometeR mutational
    additive R2  fraction reducible to c + a_i + a_j (per-position dummies)
    resid rho    Spearman after regressing that additive part out of BOTH sides
    AUC          rank separation of fR 'minimally' vs 'highly'

WHAT WOULD COUNT AS A WIN. Not a higher rho on its own -- w=1 and the neighbourhood
readout both bought that by becoming degenerate. The pair-local scheme wins only if
resid_rho improves without additive_R2 rising to meet it, i.e. if it gains
CONTACT-SPECIFIC agreement. AUC is reported because it motivated the original claim, and
is explicitly NOT treated as evidence on its own: it moved 0.674 -> 0.843 under a pure
readout change with no decoy involved.

Usage:
    .venv/bin/python scripts/pair_local_analysis.py [PDB_ID]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from readout_scope import additive_fit, auc, index_from, spearman


def main(pdb_id="1XTQ"):
    enum = np.load(f"results/pair_local/{pdb_id}_enum.npz", allow_pickle=True)
    ws = np.load(f"results/fig2/decoys/{pdb_id}_min500.npz", allow_pickle=True)
    fr_pdb = (f"results/fig2/frustra_mutational/{pdb_id}_A.done/FrustrationData/{pdb_id}_A.pdb")
    fr = pd.read_csv(fr_pdb + "_mutational", sep=r"\s+")
    key = {(min(a, b), max(a, b)): f for a, b, f in zip(fr.Res1, fr.Res2, fr.FrstIndex)}

    i, j, rs = enum["i"], enum["j"], ws["resseq"]
    # Pair-local index: mean and sd are EXACT under the enumeration, not estimated.
    with np.errstate(divide="ignore", invalid="ignore"):
        f_local = np.where(enum["sd"] > 1e-12,
                           (enum["mean"] - enum["e0"]) / enum["sd"], np.nan)
    # Whole-sequence index for the SAME contacts, from the 500-decoy tensor.
    f_whole = index_from(ws["decoys"][:, i, j], ws["native"][i, j])

    fv = np.array([key.get((min(rs[a], rs[b]), max(rs[a], rs[b])), np.nan)
                   for a, b in zip(i, j)])
    ok = np.isfinite(f_local) & np.isfinite(f_whole) & np.isfinite(fv)
    print(f"{ok.sum()} of {len(ok)} contacts usable "
          f"({(~np.isfinite(f_local)).sum()} pair-local sigma=0, "
          f"{(~np.isfinite(f_whole)).sum()} whole-sequence sigma=0)")

    ii, jj = list(i[ok]), list(j[ok])
    _, rf = additive_fit(fv[ok], ii, jj)
    state = enum["state"][ok]
    rows = []
    for lab, y in [("whole-sequence", f_whole[ok]), ("pair-local", f_local[ok])]:
        r2, res = additive_fit(y, ii, jj)
        rows.append(dict(scheme=lab, n=int(ok.sum()), rho=spearman(y, fv[ok]),
                         additive_r2=r2, resid_rho=spearman(res, rf),
                         auc=auc(y, state == "minimally"),
                         median_index=float(np.median(y)), sd_index=float(np.std(y))))
    df = pd.DataFrame(rows)
    df.to_csv(f"results/pair_local/{pdb_id}_comparison.csv", index=False)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    d_r = df.resid_rho[1] - df.resid_rho[0]
    d_a = df.additive_r2[1] - df.additive_r2[0]
    print(f"\nresid_rho {d_r:+.4f}, additive_R2 {d_a:+.4f}")
    print("VERDICT:", "contact-specific gain" if d_r > 0.05 and d_a < 0.15 else
          "gain is additive, not contact-specific" if d_r > 0.05 else
          "no contact-specific gain")
    print(f"\n-> results/pair_local/{pdb_id}_comparison.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1XTQ")
