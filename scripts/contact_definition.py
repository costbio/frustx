"""Contact definition: does CB-CB fix anything a Ca-Ca cutoff gets wrong?

The paper defines a contact as Ca-Ca <= 10 A. That admits pairs whose side chains point
away from each other, for which no REF2015 term fires at all -- 23.7% of contacts across
the six GTPase runs have decoy sigma <= 0.02, where Eq. 1 is 0/0 or nearly so. Those
contacts are measurably less reliable (split-half 0.897 vs 0.988 on 1XTQ), and they are
also part of why FrustX and frustratometeR disagree about which contacts exist (475 vs
389 on 1UBQ).

This script tests whether switching the representative atom fixes it, on two endpoints
that must be kept apart: agreement with frustratometeR, and FrustX own internal quality.
The answer differs between them -- see "Contact definition" in docs/method.md.

Reads existing tensors only; no PyRosetta.

Usage:
    .venv/bin/python scripts/contact_definition.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from frustx.contacts import Residue, contact_pairs
from readout_scope import additive_fit, spearman, auc, index_from

def atoms(pdb, which):
    # (chain, resseq) -> representative coord. CB falls back to CA for Gly.
    ca, cb = {}, {}
    for l in open(pdb):
        if not l.startswith("ATOM") or l[16] not in (" ", "A"): continue
        k = (l[21], int(l[22:26])); nm = l[12:16].strip()
        xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        if nm == "CA": ca[k] = xyz
        if nm == "CB": cb[k] = xyz
    return {k: (cb.get(k, v) if which == "CB" else v) for k, v in ca.items()}

RUNS = [("1UBQ", "results/validation/decoy_samples/1ubq_min200.npz",
         "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb")] + \
       [(p, f"results/fig2/decoys/{p}_min500.npz",
         f"results/fig2/frustra_mutational/{p}_A.done/FrustrationData/{p}_A.pdb")
        for p in ["1XTQ","1XTS","1KAO","2RAP","1OIV","1OIW"]]

rows = []
for lab, npz, frp in RUNS:
    d = np.load(npz, allow_pickle=True)
    D, N, rs, ch, rn = d["decoys"], d["native"], d["resseq"], d["chain"], d["resname"]
    fr = pd.read_csv(frp + "_mutational", sep=r"\s+")
    key = {(min(a,b),max(a,b)):(f,s) for a,b,f,s in zip(fr.Res1,fr.Res2,fr.FrstIndex,fr.FrstState)}
    res = [Residue(index=k, chain=str(c), resseq=int(r), icode=" ", resname=str(m))
           for k,(c,r,m) in enumerate(zip(ch, rs, rn))]
    for which, cut in [("CA",10.0), ("CB",9.5), ("CB",9.0)]:
        co = atoms(frp, which)
        X = np.array([co[(str(c), int(r))] for c, r in zip(ch, rs)])
        P = contact_pairs(res, X, cutoff=cut, min_seq_sep=2)
        i, j = P[:,0], P[:,1]
        idx = index_from(D[:, i, j], N[i, j])
        sig = D[:, i, j].std(0)
        keep, ii, jj, y, st = [], [], [], [], []
        for k in range(len(P)):
            kk = (min(rs[i[k]], rs[j[k]]), max(rs[i[k]], rs[j[k]]))
            if kk in key and np.isfinite(idx[k]):
                y.append(idx[k]); keep.append(key[kk][0]); st.append(key[kk][1])
                ii.append(int(i[k])); jj.append(int(j[k]))
        y = np.array(y); f = np.array(keep); st = np.array(st)
        _, rf = additive_fit(f, ii, jj); r2, rs_ = additive_fit(y, ii, jj)
        m = np.isin(st, ["minimally","highly"])
        rows.append(dict(structure=lab, defn=f"{which}<={cut}", n_contacts=len(P),
                         frac_dead=float((sig <= 0.02).mean()), n_shared=len(y),
                         rho=spearman(y, f), resid_rho=spearman(rs_, rf),
                         auc=auc(y[m], st[m] == "minimally")))
df = pd.DataFrame(rows)
df.to_csv("results/contact_definition.csv", index=False)
print(df.groupby("defn")[["n_contacts","frac_dead","n_shared","rho","resid_rho","auc"]]
        .mean().to_string(float_format=lambda v: f"{v:+.4f}"))
print("\n-> results/contact_definition.csv")
