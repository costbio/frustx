"""Build the JSON the explainer page renders, entirely from runs already on disk.

Source of truth is the decoy *energy tensor* (200 decoys x 76 x 76) rather than a
finished contact table, because the page draws real decoy histograms -- and a histogram
cannot be reconstructed from a mean and a sigma. Recomputing Eq. 1 from the tensor
means every number on the page (index, decomposition, histogram, the fR comparison)
comes from one self-consistent ensemble.

  results/validation/decoy_samples/1ubq_min200.npz   FrustX, 1UBQ, 200 decoys, min, w=0
  results/validation/frustra*/                       frustratometeR, same input PDB

Written to results/artefact/payload.json so the page can be rebuilt without PyRosetta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_frustratometer import spearman, spearman_p  # noqa: E402

from frustx.additivity import additive_decomposition  # noqa: E402
from frustx.contacts import contact_pairs, load_ca  # noqa: E402
from frustx.config import MINIMALLY_FRUSTRATED, HIGHLY_FRUSTRATED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "artefact"
NPZ = ROOT / "results/validation/decoy_samples/1ubq_min200.npz"
PDB = ROOT / "results/validation/frustra/1ubq_A.done/FrustrationData/1ubq_A.pdb"
FR = {
    "configurational": ROOT / "results/validation/frustra/1ubq_A.done/FrustrationData/1ubq_A.pdb_configurational",
    "mutational": ROOT / "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb_mutational",
}
CUTOFF, MIN_SEP = 10.0, 2          # matches the archived run these decoys came from


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ, allow_pickle=True)
    decoys, native, resseq, resname = d["decoys"], d["native"], d["resseq"], d["resname"]
    L, n_dec = len(resseq), decoys.shape[0]

    # The tensor is dense over all 76x76 residue pairs, but a *contact* is a geometric
    # object -- Ca-Ca <= 10 A. Recovering the set by "which pairs have a non-zero
    # energy" would silently admit 881 pairs instead of 475, so the contact map is
    # rebuilt from the same PDB the run used. frustx.contacts has no PyRosetta
    # dependency, which is why this is cheap.
    residues, ca = load_ca(PDB)
    assert [r.resseq for r in residues] == list(resseq), "PDB/tensor residue mismatch"
    pairs = contact_pairs(residues, ca, cutoff=CUTOFF, min_seq_sep=MIN_SEP)
    ii, jj = pairs[:, 0], pairs[:, 1]
    e_nat = native[ii, jj]
    e_dec = decoys[:, ii, jj]                           # (n_dec, n_contacts)

    mu, sd = e_dec.mean(axis=0), e_dec.std(axis=0, ddof=1)
    # Eq. 1 with FrustX's sign flip: positive = native below the decoy mean = minimally
    # frustrated. sigma == 0 is a real degeneracy (a pair with no atomistic interaction
    # in any sequence, e.g. two backbone-only Gly), not a numerical accident -- it is
    # dropped rather than clamped, which is what compute_frustration does.
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(sd > 0, (mu - e_nat) / sd, np.nan)

    dec = additive_decomposition(index, pairs, L)

    ok = np.isfinite(index)
    contacts = pd.DataFrame({
        "i": ii, "j": jj,
        "ri": resseq[ii], "rj": resseq[jj],
        "ni": resname[ii], "nj": resname[jj],
        "e_nat": e_nat, "mu": mu, "sd": sd,
        "fidx": index, "onebody": dec.fitted, "specific": dec.residual,
    })
    contacts["key"] = list(zip(contacts.ri, contacts.rj))

    comparison = {}
    for mode, path in FR.items():
        fr = pd.read_csv(path, sep=r"\s+")
        fr["key"] = list(zip(fr.Res1, fr.Res2))
        m = contacts.merge(fr[["key", "FrstIndex", "Welltype"]], on="key").dropna(subset=["fidx"])
        rho = spearman(m.fidx, m.FrstIndex)
        comparison[mode] = {
            "n_fr": int(len(fr)), "n_shared": int(len(m)),
            "rho": rho, "p": spearman_p(rho, len(m)),
            "rho_specific": spearman(m.specific, m.FrstIndex),
            "median": float(fr.FrstIndex.median()), "sd": float(fr.FrstIndex.std()),
            "min": float(fr.FrstIndex.min()), "max": float(fr.FrstIndex.max()),
            "n_frustrated": int((fr.FrstIndex < HIGHLY_FRUSTRATED).sum()),
            "n_minimal": int((fr.FrstIndex > MINIMALLY_FRUSTRATED).sum()),
            "points": [[round(a, 3), round(b, 3)] for a, b in zip(m.fidx, m.FrstIndex)],
        }

    # Three worked contacts: the largest contact-specific value, a frustrated one, and a
    # neutral one -- so the page can show the same machinery producing three answers.
    finite = contacts[ok]
    picks = {
        "minimal": finite.loc[finite.specific.idxmax()],
        "frustrated": finite.loc[finite.fidx.idxmin()],
        "neutral": finite.iloc[(finite.fidx - 0.0).abs().argsort().iloc[0]],
    }
    worked = {}
    for label, row in picks.items():
        col = int(np.flatnonzero((ii == row.i) & (jj == row.j))[0])
        samples = e_dec[:, col]
        lo, hi = samples.min(), samples.max()
        pad = 0.15 * (hi - lo) if hi > lo else 1.0
        edges = np.linspace(min(lo, row.e_nat) - pad, max(hi, row.e_nat) + pad, 33)
        counts, _ = np.histogram(samples, bins=edges)
        worked[label] = {
            "label": f"{row.ni}{row.ri}-{row.nj}{row.rj}",
            "e_nat": float(row.e_nat), "mu": float(row.mu), "sd": float(row.sd),
            "index": float(row.fidx), "onebody": float(row.onebody),
            "specific": float(row.specific),
            "bins": [round(float(x), 4) for x in edges],
            "counts": [int(c) for c in counts],
        }

    payload = {
        "protein": "1UBQ", "n_residues": int(L), "n_decoys": int(n_dec),
        "protocol": "min", "cutoff": CUTOFF, "min_seq_sep": MIN_SEP,
        "n_contacts": int(ok.sum()), "n_dropped_sigma0": int((~ok).sum()),
        "additive_r2": float(dec.r2),
        "thresholds": {"minimal": MINIMALLY_FRUSTRATED, "frustrated": HIGHLY_FRUSTRATED},
        "stats": {
            "median": float(finite.fidx.median()), "sd": float(finite.fidx.std()),
            "min": float(finite.fidx.min()), "max": float(finite.fidx.max()),
            "n_minimal": int((finite.fidx > MINIMALLY_FRUSTRATED).sum()),
            "n_frustrated": int((finite.fidx < HIGHLY_FRUSTRATED).sum()),
        },
        "residues": [{"n": int(a), "aa": str(b)} for a, b in zip(resseq, resname)],
        "contacts": [
            [int(r.ri), int(r.rj), round(float(r.fidx), 3),
             round(float(r.onebody), 3), round(float(r.specific), 3)]
            for r in finite.itertuples()
        ],
        "top_specific": [
            {"pair": f"{r.ni}{r.ri}-{r.nj}{r.rj}", "index": round(float(r.fidx), 2),
             "onebody": round(float(r.onebody), 2), "specific": round(float(r.specific), 2)}
            for r in finite.nlargest(6, "specific").itertuples()
        ],
        "most_frustrated": [
            {"pair": f"{r.ni}{r.ri}-{r.nj}{r.rj}", "index": round(float(r.fidx), 2),
             "onebody": round(float(r.onebody), 2), "specific": round(float(r.specific), 2)}
            for r in finite.nsmallest(6, "fidx").itertuples()
        ],
        "worked": worked,
        "comparison": comparison,
    }
    (OUT / "payload.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"contacts={payload['n_contacts']} dropped={payload['n_dropped_sigma0']} "
          f"R2={payload['additive_r2']:.4f}")
    print(json.dumps({k: payload[k] for k in ("stats", "top_specific", "most_frustrated")}, indent=2))
    print(json.dumps({m: {k: v for k, v in c.items() if k != "points"}
                      for m, c in comparison.items()}, indent=2))
    print("bytes:", (OUT / "payload.json").stat().st_size)


if __name__ == "__main__":
    main()
