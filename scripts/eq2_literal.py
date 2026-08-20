"""The literal Eq. 2 experiment: why the paper's own index reports no frustration.

Chen et al. Eq. 2 is

    E_ij = e_ij + 1/2 sum_{k!=j} e_ik + 1/2 sum_{l!=i} e_jl

The two exclusions make the direct term cancel exactly, leaving E_ij = 1/2 (R_i + R_j)
with R_i = sum_k e_ik. Verified here term-by-term rather than asserted (step 1 below),
because every downstream conclusion rests on it.

FrustX's own w parameter is NOT that. frustation.contact_energy_matrix computes
e_ij + w*1/2(R_i+R_j), which keeps e_ij on top and so overshoots the literal form by
exactly e_ij at w=1. Here we use the interpolation that actually lands on the paper:

    E_ij(w) = (1-w) * e_ij  +  w * 1/2 (R_i + R_j)

w=0 is FrustX's default (bare pair energy), w=1 is Eq. 2 verbatim. Both endpoints are
then the thing they claim to be, which the production path's parameterisation cannot say.

Everything is recomputed from the retained decoy tensor, so all w share identical decoys
and no difference between them is decoy noise. No Rosetta run required.

Usage:
    .venv/bin/python scripts/eq2_literal.py [npz] [pdb]
"""
import sys

import numpy as np


def ca_coords_from_pdb(path):
    """CA coordinates and (chain, resseq) labels, in file order.

    Parsed directly rather than via PyRosetta: the tensor's residue labels came from the
    pose, so we assert agreement with them in main() instead of trusting either source.
    """
    xyz, labels = [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16] == " CA ":
            # altloc: keep only the primary conformer, else a residue appears twice
            if line[16] not in (" ", "A"):
                continue
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            labels.append((line[21], int(line[22:26])))
    return np.array(xyz), labels


def index_at_w(e_dec, e_nat, w, contacts):
    """Eq. 1 index at background weight w, for the given contact list.

    Returns (index, numerator, sigma), each of shape (n_contacts,). Sign is FrustX's
    convention: positive = minimally frustrated.
    """
    # R_i per decoy and for the native. axis=-1 sums over partners k.
    R_dec = e_dec.sum(axis=2)                     # (n_decoys, n_res)
    R_nat = e_nat.sum(axis=1)                     # (n_res,)

    B_dec = 0.5 * (R_dec[:, :, None] + R_dec[:, None, :])   # (n_decoys, n, n)
    B_nat = 0.5 * (R_nat[:, None] + R_nat[None, :])

    E_dec = (1.0 - w) * e_dec + w * B_dec
    E_nat = (1.0 - w) * e_nat + w * B_nat

    i, j = contacts[:, 0], contacts[:, 1]
    d = E_dec[:, i, j]                            # (n_decoys, n_contacts)
    num = d.mean(axis=0) - E_nat[i, j]            # <E_decoy> - E_native
    sig = d.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.where(sig > 0, num / sig, np.nan)
    return idx, num, sig


def main(npz_path, pdb_path, cutoff=10.0, min_seq_sep=2):
    d = np.load(npz_path, allow_pickle=True)
    e_dec, e_nat = d["decoys"], d["native"]
    n = e_nat.shape[0]

    # ---- step 1: is the collapse OUR ARITHMETIC, or is it real? -------------------
    lit = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s_i = sum(e_nat[i, k] for k in range(n) if k not in (i, j))
            s_j = sum(e_nat[j, l] for l in range(n) if l not in (i, j))
            lit[i, j] = e_nat[i, j] + 0.5 * s_i + 0.5 * s_j
    R = e_nat.sum(axis=1)
    coll = 0.5 * (R[:, None] + R[None, :])
    off = ~np.eye(n, dtype=bool)
    print(f"[1] Eq.2 written out vs 1/2(R_i+R_j): max abs diff = "
          f"{np.abs(lit - coll)[off].max():.2e}   (float noise => collapse is exact)")

    # ---- contacts, from the structure the tensor was built from -------------------
    xyz, labels = ca_coords_from_pdb(pdb_path)
    assert len(xyz) == n, f"PDB has {len(xyz)} CA, tensor has {n} residues"
    assert [l[1] for l in labels] == list(d["resseq"]), "PDB/tensor numbering disagree"
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    ii, jj = np.triu_indices(n, k=min_seq_sep)
    keep = dist[ii, jj] <= cutoff
    contacts = np.column_stack([ii[keep], jj[keep]])
    print(f"    contacts: {len(contacts)}  (CaCa <= {cutoff} A, |i-j| >= {min_seq_sep})")

    # ---- step 2: is our e_ij EXTRACTION sane in magnitude? ------------------------
    ci, cj = contacts[:, 0], contacts[:, 1]
    B_nat = np.abs(coll[ci, cj])
    ev = np.abs(e_nat[ci, cj])
    # MEAN, not median. 42% of contacts sit near the 10 A cutoff with |e_ij| < 0.05 REU,
    # so the median of |e_ij| describes the cutoff edge rather than a typical contact and
    # inflates the ratio to ~68x. The mean is the honest comparison and reproduces the
    # ~14x quoted in config.py.
    print(f"[2] on contacts: mean |e_ij| = {ev.mean():.3f} REU"
          f"  (median {np.median(ev):.3f}; {100*(ev<0.05).mean():.0f}% are < 0.05, cutoff edge)")
    print(f"    mean |1/2(R_i+R_j)| = {B_nat.mean():.3f} REU"
          f"  -> background is {B_nat.mean()/ev.mean():.1f}x the direct term")

    # ---- step 3: the sweep -------------------------------------------------------
    print(f"\n[3] index vs w   (n={len(contacts)} contacts, {e_dec.shape[0]} decoys)")
    print(f"{'w':>5} {'min F':>8} {'median F':>9} {'max F':>8} "
          f"{'med num':>9} {'med sigma':>10} {'#high':>6} {'#neut':>6} {'#min':>5} {'num>0':>7}")
    for w in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        idx, num, sig = index_at_w(e_dec, e_nat, w, contacts)
        ok = np.isfinite(idx)
        nh = int((idx[ok] <= -1.0).sum())
        nm = int((idx[ok] >= 0.78).sum())
        nn = int(ok.sum()) - nh - nm
        print(f"{w:>5.2f} {idx[ok].min():>8.2f} {np.median(idx[ok]):>9.2f} "
              f"{idx[ok].max():>8.2f} {np.median(num):>9.2f} {np.median(sig):>10.2f} "
              f"{nh:>6} {nn:>6} {nm:>5} {100*(num>0).mean():>6.1f}%")

    # ---- step 4: WHY. the one-body tautology --------------------------------------
    # At w=1 the contact energy depends on i and j only through their own totals R.
    # Shuffling the sequence degrades a real protein's energy at essentially every
    # position, so R_nat < <R_decoy> almost everywhere -- which forces the numerator
    # positive for every contact, independent of whether that contact is frustrated.
    R_dec_mean = e_dec.sum(axis=2).mean(axis=0)
    better = (R.sum() and (R < R_dec_mean))
    print(f"\n[4] residues where native R_i beats the decoy mean: "
          f"{better.sum()}/{n} ({100*better.mean():.1f}%)")
    print(f"    native   sum_i R_i = {R.sum():>9.2f} REU")
    print(f"    decoy    sum_i R_i = {R_dec_mean.sum():>9.2f} REU  (mean over decoys)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/validation/decoy_samples/1ubq_min200.npz",
         sys.argv[2] if len(sys.argv) > 2 else "results/validation/1ubq.pdb")
