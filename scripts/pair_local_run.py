"""Tier 2: the pair-local decoy experiment, parallel and checkpointed.

WHAT IS BEING TESTED. FrustX's production decoy shuffles the whole sequence, so a decoy
energy for contact (i,j) absorbs every other position changing too. The pair-local decoy
changes ONLY i and j and repacks only their 10 A shell. The packer-noise control
(scripts/packer_noise.py) established that the variance this removes is context rather
than Monte-Carlo scatter -- 5.4% pooled packer noise -- so there is real information to
recover. This measures whether recovering it produces a better index.

DESIGN, pre-registered before the run:
  * balanced contact set: N_HIGH frustratometeR-'highly' + N_HIGH fR-'minimally' contacts,
    sampled from the fR reference. Balanced because SE(Cohen d) is set by the smaller
    group, and 1UBQ's 29 'highly' contacts cannot support the comparison at all.
  * NO SAMPLING. The pair-local energy turned out to be an essentially deterministic
    function of (i, j, a_i, a_j) -- see "determinism" below -- and there are only ~400
    ordered identity pairs. So every cell is ENUMERATED once and the decoy mean and sd are
    computed EXACTLY under the known weights q(a,b), rather than estimated from a sample:

        E[e] = SUM_ab q(a,b) e(a,b)        Var[e] = SUM_ab q(a,b) (e(a,b) - E[e])^2

    q is the law the whole-sequence shuffle induces (rao_blackwell.exact_pair_weights), so
    the two schemes differ only in locality, which is the one variable under test. This
    removes cell-selection sampling error entirely and retires the "how many decoys per
    contact" question: the answer is all of them, exactly, at ~2x the cost of 200 sampled.
  * a matched shell-restricted native reference per contact (local_decoy
    .local_native_reference) -- E0 prepared exactly as the decoys are.
  * four numbers reported, never a scatterplot alone: rho vs fR, additive R^2,
    one-body-residualised rho, AUC over fR's classes.

PARALLELISM. The codebase had none; every decoy loop was serial. Contacts are independent,
so the work is split across processes by contact.

  RNG. Rosetta's packer draws from a global RNG, and forked workers would inherit identical
  state. Each worker therefore initialises Rosetta itself with '-constant_seed -jran
  <distinct>', making the run reproducible for a fixed worker count. MEASURED CAVEAT: this
  matters far less than expected for pair-local poses. Repacking ~23 shell residues with
  two fixed identities is a small enough problem that the annealer lands on the same
  optimum every time -- 11 of 12 (contact, identity-pair) combinations tested were
  bit-identical across processes and seeds, the twelfth spread 0.21 REU. The whole-sequence
  decoy, which redesigns every position, is genuinely stochastic by comparison (spread
  ~0.012 REU on the same contact). So enumeration is exact up to a small residual in the
  minority of cells where the packing is ambiguous.

DETERMINISM is what licenses enumeration, and it was measured, not assumed:
scripts/pair_local_run.py was written to sample until that check came back.

Usage:
    .venv/bin/python scripts/pair_local_run.py <PDB_ID> [n_per_class] [workers]
"""
import os
import sys
import time
from multiprocessing import Pool, Value

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

_CTX = {}
_counter = None


def _init(pdb, seed0):
    """Per-worker setup: a distinct Rosetta RNG stream, one pose, one score function pair."""
    global _CTX
    with _counter.get_lock():
        wid = _counter.value
        _counter.value += 1
    import pyrosetta
    from frustx.energies import make_score_function
    # -constant_seed makes packing reproducible; -jran differs per worker so the streams
    # are independent rather than inherited-identical.
    pyrosetta.init(f"-mute all -ignore_unrecognized_res -constant_seed -jran {seed0 + wid}",
                   silent=True)
    from frustx.frustration import residues_from_pose
    from vicinity_profile import load_atoms
    pose = pyrosetta.pose_from_pdb(pdb)
    res = residues_from_pose(pose)
    ca, _ = load_atoms(pdb, atom="CA")
    _CTX = dict(pose=pose,
                xyz=np.array([ca[(r.chain, r.resseq)] for r in res]),
                sf_pack=make_score_function(remove_fa_rep=False),
                sf_measure=make_score_function(remove_fa_rep=True))


def _one_contact(arg):
    """Enumerate every reachable identity pair for one contact, plus its native reference."""
    k, i, j = arg
    from frustx.frustration import contact_energy_matrix
    from local_decoy import local_decoy, local_native_reference
    from rao_blackwell import exact_pair_weights
    c = _CTX
    # Only cells with q > 0 are reachable under the shuffle: an amino acid absent from the
    # sequence can never appear, and a singleton cannot occupy both positions at once.
    q = {ab: w for ab, w in exact_pair_weights(c["pose"].sequence()).items() if w > 0}
    cells = sorted(q)
    t0 = time.perf_counter()
    nat = local_native_reference(c["pose"], c["xyz"], i, j, c["sf_pack"])
    e0 = contact_energy_matrix(nat, c["sf_measure"])[i, j]
    e = np.empty(len(cells))
    for m, (a, b) in enumerate(cells):
        d = local_decoy(c["pose"], c["xyz"], i, j, a, b, c["sf_pack"])
        e[m] = contact_energy_matrix(d, c["sf_measure"])[i, j]
    w = np.array([q[ab] for ab in cells])
    w = w / w.sum()                      # renormalise: q over reachable cells sums to 1
    mean = float(w @ e)
    sd = float(np.sqrt(w @ (e - mean) ** 2))
    return dict(k=k, i=i, j=j, e0=float(e0), mean=mean, sd=sd,
                secs=time.perf_counter() - t0, energies=e,
                cells=np.array(["".join(ab) for ab in cells]), weights=w)


def pick_contacts(pdb_id, fr_pdb, npz, n_per_class, seed=0):
    """Balanced sample of fR-'highly' and fR-'minimally' contacts that FrustX also has."""
    from frustx.contacts import Residue, contact_pairs
    from vicinity_profile import load_atoms
    d = np.load(npz, allow_pickle=True)
    rs, ch, rn = d["resseq"], d["chain"], d["resname"]
    ca, _ = load_atoms(fr_pdb, atom="CA")
    res = [Residue(index=k, chain=str(c), resseq=int(r), icode=" ", resname=str(m))
           for k, (c, r, m) in enumerate(zip(ch, rs, rn))]
    P = contact_pairs(res, np.array([ca[(str(c), int(r))] for c, r in zip(ch, rs)]),
                      cutoff=10.0, min_seq_sep=2)
    fr = pd.read_csv(fr_pdb + "_mutational", sep=r"\s+")
    state = {(min(a, b), max(a, b)): s for a, b, s in zip(fr.Res1, fr.Res2, fr.FrstState)}
    rows = []
    for k, (i, j) in enumerate(P):
        st = state.get((min(rs[i], rs[j]), max(rs[i], rs[j])))
        if st in ("highly", "minimally"):
            rows.append((k, int(i), int(j), st))
    df = pd.DataFrame(rows, columns=["k", "i", "j", "state"])
    rng = np.random.default_rng(seed)
    out = []
    for st in ("highly", "minimally"):
        sub = df[df.state == st]
        take = min(n_per_class, len(sub))
        out.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
        print(f"  fR '{st}': {len(sub)} available, taking {take}")
    return pd.concat(out).reset_index(drop=True)


def main(pdb_id, n_per_class, workers):
    global _counter
    fr_pdb = (f"results/fig2/frustra_mutational/{pdb_id}_A.done/FrustrationData/{pdb_id}_A.pdb"
              if pdb_id != "1UBQ" else
              "results/validation/frustra_mutational/1ubq_A.done/FrustrationData/1ubq_A.pdb")
    npz = (f"results/fig2/decoys/{pdb_id}_min500.npz" if pdb_id != "1UBQ"
           else "results/validation/decoy_samples/1ubq_min200.npz")
    sel = pick_contacts(pdb_id, fr_pdb, npz, n_per_class)
    out = f"results/pair_local/{pdb_id}_enum.npz"
    os.makedirs("results/pair_local", exist_ok=True)

    tasks = [(int(r.k), int(r.i), int(r.j)) for r in sel.itertuples()]
    print(f"{len(tasks)} contacts, all reachable identity cells enumerated, "
          f"on {workers} workers", flush=True)

    _counter = Value("i", 0)
    t0 = time.perf_counter()
    done, results = 0, []
    with Pool(workers, initializer=_init, initargs=(fr_pdb, 12345)) as pool:
        for r in pool.imap_unordered(_one_contact, tasks, chunksize=1):
            results.append(r)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {done}/{len(tasks)}  {el/60:.1f} min elapsed, "
                      f"~{el/done*(len(tasks)-done)/60:.1f} min left", flush=True)
                # Checkpoint every 10 contacts -- same lesson as dump_decoy_samples.py.
                np.savez_compressed(out + ".partial.npz",
                                    **_pack(results, sel))
    np.savez_compressed(out, **_pack(results, sel))
    if os.path.exists(out + ".partial.npz"):
        os.remove(out + ".partial.npz")
    print(f"wrote {out}  ({(time.perf_counter()-t0)/60:.1f} min)")


def _pack(results, sel):
    r = sorted(results, key=lambda x: x["k"])
    state = dict(zip(sel.k, sel.state))
    return dict(
        k=np.array([x["k"] for x in r]), i=np.array([x["i"] for x in r]),
        j=np.array([x["j"] for x in r]), e0=np.array([x["e0"] for x in r]),
        energies=np.array([x["energies"] for x in r]),
        mean=np.array([x["mean"] for x in r]), sd=np.array([x["sd"] for x in r]),
        cells=r[0]["cells"], weights=r[0]["weights"],
        state=np.array([state[x["k"]] for x in r]),
        secs=np.array([x["secs"] for x in r]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1XTQ",
         int(sys.argv[2]) if len(sys.argv) > 2 else 112,
         int(sys.argv[3]) if len(sys.argv) > 3 else 8)
