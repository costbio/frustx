"""Dump the full per-decoy contact energy tensor, not just its mean and sd.

`compute_frustration` accumulates running sums and throws each decoy away, which is
right for production but leaves us unable to ask what *shape* the decoy energy
distribution has. That shape is the whole question behind the classification
thresholds: FrustX's index is a genuine Z-score, so a cut at Z = 1 only carries a
probabilistic meaning ("fewer than ~16% of random substitutions do this well") if the
decoy energies are approximately normal. Borrowing AWSEM's 0.78 / -1.0 assumes a
distribution shape we have never checked.

Output is an .npz holding:
    decoys  (n_decoys, n_res, n_res)  E_ij for every decoy
    native  (n_res, n_res)            E_ij for the repacked native reference
    resseq, chain, resname            residue labels, so contacts can be joined back

Memory is not a concern: 1UBQ is 76 residues, so 200 decoys is ~9 MB. RAM does become a
wall for big proteins -- the tensor is 8*N*n^2 bytes, so n=500 at N=1000 is 2 GB.

CHECKPOINTING. This script used to build the whole tensor in RAM and write only after the
final decoy, so a job killed at 499/500 lost everything -- which is exactly what happened
to a fig2 run. It now writes <out>.partial.npz every CHECKPOINT_EVERY decoys and resumes
from it, re-entering the loop at the first unfinished decoy.

Resumption reproduces the SEQUENCE ensemble exactly -- decoy k's shuffle depends only on
seed=k (frustx/decoys.py:130) -- but NOT the energies bit-for-bit. Rosetta is initialised
without -constant_seed (frustx/energies.py:37), so the packer draws from a global RNG that
differs between processes: decoy k built after a resume has the same sequence as decoy k
built before it, and a slightly different packing. A resumed run is therefore statistically
equivalent to an uninterrupted one, not identical to it. That is the correct standard here
(the decoy ensemble is a sample, not a fixed object), but it does mean a resumed run cannot
be used to reproduce an earlier run's exact numbers.

The native reference is stored in the checkpoint and NOT recomputed on resume. It goes
through a repack whose packer draws from Rosetta's global RNG, which is not seeded per
call, so recomputing it would silently give a different E0 for the second half of the run.

PARALLEL. Decoys are independent, so this forks a worker pool. Two things change from
the serial version and both matter:

  * Completion is OUT OF ORDER, so the checkpoint cannot be a "first n_done decoys are
    finished" high-water mark any more -- that would be meaningless. It is a boolean
    `done` mask of length n_decoys, and a resume submits exactly the unfinished indices.
  * The tensor is allocated with np.full(nan) rather than np.empty. With a prefix count
    and np.empty, a checkpoint claiming n_done=35 while decoys 25-29 were still in flight
    would have shipped UNINITIALISED MEMORY into the final .npz, and nothing downstream
    would have noticed.

Usage:
    .venv/bin/python scripts/dump_decoy_samples.py <pdb> <out.npz> [n_decoys] [protocol] [n_jobs]

    Re-running the same command after an interruption resumes; delete <out>.partial.npz
    to force a clean start.
"""
import json
import os
import sys

import numpy as np
import pyrosetta

from frustx.decoys import make_decoy, native_reference
from frustx.energies import init_rosetta, make_score_function
from frustx.config import DEFAULT_BACKGROUND_WEIGHT
from frustx.frustration import contact_energy_matrix, residues_from_pose
from frustx import provenance
from frustx.energies import DEFAULT_INIT_FLAGS, DEFAULT_WEIGHTS


# Every 25 decoys is ~50 s of work at the measured 'min' cost on a 169-residue protein --
# small enough that a teardown costs little, large enough that rewriting the partial
# tensor stays a rounding error on the total.
CHECKPOINT_EVERY = 25

# Set by main() before any checkpoint is touched. Module-level only so the loader can
# name which settings differ; nothing else should read it.
_SETTINGS = {}


def _load_checkpoint(path, n_decoys, n, key):
    """Return (decoys, native, done_mask) from a partial run, or (None, None, None).

    The shape check alone was NOT enough, and the gap was a live corruption path rather
    than a theoretical one. It compares two numbers -- decoy count and residue count --
    so resuming a `min` checkpoint with `relax` on the command line passed it and mixed
    two ensembles into one tensor. Worse, `native` is loaded from the checkpoint and
    deliberately not recomputed (see the module docstring), so E0 stayed pinned to the
    FIRST protocol while every post-resume decoy used the second: the reference and the
    ensemble end up prepared differently, which is the exact failure decoys.py:16-19
    exists to prevent.

    `key` is the "ensemble" regeneration key -- see frustx/provenance.py. It covers the
    settings that determine what a decoy structure IS, which is what has to match for
    two halves of a tensor to belong together.

    A checkpoint written before this check existed carries no key. It is refused rather
    than accepted, because "no key" is indistinguishable from "written under different
    settings", and the whole point here is to stop guessing.
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        d = np.load(path, allow_pickle=True)
        done = d["done"].astype(bool)
        if d["decoys"].shape != (n_decoys, n, n) or done.shape != (n_decoys,):
            print(f"ignoring {path}: shape mismatch", flush=True)
            return None, None, None
        old_key = str(d["regeneration_key"]) if "regeneration_key" in d else None
        if old_key != key:
            what = "predates the settings check" if old_key is None else "settings differ"
            print(f"REFUSING to resume {path}: {what}.", flush=True)
            if old_key is not None and "settings" in d:
                for line in provenance.describe_mismatch(
                        json.loads(str(d["settings"])), _SETTINGS, "ensemble"):
                    print(f"    {line}", flush=True)
            print("    Delete it to start over, or fix the command line.", flush=True)
            # sys.exit, not "start fresh": silently overwriting hours of decoys because
            # a flag was mistyped is the more expensive mistake of the two.
            sys.exit(1)
        print(f"resuming from {path}: {done.sum()}/{n_decoys} decoys already done",
              flush=True)
        return d["decoys"], d["native"], done
    except Exception as exc:                       # a truncated write is not fatal
        print(f"ignoring unreadable checkpoint {path}: {exc}", flush=True)
        return None, None, None


def main(pdb, out, n_decoys, protocol, n_jobs=1):
    init_rosetta()
    pose = pyrosetta.pose_from_pdb(pdb)

    # Same two score functions the production path uses: fa_rep ON for packing so
    # decoys are physically buildable, OFF for measurement so a single clash cannot
    # dominate the contact energy.
    sf_pack = make_score_function(remove_fa_rep=False)
    sf_measure = make_score_function(remove_fa_rep=True)

    n = pose.total_residue()
    partial = str(out) + ".partial.npz"

    # The settings that decide what a decoy structure IS. This script hardcodes several
    # of them (seed=k per decoy, repeats=1, readout="pair", background_weight at its
    # default), so they are written out literally rather than plumbed -- but they are
    # written out, because a later edit to any of them must invalidate old checkpoints.
    global _SETTINGS
    _SETTINGS = {
        "structure_key": provenance.file_key(pdb),
        "init_flags": DEFAULT_INIT_FLAGS,
        "weights": DEFAULT_WEIGHTS,
        "protocol": protocol,
        "repeats": 1,
        "seed": 0,
        "n_decoys": n_decoys,
        "packing_seed": None,
        # This script never builds a ligand pose today, but the field is required rather
        # than defaulted: regeneration_key raises on a missing setting precisely so a key
        # computed without it cannot compare equal to one computed with it.
        "freeze_ligand": True,
        # Same reasoning: required, not defaulted. A list of ligand content keys; empty
        # because this script scores a protein-only pose.
        "ligands": (),
    }
    key = provenance.regeneration_key(_SETTINGS, "ensemble")
    decoys, native, done = _load_checkpoint(partial, n_decoys, n, key)

    if decoys is None:
        # The native goes through the same repack as the decoys -- comparing an
        # unrelaxed native against relaxed decoys would bias every index positive. It is
        # built ONCE, in the parent, and stored in the checkpoint: a worker rebuilding it
        # under a different RNG stream would give a different E0 for part of the run.
        nat_pose = native_reference(pose, sf_pack, protocol=protocol)
        native = contact_energy_matrix(nat_pose, sf_measure)
        # NaN, not np.empty: an unfinished slot must be obviously unfinished. See the
        # module docstring on why np.empty plus a prefix count was actively dangerous.
        decoys = np.full((n_decoys, n, n), np.nan, dtype=np.float64)
        done = np.zeros(n_decoys, dtype=bool)

    res = residues_from_pose(pose)
    labels = dict(
        resseq=np.array([r.resseq for r in res]),
        chain=np.array([r.chain for r in res]),
        resname=np.array([r.resname for r in res]),
    )

    todo = np.flatnonzero(~done).tolist()
    n_jobs = max(1, min(n_jobs, len(todo))) if todo else 1

    def save_checkpoint():
        # Write to a temp path and rename: a kill DURING the checkpoint write would
        # otherwise leave a truncated file where a good one used to be.
        tmp = partial + ".tmp.npz"
        # The key and the settings that produced it travel WITH the tensor. Storing the
        # key alone would make a mismatch undiagnosable -- you would know the checkpoint
        # was wrong without knowing which flag to fix.
        np.savez_compressed(tmp, decoys=decoys, native=native, done=done,
                            regeneration_key=key, settings=json.dumps(_SETTINGS),
                            **labels)
        os.replace(tmp, partial)

    def record(k, E):
        # Indexed by the k carried in the RESULT, never by arrival order -- under
        # out-of-order completion those are different, and using arrival order would
        # permute the tensor rows. Nothing downstream would complain: mean, sigma and
        # index are all order-invariant, but scripts/packer_noise.py and the
        # Rao-Blackwell analysis join row k to seed k and would be silently wrong.
        decoys[k] = E
        done[k] = True

    print(f"{len(todo)} decoys to build on {n_jobs} worker(s)", flush=True)
    if n_jobs == 1:
        for count, k in enumerate(todo, 1):
            record(k, contact_energy_matrix(
                make_decoy(pose, sf_pack, seed=k, protocol=protocol), sf_measure))
            if count % 10 == 0:
                print(f"{int(done.sum())}/{n_decoys}", flush=True)
            if count % CHECKPOINT_EVERY == 0 and count < len(todo):
                save_checkpoint()
                print(f"  checkpoint at {int(done.sum())}", flush=True)
    else:
        import multiprocessing as mp

        from frustx.energies import make_fa_rep_score_function
        from frustx.frustration import _WORKER, _worker_init, _worker_decoy, contact_mask
        ctx = mp.get_context("fork")
        # readout="pair" with an all-True mask: this script wants the raw contact energy
        # matrix, the same quantity the serial version stored.
        # sf_fa_rep is required by _worker_decoy even though this script discards the
        # fa_rep matrix below: the worker computes both parts unconditionally, and
        # omitting the key is a KeyError raised inside a forked child, hours in.
        _WORKER.update(pose=pose, sf_pack=sf_pack, sf_measure=sf_measure,
                       sf_fa_rep=make_fa_rep_score_function(),
                       mask=None, seed=0, protocol=protocol, repeats=1,
                       background_weight=DEFAULT_BACKGROUND_WEIGHT, readout="pair",
                       packing_seed=None)
        counter = ctx.Value("i", 0)
        # Fresh RNG base per invocation. A hardcoded base would make every run of this
        # command produce a bit-identical tensor, and would make a RESUME redraw packer
        # randomness from stream positions the pre-crash run already consumed.
        jran_base = int.from_bytes(os.urandom(4), "little") % (2 ** 31 - n_jobs - 1)
        try:
            with ctx.Pool(n_jobs, initializer=_worker_init,
                          initargs=(jran_base, counter)) as pool:
                # Three-tuple: the fa_rep part is dropped here. This script's tensor
                # is the plain measured energy, and widening its on-disk format is a
                # separate decision from computing the extra column.
                for count, (k, E, _) in enumerate(
                        pool.imap_unordered(_worker_decoy, todo, chunksize=1), 1):
                    record(k, E)
                    if count % 10 == 0:
                        print(f"{int(done.sum())}/{n_decoys}", flush=True)
                    if count % CHECKPOINT_EVERY == 0 and count < len(todo):
                        save_checkpoint()
                        print(f"  checkpoint at {int(done.sum())}", flush=True)
        finally:
            _WORKER.clear()

    if not done.all():
        raise RuntimeError(f"only {int(done.sum())}/{n_decoys} decoys completed")
    assert not np.isnan(decoys).any(), "a decoy slot was never written"

    np.savez_compressed(out, decoys=decoys, native=native, **labels)
    if os.path.exists(partial):
        os.remove(partial)          # only once the real output is safely on disk
    print(f"wrote {out}  decoys={decoys.shape}")


if __name__ == "__main__":
    # Default protocol is "min" so this matches results/validation/frustx_min500,
    # the run whose index distribution motivated the question.
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 200,
         sys.argv[4] if len(sys.argv) > 4 else "min",
         int(sys.argv[5]) if len(sys.argv) > 5 else 1)
