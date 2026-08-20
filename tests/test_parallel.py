"""Parallel decoy generation: correctness, not speed.

The failure mode this guards against is not a crash. If forked workers share Rosetta's
global RNG they produce correlated packings, sigma is quietly understated, and every
number still looks plausible. Packer noise is only ~5.4% of decoy variance
(docs/method.md), so even total correlation across 8 workers shifts sigma by ~2.7% --
far too small to catch statistically in a fast test. So these checks are EXACT.
"""
import numpy as np
import pytest

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.energies import init_rosetta, make_score_function  # noqa: E402
from frustx.frustration import _check_accounting, compute_frustration  # noqa: E402


def _report_seed(q):
    """Report this worker's RNG seed WITHOUT the real initializer (negative control)."""
    from pyrosetta.rosetta.numeric.random import rg
    q.put(rg().get_seed())


def _init_and_report(jran_base, counter, q):
    """Run the production initializer, then report the seed it produced.

    Reported from the INITIALIZER, not from a task: with trivial tasks a single worker
    can grab all of them and report its own seed N times, which looks like every worker
    sharing a stream. The initializer runs exactly once per worker, so this counts right.
    """
    from frustx.frustration import _worker_init
    _worker_init(jran_base, counter)
    from pyrosetta.rosetta.numeric.random import rg
    q.put(rg().get_seed())

HELIX_SEQ = "AEALKKLAEELKKG"


@pytest.fixture(scope="module")
def helix():
    init_rosetta()
    pose = pyrosetta.pose_from_sequence(HELIX_SEQ)
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    return pose


@pytest.fixture(scope="module")
def sfs():
    return make_score_function(remove_fa_rep=False), make_score_function(remove_fa_rep=True)


@pytest.fixture(scope="module")
def pinned_serial(helix, sfs):
    return compute_frustration(helix, *sfs, n_decoys=6, seed=0, protocol="min",
                               packing_seed=4242, n_jobs=1)


@pytest.mark.parametrize("n_jobs", [2, 3, 8])
def test_parallel_is_bit_identical_to_serial_when_packing_is_pinned(
        helix, sfs, pinned_serial, n_jobs):
    """The strongest available check, and the only one with real power against the RNG bug.

    With packing_seed set, decoy k is a pure function of (packing_seed, k), and the parent
    accumulates via imap (ordered), so float addition happens in the same order as serial.
    Bit-identical, not merely close. n_jobs=8 exceeds n_decoys=6 on purpose -- it must
    clamp rather than fork idle workers.
    """
    r = compute_frustration(helix, *sfs, n_decoys=6, seed=0, protocol="min",
                            packing_seed=4242, n_jobs=n_jobs)
    for name in ("decoy_mean", "decoy_std", "index", "native_energy"):
        a = np.nan_to_num(getattr(pinned_serial, name))
        b = np.nan_to_num(getattr(r, name))
        assert np.array_equal(a, b), f"{name} differs at n_jobs={n_jobs}"


def test_pinning_the_native_reference_is_not_forgotten(helix, sfs):
    """E0 has its own stochastic repack.

    Regression guard on a real bug: pinning only the decoys left decoy_mean and decoy_std
    bit-identical while `index` still wandered, because index = (mean - E0)/sigma.
    """
    a = compute_frustration(helix, *sfs, n_decoys=4, protocol="min", packing_seed=99)
    b = compute_frustration(helix, *sfs, n_decoys=4, protocol="min", packing_seed=99)
    assert np.array_equal(np.nan_to_num(a.native_energy), np.nan_to_num(b.native_energy))


def test_unpinned_runs_are_independent_samples(helix, sfs):
    """Without packing_seed the packer must stay stochastic.

    This is a FEATURE, not laxity: scripts/packer_noise.py measures packer noise by
    rebuilding one sequence repeatedly and would read exactly zero if seeding leaked into
    the default path.
    """
    a = compute_frustration(helix, *sfs, n_decoys=4, protocol="min")
    b = compute_frustration(helix, *sfs, n_decoys=4, protocol="min")
    assert not np.array_equal(np.nan_to_num(a.decoy_mean), np.nan_to_num(b.decoy_mean))


def test_pool_is_actually_wired_to_the_rng_initializer(helix, sfs, tmp_path, monkeypatch):
    """END-TO-END guard that compute_frustration passes the initializer to its Pool.

    THIS TEST EXISTS BECAUSE THE REST OF THE FILE DID NOT CATCH IT. An adversarial review
    deleted `initializer=_worker_init` from the Pool call and all 73 tests still passed.
    The bit-identity tests set packing_seed, and _seed_packer overrides the worker's
    stream as the first statement of every decoy, so they are blind by construction.

    IT MUST OBSERVE THE WIRING, NOT THE OUTPUT. Two output-comparison versions of this
    test were written and BOTH passed against the sabotaged copy. The reason is worth
    recording: without the initializer the workers do start from identical inherited
    state, but which worker draws which decoy varies between runs, so the outputs differ
    anyway. No assertion about output values can separate "workers have distinct streams"
    from "the scheduler dealt the decoys differently".

    So the spy records the seed each worker actually ends up with. Under fork the patched
    module is inherited by the children, so the spy runs in each worker; it reports
    through a file because a module global would not propagate back across the fork.
    """
    import frustx.frustration as F

    log = tmp_path / "worker_seeds.txt"
    real = F._worker_init

    def spy(jran_base, counter):
        real(jran_base, counter)
        from pyrosetta.rosetta.numeric.random import rg
        with open(log, "a") as fh:
            fh.write(f"{rg().get_seed()}\n")

    monkeypatch.setattr(F, "_worker_init", spy)
    compute_frustration(helix, *sfs, n_decoys=4, protocol="none", n_jobs=2,
                        jran_base=7000)
    seeds = sorted(int(x) for x in log.read_text().split())
    assert seeds == [7000, 7001], (
        f"expected one distinct RNG stream per worker, got {seeds} -- "
        "compute_frustration is not passing _worker_init to its Pool"
    )


def test_parallel_reruns_are_independent_samples(helix, sfs):
    """Re-running must grow the ensemble in parallel exactly as it does serially.

    Regression guard on a real bug: jran_base defaulted to a fixed 1, so two invocations
    of the same parallel command returned a BIT-IDENTICAL ensemble (measured maxdiff 0.0
    against 0.53 serial). Merging two such runs adds no information while the apparent
    standard error falls as though it had.

    The guard is on the SEED, not on the energies. Asserting that two unpinned runs
    produce different decoy means looked like the stronger test and was in fact a flaky
    one: on a 14-residue helix the packer has few enough rotamer choices that it lands
    on the same answer from either stream, and the assertion failed about 1 run in 6
    (measured, in isolation -- not an ordering effect). A flaky guard is one that gets
    ignored, so the two directions are tested separately and exactly instead: different
    seeds must be DRAWN, and a pinned seed must REPRODUCE. Together those say the seed
    is what controls the ensemble, which is what the bug broke.
    """
    kw = dict(n_decoys=4, seed=0, protocol="min", n_jobs=2)
    a = compute_frustration(helix, *sfs, **kw)
    b = compute_frustration(helix, *sfs, **kw)
    assert a.jran_base != b.jran_base, (
        f"both runs drew jran_base={a.jran_base} -- re-running is not an independent "
        "sample, it is the same ensemble twice"
    )

    pinned = dict(kw, jran_base=4242, packing_seed=11)
    c = compute_frustration(helix, *sfs, **pinned)
    d = compute_frustration(helix, *sfs, **pinned)
    assert np.array_equal(np.nan_to_num(c.decoy_mean), np.nan_to_num(d.decoy_mean))


@pytest.mark.parametrize("bad", [0, -1, -4, None])
def test_nonsense_job_counts_raise_rather_than_running_serially(helix, sfs, bad):
    """`-1` is joblib's "all cores". Clamping it up to 1 would turn a familiar flag into
    a silent 8x slowdown on an hour-long run, with run.json recording n_jobs=-1."""
    with pytest.raises(ValueError, match="n_jobs must be"):
        compute_frustration(helix, *sfs, n_decoys=3, protocol="none", n_jobs=bad)


def test_progress_counter_is_monotone_in_parallel(helix, sfs):
    """`done` is a count of completions, never a decoy index.

    Decoys finish out of order, so reporting k would make the CLI's counter jump around.
    The contract must be identical to the serial path (tests/test_frustration.py).
    """
    seen = []
    compute_frustration(helix, *sfs, n_decoys=5, protocol="none", n_jobs=3,
                        progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_accounting_guard_catches_drops_and_duplicates():
    """Hand-built cases for the guard itself -- these are the shapes a chunking or
    slot-assignment bug produces, and all three are invisible in the output otherwise."""
    _check_accounting([0, 1, 2], 3)                      # correct: no raise
    _check_accounting([2, 0, 1], 3)                      # arrival order is irrelevant
    with pytest.raises(RuntimeError, match="accounting failed"):
        _check_accounting([0, 1], 3)                     # one decoy lost
    with pytest.raises(RuntimeError, match="accounting failed"):
        _check_accounting([0, 1, 1], 3)                  # one counted twice, one lost
    with pytest.raises(RuntimeError, match="accounting failed"):
        _check_accounting([0, 1, 2, 2], 3)               # a duplicate on top of a full set


def test_workers_get_distinct_rng_streams():
    """Direct observation of the hazard, plus its negative control.

    get_seed() reports the seed a worker's generator was initialised with. Under the
    correct initializer the seeds are distinct; with NO initializer, forked children
    inherit one stream and report the SAME seed. The negative control is the point: it
    fails loudly if a future PyRosetta changes fork behaviour and quietly invalidates the
    whole design rationale.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    init_rosetta()

    q = ctx.Queue()
    counter = ctx.Value("i", 0)
    with ctx.Pool(4, initializer=_init_and_report, initargs=(9000, counter, q)):
        seeds = [q.get(timeout=60) for _ in range(4)]
    assert set(seeds) == {9000, 9001, 9002, 9003}, f"workers share an RNG stream: {seeds}"

    q2 = ctx.Queue()
    with ctx.Pool(4, initializer=_report_seed, initargs=(q2,)):   # no real init
        inherited = [q2.get(timeout=60) for _ in range(4)]
    assert len(set(inherited)) == 1, (
        "forked workers no longer inherit one RNG stream -- the initializer's rationale "
        f"needs rechecking; got {inherited}"
    )
