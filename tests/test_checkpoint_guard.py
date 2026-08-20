"""The checkpoint resume guard in scripts/dump_decoy_samples.py.

This is a test of a SCRIPT, which is unusual here, and it earns the exception: the thing
being guarded is a silent corruption of a multi-hour decoy tensor. Before the guard, the
resume check compared two numbers -- decoy count and residue count -- so resuming a `min`
checkpoint with `relax` on the command line passed it, mixed two ensembles into one
tensor, and left the native reference pinned to the first protocol while every
post-resume decoy used the second.

The checkpoint is FABRICATED here rather than produced by a real run. A real run writes
one only every CHECKPOINT_EVERY=25 decoys, so producing one honestly would mean building
25 decoys per test. What matters is the file's contents, and those are written out
explicitly below -- which also means this test states the on-disk format in one place.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx import provenance  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dump_decoy_samples.py"


@pytest.fixture(scope="module")
def dds():
    """Import the script as a module. It guards its own entry point with __main__, so
    importing runs no work."""
    spec = importlib.util.spec_from_file_location("dds", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _settings(protocol="min", n_decoys=4, structure_key="abc"):
    return {
        "structure_key": structure_key, "init_flags": "-mute all", "weights": "ref2015",
        "protocol": protocol, "repeats": 1, "seed": 0, "n_decoys": n_decoys,
        "packing_seed": None, "freeze_ligand": True, "ligands": (),
    }


def _write_checkpoint(path, settings, n_decoys=4, n=8, key=None):
    """Exactly what save_checkpoint writes."""
    if key is None:
        key = provenance.regeneration_key(settings, "ensemble")
    np.savez_compressed(
        path,
        decoys=np.zeros((n_decoys, n, n)),
        native=np.zeros((n, n)),
        done=np.array([True, True, False, False]),
        regeneration_key=key,
        settings=json.dumps(settings),
        resseq=np.arange(n), chain=np.array(["A"] * n), resname=np.array(["ALA"] * n),
    )


def test_matching_settings_resume(dds, tmp_path):
    """The positive case. Without this the guard could 'pass' by refusing everything."""
    s = _settings()
    p = tmp_path / "c.npz"
    _write_checkpoint(p, s)
    dds._SETTINGS = s
    decoys, native, done = dds._load_checkpoint(
        str(p), 4, 8, provenance.regeneration_key(s, "ensemble"))
    assert decoys is not None
    assert done.tolist() == [True, True, False, False]


def test_a_changed_protocol_refuses_to_resume(dds, tmp_path, capsys):
    """The bug. `min` on disk, `relax` requested."""
    on_disk = _settings(protocol="min")
    requested = _settings(protocol="relax")
    p = tmp_path / "c.npz"
    _write_checkpoint(p, on_disk)
    dds._SETTINGS = requested

    with pytest.raises(SystemExit) as exc:
        dds._load_checkpoint(str(p), 4, 8,
                             provenance.regeneration_key(requested, "ensemble"))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "REFUSING to resume" in out
    # and it must say WHICH setting, or the operator cannot tell whether to delete the
    # checkpoint or fix the command line
    assert "protocol: 'min' -> 'relax'" in out


def test_a_changed_structure_refuses_to_resume(dds, tmp_path):
    """The shape check could never catch this: a different protein with the same residue
    count passes it cleanly."""
    p = tmp_path / "c.npz"
    _write_checkpoint(p, _settings(structure_key="protein-one"))
    requested = _settings(structure_key="protein-two")
    dds._SETTINGS = requested
    with pytest.raises(SystemExit):
        dds._load_checkpoint(str(p), 4, 8,
                             provenance.regeneration_key(requested, "ensemble"))


def test_a_keyless_checkpoint_is_refused_not_trusted(dds, tmp_path, capsys):
    """Checkpoints written before the guard existed carry no key. 'No key' is
    indistinguishable from 'written under different settings', so it must not be
    accepted -- the whole point is to stop guessing."""
    p = tmp_path / "c.npz"
    np.savez_compressed(p, decoys=np.zeros((4, 8, 8)), native=np.zeros((8, 8)),
                        done=np.zeros(4, dtype=bool))
    s = _settings()
    dds._SETTINGS = s
    with pytest.raises(SystemExit):
        dds._load_checkpoint(str(p), 4, 8, provenance.regeneration_key(s, "ensemble"))
    assert "predates the settings check" in capsys.readouterr().out


def test_it_exits_rather_than_silently_starting_fresh(dds, tmp_path):
    """Overwriting hours of decoys because a flag was mistyped is the more expensive of
    the two mistakes, so a mismatch must stop the run, not quietly restart it."""
    p = tmp_path / "c.npz"
    _write_checkpoint(p, _settings(protocol="min"))
    requested = _settings(protocol="relax")
    dds._SETTINGS = requested
    with pytest.raises(SystemExit):
        dds._load_checkpoint(str(p), 4, 8,
                             provenance.regeneration_key(requested, "ensemble"))
    assert p.exists(), "the checkpoint must not be deleted or overwritten on refusal"


def test_shape_mismatch_still_ignored_quietly(dds, tmp_path, capsys):
    """The pre-existing behaviour must survive: a wrong-shaped checkpoint is IGNORED
    (start fresh), not a hard exit. It cannot be a partially-completed version of this
    request, so there is nothing to protect."""
    s = _settings(n_decoys=4)
    p = tmp_path / "c.npz"
    _write_checkpoint(p, s, n_decoys=4, n=8)
    dds._SETTINGS = s
    got = dds._load_checkpoint(str(p), 4, 99,   # n=99, so shape disagrees
                               provenance.regeneration_key(s, "ensemble"))
    assert got == (None, None, None)
    assert "shape mismatch" in capsys.readouterr().out


def test_an_unreadable_checkpoint_is_ignored_not_fatal(dds, tmp_path, capsys):
    """A truncated write is normal after a kill and must not be fatal."""
    p = tmp_path / "c.npz"
    p.write_bytes(b"not an npz file at all")
    dds._SETTINGS = _settings()
    got = dds._load_checkpoint(str(p), 4, 8, "whatever")
    assert got == (None, None, None)
    assert "unreadable" in capsys.readouterr().out


def test_absent_checkpoint_is_a_clean_start(dds, tmp_path):
    got = dds._load_checkpoint(str(tmp_path / "nope.npz"), 4, 8, "k")
    assert got == (None, None, None)
