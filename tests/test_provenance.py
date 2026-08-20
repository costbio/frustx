"""Tests for regeneration keys.

The point of a key is negative: it must REFUSE work produced under different settings.
So most of these assert that two keys differ, and the ones that assert equality are the
ones guarding against over-invalidation -- a key that changes when nothing relevant
changed is not safe, it is just useless, because it throws away expensive work.

No PyRosetta needed; provenance.py is stdlib-only on purpose.
"""

import json

import pytest

from frustx.provenance import (STAGE_INPUTS, describe_mismatch, file_key,
                               regeneration_key)


def _settings(**over):
    """A complete settings dict. Written out in full rather than built from defaults,
    so a test reads as the scenario it is testing."""
    base = dict(
        structure_key="deadbeef", init_flags="-mute all", weights="ref2015",
        protocol="min", repeats=1, seed=0, n_decoys=100, packing_seed=None,
        background_weight=0.0, readout="pair", packing_frustration=False,
        contact_atom="CA", cutoff=10.0, min_seq_sep=1, ligand_cutoff=6.0,
        freeze_ligand=True, ligands=(),
    )
    base.update(over)
    return base


def test_protocol_change_invalidates_the_ensemble():
    """The live bug this exists for: resuming a `min` checkpoint under `relax`."""
    a = regeneration_key(_settings(protocol="min"), "ensemble")
    b = regeneration_key(_settings(protocol="relax"), "ensemble")
    assert a != b


def test_a_cheap_setting_does_not_invalidate_the_expensive_stage():
    """readout changes how decoys are MEASURED, not what they ARE.

    This is the whole reason the stages are partitioned rather than hashed as one blob.
    Decoy structures cost ~700 ms each; throwing them away because an analysis knob moved
    would make the key actively harmful.
    """
    a = _settings(readout="pair")
    b = _settings(readout="neighbourhood")
    assert regeneration_key(a, "ensemble") == regeneration_key(b, "ensemble")
    assert regeneration_key(a, "measurement") != regeneration_key(b, "measurement")


def test_the_contact_map_enters_the_measurement_only_at_neighbourhood_readout():
    """At readout="pair" apply_readout returns E untouched, so the contact map never
    reaches the energies and cutoff must NOT invalidate the measurement. At
    "neighbourhood" it does. A static per-stage table would get one of these wrong."""
    pair = _settings(readout="pair")
    assert regeneration_key(pair, "measurement") == \
        regeneration_key(_settings(readout="pair", cutoff=8.0), "measurement")

    nb = _settings(readout="neighbourhood")
    assert regeneration_key(nb, "measurement") != \
        regeneration_key(_settings(readout="neighbourhood", cutoff=8.0), "measurement")


@pytest.mark.parametrize("field", ["protocol", "repeats", "seed", "n_decoys",
                                   "weights", "structure_key", "init_flags",
                                   "packing_seed", "freeze_ligand", "ligands"])
def test_every_ensemble_input_actually_moves_the_key(field):
    """Guards against a field being listed in STAGE_INPUTS but silently unused -- e.g.
    if the digest were taken over a hardcoded subset."""
    base = _settings()
    changed = _settings(**{field: "SOMETHING-ELSE"})
    assert regeneration_key(base, "ensemble") != regeneration_key(changed, "ensemble")


def test_a_missing_setting_raises_rather_than_defaulting():
    """The one way this module could CAUSE the bug it prevents: if an absent field were
    treated as its default, a key computed without it would compare EQUAL to one
    computed with it present at that default."""
    s = _settings()
    del s["protocol"]
    with pytest.raises(KeyError, match="protocol"):
        regeneration_key(s, "ensemble")


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        regeneration_key(_settings(), "not-a-stage")


def test_the_key_does_not_depend_on_dict_order():
    """Two callers building the same settings in different order must agree, or a
    checkpoint would be refused for no reason."""
    a = _settings()
    b = {k: a[k] for k in reversed(list(a))}
    assert regeneration_key(a, "ensemble") == regeneration_key(b, "ensemble")


def test_settings_outside_the_stage_are_ignored():
    """An unrelated key in the dict must not move the digest, or every stage would
    invalidate on every change and the partitioning would be decorative."""
    a = _settings()
    b = _settings(some_unrelated_thing="whatever")
    assert regeneration_key(a, "ensemble") == regeneration_key(b, "ensemble")


def test_file_key_is_content_not_path(tmp_path):
    """A path proves nothing about what was in the file. Two runs against different
    edits of the same filename are otherwise indistinguishable."""
    p = tmp_path / "s.pdb"
    p.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    first = file_key(p)

    same_content_other_name = tmp_path / "other.pdb"
    same_content_other_name.write_text(p.read_text())
    assert file_key(same_content_other_name) == first

    p.write_text("ATOM      1  CA  ALA A   1       9.999   0.000   0.000\n")
    assert file_key(p) != first, "editing the file did not change its key"


def test_file_key_is_chunk_size_independent(tmp_path):
    """Chunked reading must not change the digest -- it is read in 1 MB blocks."""
    p = tmp_path / "s.pdb"
    p.write_text("X" * 5000)
    assert file_key(p, n_bytes=7) == file_key(p, n_bytes=1 << 20)


def test_describe_mismatch_names_the_field_that_moved():
    """A key comparison can only say yes or no. Being told a checkpoint does not match,
    without being told protocol went min -> relax, is not enough to decide whether to
    delete the file or fix the command line."""
    got = describe_mismatch(_settings(protocol="min"),
                            _settings(protocol="relax"), "ensemble")
    assert got == ["protocol: 'min' -> 'relax'"]


def test_describe_mismatch_is_quiet_when_nothing_moved():
    assert describe_mismatch(_settings(), _settings(), "ensemble") == []


def test_describe_mismatch_reports_an_absent_field():
    s = _settings()
    del s["seed"]
    got = describe_mismatch(s, _settings(), "ensemble")
    assert any("seed" in line and "absent" in line for line in got)


def test_every_stage_is_hashable_with_a_full_settings_dict():
    """Cheap completeness guard: a stage added to STAGE_INPUTS with a field that
    _settings() does not supply would fail here rather than at the call site."""
    for stage in STAGE_INPUTS:
        assert len(regeneration_key(_settings(), stage)) == 64


def test_measurement_contains_every_ensemble_field():
    """A measured artefact is produced FROM decoy structures, so anything that changes
    what a decoy IS must invalidate the measurement too. Adding a field to `ensemble` and
    forgetting it here would let a measurement be reused across structurally different
    ensembles, silently."""
    assert set(STAGE_INPUTS["ensemble"]) <= set(STAGE_INPUTS["measurement"])


def test_freeze_ligand_invalidates_both_stages():
    a, b = _settings(freeze_ligand=True), _settings(freeze_ligand=False)
    for stage in ("ensemble", "measurement"):
        assert regeneration_key(a, stage) != regeneration_key(b, stage)


def test_ligand_cutoff_invalidates_the_contact_map():
    a, b = _settings(ligand_cutoff=6.0), _settings(ligand_cutoff=7.0)
    assert regeneration_key(a, "contacts") != regeneration_key(b, "contacts")


def test_ligand_cutoff_invalidates_the_measurement_only_at_a_neighbourhood_readout():
    """Non-vacuous in both directions, which is why both halves are asserted.

    At readout="pair" apply_readout returns the energies untouched, so the contact map
    never enters the measurement and ligand_cutoff MUST NOT invalidate it -- a key that
    over-invalidates throws away work for no reason. At readout="neighbourhood" the map
    enters through row_i + row_j and it MUST.

    Measured before the fix: `contacts differs=True, measurement differs=False`, while the
    protein `cutoff` control on the same dict already differed. So the omission was
    specific to ligand_cutoff rather than to the conditional as a whole.
    """
    nb = _settings(readout="neighbourhood")
    assert regeneration_key(nb, "measurement") != \
        regeneration_key(_settings(readout="neighbourhood", ligand_cutoff=9.0),
                         "measurement")

    pair = _settings(readout="pair")
    assert regeneration_key(pair, "measurement") == \
        regeneration_key(_settings(readout="pair", ligand_cutoff=9.0), "measurement")


def test_the_dump_decoy_samples_settings_dict_is_still_hashable():
    """scripts/dump_decoy_samples.py builds a settings dict that no test covers.

    regeneration_key raises on a missing field, so adding one to STAGE_INPUTS breaks that
    script -- and the eight checkpoint tests exercise `_load_checkpoint` only, never the
    construction, so the script can be dead while pytest is green. It has happened twice
    already for other reasons: it is an external consumer of a private structure.

    This asserts the KEYS agree with what the stages require, without importing the
    script (it pulls in PyRosetta at module scope).
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "scripts" / "dump_decoy_samples.py"
    tree = ast.parse(source.read_text())
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                and any(getattr(t, "id", "") == "_SETTINGS" for t in n.targets)]
    assert literals, "_SETTINGS is no longer a plain assignment; update this test"

    # There are two: an empty module-level placeholder, and the real one main() fills in.
    # Only the populated one can be checked, and requiring exactly one populated dict is
    # what keeps this test honest if a third appears.
    populated = [d for d in literals if d.keys]
    assert len(populated) == 1, f"expected one populated _SETTINGS, found {len(populated)}"

    keys = {k.value for k in populated[0].keys if isinstance(k, ast.Constant)}
    required = set(STAGE_INPUTS["ensemble"])
    assert not (required - keys), (
        f"scripts/dump_decoy_samples.py is missing {sorted(required - keys)} and will "
        f"raise KeyError at run time")
