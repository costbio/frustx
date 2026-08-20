"""Tests for the command-line entry point."""

import json
import subprocess
import sys

import pytest

from frustx.cli import build_parser, main


def test_parser_defaults_match_the_paper():
    args = build_parser().parse_args(["x.pdb", "-o", "out"])
    assert args.decoys == 1000          # "1000 appropriately distributed decoys"
    assert args.cutoff == 10.0          # "a cutoff of 10 A is used"
    assert args.protocol == "relax"     # the literal reading of "Monte-Carlo relaxation"
    assert args.packing_frustration is False


def test_importing_cli_does_not_load_pyrosetta():
    """Guards the lazy import that keeps `frustx --help` instant.

    frustx.cli must not pull in PyRosetta transitively -- it previously did, via
    frustration -> decoys, which made --help take 2.8 s instead of 47 ms.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import frustx.cli, sys; print('pyrosetta' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False", "frustx.cli imported PyRosetta at module scope"


def test_missing_input_file_exits_2(tmp_path):
    assert main([str(tmp_path / "absent.pdb"), "-o", str(tmp_path / "out")]) == 2


def test_too_few_decoys_exits_2(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
    assert main([str(pdb), "-o", str(tmp_path / "out"), "-n", "1"]) == 2


# --- end-to-end, needs PyRosetta -----------------------------------------

def test_end_to_end_writes_all_outputs(tmp_path):
    pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")
    from frustx.energies import init_rosetta

    init_rosetta()
    pose = pyrosetta.pose_from_sequence("AEALKKLAEELKKG")
    for i in range(1, pose.total_residue() + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)
    pose.pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(pose))

    pdb = tmp_path / "helix.pdb"
    pose.dump_pdb(str(pdb))
    out = tmp_path / "results"

    # protocol=none and 3 decoys: exercising the plumbing, not the science.
    rc = main([str(pdb), "-o", str(out), "-n", "3", "--protocol", "none", "--quiet"])
    assert rc == 0

    for name in ("contacts.csv", "residues.csv", "frustration.pdb", "run.json"):
        assert (out / name).is_file(), f"{name} not written"

    meta = json.loads((out / "run.json").read_text())
    assert meta["n_decoys"] == 3
    assert meta["protocol"] == "none"
    assert meta["n_residues"] == pose.total_residue()
    assert meta["n_contacts"] > 0
    # Provenance must be enough to reproduce the run.
    assert {"seed", "cutoff", "min_seq_sep", "frustx_version"} <= set(meta)
