"""Tests for the Rosetta init flags.

The behavioural test here runs PyRosetta in a SUBPROCESS. Rosetta's init is global and
effectively once-per-process, so the two flag settings cannot both be exercised inside
one pytest session -- and the point of the test is precisely to compare them.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.energies import DEFAULT_INIT_FLAGS  # noqa: E402


# A real CCD three-letter code (benzene) for which this repo ships no .params file.
# That is the whole setup: nothing local defines BNZ, so anything that builds it is
# building it from Rosetta's bundled dictionary.
_LIGAND = [
    "HETATM 9990  C1  BNZ A 900      20.000  15.000  10.000  1.00  0.00           C",
    "HETATM 9991  C2  BNZ A 900      21.400  15.000  10.000  1.00  0.00           C",
    "HETATM 9992  C3  BNZ A 900      22.100  16.200  10.000  1.00  0.00           C",
    "HETATM 9993  C4  BNZ A 900      21.400  17.400  10.000  1.00  0.00           C",
    "HETATM 9994  C5  BNZ A 900      20.000  17.400  10.000  1.00  0.00           C",
    "HETATM 9995  C6  BNZ A 900      19.300  16.200  10.000  1.00  0.00           C",
]

_LOAD = textwrap.dedent("""
    import pyrosetta
    pyrosetta.init({flags!r}, silent=True)
    p = pyrosetta.pose_from_file({pdb!r})
    print("NAMES", [p.residue(i).name3() for i in range(1, p.total_residue() + 1)])
""")


def _residue_names(pdb, flags):
    """Load `pdb` under `flags` in a fresh interpreter, return the residue names."""
    src = _LOAD.format(flags=flags, pdb=str(pdb))
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("NAMES")), None)
    assert line is not None, f"probe produced no result:\n{out.stdout}\n{out.stderr}"
    return eval(line[len("NAMES"):])


@pytest.fixture(scope="module")
def peptide_with_ligand(tmp_path_factory):
    """Four alanines with an unparametrised BNZ HETATM appended.

    The peptide is dumped by Rosetta rather than hand-written so the backbone geometry
    is real -- hand-built colinear atoms make Rosetta fail on a degenerate stub before
    it ever reaches the question under test.
    """
    import pyrosetta
    # DEFAULT_INIT_FLAGS, not a weaker string: this init lands in the pytest process
    # itself. energies.init_rosetta() has a module-level _INITIALISED guard, so if it
    # already ran, re-initing here silently resets the global Rosetta options while the
    # guard stays True -- and every test module collected after this one would then run
    # without the flags, with no way to restore them. Harmless for poses built from
    # sequence, but it is the kind of cross-test coupling that makes a later failure
    # impossible to reproduce in isolation.
    pyrosetta.init(DEFAULT_INIT_FLAGS, silent=True)
    pose = pyrosetta.pose_from_sequence("AAAA")
    d = tmp_path_factory.mktemp("lig")
    pose.dump_pdb(str(d / "pep.pdb"))
    atoms = [l.rstrip("\n") for l in (d / "pep.pdb").read_text().splitlines()
             if l.startswith("ATOM")]
    pdb = d / "pep_lig.pdb"
    pdb.write_text("\n".join(atoms + _LIGAND) + "\nEND\n")
    return pdb


def test_the_flag_is_in_the_defaults():
    """Cheap guard. The behavioural test below is what gives this one its meaning, but
    this is the one that fails fast if the flag is ever dropped from the string."""
    assert "-in:file:load_PDB_components false" in DEFAULT_INIT_FLAGS


def test_ccd_fallback_admits_an_unparametrised_ligand(peptide_with_ligand):
    """The hazard, demonstrated rather than asserted from documentation.

    With the CCD lookup at its default, -ignore_unrecognized_res does NOT ignore BNZ:
    Rosetta builds it from its bundled dictionary, with atom types and charges that no
    one in this project chose. If this test ever fails, the fallback is gone from
    Rosetta and the flag below has become unnecessary.
    """
    names = _residue_names(peptide_with_ligand, "-mute all -ignore_unrecognized_res")
    assert names == ["ALA", "ALA", "ALA", "ALA", "BNZ"]


def test_our_defaults_drop_it_instead(peptide_with_ligand):
    """And with FrustX's actual flags, the ligand is dropped, which is honest: we have
    no parameters for it, so we should not pretend to have scored it."""
    names = _residue_names(peptide_with_ligand, DEFAULT_INIT_FLAGS)
    assert names == ["ALA", "ALA", "ALA", "ALA"]
