"""Tests for Eq. 1 / Eq. 2 and the output tables.

Runs use protocol="min" and a handful of decoys: these check mechanics and algebra, not
scientific convergence. A real run is 1000 decoys with protocol="relax".
"""

import numpy as np
import pytest

pyrosetta = pytest.importorskip("pyrosetta", reason="PyRosetta not installed")

from frustx.energies import init_rosetta, make_score_function, pair_energy_matrix  # noqa: E402
from frustx.frustration import (  # noqa: E402
    HIGHLY_FRUSTRATED,
    MINIMALLY_FRUSTRATED,
    classify,
    compute_frustration,
    contact_energy_matrix,
    residues_from_pose,
)
from frustx.output import contact_table, residue_table, write_bfactor_pdb  # noqa: E402

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
def score_functions():
    return make_score_function(remove_fa_rep=False), make_score_function(remove_fa_rep=True)


@pytest.fixture(scope="module")
def result(helix, score_functions):
    sf_pack, sf_measure = score_functions
    return compute_frustration(
        helix, sf_pack, sf_measure, n_decoys=6, seed=0, protocol="min"
    )


# --- Eq. 2 --------------------------------------------------------------

def test_eq2_simplification_matches_literal_formula(helix, score_functions):
    """E_ij = 1/2(R_i + R_j) must equal a literal transcription of Eq. 2.

    This is the regression guard on the algebra in docs/method.md. If the simplification
    is ever wrong, every frustration index is wrong, and nothing else in the suite would
    catch it.
    """
    _, sf_measure = score_functions
    e = pair_energy_matrix(helix, sf_measure)
    n = e.shape[0]

    literal = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s1 = sum(e[i, k] for k in range(n) if k != j)
            s2 = sum(e[j, l] for l in range(n) if l != i)
            literal[i, j] = e[i, j] + 0.5 * s1 + 0.5 * s2

    # background_weight=1 plus the direct term is the paper's Eq. 2 up to e_ij itself;
    # subtracting it recovers the literal 1/2(R_i + R_j).
    e = pair_energy_matrix(helix, sf_measure)
    at_w1 = contact_energy_matrix(helix, sf_measure, background_weight=1.0)
    recovered = at_w1 - e
    np.fill_diagonal(recovered, 0.0)
    np.fill_diagonal(literal, 0.0)
    np.testing.assert_allclose(recovered, literal, atol=1e-9)


def test_default_background_weight_is_zero_direct_pair_energy(helix, score_functions):
    """The shipped default must be the direct pair energy, not Eq. 2.

    Eq. 2 as written was measured against frustratometeR to give a degenerate index
    (zero frustrated contacts on ubiquitin, 86% reducible to a residue-level quantity).
    If this default is ever changed back, that regression returns silently.
    """
    _, sf_measure = score_functions
    default = contact_energy_matrix(helix, sf_measure)
    direct = pair_energy_matrix(helix, sf_measure).copy()
    np.fill_diagonal(direct, 0.0)
    np.testing.assert_allclose(default, direct, atol=1e-12)

    # And w != 0 must actually differ, or the parameter is doing nothing.
    assert not np.allclose(default, contact_energy_matrix(helix, sf_measure, 1.0))


def test_contact_energy_matrix_is_symmetric_zero_diagonal(helix, score_functions):
    _, sf_measure = score_functions
    E = contact_energy_matrix(helix, sf_measure)
    np.testing.assert_allclose(E, E.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(E), 0.0, atol=1e-12)


# --- Eq. 1 --------------------------------------------------------------

def test_index_is_zscore_of_native_against_decoys(result):
    """The stored index must be exactly (decoy_mean - native) / decoy_std."""
    ok = result.decoy_std > 0
    expected = (result.decoy_mean[ok] - result.native_energy[ok]) / result.decoy_std[ok]
    np.testing.assert_allclose(result.index[ok], expected, atol=1e-12)


def test_sign_convention_native_better_than_decoys_is_positive(result):
    """A designed native beats shuffled decoys, so F must be POSITIVE.

    This is the guard on the deliberate sign flip away from the paper's Eq. 1. If someone
    "corrects" the sign back, this fails.
    """
    contacts = [result.index[i, j] for i, j in result.contacts]
    finite = np.array([v for v in contacts if np.isfinite(v)])
    assert finite.size > 0
    assert finite.mean() > 0, (
        "native should score better than random-sequence decoys, giving positive F "
        "under the conventional sign"
    )


def test_too_few_decoys_raises(helix, score_functions):
    sf_pack, sf_measure = score_functions
    with pytest.raises(ValueError, match="at least 2"):
        compute_frustration(helix, sf_pack, sf_measure, n_decoys=1)


def test_progress_callback_fires_once_per_decoy(helix, score_functions):
    sf_pack, sf_measure = score_functions
    seen = []
    compute_frustration(
        helix, sf_pack, sf_measure, n_decoys=3, protocol="none",
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


# --- classification -----------------------------------------------------

def test_classify_thresholds():
    values = np.array([MINIMALLY_FRUSTRATED + 0.01, 0.0, HIGHLY_FRUSTRATED - 0.01, np.nan])
    assert list(classify(values)) == ["minimally", "neutral", "highly", "undefined"]


# --- pose adapters ------------------------------------------------------

def test_residues_from_pose_matches_sequence(helix):
    residues = residues_from_pose(helix)
    assert len(residues) == helix.total_residue()
    assert [r.index for r in residues] == list(range(helix.total_residue()))
    # name3 of the first residue should match the one-letter sequence we built from.
    assert residues[0].resname == helix.residue(1).name3()


# --- output tables ------------------------------------------------------

def test_contact_table_has_one_row_per_contact(result):
    df = contact_table(result)
    assert len(df) == len(result.contacts)
    assert {"resnum_i", "resnum_j", "frustration_index", "frustration_class"} <= set(df.columns)
    assert df["frustration_index"].notna().any()


def test_residue_table_has_one_row_per_residue(result):
    df = residue_table(result)
    assert len(df) == len(result.residues)
    # Every contact contributes to exactly two residues.
    assert df["n_contacts"].sum() == 2 * len(result.contacts)
    counts = df[["n_minimally_frustrated", "n_neutral", "n_highly_frustrated"]].sum(axis=1)
    assert (counts <= df["n_contacts"]).all()


def test_write_bfactor_pdb_roundtrips(result, helix, tmp_path):
    """B-factors must land in the file and be finite."""
    pose = helix.clone()
    # A pose built from sequence has no PDB info until we give it some.
    pose.pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(pose))
    out = write_bfactor_pdb(pose, result, tmp_path / "painted.pdb")
    text = out.read_text()
    bfactors = [float(line[60:66]) for line in text.splitlines()
                if line.startswith("ATOM")]
    assert bfactors, "no ATOM records written"
    assert all(np.isfinite(b) for b in bfactors)
    assert any(b != 0.0 for b in bfactors), "all B-factors zero; nothing was painted"
