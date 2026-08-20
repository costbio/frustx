"""Turning a FrustrationResult into things you can look at.

Three views, in increasing order of lossiness:

  * `contact_table`  -- one row per contact. The primary result; nothing is aggregated.
  * `residue_table`  -- one row per residue, aggregated over its contacts. Convenient
                        for plotting a profile along the sequence, but note the paper
                        defines frustration per CONTACT, not per residue: this
                        aggregation is ours, not theirs.
  * `write_bfactor_pdb` -- per-residue values painted into the B-factor column, for
                        colouring in PyMOL/ChimeraX.
"""

import numpy as np
import pandas as pd

from frustx.additivity import additive_decomposition
from frustx.frustration import classify


def contact_table(result):
    """One row per contact. This is the primary output.

    Three index columns, not one, because "is this contact frustrated?" is two questions:

        frustration_index           the index itself
        frustration_index_onebody   the part predictable from the two residues alone
                                    (c + a_i + a_j) -- burial, exposure, local packing
        frustration_index_specific  what is left, i.e. what is particular to THIS pair

    Neither part is the "real" one. A residue-level question wants the one-body column; a
    claim about a particular contact needs the specific column, because an index that is
    largely one-body cannot support one however well it correlates with anything. See
    frustx/additivity.py, and "Reporting both parts" in docs/method.md.

    frustration_class stays on the raw index: the 0.78 / -1.0 thresholds are inherited
    from AWSEM and calibrated against that quantity, not against the residual.
    """
    rows = []
    for i, j in result.contacts:
        ri, rj = result.residues[i], result.residues[j]
        rows.append(
            {
                "chain_i": ri.chain, "resnum_i": ri.resseq, "resname_i": ri.resname,
                "chain_j": rj.chain, "resnum_j": rj.resseq, "resname_j": rj.resname,
                "native_energy": result.native_energy[i, j],
                "decoy_mean": result.decoy_mean[i, j],
                "decoy_std": result.decoy_std[i, j],
                "frustration_index": result.index[i, j],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        # The fa_rep part of the same quantity, carried through so the repulsive term
        # is a reporting choice rather than something fixed when the decoys were built.
        # Raw parts, not a second index: recombining them needs the covariance, which
        # lives on the result -- see FrustrationResult.index_at_fa_rep. Putting a
        # ready-made "index with fa_rep" column here would invite the wrong arithmetic
        # (adding two indices, or two sigmas) on the CSV downstream.
        if result.fa_rep_native is not None:
            ij = (result.contacts[:, 0], result.contacts[:, 1])
            df["fa_rep_native"] = result.fa_rep_native[ij]
            df["fa_rep_decoy_mean"] = result.fa_rep_mean[ij]
            df["fa_rep_decoy_std"] = result.fa_rep_std[ij]
        df["frustration_class"] = classify(df["frustration_index"].to_numpy())
        d = additive_decomposition(
            df["frustration_index"].to_numpy(), result.contacts, len(result.residues)
        )
        df["frustration_index_onebody"] = d.fitted
        df["frustration_index_specific"] = d.residual
    return df


def residue_table(result):
    """One row per residue, aggregating the contacts it participates in.

    `mean_frustration` is the mean index over that residue's contacts. The counts
    mirror what frustratometeR reports, and are often more informative than the mean:
    a residue with many minimally frustrated contacts and a few highly frustrated ones
    is a different thing from a uniformly neutral residue, and averaging hides that.

    `onebody_coefficient` is this residue's a_i from the additive fit -- the amount it
    shifts every contact it takes part in. IT IS IDENTIFIED ONLY UP TO A CONSTANT (the
    fit is rank deficient), so compare residues WITHIN a run and never across runs, and
    read differences rather than absolute values.
    """
    n = len(result.residues)
    decomposition = additive_decomposition(
        np.array([result.index[i, j] for i, j in result.contacts], dtype=float),
        result.contacts, n,
    ) if len(result.contacts) else None
    per_residue = [[] for _ in range(n)]
    for i, j in result.contacts:
        value = result.index[i, j]
        per_residue[i].append(value)
        per_residue[j].append(value)

    rows = []
    for idx, res in enumerate(result.residues):
        values = np.array(per_residue[idx], dtype=float)
        finite = values[np.isfinite(values)]
        labels = classify(finite) if finite.size else np.array([], dtype=object)
        rows.append(
            {
                "chain": res.chain,
                "resnum": res.resseq,
                "resname": res.resname,
                "onebody_coefficient": (decomposition.coefficients[idx]
                                        if decomposition is not None else np.nan),
                "n_contacts": len(values),
                "mean_frustration": finite.mean() if finite.size else np.nan,
                "n_minimally_frustrated": int((labels == "minimally").sum()),
                "n_neutral": int((labels == "neutral").sum()),
                "n_highly_frustrated": int((labels == "highly").sum()),
            }
        )
    return pd.DataFrame(rows)


def write_bfactor_pdb(pose, result, path, column="mean_frustration"):
    """Write the pose with per-residue frustration in the B-factor column.

    Colour it in PyMOL with:  spectrum b, blue_white_red, all

    Residues with no finite value (no contacts) get 0.0 rather than NaN, since most
    viewers choke on NaN in the B-factor field.
    """
    table = residue_table(result).set_index(["chain", "resnum"])
    info = pose.pdb_info()
    if info is None:
        raise ValueError("pose has no PDB info; cannot write B-factors")

    for i in range(1, pose.total_residue() + 1):
        key = (info.chain(i), info.number(i))
        value = table[column].get(key, np.nan) if key in table.index else np.nan
        if not np.isfinite(value):
            value = 0.0
        # B-factors are per-atom, so paint every atom of the residue the same value.
        for atom in range(1, pose.residue(i).natoms() + 1):
            info.bfactor(i, atom, float(value))

    pose.dump_pdb(str(path))
    return path
