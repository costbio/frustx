"""AWSEM's contact energy, reimplemented in Python.

Why this exists: FrustX and frustratometeR disagree, and the disagreement has two
possible sources that were previously entangled -- a different energy function
(REF2015 all-atom vs AWSEM coarse-grained) and a different protocol (contact
definition, decoy construction, what enters the numerator). Reimplementing AWSEM's
energy lets us run *it* through FrustX's own machinery, which separates the two:

    FrustX-REF2015  vs  FrustX-AWSEM       -> pure force-field effect
    FrustX-AWSEM    vs  frustratometeR     -> pure protocol effect

Everything here is transcribed from the AWSEM source that frustratometeR ships,
dumped under results/reference/awsem_source/. Line references are to that copy.

    compute_water_energy   fix_backbone.cpp:5444
    compute_burial_energy  fix_backbone.cpp:5478
    compute_ro (density)   smart_matrix_lib.h:compute_ro
    get_residue_distance   fix_backbone.cpp:5558   (CB, or CA for glycine)
    gamma table layout     fix_backbone.cpp:624-700

Parameters come from AWSEMFiles/fix_backbone_coeff.data and are hard-coded below
rather than parsed, because that file's format is positional and unlabelled -- a
parser would be less readable than the constants with the source line that reads them.

NOT implemented: electrostatics. frustratometeR ships the section header as
"[DebyeHuckel]-", which fails the exact strcmp at fix_backbone.cpp:467, so huckel_flag
stays 0 and their reference energies contain no electrostatic term. Matching that is
deliberate -- see docs/method.md.
"""

import numpy as np

# --- [Water], fix_backbone.cpp:~/[Water] parser -----------------------------------
K_WATER = 1.0
WATER_KAPPA = 5.0            # steepness of the radial switching function
WATER_KAPPA_SIGMA = 7.0      # steepness of the density switching function
DENSITY_THRESHOLD = 2.6      # rho above which a residue counts as protein-surrounded
WELL_R_MIN = (4.5, 6.5)      # well 0 = direct contact, well 1 = mediated
WELL_R_MAX = (6.5, 9.5)

# --- [Burial] ---------------------------------------------------------------------
K_BURIAL = 1.0
BURIAL_KAPPA = 4.0
BURIAL_RO_MIN = (0.0, 3.0, 6.0)
BURIAL_RO_MAX = (3.0, 6.0, 9.0)

# Residue type ordering, decoded from se_map at fix_backbone.cpp:55. Index into the
# gamma tables. This is the conventional AWSEM ordering, not alphabetical.
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def load_gammas(awsem_dir):
    """Read gamma.dat and burial_gamma.dat into arrays.

    gamma.dat is read at fix_backbone.cpp:626-639 as a flat token stream:
    for each of the 2 wells, for i in 0..19, for j in i..19, two numbers. That is
    2 * 210 * 2 = 840 values. The table is then symmetrised.

    Returns
    -------
    water : (2, 20, 20, 2) array   [well][i][j][0=protein-like, 1=water-like]
    burial : (20, 3) array
    """
    vals = np.fromstring(open(f"{awsem_dir}/gamma.dat").read(), sep=" ")
    if vals.size != 840:
        raise ValueError(f"gamma.dat: expected 840 values, got {vals.size}")

    water = np.zeros((2, 20, 20, 2))
    k = 0
    for well in range(2):
        for i in range(20):
            for j in range(i, 20):
                g0, g1 = vals[k], vals[k + 1]
                k += 2
                water[well, i, j, 0] = water[well, j, i, 0] = g0 * K_WATER
                water[well, i, j, 1] = water[well, j, i, 1] = g1 * K_WATER

    burial = np.fromstring(open(f"{awsem_dir}/burial_gamma.dat").read(), sep=" ")
    if burial.size != 60:
        raise ValueError(f"burial_gamma.dat: expected 60 values, got {burial.size}")
    return water, burial.reshape(20, 3)


def load_cb(pdb_path):
    """Return (resnames, chains, resseqs, coords) using CB, or CA for glycine.

    This is AWSEM's interaction site: get_residue_distance (fix_backbone.cpp:5558)
    picks xca for 'G' and xcb otherwise. Written here rather than reusing
    contacts.load_ca because that one deliberately returns CA only.
    """
    res = {}
    order = []
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        name, resname = line[12:16].strip(), line[17:20].strip()
        if resname not in THREE_TO_ONE or name not in ("CA", "CB"):
            continue
        key = (line[21], int(line[22:26]))
        if key not in res:
            res[key] = {"resname": resname}
            order.append(key)
        res[key][name] = np.array([float(line[30:38]), float(line[38:46]),
                                   float(line[46:54])])

    resnames, chains, resseqs, coords = [], [], [], []
    for key in order:
        r = res[key]
        # Glycine has no CB; anything else missing one is a broken structure, and
        # falling back silently would corrupt every distance it takes part in.
        site = r.get("CA") if r["resname"] == "GLY" else r.get("CB")
        if site is None:
            raise ValueError(f"residue {key} ({r['resname']}) has no interaction site")
        resnames.append(r["resname"])
        chains.append(key[0])
        resseqs.append(key[1])
        coords.append(site)
    return resnames, np.array(chains), np.array(resseqs), np.array(coords)


def theta(rij, well):
    """Radial switching function for one well. fix_backbone.cpp:5466-5472."""
    t_min = np.tanh(WATER_KAPPA * (rij - WELL_R_MIN[well]))
    t_max = np.tanh(WATER_KAPPA * (WELL_R_MAX[well] - rij))
    return 0.25 * (1.0 + t_min) * (1.0 + t_max)


def densities(coords, chains, resseqs):
    """Local density rho_i. smart_matrix_lib.h:compute_ro.

    rho_i = sum_j theta(r_ij, well 0), over j that are either on another chain or
    more than one residue away in sequence. Note this uses the *direct* well only.
    """
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    same_chain = chains[:, None] == chains[None, :]
    near = np.abs(resseqs[:, None] - resseqs[None, :]) <= 1
    excluded = same_chain & near              # includes the i == j diagonal
    t = theta(d, 0)
    t[excluded] = 0.0
    return t.sum(axis=1)


def water_energy(rij, ti, tj, rho_i, rho_j, water_gamma):
    """Pair contact energy. fix_backbone.cpp:5444-5476.

    The direct well uses the plain average of the two gammas; only the mediated
    well is switched by the local densities, between a protein-mediated and a
    water-mediated gamma. Vectorised over arrays of contacts.
    """
    sigma_wat = 0.25 * (1.0 - np.tanh(WATER_KAPPA_SIGMA * (rho_i - DENSITY_THRESHOLD))) \
                     * (1.0 - np.tanh(WATER_KAPPA_SIGMA * (rho_j - DENSITY_THRESHOLD)))
    sigma_prot = 1.0 - sigma_wat

    direct = 0.5 * (water_gamma[0, ti, tj, 0] + water_gamma[0, ti, tj, 1])
    mediated = sigma_prot * water_gamma[1, ti, tj, 0] + sigma_wat * water_gamma[1, ti, tj, 1]

    return -(direct * theta(rij, 0) + mediated * theta(rij, 1))


def burial_energy(ti, rho_i, burial_gamma):
    """One-body burial term. fix_backbone.cpp:5478-5500.

    Three overlapping density windows (low / medium / high burial), each a pair of
    tanh ramps. Vectorised over arrays of residues.
    """
    e = np.zeros(np.shape(rho_i), dtype=float)
    for w in range(3):
        t0 = np.tanh(BURIAL_KAPPA * (rho_i - BURIAL_RO_MIN[w]))
        t1 = np.tanh(BURIAL_KAPPA * (BURIAL_RO_MAX[w] - rho_i))
        e += -0.5 * K_BURIAL * burial_gamma[ti, w] * (t0 + t1)
    return e


def types_from_resnames(resnames):
    """Map three-letter residue names to AWSEM gamma-table indices."""
    return np.array([AA_INDEX[THREE_TO_ONE[r]] for r in resnames])
