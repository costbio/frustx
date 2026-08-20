"""Turn an arbitrary small molecule into a Rosetta residue type FrustX can score.

The ligands this project cares about now are the output of docking or co-folding --
arbitrary novel chemistry, not the two dozen coenzymes (ATP, ZN, MG, SAH, UDP...) that
ship in Rosetta's ``fa_standard`` set. Those need parameters, and the conventional way to
get them is ``molfile_to_params.py``, which is Rosetta source we may not redistribute.

We do not need it. Rosetta's own SDF/MOL reader is exposed in PyRosetta as
``core::chemical::sdf::convert_to_ResidueTypes``, and it does the whole job -- atom types,
partial charges, connectivity, internal coordinates -- from a plain SDF. Measured on
erlotinib (PDB chemical component AQ4, 52 atoms / 29 heavy):

    rts = convert_to_ResidueTypes("AQ4.sdf", False)   # False = do not load rotamers
    -> name 'AQ4', natoms 52, is_ligand True, atom types ['COO','COO','aroC',...]

Registration is PER POSE, into a ``PoseResidueTypeSet`` layered over ``fa_standard``.
That is not a stylistic choice:

  * The ChemicalManager's global ``fa_standard`` set is a const owning pointer and cannot
    be added to.
  * ``-extra_res_fa`` -- the usual route -- is honoured only at the FIRST ``pyrosetta.init``
    of a process, and FrustX's decoy workers re-init after fork (frustration.py:381-384).
  * Because the type travels inside the pose, those workers need to know nothing about
    ligands at all, and ``DEFAULT_INIT_FLAGS`` (energies.py:51-52) is untouched. Verified
    bit-identical at n_jobs 2, 3 and 8; see tests/test_ligand_params.py.

A user-supplied ``.params`` file registers through the SAME call, by filename, and is the
escape hatch for when auto-typing is not good enough -- which is a real concern, see
``check_chemistry`` below.

WHAT THIS MODULE DELIBERATELY WILL NOT DO
-----------------------------------------
Add hydrogens. That is cheminformatics at a chosen pH, it needs RDKit or OpenBabel, and
neither is a dependency of this project (same reasoning as dvc and matplotlib in
CLAUDE.md -- infrastructure for preparing inputs is not something ``frustx`` imports). A
ligand that arrives without its hydrogens is REFUSED with an instruction, never silently
repaired: see ``require_hydrogens`` for what the alternative costs.
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from pathlib import Path


class LigandParametrisationError(ValueError):
    """A ligand file cannot be turned into a residue type we are willing to score.

    Its own class because the CLI has to catch it alongside ``RuntimeError``: Rosetta's
    C++ ``utility_exit`` surfaces in Python as a bare RuntimeError, so a single
    ``except LigandParametrisationError`` would let real failures escape as tracebacks.
    """


# Only formats Rosetta's MolFileIOReader actually parses. `.mol2` is NOT among them, and
# this is the reason the check is by extension rather than by trying and catching: handed
# a mol2, the reader returns an EMPTY vector and raises nothing at all.
#     fix/x.mol2: n_types = 0 (no exception raised)
# An empty vector one frame later is an IndexError about a vector1, which tells a user
# nothing about what they did wrong.
SDF_SUFFIXES = (".sdf", ".mol", ".mdl")

# Valence a neutral heavy atom saturates at. Used only to decide whether hydrogens are
# MISSING, never to type anything -- Rosetta does the typing.
_VALENCE_CAP = {"Carbon": 4, "Nitrogen": 3, "Oxygen": 2, "Sulfur": 2}


@dataclass(frozen=True)
class LigandSpec:
    """One ligand file, plus the identity overrides the user may need to supply.

    `key` is a content hash, not a path, and it is what goes into the regeneration key:
    `--ligand-sdf hit.sdf` is a DIFFERENT MOLECULE tomorrow if the screen was re-run, and
    a cached ensemble keyed on the filename would be silently reused for it.

    `name` and `name3` are kept SEPARATE because Rosetta keys them separately, and
    conflating them is the single easiest way to write this module wrong -- see
    `_names_of_file`.
    """

    path: Path
    source: str  # "sdf" or "params"
    key: str  # sha256 of the file contents
    name: str = ""  # override for the registration/lookup key
    name3: str = ""  # override for the 3-letter code written into HETATM records
    mode: str = "placed"  # "placed" (append from the file) or "topology" (already in the PDB)


def file_key(path):
    """sha256 of a ligand file's contents, chunked so a big multi-conformer SDF is fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_spec(path, name="", name3="", mode="placed"):
    """Build a LigandSpec from a path, deciding source by extension."""
    path = Path(path)
    if not path.is_file():
        raise LigandParametrisationError(f"{path}: no such file")
    suffix = path.suffix.lower()
    if suffix == ".params":
        source = "params"
    elif suffix in SDF_SUFFIXES:
        source = "sdf"
    else:
        raise LigandParametrisationError(
            f"{path}: unsupported extension {suffix!r}. Rosetta's reader handles "
            f"{', '.join(SDF_SUFFIXES)} and .params. Note .mol2 is NOT supported -- handed "
            f"one, the reader returns zero residue types and raises nothing, so it is "
            f"refused here rather than failing later as an empty-vector IndexError. "
            f"Convert it to SDF first (obabel -imol2 in.mol2 -osdf -O out.sdf -h)."
        )
    return LigandSpec(path=path, source=source, key=file_key(path), name=name,
                      name3=name3, mode=mode)


# ---------------------------------------------------------------- reading


def read_mutable_types(path):
    """Every MutableResidueType in an SDF, as a plain Python list.

    Rosetta returns a vector1, which is 1-INDEXED; `rts[0]` is an error, not the first
    molecule. Converting here means no caller has to remember that.
    """
    from pyrosetta.rosetta.core.chemical import sdf

    types = sdf.convert_to_ResidueTypes(str(path), False)
    return [types[i] for i in range(1, len(types) + 1)]


def single_mutable_type(path):
    """The one molecule in an SDF, refusing a file that holds several.

    A multi-molecule SDF is the normal output of a docking run -- one entry per pose or
    per hit -- and silently taking the first would score an arbitrary member of the set.
    Measured: fix/multi.sdf -> n_types=2 names=['MOH','XYZ'].
    """
    types = read_mutable_types(path)
    if not types:
        raise LigandParametrisationError(
            f"{path}: Rosetta's reader produced no residue type from this file. If it is "
            f"a .mol2 renamed to .sdf, that is the cause -- the reader returns an empty "
            f"result for mol2 without raising."
        )
    if len(types) > 1:
        names = ", ".join(t.name() for t in types[:5])
        raise LigandParametrisationError(
            f"{path}: holds {len(types)} molecules ({names}...). FrustX will not guess "
            f"which one you docked. Split the file and pass the entry you mean."
        )
    return types[0]


def _names_of_file(spec):
    """(name, name3) as the file declares them, read WITHOUT registering anything.

    Both, keyed separately, because Rosetta indexes them separately and the obvious
    simplification -- key everything on the 3-letter code -- is broken for every real
    docking output. An SDF title longer than three characters gets a name3 that is merely
    a truncation, and that truncation is not a lookup key:

        fix/longtitle.sdf -> name='ligand_1'  name3='lig'
        prts.has_name('ligand_1') = True
        prts.has_name('lig')      = False
        prts.name_map('lig')      -> RuntimeError: The residue lig could not be generated.

    So `name` is what we register and look up under; `name3` is what lands in the HETATM
    record, in contacts.csv and in frustration.pdb.
    """
    if spec.source == "params":
        name = name3 = None
        for line in Path(spec.path).read_text().splitlines():
            if line.startswith("NAME "):
                name = line.split()[1]
            elif line.startswith("IO_STRING"):
                name3 = line.split()[1]
        if name is None or name3 is None:
            raise LigandParametrisationError(
                f"{spec.path}: a params file needs both a NAME and an IO_STRING line. "
                f"Without them it declares no residue identity and nothing in the "
                f"structure can be matched to it."
            )
        return (spec.name or name), (spec.name3 or name3)
    mutable = single_mutable_type(spec.path)
    return (spec.name or mutable.name()), (spec.name3 or mutable.name3())


# ---------------------------------------------------------------- gates


def real_heavy_atom_count(mutable):
    """Heavy atoms that physically exist.

    NOT `nheavyatoms()`. That is a prefix count and it INCLUDES VIRTUAL atoms, which
    Rosetta adds to give small molecules enough atoms to define a coordinate frame.
    Measured on bromide: natoms 3, nheavyatoms() 3, virtual indices [2, 3] -- one real
    atom. A bromide is one atom. This project has already shipped one bug from trusting
    that count (frustration.py:250 carries the same filter for the same reason).
    """
    n = 0
    for vd in mutable.all_atoms():
        atom = mutable.atom(vd)
        if not atom.is_hydrogen() and not atom.is_virtual():
            n += 1
    return n


def missing_hydrogens(mutable):
    """Heavy atoms that are short of valence and carry no hydrogen, by name.

    A GLOBAL "are there any hydrogens" TEST IS NOT ENOUGH, and this is the whole reason
    the function is written per-atom. Polar-hydrogens-only is exactly what AutoDock and
    PDBQT-derived pipelines emit, and one polar hydrogen defeats a global count while the
    molecule is still badly wrong. All three of these parse without complaint:

        methanol.sdf         n_H=4   C1 CH3 -0.234   O1 OH  -0.624   (correct)
        methanol_polarH.sdf  n_H=1   C1 CH1 +0.017   O1 OH  -0.553   (corrupt)
        methanol_noH.sdf     n_H=0   C1 CH1 +0.335   O1 OOC -0.335   (corrupt)

    The carbon goes CH3 -> CH1 and every charge moves. So the test is per-atom valence
    saturation: sum the bond ORDERS (a triple bond counts 3 -- erlotinib's terminal
    alkyne carbon is C#CH, used = 3 + 1 = 4, and must not be flagged), and a heavy atom
    below its element's cap with no hydrogen on it is missing hydrogens.

    Bond order comes from `int(mutable.bond(a, b).bond_name())`, which is the BondName
    enum: Single 1, Double 2, Triple 3, Aromatic 4, Unknown 0. Aromatic and unknown are
    counted as 1 -- charitable on purpose, since this gate exists to catch a stripped
    file, not to audit a chemist's bond perception. Verified to flag nothing across all
    29 heavy atoms of erlotinib.
    """
    bare = []
    for vd in mutable.all_atoms():
        atom = mutable.atom(vd)
        if atom.is_hydrogen() or atom.is_virtual():
            continue
        element = atom.element_type().get_chemical_name()
        cap = _VALENCE_CAP.get(element)
        if cap is None:
            continue  # halogens, metals, phosphorus: no simple neutral valence to assume
        has_hydrogen = False
        used = 0
        for other in mutable.bonded_neighbors(vd):
            neighbour = mutable.atom(other)
            if neighbour.is_virtual():
                continue
            if neighbour.is_hydrogen():
                has_hydrogen = True
                used += 1
                continue
            order = int(mutable.bond(vd, other).bond_name())
            used += order if order in (1, 2, 3) else 1
        if used < cap and not has_hydrogen:
            bare.append(atom.name().strip())
    return bare


def require_hydrogens(mutable, path):
    """Refuse a ligand whose hydrogens are absent or only partially present."""
    bare = missing_hydrogens(mutable)
    if bare:
        shown = ", ".join(bare[:8])
        more = f" (and {len(bare) - 8} more)" if len(bare) > 8 else ""
        raise LigandParametrisationError(
            f"{path}: {len(bare)} heavy atom(s) are below valence with no hydrogen "
            f"attached: {shown}{more}. Rosetta types atoms from their connectivity, so a "
            f"stripped file gets systematically wrong types and charges -- a methanol "
            f"carbon types as CH1 instead of CH3 -- and every energy computed from it is "
            f"wrong in a way nothing downstream can detect. FrustX will not add the "
            f"hydrogens for you: the protonation state is a choice about pH that belongs "
            f"to whoever prepared the ligand. Add them first, e.g. "
            f"`obabel in.sdf -osdf -O out.sdf -h -p 7.4`."
        )


def check_chemistry(mutable):
    """Warnings about chemistry Rosetta's fa_standard set cannot represent faithfully.

    Returned rather than raised: none of these is a reason to refuse a run, but all of
    them are a reason to prefer a curated .params file if the contact matters.

    The sp-carbon case is the one measured here. fa_standard has no linear-carbon type,
    so erlotinib's terminal alkyne comes back as COO -- a CARBOXYL carbon -- carrying
    +0.683 where the true partial charge is near zero:

        C1  Carbon  type=COO  q=+0.683  nH=1  (this is C#CH)
        C2  Carbon  type=COO  q=+0.683  nH=0

    That is a ~0.7 e error on two atoms, and fa_elec is a pairwise product of charges, so
    it does not stay local to the ligand. Whether it matters for a FRUSTRATION index is a
    separate question -- the ligand is frozen and identically typed in the native and in
    every decoy, so a constant offset cancels in the Z-score of Eq. 1 -- but the
    cancellation is only exact if the mutated protein residue does not itself change how
    that atom is screened. Not resolved; see docs/method.md.
    """
    warnings = []
    for vd in mutable.all_atoms():
        atom = mutable.atom(vd)
        if atom.is_virtual() or atom.is_hydrogen():
            continue
        element = atom.element_type().get_chemical_name()
        type_name = mutable.atom_type(vd).name().strip()
        name = atom.name().strip()
        if element == "Carbon":
            triple = any(
                int(mutable.bond(vd, other).bond_name()) == 3
                for other in mutable.bonded_neighbors(vd)
            )
            if triple:
                warnings.append(
                    f"{name}: sp (triple-bonded) carbon typed {type_name!r} with charge "
                    f"{atom.charge():+.3f}; fa_standard has no linear carbon type"
                )
    net = sum(mutable.atom(vd).charge() for vd in mutable.all_atoms())
    if abs(net - round(net)) > 0.05:
        warnings.append(
            f"net charge {net:+.3f} is not close to an integer, which usually means the "
            f"input's formal charges or protonation were not what the typing assumed"
        )
    return warnings


# ---------------------------------------------------------------- registration


def refuse_shadowing(global_set, name, name3, path):
    """Refuse a ligand whose name OR name3 is already defined in fa_standard.

    BOTH indices, and this is not defensive over-checking -- a name-only guard misses most
    of the real cases. `ResidueTypeSet.has_name` matches `.name()`, and many curated
    ligand types have a name that differs from their 3-letter code:

        'ATP'  has_name=True   has_name3=True
        'SAH'  has_name=False  has_name3=True     <- a name-only guard lets this through
        ' ZN'  has_name=False  has_name3=True     <- note the padding

    Registering over one is accepted WITHOUT COMPLAINT and the new type wins the lookup:

        fix/atpname.sdf registered without complaint
        name_map('ATP').natoms() = 6    (the curated ATP has 45)

    So a user who fetches ATP_ideal.sdf from RCSB and passes it would silently get the
    neutral tetraacid in place of Rosetta's fitted, charged nucleotide -- in a kinase
    P-loop, which is precisely where it would matter most. Refuse instead.
    """
    if global_set.has_name(name) or global_set.has_name3(name3):
        raise LigandParametrisationError(
            f"{path}: {name!r}/{name3!r} is already defined in Rosetta's fa_standard set. "
            f"Registering this file would SHADOW the curated type for this pose, with no "
            f"warning and no record. Rosetta's parameters for it were fitted; a typing "
            f"derived from an SDF is a lookup table. Drop the flag to use the curated "
            f"type -- or, if you really do mean to replace it, rename the molecule with "
            f"--ligand-name/--ligand-name3 so the substitution is visible in the output."
        )


def prepared_mutable(spec, warn=None):
    """The MutableResidueType for a spec, gated, renamed and ready to register."""
    mutable = single_mutable_type(spec.path)
    require_hydrogens(mutable, spec.path)
    if warn is not None:
        for message in check_chemistry(mutable):
            warn(f"{spec.path.name}: {message}")
    if spec.name:
        mutable.name(spec.name)
    if spec.name3:
        mutable.name3(spec.name3)
    # Match HETATM records to atoms by GEOMETRY rather than by name. An SDF carries no PDB
    # atom names, so Rosetta invents C1/C2/...; a PDB's HETATM block for the same molecule
    # says C13/O2/N4. Without this, atom matching fails and the pose builder tries to
    # rebuild the whole ligand from internal coordinates:
    #     remap=True  -> ligand coordinates match the HETATM block to 0.0 A
    #     remap=False -> RuntimeError: too many tries in fill_missing_atoms!
    # The constructor default is already True; it is set explicitly because the entire
    # topology mode depends on it and a default is not a guarantee.
    mutable.remap_pdb_atom_names(True)
    return mutable


def build_residue_type_set(specs, warn=None):
    """A PoseResidueTypeSet carrying every ligand type, layered over fa_standard."""
    import pyrosetta.rosetta.core.chemical as chem

    global_set = chem.ChemicalManager.get_instance().residue_type_set("fa_standard")

    # Collisions among the user's OWN files, checked in Python before Rosetta sees them.
    # Rosetta does refuse a duplicate, loudly -- but as a C++ exit that surfaces as a bare
    # RuntimeError:
    #     ERROR: ... residue type 'MOH' already exists in the cache.
    #     ERROR:: Exit from: .../core/chemical/ResidueTypeSetCache.cc line: 107
    # and this is the NORMAL case, not an edge case: Vina, Glide and Boltz all name every
    # hit UNL or LIG. name3 is checked as well as name, because name3 is the key that
    # contacts.csv and frustration.pdb are written under -- two molecules sharing it would
    # be indistinguishable in the results even if Rosetta kept them apart internally.
    seen = {"name": {}, "name3": {}}
    resolved = []
    for spec in specs:
        name, name3 = _names_of_file(spec)
        refuse_shadowing(global_set, name, name3, spec.path)
        for label, value in (("name", name), ("name3", name3)):
            if value in seen[label]:
                raise LigandParametrisationError(
                    f"{spec.path} and {seen[label][value]} both declare {label} {value!r}. "
                    f"Rename one -- the SDF title line, or NAME/IO_STRING in a params "
                    f"file, or --ligand-name/--ligand-name3 -- so the two molecules stay "
                    f"distinguishable in contacts.csv and in the regeneration key."
                )
            seen[label][value] = spec.path
        resolved.append((spec, name, name3))

    pose_set = chem.PoseResidueTypeSet(global_set)
    report = []
    for spec, name, name3 in resolved:
        if spec.source == "params":
            # A params file registers BY FILENAME through the same call -- Rosetta parses
            # it itself. This is the escape hatch when auto-typing is not good enough.
            pose_set.add_base_residue_type(str(spec.path))
        else:
            # add_base_residue_type takes a MUTABLE type. Handing it a finalised
            # ResidueType is a TypeError, not a conversion.
            pose_set.add_base_residue_type(prepared_mutable(spec, warn=warn))
        report.append({"mode": spec.mode, "source": spec.source, "name": name,
                       "name3": name3, "key": spec.key, "path": str(spec.path)})
    return pose_set, report


def ligands_setting(report):
    """The ligand identity as a provenance settings value: content keys, sorted.

    Sorted so that the order flags were typed in does not invalidate a cached ensemble.
    `mode` is part of it because the same file used as `topology` and as `placed` puts the
    ligand in two different places, and `key` is a content hash rather than a path because
    a re-docked hit.sdf is a different molecule under the same name.
    """
    return sorted(
        (e["mode"], e["source"], e["name"], e["name3"], e["key"]) for e in report
    )


# ---------------------------------------------------------------- pose loading


def append_ligand(pose, residue_type, chain=None, number=1):
    """Append a ligand by jump at the coordinates its own file carried, WITH a PDB identity.

    No coordinate-setting loop is needed: `Residue(rt, True)` is built from the type's own
    conformer, which for an SDF-derived type is the SDF coordinate block, atom for atom.

    The PDBInfo assignment is not cosmetic, and leaving it out is a real bug.
    `append_residue_by_jump` leaves the record at Rosetta's unknown-chain sentinel:

        res 16 MOH chain='^' num=0
        res 17 XYZ chain='^' num=0

    while `dump_pdb` INVENTS a different identity at write time (`MOH B 16`, `XYZ C 17`).
    So contacts.csv, which reads pdb_info, and frustration.pdb disagree about which residue
    is which for a SINGLE ligand -- and for TWO, the ('^', 0) key is duplicated, the
    residue table's index stops being unique, and write_bfactor_pdb dies:

        duplicated (chain,resnum) rows: 1
        write_bfactor_pdb RAISED: ValueError The truth value of a Series is ambiguous

    after the entire decoy ensemble has been computed, which is the most expensive place a
    run can fail. Ligands loaded in `topology` mode need none of this: they come from the
    PDB and keep its record.
    """
    import pyrosetta

    residue = pyrosetta.rosetta.core.conformation.Residue(residue_type, True)
    pose.append_residue_by_jump(residue, pose.total_residue(), "", "", True)
    index = pose.total_residue()
    info = pose.pdb_info()
    if info is not None:
        if chain is None:
            used = {info.chain(k) for k in range(1, index)}
            chain = next((c for c in string.ascii_uppercase if c not in used), "X")
        info.chain(index, chain)
        info.number(index, number)
        info.icode(index, " ")
        info.obsolete(False)
    return index


def load_pose_with_ligands(structure, specs, warn=None):
    """Load a structure whose ligands are typed from user-supplied files.

    Two modes, and the difference is where the ligand's coordinates come from:

    `topology` -- the ligand is already a HETATM block in the structure, and the file only
    supplies its chemistry. This is the post-crystallography / co-folding case, and it is
    the one to prefer: the pose keeps the structure's own chain and residue number.

    `placed` -- the ligand is NOT in the structure and is appended from its own file's
    coordinates. This is the post-docking case, where the docking program wrote a separate
    SDF for the pose it found.

    Returns (pose, report). The report is built from the POSE's residue types, not from the
    type set, so it records what was actually scored rather than what we intended to
    register.
    """
    import pyrosetta

    pose_set, report = build_residue_type_set(specs, warn=warn)

    pose = pyrosetta.rosetta.core.pose.Pose()
    pose.conformation().reset_residue_type_set_for_conf(pose_set)
    pyrosetta.rosetta.core.import_pose.pose_from_file(
        pose, str(structure), False,
        pyrosetta.rosetta.core.import_pose.FileType.PDB_file,
    )

    present = {pose.residue_type(i).name() for i in range(1, pose.total_residue() + 1)}
    for entry in report:
        if entry["mode"] == "topology":
            if entry["name"] not in present:
                raise LigandParametrisationError(
                    f"{entry['path']}: declared as a topology ligand, but no residue named "
                    f"{entry['name']!r} is present in {structure} after loading. Either the "
                    f"structure has no HETATM block for it, or its 3-letter code there is "
                    f"not {entry['name3']!r}. Pass --ligand-name3 to match the structure, or "
                    f"use placed mode to append it from the file's own coordinates."
                )
        else:
            if entry["name"] in present:
                raise LigandParametrisationError(
                    f"{entry['path']}: declared as a placed ligand to be appended, but "
                    f"{entry['name']!r} is ALREADY in {structure}. Appending would score it "
                    f"twice. Use topology mode to type the copy that is already there."
                )
            index = append_ligand(pose, pose_set.name_map(entry["name"]))
            entry["pose_index"] = index

    info = pose.pdb_info()
    for entry in report:
        for i in range(1, pose.total_residue() + 1):
            if pose.residue_type(i).name() == entry["name"]:
                entry["pose_index"] = i
                if info is not None:
                    entry["chain"] = info.chain(i)
                    entry["resnum"] = int(info.number(i))
                entry["n_heavy_atoms"] = sum(
                    not pose.residue(i).is_virtual(k)
                    for k in range(1, pose.residue(i).nheavyatoms() + 1)
                )
                break
    return pose, report
