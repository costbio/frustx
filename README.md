# FrustX

A tool for calculation of local frustration levels in protein structures at atomistic
resolution.

Reimplements the protocol of Chen et al., *Nat Commun* 11:5944 (2020),
["Surveying biomolecular frustration at atomic resolution"](https://pubmed.ncbi.nlm.nih.gov/33230150/),
which scores frustration with the all-atom Rosetta REF2015 energy function rather than a
coarse-grained one.

See [`docs/method.md`](docs/method.md) for the method spec and
[`CLAUDE.md`](CLAUDE.md) for how the repo is organised.

**Status: early, not yet validated per-contact.** Full pipeline runs end to end.
Against frustratometeR (AWSEM) on 1UBQ, always in `mutational`/`singleresidue` mode —
FrustX's decoy randomises identity at fixed native geometry, which is a *mutational*
decoy: residue-level agreement is ρ = **+0.48** (p = 1e-5, n = 76) vs `singleresidue`
mode using a true per-residue Z-score (ρ = +0.41 using the CLI's `mean_frustration`),
and per-contact ρ = +0.32 (p = 1e-9, n = 344) vs `mutational` mode, rising to +0.40 on
direct contacts alone. The per-contact comparison against `configurational` mode has been
withdrawn — that mode's decoy discards the contact's own geometry, so one decoy mean/sd
covers the whole protein and its index is a rescaled energy rather than a per-contact
Z-score. It is also the wrong mode to compare against: FrustX's decoy is a mutational one.

Running AWSEM's own energy through FrustX's machinery separates the two things that
differ (`frustx/awsem.py`, verified exact against frustratometeR's output to 1e-3):
force field alone ρ = +0.18, protocol alone ρ = +0.27. The force field is the larger
difference, and after removing one-body content it is the only comparison that stays
positive.

The deeper limitation: frustratometeR's indices are largely residue-additive (56%
configurational, 95% mutational), while FrustX's is 23%. That 95% is a property of their
*protocol*, not of AWSEM — the same energy through FrustX's protocol is only 39%
additive. **frustratometeR cannot validate
per-contact specificity, which is the property FrustX exists to provide.** Reproducing a figure from Chen et al. (2020) is **not** available as a test: the paper
publishes no per-contact value for any system, and its own Eq. 2 index is ~76%
residue-additive by construction. See `docs/method.md`.

What *is* available is the paper's Fig. 2 quantity — a per-residue 5 Å vicinity count,
whose "vicinity" rule was recovered and verified bit-for-bit against frustratometeR. That
pipeline runs (`scripts/fig2_profile.py`, `scripts/vicinity_profile.py`) and reproduces
sensible profiles on six small-GTPase structures. The *biological* reading first drawn from
it — that GTPase functional elements shift toward frustration on activation — **did not
replicate** across two further conformer pairs (pooled p = 0.86 / 0.19, inconsistent signs;
`scripts/gtpase_replication.py`). The machinery is validated at residue resolution; the
claim is withdrawn. See `docs/method.md`.

## Setup on a new machine

Requires Python ≥ 3.11 and ~4 GB of disk (PyRosetta is 3.2 GB).

```bash
git clone https://github.com/osercinoglu/frustx.git
cd frustx

python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# PyRosetta is not on PyPI and cannot be a declared dependency.
# The PATH prefix is REQUIRED: pyrosetta_installer shells out to a bare `pip`, so
# without it the 3.2 GB lands in whichever python `pip` resolves to -- not the venv.
PATH=".venv/bin:$PATH" .venv/bin/python -m pip install pyrosetta-installer
PATH=".venv/bin:$PATH" .venv/bin/python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(serialization=True)"

.venv/bin/python -m pytest tests/ -q     # 232 tests; all but tests/test_contacts.py need PyRosetta
```

### Data (DVC + GCS)

No input or output data is in git. Structures, decoy runs and result tables live in
`gs://frustx` and are tracked by `results.dvc`.

DVC is intentionally not in `pyproject.toml` — it is data infrastructure, not something
`frustx` imports, and it would make `pip install -e ".[dev]"` ~200 MB heavier:

```bash
.venv/bin/python -m pip install "dvc[gs]"
```

Authenticate with a GCP service-account key that has `roles/storage.objectAdmin` on the
bucket. **Keep the key outside the repo** and readable only by you:

```bash
mkdir -p ~/.gcp && chmod 700 ~/.gcp
# copy your key to ~/.gcp/<key>.json, then:
chmod 600 ~/.gcp/<key>.json
.venv/bin/dvc remote modify --local storage credentialpath ~/.gcp/<key>.json
```

`--local` is not optional: it writes `.dvc/config.local`, which `.dvc/.gitignore`
excludes. Without it the path goes into the tracked `.dvc/config`.

Then fetch the data:

```bash
.venv/bin/dvc pull          # populates results/
.venv/bin/dvc status -c     # "in sync" = local and bucket agree
```

After producing new results, `dvc add results/ && dvc push`, then commit the updated
`results.dvc`. Alternatively `gcloud auth application-default login` works instead of a
key file, if you have the SDK installed.

## Usage

```bash
frustx structure.pdb -o results/
```

Writes `contacts.csv` (one row per contact, the primary result), `residues.csv`
(aggregated per residue), `frustration.pdb` (indices painted into the B-factor column)
and `run.json` (full provenance — results are only comparable between runs with the same
decoy count and relaxation protocol).

Useful flags:

| flag | what |
|---|---|
| `-n, --decoys` | decoy ensemble size (default 1000, the paper's figure) |
| `--protocol` | `relax` (default, FastRelax) or `min` (~10× faster) — this sets Eq. 1's denominator, so it moves every index |
| `--packing-frustration` | keep `fa_rep`, giving the paper's separate structure-quality measure |
| `--cutoff` | Cα–Cα contact cutoff, default 10 Å |
| `-j, --jobs` | decoy-building processes (default 1). ~3.2x at `-j 4`; `-j 8` on 4 physical cores buys only 4.4x |
| `--contact-atom` | `CA` (default, the paper's definition) or `CB`, which admits far fewer pairs with no atomistic interaction |
| `--readout` | `pair` (default) or `neighbourhood`, matching what frustratometeR sums. Diagnostic |
| `--background-weight` | weight `w` on the Eq. 2 background term, default `0` (direct pair energy only). `w=1` is Eq. 2 as literally written, which makes the index degenerate — see `docs/method.md` |

### Ligands

Rosetta already knows a few dozen cofactors (ATP, ZN, MG, SAH, UDP…), and those need no
flags at all — just leave them in the PDB. Anything else — a screening hit, a docked pose,
a co-folding output — needs a file describing its chemistry:

```bash
# the ligand is a HETATM block in the structure; the SDF supplies only its chemistry
frustx complex.pdb -o results/ --ligand ligand.sdf

# the ligand is NOT in the structure and is appended at its own file's coordinates
frustx apo.pdb -o results/ --ligand-placed docked_pose.sdf

# several ligands, and a Rosetta .params file where auto-typing is not good enough
frustx complex.pdb -o results/ --ligand lig_a.sdf --ligand lig_b.params
```

`.sdf`, `.mol` and `.mdl` are read directly; `.mol2` is not supported, convert it first
(`obabel -imol2 in.mol2 -osdf -O out.sdf -h`). Atom names need not match the structure —
they are remapped geometrically.

**The ligand must carry its hydrogens.** A file with none, or with polar hydrogens only,
is refused: Rosetta types atoms from connectivity, so a stripped ligand gets wrong types
and charges silently. Add them first with `obabel in.sdf -osdf -O out.sdf -h -p 7.4`.

Ligands enter the contact map and are never mutated, repacked or moved during decoy
generation. Contacts involving one use a minimum heavy-atom distance instead of Cα–Cα,
since a ligand has no Cα.

| flag | what |
|---|---|
| `--ligand FILE` | ligand already in the structure; repeat for several |
| `--ligand-placed FILE` | ligand appended from its own file's coordinates |
| `--ligand-name NAME` | override the residue name a file declares, applied in order. Needed when two hits are both called `UNL` or `LIG` |
| `--ligand-name3 XXX` | override the 3-letter code, applied in order. For `--ligand` it must match the HETATM residue name |
| `--ligand-cutoff` | minimum heavy-atom distance for a contact involving a ligand, default 6 Å |
| `--no-freeze-ligand` | let ligands move during decoy generation. Not recommended — decoys shuffle amino-acid identity only |

Two warnings are worth acting on. A ligand reported as making **no contacts** usually means
`--ligand-placed` was given coordinates from the wrong frame — an RCSB `_ideal.sdf` is a
generated conformer near the origin, not a docked pose. A **typing warning** means Rosetta's
atom types cannot represent that chemistry faithfully (it has no sp carbon, for instance);
supply a curated `.params` file if the affected contacts matter.

Passing a file whose name collides with one of Rosetta's built-in types is refused rather
than silently preferred — drop the flag to use the curated version, or rename with
`--ligand-name`.

`contacts.csv` carries the index split into the part predictable from the two residues
alone (`frustration_index_onebody` — burial, exposure, packing) and what is particular to
that pair (`frustration_index_specific`). Both are real quantities and neither is the
"true" one: a residue-level question wants the first, while a claim about a *particular*
contact needs the second. `run.json` records `additive_r2`, the fraction of the index the
one-body part explains. See `docs/method.md`.

Colour the output in PyMOL with `spectrum b, blue_white_red, all`. High positive =
minimally frustrated, matching frustratometeR's convention (the paper's Eq. 1 uses the
opposite sign; see `docs/method.md`).

Roughly 12 min per 1000 decoys on a 76-residue protein with `--protocol min`, ~1 h with
`relax`. Progress goes to **stderr**, so it never contaminates a redirected stdout:

```
  decoys ████████████░░░░░░░░░░░░░░  48%  478/1000  28m41s elapsed, 31m18s left
```

The ETA extrapolates the mean time per completed decoy. When stderr is not a terminal the
same information is written as plain lines at each 10% mark instead, since a log file full
of carriage returns is one unreadable line. `--quiet` silences all of it.
