# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local frustration in protein structures at **atomistic** resolution, reimplementing Chen
et al., *Nat Commun* 11:5944 (2020), which scores frustration with the all-atom Rosetta
REF2015 force field rather than a coarse-grained one (frustratometeR/AWSEM).
**Read `docs/method.md` first** — it is the spec: the paper's formulas quoted verbatim,
plus every implementation decision and validation experiment run against it since. It is
long (3000+ lines) and append-only; grep it rather than reading start to end, and check
the bottom for the most recent conclusions — several early sections are explicitly
superseded or withdrawn later in the file (search for "CORRECTED", "withdrawn",
"supersedes").

**Status: early, not yet validated per-contact** (see `README.md` for the current
validation numbers against frustratometeR — read it before quoting any correlation
figure, since several were later revised or withdrawn).

## How to work on this repo

- **Baby steps.** One reviewable increment per turn. Do not build a whole subsystem
  silently.
- **Show the code.** Paste new logic into the response. The user needs to stay in command
  of every line; this is research tooling whose correctness he has to vouch for.
- **Mini-test before analysis.** Before trusting an assumption about a library, file
  format, or energy term, write a few-line probe and run it. Never run a real calculation
  on an unverified assumption.
- **Say where to look.** Reference `file.py:line` when reporting.
- **Comment informatively.** Explain *why*, especially where the code encodes a choice
  from the paper. Where the paper is ambiguous, say so in the comment and expose a
  parameter rather than silently picking (see `frustx/config.py` for the pattern — every
  default there carries a comment on what was tried and why it won).
- **Keep it simple.** No abstraction that is not paying for itself right now.

## Commands

```bash
# tests (206 test functions; all but tests/test_contacts.py need PyRosetta on PATH)
.venv/bin/python -m pytest tests/ -q

# a single test file / test
.venv/bin/python -m pytest tests/test_frustration.py -q
.venv/bin/python -m pytest tests/test_frustration.py::test_native_beats_decoys -q

# run the CLI
.venv/bin/frustx structure.pdb -o results/some_run
.venv/bin/frustx structure.pdb -o results/some_run --protocol min -n 200   # fast, for iterating

# with a ligand (see README.md "Ligands" for the full flag set)
.venv/bin/frustx complex.pdb -o results/some_run --ligand ligand.sdf
```

Install: `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"`, then
PyRosetta per the **Environment** section below. There is no lint config in the repo
(no ruff/flake8/black entry) — don't invent one.

## Architecture

### Compute pipeline (the dependency order that matters)

```
contacts.py   ── geometric contact map (Cα–Cα ≤ 10 Å, or min heavy-atom dist for
                  ligand contacts). No PyRosetta import — unit-testable on bare
                  coordinates. Fixes the meaning of "a contact" for everything below.
energies.py   ── e_ij: pairwise REF2015 interaction energy for one pose. The two
                  non-obvious extraction steps (bb-bb hbonds off the default energy
                  graph, rama_prepro in a long-range container) are the load-bearing
                  logic here — see the module docstring and docs/method.md
                  "Implementation notes" before touching this file.
decoys.py     ── the randomised reference ensemble for Eq. 1: shuffle the sequence,
                  repack side chains onto the fixed backbone, relax. native_reference()
                  runs the native sequence through the *same* repack+relax (only
                  without shuffling) — do not compute e_ij on a bare input pose and
                  call it E0, that comparison is invalid (see "Caveat on raw input
                  structures" in docs/method.md).
frustration.py── Eq. 1: F_ij = (⟨E_decoy⟩ − E_native) / σ(E_decoy), with the sign
                  flipped from the paper so output matches the rest of the field
                  (frustratometeR convention: high positive = minimally frustrated).
                  E_ij itself is e_ij + w·½(R_i+R_j) (config.DEFAULT_BACKGROUND_WEIGHT,
                  default w=0 — w=1 is Eq. 2 as literally written but makes the index
                  degenerate, see below).
additivity.py ── splits any per-contact index into a one-body part (predictable from
                  the two residues alone: burial/exposure/packing) and a
                  contact-specific residual. Both are reported; neither is "the real
                  one" (README's frustration_index_onebody / _specific).
output.py     ── FrustrationResult -> contact_table / residue_table / bfactor-painted
                  PDB. The paper defines frustration per contact, not per residue —
                  residue_table is FrustX's own aggregation, not the paper's.
cli.py        ── argparse entry point; writes run.json (full provenance) alongside
                  every result. config.py is deliberately import-light (no numpy/
                  Bio/pyrosetta) so `frustx --help` doesn't pay PyRosetta's ~2.5s
                  import cost.
provenance.py ── content-hash keys deciding whether a checkpointed intermediate
                  (decoy structures, tensors) is still valid for reuse — settings
                  changes invalidate different stages independently (e.g. --readout
                  invalidates scoring but not the decoy structures themselves).
ligand_params.py ── turns an arbitrary SDF into a Rosetta residue type via PyRosetta's
                  own SDF reader (core::chemical::sdf::convert_to_ResidueTypes),
                  registered per-pose into a PoseResidueTypeSet layered over
                  fa_standard. Ligands enter the contact map but are frozen (never
                  shuffled/repacked) during decoy generation.
awsem.py      ── independent reimplementation of AWSEM's contact energy, transcribed
                  from frustratometeR's bundled LAMMPS source. Exists purely to
                  separate "different force field" from "different protocol" when
                  comparing FrustX against frustratometeR — not part of the main
                  REF2015 pipeline.
```

`scripts/` holds one-off analysis/validation drivers (decoy sweeps, comparisons against
frustratometeR, figure reproduction) — these are how the conclusions in `docs/method.md`
were produced, not part of the package. `tests/fixtures/` holds tiny hand-built SDFs
(methanol, bromide, etc.) used to unit-test `ligand_params.py` without needing
PyRosetta-scale structures.

### Things worth knowing before changing the math

- **Eq. 2's contact energy algebraically simplifies**: `E_ij = e_ij + ½Σ_{k≠j}e_ik +
  ½Σ_{l≠i}e_jl` reduces exactly to `½(R_i + R_j)` where `R_i = Σ_k e_ik` — the direct
  `e_ij` term cancels. This is why `w` (the background weight) exists as a knob rather
  than Eq. 2 being hardcoded: `w=1` reproduces the paper literally but makes the index
  ~86-95% reducible to a residue-level quantity and reports zero highly-frustrated
  contacts on ubiquitin; `w=0` (the default) keeps the index pair-specific. See
  `docs/method.md` for the full derivation and the w-sweep experiments.
- **`--protocol`** (`min` vs `relax`) sets Eq. 1's denominator (decoy spread), so it
  moves every index — results at different protocols are not comparable, which is why
  `run.json` records it and `provenance.py` treats it as invalidating decoy structures.
- PyRosetta is required by everything except `contacts.py` and `ligand_params.py`'s
  pure-SDF-parsing parts; that's the reasoning behind which tests need it.

## Why the old prototype failed

`FrustX.py` (root of the repo) is a **dead prototype, kept for reference only — do not
extend it.** It used FASPR to build whole-sequence mutants and GROMACS to score them,
then read the **total system potential** from `.edr` files with gRINN's pairwise mode
switched off (`--nointeraction`). Frustration is local; a single side-chain change moves
the total potential by a few kJ/mol inside ~10⁵ kJ/mol dominated by solvent — the signal
was below the noise floor. Eq. 2 (the many-body background term) is the fix, and it's
why `energies.py`/`frustration.py` exist. `FrustX.py` also crashes on a second input
file, since `add_argument` is called inside the per-PDB loop (`FrustX.py:23-27`).

## Scope right now

Protein monomers/complexes, **plus ligands** (see README's "Ligands" section for the
CLI surface — this is active, not deferred). Still deferred by explicit decision: **no
packaging, no OpenMM backend** (OpenMM cannot reproduce these numbers — different force
field, different decoy semantics — so it would be a separate experiment, not a
prerequisite).

## Data hygiene

**No input or output data is tracked in git.** Structures, energies, decoy ensembles and
result tables are versioned with DVC against a `gs://frustx` remote (already `dvc
init`-ed and configured — check `.dvc/config` before assuming otherwise). `.gitignore`
excludes `data/`, `results/`, `*.pdb`, `*.csv`, `*.parquet`; the one exception is tiny
hand-built test fixtures under `tests/`, which are code, not data. `results.dvc` is the
git-tracked pointer to what's currently in the bucket — `dvc pull` populates `results/`,
`dvc add results/ && dvc push` after producing new output, then commit the updated
`results.dvc`.

Scratch/probe files go in the session scratchpad, not the repo. **But the scratchpad is
wiped between sessions** — anything worth regenerating (reference outputs, validation
runs, decoy ensembles) goes in `results/` (gitignored, DVC-tracked), not the scratchpad.
A frustratometeR reference run and a 500-decoy FrustX run were lost this way once; both
were reproducible, but the compute was not free. Conclusions belong in `docs/method.md`
immediately, not held in a scratch file.

## Environment

`.venv/` (gitignored). Deps via `pip install -e ".[dev]"`.

PyRosetta is **not** a PyPI package and is not in `pyproject.toml`; install it with:

```bash
# NOTE the PATH prefix. pyrosetta_installer shells out to a bare `pip` via
# shell=True (pyrosetta_installer/__init__.py:89) rather than
# `sys.executable -m pip`, so without this it silently installs 3.2 GB into
# whichever python `pip` resolves to on PATH -- NOT into the venv.
PATH=".venv/bin:$PATH" .venv/bin/python -m pip install pyrosetta-installer
PATH=".venv/bin:$PATH" .venv/bin/python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(serialization=True)"
```

DVC is installed in the venv with the GCS backend (`dvc[gs]`), authenticated with a
GCP service-account key kept **outside the repo** (`~/.gcp/`, mode 700) and registered
with `dvc remote modify --local storage credentialpath ~/.gcp/<key>.json` — `--local` is
not optional, it keeps the path out of the tracked `.dvc/config`. It is deliberately
**not** in `pyproject.toml` — it is infrastructure for moving data around, not something
`frustx` or the tests import, and putting it in the `dev` extra would force a ~200 MB
install on anyone who just wants to run pytest. (`matplotlib` and `latex2mathml` *are* in
the `dev` extra, since `scripts/plot_*.py` and `docs/explainer/build.py` genuinely import
them.)
