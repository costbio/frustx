# FrustX — working agreement

Local frustration in protein structures at **atomistic** resolution, reproducing
Chen et al., *Nat Commun* 11:5944 (2020). **Read `docs/method.md` first** — it is the
spec, with the paper's formulas and protocol quoted verbatim.

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
  parameter rather than silently picking.
- **Keep it simple.** No abstraction that is not paying for itself right now.

## Data hygiene

**No input or output data is tracked in git.** Structures, energies, decoy ensembles and
result tables are versioned with DVC against a `gs://` remote, which the user configures.
`.gitignore` already excludes `data/`, `results/`, `*.pdb`, `*.csv`, `*.parquet`. The one
exception is tiny hand-built test fixtures under `tests/`, which are code, not data.

Scratch/probe files go in the session scratchpad, not the repo.

**But the session scratchpad is wiped between sessions.** Anything worth regenerating --
reference outputs, validation runs, decoy ensembles -- goes in `results/` (gitignored,
DVC-tracked), not the scratchpad. A frustratometeR reference run and a 500-decoy FrustX
run were lost this way once; both were reproducible, but the compute was not free.
Conclusions belong in `docs/method.md` immediately, not held in a scratch file.

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

Run tests with `.venv/bin/python -m pytest tests/ -q`.

DVC is installed in the venv with the GCS backend (`dvc 3.67.1`, `dvc_gs.GSFileSystem`):

```bash
.venv/bin/python -m pip install "dvc[gs]"
```

It is deliberately **not** in `pyproject.toml` — it is infrastructure for moving data
around, not something `frustx` or the tests import, and putting it in the `dev` extra
would force a ~200 MB install on anyone who just wants to run pytest. (`matplotlib` *is*
in the `dev` extra, since `scripts/plot_*.py` genuinely import it.)

**The repo is not `dvc init`-ed yet and no remote is configured** — that needs the user's
bucket URL and GCS credentials.

## Layout

| Path | What |
|---|---|
| `docs/method.md` | The spec. Paper formulas + protocol, verbatim. Start here. |
| `frustx/contacts.py` | Contact map, Cα–Cα ≤ 10 Å. No PyRosetta dependency, so it is unit-testable on hand-built coordinates. |
| `tests/` | Hand-verifiable toy cases — coordinates on a line, checkable by mental arithmetic. |
| `FrustX.py` | **Dead prototype, kept for reference only.** Do not extend it. Its frustration formula at `:310` is the right shape; its energy source is not (see below). |

## Why the old prototype failed

`FrustX.py` used FASPR to build whole-sequence mutants and GROMACS to score them, then
read the **total system potential** from `.edr` files, with gRINN's pairwise interaction
mode switched *off* (`FrustX.py:207`, `--nointeraction`). Frustration is a local quantity;
a single side-chain change moves the total potential by a few kJ/mol inside ~10⁵ kJ/mol
dominated by solvent. The signal was below the noise floor. Eq. 2 in `docs/method.md` is
the fix. It also crashes on the second input file — `add_argument` is called inside the
per-PDB loop (`FrustX.py:23-27`).

## Scope right now

Protein monomers/complexes, **plus ligands**. Ligand work was deferred for a long time
and is now open, on the strength of the recon in `docs/method.md` — the EGFR
`atomfrust-implementation` branch has already paid the cost of finding out where a
heteroatom breaks a protein-only pipeline, so we are not starting from zero. Read that
section before touching anything ligand-shaped.

Still deferred by explicit decision: **no packaging, no OpenMM backend**.
