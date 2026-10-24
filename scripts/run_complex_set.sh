#!/usr/bin/env bash
# Decoy generation for the KLIFS-first complex set: 91 complexes, 8 ligands, 71 kinases.
#
# The question (docs/kinome_dataset.md section 7, docs/method.md "KLIFS-first complex
# set"): across one ligand's co-crystal complexes with many kinases, does the frustration
# of its contacts follow the measured affinity (selectivity)?
#
# scripts/fetch_complex_set.py wrote runs.csv, the only source of which complex, which
# chain, which ligand copy and which ligand code is scored. Only rows with ready=True run.
# Every run uses --ligand: the SDF supplies the chemistry, the pose is the crystal's.
# B96 (doramapimod) runs as B9X because fa_standard already has a "B96" -- a
# non-canonical amino acid -- see ROSETTA_CODE_CLASH in fetch_complex_set.py.
#
#   ./scripts/run_complex_set.sh --dry-run     # print the commands, run nothing
#   DECOYS=50 ./scripts/run_complex_set.sh     # quick pass (into a separate RESULTS!)
#   ./scripts/run_complex_set.sh               # the real thing: relax, 1000 decoys
#
# Settings are PINNED to the pilot and the COX series (relax / 1000 / seed 0 / w=0 /
# readout pair / cutoff 10 / ligand-cutoff 6 / contact-atom CA): --protocol sets Eq. 1's
# denominator and n_decoys its sampling error, so a frustration compared across
# mismatched settings is not a number. The last step re-reads every run.json and refuses
# to call the batch comparable if they disagree (COMPARABLE_SETTINGS, cox_selectivity.py).
#
# Resumable: a complex whose run.json already records the current DECOYS and PROTOCOL is
# skipped, so after an interrupt or a reboot just start it again.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRUSTX="$ROOT/.venv/bin/frustx"
PYTHON="$ROOT/.venv/bin/python"
MANIFEST="${MANIFEST:-$ROOT/data/kinome/complex_set/runs.csv}"
RESULTS="${RESULTS:-$ROOT/results/complex_set}"
LOGS="$RESULTS/logs"

DECOYS="${DECOYS:-1000}"
PROTOCOL="${PROTOCOL:-relax}"
# This machine: 2 x Xeon Silver 4314, 32 physical cores / 64 threads, 125 GB. Decoys are
# independent, so --jobs is the knob that only costs wall clock. 60 keeps a few threads
# for the parent process and the OS; the pilot used 50 (~10 min per ~270-residue complex).
JOBS="${JOBS:-60}"
SEED="${SEED:-0}"
# Per-run wall-clock ceiling, so one pathological complex cannot stall the batch. The
# largest structure (p110g, 839 residues) is ~3x a typical kinase domain.
RUN_TIMEOUT="${RUN_TIMEOUT:-21600}"

if [[ ! -x "$FRUSTX" ]]; then
    echo "Missing executable: $FRUSTX" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST  (run scripts/fetch_complex_set.py first)" >&2
    exit 2
fi

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LOGS"

echo "complex-set batch: decoys=$DECOYS protocol=$PROTOCOL jobs=$JOBS seed=$SEED"
echo "manifest:          $MANIFEST"
echo "results:           $RESULTS"
echo

total=0
failed=0
skipped=0
expected=$("$PYTHON" -c 'import sys,pandas as pd; r=pd.read_csv(sys.argv[1]); print(int(r["ready"].sum()))' "$MANIFEST")

# pandas parses the CSV and hands bash clean tab-separated columns
while IFS=$'\t' read -r name ligand name3 structure sdf; do
    [[ -n "$name" ]] || continue
    total=$((total + 1))
    out="$RESULTS/$name"
    log="$LOGS/$name.log"

    if [[ ! -f "$structure" ]]; then
        echo "MISSING STRUCTURE: $structure" | tee -a "$LOGS/batch_errors.log" >&2
        failed=$((failed + 1))
        continue
    fi

    if [[ -f "$out/run.json" ]] \
       && grep -q "\"n_decoys\": *$DECOYS\b" "$out/run.json" \
       && grep -q "\"protocol\": *\"$PROTOCOL\"" "$out/run.json"; then
        echo "SKIP completed: $name"
        skipped=$((skipped + 1))
        continue
    fi

    cmd=("$FRUSTX" "$structure" --ligand "$sdf")
    # renamed only where the PDB code collides with a Rosetta residue type
    [[ "$name3" != "$ligand" ]] && cmd+=(--ligand-name "$name3")
    cmd+=(--ligand-name3 "$name3"
          -n "$DECOYS"
          --protocol "$PROTOCOL"
          --seed "$SEED"
          -o "$out"
          --jobs "$JOBS")

    if (( DRY_RUN )); then
        printf '%q ' "${cmd[@]}"; printf '\n'
        continue
    fi

    echo "START $name  ($total/$expected)  $(date -Is)"
    # a partial directory from an interrupted run would look like a valid checkpoint
    rm -rf "$out"
    run_start=$SECONDS
    if timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" "${cmd[@]}" > "$log" 2>&1; then
        echo "DONE  $name  $(( (SECONDS - run_start) / 60 )) min  $(date -Is)"
    else
        status=$?
        if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
            msg="TIMEOUT $name after ${RUN_TIMEOUT}s; see $log"
        else
            msg="FAILED  $name (exit=$status); see $log"
        fi
        echo "$msg" | tee -a "$LOGS/batch_errors.log" >&2
        failed=$((failed + 1))
    fi
done < <("$PYTHON" - "$MANIFEST" <<'PY'
import sys
import pandas as pd
r = pd.read_csv(sys.argv[1], keep_default_na=False)
r = r[r["ready"].astype(str) == "True"]
for x in r.itertuples():
    print("\t".join([x.complex, x.ligand_code, x.ligand_name3, x.structure, x.ligand_sdf]))
PY
)

echo
if (( DRY_RUN )); then
    echo "dry run: $total commands printed, nothing executed."
    exit 0
fi

"$PYTHON" - "$RESULTS" "$expected" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
from cox_selectivity import COMPARABLE_SETTINGS

results = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
runs = {rj.parent.name: json.loads(rj.read_text())
        for rj in sorted(results.glob("*/run.json"))}

print(f"{len(runs)}/{expected} runs have a run.json")
if not runs:
    sys.exit(1)

disagree = {}
for key in COMPARABLE_SETTINGS:
    values = {name: cfg.get(key) for name, cfg in runs.items()}
    if len(set(map(repr, values.values()))) > 1:
        disagree[key] = values
first = next(iter(runs.values()))
print("settings:", {k: first.get(k) for k in COMPARABLE_SETTINGS})
for name, cfg in sorted(runs.items()):
    print(f"  {name:16} contacts={cfg.get('n_contacts'):>5}  "
          f"{cfg.get('elapsed_seconds', 0) / 60:6.1f} min")

if disagree:
    print("\nREFUSING to call this batch comparable -- runs disagree on:")
    for key, values in disagree.items():
        print(f"  {key}: {values}")
    sys.exit(1)
print("\nall runs agree on every comparability setting.")
PY
verify=$?

completed=$(find "$RESULTS" -mindepth 2 -maxdepth 2 -name run.json -type f 2>/dev/null | wc -l)
echo
echo "Batch finished: $completed/$total runs have a run.json" \
     "($skipped skipped as already complete, $failed failed)."
if (( failed )); then
    echo "Re-run this script to retry anything that failed; completed runs are skipped."
fi
echo "Next: dvc add results && dvc push, then the per-ligand frustration-vs-affinity analysis."
exit $(( failed > 0 || verify != 0 ))
