#!/usr/bin/env bash
# Decoy generation for the paralog pilot: 5 pairs, 10 complexes.
#
# The question (docs/kinome_dataset.md, docs/method.md "The kinome dataset"): for one
# ligand bound to two closest-relative kinases with a known affinity difference, is that
# difference carried by the few KLIFS pocket positions where the two kinases differ?
#
# Unlike the COX batch, these are real co-crystal poses, so every run uses --ligand: the
# SDF supplies only the chemistry and the coordinates come from the structure's HETATM
# block. scripts/fetch_pilot.py produced both, and pilot_runs.csv is the manifest this
# script reads -- it is the only source of which complex, which chain and which ligand
# copy is being scored.
#
#   ./scripts/run_pilot.sh --dry-run       # print the commands, run nothing
#   DECOYS=50 ./scripts/run_pilot.sh       # quick smoke pass over all ten
#   ./scripts/run_pilot.sh                 # the real thing: relax, 1000 decoys
#
# Settings are PINNED and identical across all ten runs, because --protocol sets Eq. 1's
# denominator and n_decoys its sampling error: a frustration difference taken across
# mismatched settings is not a number (see COMPARABLE_SETTINGS in
# scripts/cox_selectivity.py). The last thing this script does is re-read every run.json
# and refuse to call the batch complete if they disagree.
#
# They match the COX series on purpose -- relax / 1000 / w=0 / readout pair / cutoff 10 /
# ligand-cutoff 6 / contact-atom CA -- so the two experiments stay on one scale.
#
# Re-running skips complexes already finished at the current DECOYS and PROTOCOL, so the
# batch is resumable after an interrupt or a machine reboot.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRUSTX="$ROOT/.venv/bin/frustx"
PYTHON="$ROOT/.venv/bin/python"
MANIFEST="${MANIFEST:-$ROOT/data/kinome/pilot/pilot_runs.csv}"
RESULTS="${RESULTS:-$ROOT/results/pilot}"
LOGS="$RESULTS/logs"

DECOYS="${DECOYS:-1000}"
PROTOCOL="${PROTOCOL:-relax}"
# 50 is what the 98-run COX batch used on this machine (64 logical / 16 physical cores).
# Decoys are independent, so this is the one knob that only costs wall clock.
JOBS="${JOBS:-50}"
SEED="${SEED:-0}"
# Wall-clock ceiling per run. A 1000-decoy relax job on a ~270-residue kinase domain at
# --jobs 50 should land far inside this; the timeout exists so that one pathological
# PDB-SDF atom mapping cannot stall the whole batch overnight.
RUN_TIMEOUT="${RUN_TIMEOUT:-21600}"

if [[ ! -x "$FRUSTX" ]]; then
    echo "Missing executable: $FRUSTX" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST  (run scripts/fetch_pilot.py first)" >&2
    exit 2
fi

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LOGS"

echo "pilot batch: decoys=$DECOYS protocol=$PROTOCOL jobs=$JOBS seed=$SEED"
echo "manifest:    $MANIFEST"
echo "results:     $RESULTS"
echo

total=0
failed=0
skipped=0
started_at="$(date -Is)"
# how many complexes the manifest holds, so progress counts are not hard-coded
expected=$("$PYTHON" -c 'import sys,pandas as pd; print(len(pd.read_csv(sys.argv[1])))' "$MANIFEST")

# The manifest is CSV with quoted fields; let pandas parse it and hand bash clean
# tab-separated columns rather than parsing CSV in shell.
while IFS=$'\t' read -r ligand kinase pair role structure sdf; do
    [[ -n "$ligand" ]] || continue
    total=$((total + 1))
    name="${ligand}_${kinase}"
    out="$RESULTS/$name"
    log="$LOGS/$name.log"

    if [[ ! -f "$structure" ]]; then
        echo "MISSING STRUCTURE: $structure" | tee -a "$LOGS/batch_errors.log" >&2
        failed=$((failed + 1))
        continue
    fi

    # Resume: a run.json recording the current decoy count AND protocol means this
    # complex is already done at these settings. Anything else is redone.
    if [[ -f "$out/run.json" ]] \
       && grep -q "\"n_decoys\": *$DECOYS\b" "$out/run.json" \
       && grep -q "\"protocol\": *\"$PROTOCOL\"" "$out/run.json"; then
        echo "SKIP completed: $name"
        skipped=$((skipped + 1))
        continue
    fi

    cmd=("$FRUSTX" "$structure"
         --ligand "$sdf"
         --ligand-name3 "$ligand"
         -n "$DECOYS"
         --protocol "$PROTOCOL"
         --seed "$SEED"
         -o "$out"
         --jobs "$JOBS")

    if (( DRY_RUN )); then
        printf '%q ' "${cmd[@]}"; printf '\n'
        continue
    fi

    echo "START $name  ($total/$expected)  pair=$pair role=$role  $(date -Is)"
    # A partial output directory from an interrupted run would otherwise be picked up as
    # a valid checkpoint by provenance.py.
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
for x in r.itertuples():
    print("\t".join([x.ligand_group, x.kinase_name, x.pair, x.role,
                     x.structure, x.ligand_sdf]))
PY
)

echo
if (( DRY_RUN )); then
    echo "dry run: $total commands printed, nothing executed."
    exit 0
fi

# Every run must agree on the settings that set the index's scale. Checking here rather
# than at analysis time means a mismatch is caught while the machine is still warm.
"$PYTHON" - "$RESULTS" "$expected" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
from cox_selectivity import COMPARABLE_SETTINGS

results = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
runs = {}
for rj in sorted(results.glob("*/run.json")):
    runs[rj.parent.name] = json.loads(rj.read_text())

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
    print(f"  {name:14} contacts={cfg.get('n_contacts'):>5}  "
          f"additive_r2={cfg.get('additive_r2')}  "
          f"{cfg.get('elapsed_seconds', 0) / 60:6.1f} min")

if disagree:
    print("\nREFUSING to call this batch comparable -- runs disagree on:")
    for key, values in disagree.items():
        print(f"  {key}: {values}")
    sys.exit(1)
print("\nall runs agree on every comparability setting; the batch can be differenced.")
PY
verify=$?

completed=$(find "$RESULTS" -mindepth 2 -maxdepth 2 -name run.json -type f 2>/dev/null | wc -l)
echo
echo "Batch finished: $completed/$total runs have a run.json" \
     "($skipped skipped as already complete, $failed failed)."
if (( failed )); then
    echo "Re-run this script to retry anything that failed; completed runs are skipped."
fi
echo "Next: dvc add results && dvc push, then the position-resolved comparison."
exit $(( failed > 0 || verify != 0 ))
