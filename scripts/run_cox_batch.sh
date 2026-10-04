#!/usr/bin/env bash
# Decoy generation for the COX-1/COX-2 selectivity series.
#
# Each of the 49 inhibitors was docked separately into 1EQG (COX-1) and 3LN1
# (COX-2), so the *chemistry* is shared across the two targets but the *pose*
# is not -- data/docking/ligands/{cox1,cox2}/<ligand>_withH.sdf hold the same
# molecule at target-specific coordinates. Both PDBs are protein-only (no
# HETATM ligand), so every run uses --ligand-placed: the SDF supplies both the
# chemistry and the docked coordinates, and frustx appends it to the pose.
#
# Hydrogens were added beforehand with (Open Babel 3.1.0, env `frustrato`):
#     obabel <lig>.sdf -osdf -O <lig>_withH.sdf -h -p 7.4
# NOTE: with both flags set Open Babel applies -h and ignores -p, so these
# ligands are NEUTRAL (carboxylic acids stay protonated). This was a deliberate
# choice; re-running with -p 7.4 alone would give anionic NSAIDs and different
# numbers.
#
#   ./scripts/run_cox_batch.sh --dry-run   # print the commands, run nothing
#   DECOYS=200 ./scripts/run_cox_batch.sh  # quick smoke run
#
# Re-running skips runs that already finished at the current DECOYS count, so
# the batch is resumable after an interrupt.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRUSTX="$ROOT/.venv/bin/frustx"
DOCKING="$ROOT/data/docking"
LIGANDS="$DOCKING/ligands"
RESULTS="${RESULTS:-$DOCKING/cox_results/decoy_results}"
LOGS="$RESULTS/logs"

DECOYS="${DECOYS:-1000}"
JOBS="${JOBS:-50}"
# Wall-clock ceiling per run. A 1000-decoy job at --jobs 50 lands well inside
# this; the timeout is here to stop one pathological PDB-SDF atom mapping from
# stalling the whole 98-run batch.
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

# target key -> protein structure
declare -A PDB=(
    [cox1]="$DOCKING/cox_results/1EQG.pdb"
    [cox2]="$DOCKING/cox_results/3LN1.pdb"
)

if [[ ! -x "$FRUSTX" ]]; then
    echo "Missing executable: $FRUSTX" >&2
    exit 2
fi

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LOGS"

# The two ligand directories do not agree on filename case/punctuation
# (cox1/GSK-644784.sdf vs cox2/gsk-644784-.sdf are the same compound), so the
# output directory is keyed on a normalised name. That keeps the cox1 and cox2
# run of one inhibitor named identically apart from the target prefix, which is
# what the downstream selectivity comparison pairs on.
# extglob must be enabled before this function is *parsed*, not just called.
shopt -s extglob
normalise() {
    local n="${1,,}"      # lowercase
    n="${n%%+(-)}"        # strip trailing dashes (see extglob below)
    echo "$n"
}


total=0
failed=0
skipped=0

for target in cox1 cox2; do
    structure="${PDB[$target]}"
    if [[ ! -f "$structure" ]]; then
        echo "MISSING STRUCTURE: $structure" | tee -a "$LOGS/batch_errors.log" >&2
        continue
    fi

    for sdf in "$LIGANDS/$target"/*_withH.sdf; do
        [[ -f "$sdf" ]] || continue
        base="$(basename "$sdf" _withH.sdf)"
        ligand="$(normalise "$base")"

        out="$RESULTS/${target}_${ligand}"
        log="$LOGS/${target}_${ligand}.log"
        total=$((total + 1))

        # Resume: a run.json recording the current decoy count means this
        # protein-ligand pair is already done at these settings.
        if [[ -f "$out/run.json" ]] && grep -q "\"n_decoys\": *$DECOYS\b" "$out/run.json"; then
            echo "SKIP completed: ${target}_${ligand}"
            skipped=$((skipped + 1))
            continue
        fi

        cmd=("$FRUSTX" "$structure"
             --ligand-placed "$sdf"
             --ligand-name UNL --ligand-name3 UNL
             -n "$DECOYS"
             -o "$out"
             --jobs "$JOBS")

        if (( DRY_RUN )); then
            printf '%q ' "${cmd[@]}"; printf '\n'
            continue
        fi

        echo "START ${target}_${ligand}  ($((total)))  decoys=$DECOYS jobs=$JOBS  $(date -Is)"
        # A partial output directory from an interrupted run would otherwise be
        # reused as if it were a valid checkpoint.
        rm -rf "$out"
        if timeout --signal=TERM --kill-after=30 "$RUN_TIMEOUT" "${cmd[@]}" > "$log" 2>&1; then
            echo "DONE  ${target}_${ligand}  $(date -Is)"
        else
            status=$?
            if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
                msg="TIMEOUT ${target}_${ligand} after ${RUN_TIMEOUT}s; see $log"
            else
                msg="FAILED  ${target}_${ligand} (exit=$status); see $log"
            fi
            echo "$msg" | tee -a "$LOGS/batch_errors.log" >&2
            failed=$((failed + 1))
        fi
    done
done

completed=$(find "$RESULTS" -mindepth 2 -maxdepth 2 -name run.json -type f 2>/dev/null | wc -l)
echo
echo "Batch finished: $completed/$total runs have a run.json ($skipped skipped as already complete, $failed failed)."
echo "Re-run this script to retry anything that failed; completed runs are skipped."
