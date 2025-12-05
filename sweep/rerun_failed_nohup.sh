#!/bin/bash
# sweep/rerun_failed_nohup.sh
# Non-interactive version for nohup usage
#
# Usage:
#   nohup ./sweep/rerun_failed_nohup.sh --task autoencoding > rerun.log 2>&1 & disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
TASK_TYPE=""
NUM_PARALLEL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK_TYPE="$2"; shift 2 ;;
        --num_parallel) NUM_PARALLEL_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TASK_TYPE" ]; then
    echo "ERROR: --task is required"
    exit 1
fi

# Map task type to sweep script
SWEEP_SCRIPT="$SCRIPT_DIR/run_sweep_${TASK_TYPE}.sh"

if [ ! -f "$SWEEP_SCRIPT" ]; then
    echo "ERROR: Sweep script not found: $SWEEP_SCRIPT"
    exit 1
fi

# Check if joblog exists
JOBS_FILE="$SCRIPT_DIR/jobs_${TASK_TYPE}.txt"
JOBLOG="${JOBS_FILE%.txt}_joblog.txt"

if [ ! -f "$JOBLOG" ]; then
    echo "ERROR: Job log not found: $JOBLOG"
    exit 1
fi

if [ ! -f "$JOBS_FILE" ]; then
    echo "ERROR: Jobs file not found: $JOBS_FILE"
    exit 1
fi

# Count failed jobs
FAILED_COUNT=$(grep -c "^FAILED|" "$JOBLOG" 2>/dev/null || echo "0")
COMPLETED_COUNT=$(grep -c "^COMPLETED|" "$JOBLOG" 2>/dev/null || echo "0")
TOTAL_JOBS=$(wc -l < "$JOBS_FILE")

echo "========================================"
echo "RERUN FAILED JOBS: ${TASK_TYPE^^}"
echo "========================================"
echo "Job log: $JOBLOG"
echo "Status: ✓$COMPLETED_COUNT ✗$FAILED_COUNT / $TOTAL_JOBS"
echo ""

if [ "$FAILED_COUNT" -eq 0 ]; then
    echo "No failed jobs found! All jobs completed successfully."
    exit 0
fi

# Extract config from sweep script
OMP_THREADS=$(grep -oP '(?<=export OMP_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
OPENBLAS_THREADS=$(grep -oP '(?<=export OPENBLAS_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
MKL_THREADS=$(grep -oP '(?<=export MKL_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
NUMEXPR_THREADS=$(grep -oP '(?<=export NUMEXPR_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
NUM_PARALLEL=$(grep -oP '(?<=NUM_PARALLEL=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "4")

if [ -n "$NUM_PARALLEL_OVERRIDE" ]; then
    NUM_PARALLEL="$NUM_PARALLEL_OVERRIDE"
fi

echo "Config: NUM_PARALLEL=$NUM_PARALLEL, OMP=$OMP_THREADS"
echo "Re-running $FAILED_COUNT failed jobs..."
echo ""

# Set environment variables
export OMP_NUM_THREADS="$OMP_THREADS"
export OPENBLAS_NUM_THREADS="$OPENBLAS_THREADS"
export MKL_NUM_THREADS="$MKL_THREADS"
export NUMEXPR_NUM_THREADS="$NUMEXPR_THREADS"

# Run the sweep engine with --resume-failed
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$JOBS_FILE" \
    --task "$TASK_TYPE" \
    --num_parallel "$NUM_PARALLEL" \
    --logdir "$SCRIPT_DIR/logs_${TASK_TYPE}" \
    --resume-failed

# Clean up joblog
echo ""
echo "Cleaning up job log..."

TEMP_JOBLOG="${JOBLOG}.tmp"
> "$TEMP_JOBLOG"

COMPLETED_JOBS=$(grep "^COMPLETED|" "$JOBLOG" | cut -d'|' -f2 | sort -u)

while IFS= read -r line; do
    JOB_NUM=$(echo "$line" | cut -d'|' -f2)
    STATUS=$(echo "$line" | cut -d'|' -f1)
    
    if [ "$STATUS" = "FAILED" ]; then
        if echo "$COMPLETED_JOBS" | grep -q "^${JOB_NUM}$"; then
            continue
        fi
    fi
    
    if ! grep -q "|${JOB_NUM}|" "$TEMP_JOBLOG" 2>/dev/null; then
        echo "$line" >> "$TEMP_JOBLOG"
    fi
done < "$JOBLOG"

mv "$TEMP_JOBLOG" "$JOBLOG"

FINAL_COMPLETED=$(grep -c "^COMPLETED|" "$JOBLOG" 2>/dev/null || echo "0")
FINAL_FAILED=$(grep -c "^FAILED|" "$JOBLOG" 2>/dev/null || echo "0")

echo ""
echo "========================================"
echo "RERUN COMPLETE"
echo "========================================"
echo "Final status: ✓$FINAL_COMPLETED ✗$FINAL_FAILED / $TOTAL_JOBS"
