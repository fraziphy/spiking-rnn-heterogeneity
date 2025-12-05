#!/bin/bash
# sweep/rerun_failed.sh
# Re-run ONLY failed jobs for any task type
#
# Usage:
#   ./sweep/rerun_failed.sh --task autoencoding
#   ./sweep/rerun_failed.sh --task stability
#   ./sweep/rerun_failed.sh --task categorical
#   ./sweep/rerun_failed.sh --task temporal
#
# Optional overrides:
#   ./sweep/rerun_failed.sh --task autoencoding --num_parallel 8

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
        -h|--help)
            echo "Usage: $0 --task <task_type> [--num_parallel <n>]"
            echo ""
            echo "Task types: autoencoding, categorical, temporal, stability"
            echo ""
            echo "Examples:"
            echo "  $0 --task autoencoding"
            echo "  $0 --task stability --num_parallel 8"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TASK_TYPE" ]; then
    echo "ERROR: --task is required"
    echo "Usage: $0 --task <task_type>"
    echo "Task types: autoencoding, categorical, temporal, stability"
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
    echo "You need to run the initial sweep first."
    exit 1
fi

if [ ! -f "$JOBS_FILE" ]; then
    echo "ERROR: Jobs file not found: $JOBS_FILE"
    echo "The original jobs file is needed for re-running."
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

# Show which jobs failed
echo "Failed jobs:"
grep "^FAILED|" "$JOBLOG" | head -10 | while IFS='|' read -r status job_num session duration timestamp cmd; do
    echo "  Job $job_num (Session $session)"
done

if [ "$FAILED_COUNT" -gt 10 ]; then
    echo "  ... and $((FAILED_COUNT - 10)) more"
fi
echo ""

# Extract config from sweep script
echo "Reading config from: $SWEEP_SCRIPT"

# Extract OMP_NUM_THREADS
OMP_THREADS=$(grep -oP '(?<=export OMP_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
OPENBLAS_THREADS=$(grep -oP '(?<=export OPENBLAS_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
MKL_THREADS=$(grep -oP '(?<=export MKL_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")
NUMEXPR_THREADS=$(grep -oP '(?<=export NUMEXPR_NUM_THREADS=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "1")

# Extract NUM_PARALLEL
NUM_PARALLEL=$(grep -oP '(?<=NUM_PARALLEL=)\d+' "$SWEEP_SCRIPT" | head -1 || echo "4")

# Apply override if provided
if [ -n "$NUM_PARALLEL_OVERRIDE" ]; then
    NUM_PARALLEL="$NUM_PARALLEL_OVERRIDE"
    echo "  NUM_PARALLEL: $NUM_PARALLEL (override)"
else
    echo "  NUM_PARALLEL: $NUM_PARALLEL"
fi

echo "  OMP_NUM_THREADS: $OMP_THREADS"
echo ""

# Set environment variables
export OMP_NUM_THREADS="$OMP_THREADS"
export OPENBLAS_NUM_THREADS="$OPENBLAS_THREADS"
export MKL_NUM_THREADS="$MKL_THREADS"
export NUMEXPR_NUM_THREADS="$NUMEXPR_THREADS"

# Confirm before running
echo "Will re-run $FAILED_COUNT failed jobs with $NUM_PARALLEL parallel workers."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting re-run..."
echo ""

# Run the sweep engine with --resume-failed
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$JOBS_FILE" \
    --task "$TASK_TYPE" \
    --num_parallel "$NUM_PARALLEL" \
    --logdir "$SCRIPT_DIR/logs_${TASK_TYPE}" \
    --resume-failed

# Summary
echo ""
echo "========================================"
echo "RERUN COMPLETE"
echo "========================================"

NEW_FAILED=$(grep -c "^FAILED|" "$JOBLOG" 2>/dev/null || echo "0")
NEW_COMPLETED=$(grep -c "^COMPLETED|" "$JOBLOG" 2>/dev/null || echo "0")

# Note: joblog may have duplicate entries (FAILED then COMPLETED for same job)
# Count unique completed jobs
UNIQUE_COMPLETED=$(grep "^COMPLETED|" "$JOBLOG" | cut -d'|' -f2 | sort -u | wc -l)
UNIQUE_FAILED=$(grep "^FAILED|" "$JOBLOG" | cut -d'|' -f2 | sort -u | wc -l)
# Jobs that are both FAILED and COMPLETED = now fixed
STILL_FAILED=$((UNIQUE_FAILED - (NEW_COMPLETED - COMPLETED_COUNT) ))

echo "Results:"
echo "  Previously failed: $FAILED_COUNT"
echo "  Now completed: $((NEW_COMPLETED - COMPLETED_COUNT))"
echo "  Still failing: $STILL_FAILED (approx)"
echo ""

# Clean up joblog - remove FAILED entries for jobs that now have COMPLETED
echo "Cleaning up job log..."

TEMP_JOBLOG="${JOBLOG}.tmp"
> "$TEMP_JOBLOG"

# Get list of completed job numbers
COMPLETED_JOBS=$(grep "^COMPLETED|" "$JOBLOG" | cut -d'|' -f2 | sort -u)

# Process joblog: keep only latest entry per job, prefer COMPLETED
while IFS= read -r line; do
    JOB_NUM=$(echo "$line" | cut -d'|' -f2)
    STATUS=$(echo "$line" | cut -d'|' -f1)

    # If this is a FAILED entry, check if job is now completed
    if [ "$STATUS" = "FAILED" ]; then
        if echo "$COMPLETED_JOBS" | grep -q "^${JOB_NUM}$"; then
            # Skip this FAILED entry - job is now completed
            continue
        fi
    fi

    # Check if we already have this job in temp file
    if ! grep -q "|${JOB_NUM}|" "$TEMP_JOBLOG" 2>/dev/null; then
        echo "$line" >> "$TEMP_JOBLOG"
    fi
done < "$JOBLOG"

# Replace original with cleaned version
mv "$TEMP_JOBLOG" "$JOBLOG"

FINAL_COMPLETED=$(grep -c "^COMPLETED|" "$JOBLOG" 2>/dev/null || echo "0")
FINAL_FAILED=$(grep -c "^FAILED|" "$JOBLOG" 2>/dev/null || echo "0")

echo "Final status: ✓$FINAL_COMPLETED ✗$FINAL_FAILED / $TOTAL_JOBS"
echo ""
echo "Job log cleaned: $JOBLOG"
