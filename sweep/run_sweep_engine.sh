#!/bin/bash
# sweep/run_sweep_engine.sh

set -euo pipefail

JOBS_FILE=""
TASK_TYPE=""
NUM_PARALLEL=5
LOGDIR=""
RESUME_MODE="skip_successful"

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs_file) JOBS_FILE="$2"; shift 2 ;;
        --task) TASK_TYPE="$2"; shift 2 ;;
        --num_parallel) NUM_PARALLEL="$2"; shift 2 ;;
        --logdir) LOGDIR="$2"; shift 2 ;;
        --resume-failed) RESUME_MODE="retry_failed"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$JOBS_FILE" ] || [ -z "$TASK_TYPE" ] || [ -z "$LOGDIR" ]; then
    echo "ERROR: Missing required arguments"
    exit 1
fi

if [ ! -f "$JOBS_FILE" ]; then
    echo "ERROR: Jobs file not found: $JOBS_FILE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$LOGDIR"
JOBLOG="${JOBS_FILE%.txt}_joblog.txt"
TOTAL_JOBS=$(wc -l < "$JOBS_FILE")

extract_session() {
    local cmd="$1"
    local session="?"
    if echo "$cmd" | grep -q "session-start"; then
        session=$(echo "$cmd" | grep -oP '(?<=--session-start\s)\d+' || echo "?")
    elif echo "$cmd" | grep -q "session_id"; then
        session=$(echo "$cmd" | grep -oP '(?<=--session_id\s)\d+' || echo "?")
    elif echo "$cmd" | grep -q -- "--sessions"; then
        session=$(echo "$cmd" | grep -oP '(?<=--sessions\s)\d+' || echo "?")
    fi
    echo "$session"
}

extract_params() {
    local cmd="$1"
    local params=""
    local embed_dim=$(echo "$cmd" | grep -oP '(?<=--embed-dims\s)\d+' || echo "")
    if [ -n "$embed_dim" ]; then params="k=$embed_dim"; fi
    echo "$params"
}

if [ -f "$JOBLOG" ]; then
    echo ""
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================="
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] RESUME DETECTED"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================="

    COMPLETED=$(grep "^COMPLETED" "$JOBLOG" 2>/dev/null | wc -l)
    FAILED=$(grep "^FAILED" "$JOBLOG" 2>/dev/null | wc -l)

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Previous: ✓$COMPLETED ✗$FAILED"
else
    echo "# Job log" > "$JOBLOG"
fi

echo ""
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] SWEEP: ${TASK_TYPE^^}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Jobs: $TOTAL_JOBS | Parallel: $NUM_PARALLEL"
echo ""

should_skip_job() {
    local job_num=$1
    if [ ! -f "$JOBLOG" ]; then return 1; fi
    if grep -q "^COMPLETED|$job_num|" "$JOBLOG"; then return 0; fi
    if grep -q "^FAILED|$job_num|" "$JOBLOG"; then
        if [ "$RESUME_MODE" = "retry_failed" ]; then return 1; else return 0; fi
    fi
    return 1
}

execute_job() {
    local JOB_NUM=$1
    local CMD=$2

    if should_skip_job "$JOB_NUM" "$CMD"; then return 0; fi

    local SESSION=$(extract_session "$CMD")
    local PARAMS=$(extract_params "$CMD")
    local JOB_LOGDIR="$LOGDIR/job_${JOB_NUM}"
    mkdir -p "$JOB_LOGDIR"

    local START_TIME=$(date +%s)

    if cd "$PROJECT_ROOT" && eval "$CMD" > "$JOB_LOGDIR/stdout.log" 2> "$JOB_LOGDIR/stderr.log"; then
        local DURATION=$(( $(date +%s) - START_TIME ))
        echo "COMPLETED|$JOB_NUM|$SESSION|$DURATION|now|$CMD" >> "$JOBLOG"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✓ Job $JOB_NUM/$TOTAL_JOBS | Session $SESSION | $PARAMS | $((DURATION/60))m"
    else
        local DURATION=$(( $(date +%s) - START_TIME ))
        echo "FAILED|$JOB_NUM|$SESSION|$DURATION|now|$CMD" >> "$JOBLOG"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ Job $JOB_NUM/$TOTAL_JOBS FAILED | Session $SESSION"
    fi
}

export -f execute_job should_skip_job extract_session extract_params
export JOBLOG LOGDIR PROJECT_ROOT TOTAL_JOBS RESUME_MODE

START_SWEEP=$(date +%s)
cat "$JOBS_FILE" | nl -nln | parallel -j "$NUM_PARALLEL" --colsep '\t' execute_job {1} {2}
TOTAL_DURATION=$(( $(date +%s) - START_SWEEP ))

echo ""
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] COMPLETE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Time: $((TOTAL_DURATION/60))m"

COMPLETED=$(grep "^COMPLETED" "$JOBLOG" 2>/dev/null | wc -l)
FAILED=$(grep "^FAILED" "$JOBLOG" 2>/dev/null | wc -l)

echo ""
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Results: ✓$COMPLETED ✗$FAILED / $TOTAL_JOBS"

if [ "$COMPLETED" -gt 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Timing:"
    grep "^COMPLETED" "$JOBLOG" | awk -F'|' '{print $4}' | awk '
        BEGIN {sum=0; n=0; min=999999; max=0}
        {sum+=$1; n++; if($1<min) min=$1; if($1>max) max=$1}
        END {
            if(n>0) {
                avg=sum/n;
                printf "  Mean: %ds (%dm) | Min: %ds | Max: %ds\n", avg, avg/60, min, max
            }
        }
    '
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Log: $JOBLOG"
