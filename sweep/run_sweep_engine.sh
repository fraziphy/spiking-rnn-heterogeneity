#!/bin/bash
# run_sweep_engine.sh
# Common sweep execution engine with RESUME support for system reboots

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs_file) JOBS_FILE="$2"; shift 2 ;;
        --task) TASK_TYPE="$2"; shift 2 ;;
        --num_parallel) NUM_PARALLEL="$2"; shift 2 ;;
        --logdir) LOGDIR="$2"; shift 2 ;;
        --resume) RESUME_MODE="$2"; shift 2 ;;  # NEW: resume|resume-failed|none
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate arguments
if [ -z "$JOBS_FILE" ] || [ -z "$TASK_TYPE" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 --jobs_file FILE --task TYPE --num_parallel N --logdir DIR [--resume MODE]"
    exit 1
fi

# Defaults
NUM_PARALLEL=${NUM_PARALLEL:-10}
LOGDIR=${LOGDIR:-logs}
RESUME_MODE=${RESUME_MODE:-resume-failed}  # Default: retry failed jobs

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_section() {
    echo ""
    log_message "========================================="
    log_message "$1"
    log_message "========================================="
}

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "$LOGDIR"

# Check jobs file exists
if [ ! -f "$JOBS_FILE" ]; then
    log_message "ERROR: Jobs file not found: $JOBS_FILE"
    exit 1
fi

# Count total jobs
TOTAL_JOBS=$(wc -l < "$JOBS_FILE")

# ============================================================================
# RESUME DETECTION
# ============================================================================

JOBLOG_FILE="${JOBS_FILE%.txt}_joblog.txt"

# Check if this is a resume
IS_RESUME=false
if [ -f "$JOBLOG_FILE" ]; then
    COMPLETED_JOBS=$(awk 'NR>1 && $7 == 0' "$JOBLOG_FILE" | wc -l)
    FAILED_JOBS=$(awk 'NR>1 && $7 != 0' "$JOBLOG_FILE" | wc -l)
    TOTAL_IN_LOG=$((COMPLETED_JOBS + FAILED_JOBS))

    if [ $TOTAL_IN_LOG -gt 0 ]; then
        IS_RESUME=true
        REMAINING_JOBS=$((TOTAL_JOBS - COMPLETED_JOBS))

        log_section "RESUME DETECTED"
        log_message "Previous progress found in joblog:"
        log_message "  ✓ Completed: $COMPLETED_JOBS jobs"
        log_message "  ✗ Failed: $FAILED_JOBS jobs"
        log_message "  → Remaining: $REMAINING_JOBS jobs"

        if [ "$RESUME_MODE" = "resume-failed" ]; then
            log_message ""
            log_message "Resume mode: RETRY FAILED + RUN REMAINING"
            log_message "  - Will skip $COMPLETED_JOBS successful jobs"
            log_message "  - Will retry $FAILED_JOBS failed jobs"
            log_message "  - Will run $((REMAINING_JOBS - FAILED_JOBS)) new jobs"
        elif [ "$RESUME_MODE" = "resume" ]; then
            log_message ""
            log_message "Resume mode: RUN REMAINING ONLY"
            log_message "  - Will skip ALL $TOTAL_IN_LOG jobs in joblog"
            log_message "  - Will only run $((TOTAL_JOBS - TOTAL_IN_LOG)) new jobs"
        else
            log_message ""
            log_message "Resume mode: FRESH START"
            log_message "  - Backing up old joblog"
            mv "$JOBLOG_FILE" "${JOBLOG_FILE}.backup.$(date +%s)"
            IS_RESUME=false
        fi
    fi
fi

# ============================================================================
# EXTRACT PARAMETERS
# ============================================================================

FIRST_JOB=$(head -1 "$JOBS_FILE")
N_NEURONS=$(echo "$FIRST_JOB" | grep -oP '(?<=--n_neurons )\d+' || echo "unknown")
N_TRIALS=$(echo "$FIRST_JOB" | grep -oP '(?<=--n_trials_per_pattern )\d+' || echo "unknown")
N_PATTERNS=$(echo "$FIRST_JOB" | grep -oP '(?<=--n_input_patterns )\d+' || echo "unknown")

FIRST_SESSION=$(echo "$FIRST_JOB" | grep -oP '(?<=--session_id )\d+' || echo "0")
LAST_JOB=$(tail -1 "$JOBS_FILE")
LAST_SESSION=$(echo "$LAST_JOB" | grep -oP '(?<=--session_id )\d+' || echo "0")
N_SESSIONS=$((LAST_SESSION - FIRST_SESSION + 1))

if [ $N_SESSIONS -gt 0 ]; then
    JOBS_PER_SESSION=$((TOTAL_JOBS / N_SESSIONS))
else
    JOBS_PER_SESSION=$TOTAL_JOBS
fi

# ============================================================================
# START BANNER
# ============================================================================

if [ "$IS_RESUME" = false ]; then
    log_section "PARAMETER SWEEP - ${TASK_TYPE^^} TASK"
else
    log_section "RESUMING SWEEP - ${TASK_TYPE^^} TASK"
fi

log_message "Configuration:"
log_message "  Task type: ${TASK_TYPE}"
log_message "  Sessions: ${FIRST_SESSION} to ${LAST_SESSION} (${N_SESSIONS} sessions)"
log_message "  Total jobs: ${TOTAL_JOBS}"
log_message "  Jobs per session: ${JOBS_PER_SESSION}"
log_message "  Parallel jobs: ${NUM_PARALLEL}"
log_message "  Network: ${N_NEURONS} neurons"
log_message "  Trials: ${N_TRIALS} per pattern × ${N_PATTERNS} patterns"
log_message "  Log directory: ${LOGDIR}/"

if [ "$IS_RESUME" = true ]; then
    log_message ""
    log_message "Resume Status:"
    log_message "  Already completed: ${COMPLETED_JOBS}/${TOTAL_JOBS}"
    log_message "  Remaining work: ${REMAINING_JOBS} jobs"
    log_message "  Progress: $(echo "scale=1; 100*$COMPLETED_JOBS/$TOTAL_JOBS" | bc)%"
fi

log_section "EXECUTION START"
log_message "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

OVERALL_START=$(date +%s)

# ============================================================================
# PROGRESS MONITOR (background) - Milestone-based reporting
# ============================================================================

monitor_progress() {
    local start_time=$1
    local total=$2
    local last_milestone=0

    while true; do
        sleep 30  # Check frequently but only report at milestones

        if [ -f "$JOBLOG_FILE" ]; then
            local completed=$(awk 'NR>1 && $7 == 0' "$JOBLOG_FILE" | wc -l)
            local failed=$(awk 'NR>1 && $7 != 0' "$JOBLOG_FILE" | wc -l)

            if [ $completed -gt 0 ]; then
                local progress=$(echo "scale=0; 100*$completed/$total" | bc 2>/dev/null || echo "0")

                # Report at 25%, 50%, 75%, 100% milestones
                if [ $progress -ge 25 ] && [ $last_milestone -lt 25 ]; then
                    report_milestone $start_time $completed $failed $total $progress 25
                    last_milestone=25
                elif [ $progress -ge 50 ] && [ $last_milestone -lt 50 ]; then
                    report_milestone $start_time $completed $failed $total $progress 50
                    last_milestone=50
                elif [ $progress -ge 75 ] && [ $last_milestone -lt 75 ]; then
                    report_milestone $start_time $completed $failed $total $progress 75
                    last_milestone=75
                elif [ $progress -ge 100 ] && [ $last_milestone -lt 100 ]; then
                    report_milestone $start_time $completed $failed $total $progress 100
                    last_milestone=100
                fi
            fi
        fi

        [ ! -f /tmp/sweep_running_$$ ] && break
    done
}

report_milestone() {
    local start_time=$1
    local completed=$2
    local failed=$3
    local total=$4
    local progress=$5
    local milestone=$6

    local elapsed=$(($(date +%s) - start_time))
    local rate=$(echo "scale=2; $completed / $elapsed" | bc 2>/dev/null || echo "0")

    if [ "$rate" != "0" ] && [ -n "$rate" ]; then
        local remaining=$((total - completed))
        local eta=$(echo "scale=0; $remaining / $rate" | bc 2>/dev/null || echo "0")
        local eta_min=$((eta / 60))
        local eta_hr=$((eta_min / 60))

        local sessions_done=$((completed / JOBS_PER_SESSION))

        log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_message "🎯 MILESTONE: ${milestone}% COMPLETE"
        log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_message "Progress: ${completed}/${total} | ✓${completed} ✗${failed}"
        log_message "Sessions: ${sessions_done}/${N_SESSIONS}"
        log_message "ETA: ${eta_hr}h $((eta_min % 60))m"
        log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
}

touch /tmp/sweep_running_$$
monitor_progress $OVERALL_START $TOTAL_JOBS &
MONITOR_PID=$!

# ============================================================================
# RUN SWEEP WITH RESUME SUPPORT
# ============================================================================

run_and_log_job() {
    local job_num=$1
    shift
    local cmd="$@"
    local start_time=$(date +%s)

    local session=$(echo "$cmd" | grep -oP '(?<=--session_id )\d+' || echo "?")
    local hd_in=$(echo "$cmd" | grep -oP '(?<=--input_hd_dim )\d+' || echo "?")
    local hd_out=$(echo "$cmd" | grep -oP '(?<=--output_hd_dim )\d+' || echo "?")
    local embed_in=$(echo "$cmd" | grep -oP '(?<=--embed_dim_input )\d+' || echo "?")
    local embed_out=$(echo "$cmd" | grep -oP '(?<=--embed_dim_output )\d+' || echo "?")
    local v_th=$(echo "$cmd" | grep -oP '(?<=--v_th_std )[0-9.]+' || echo "?")
    local g=$(echo "$cmd" | grep -oP '(?<=--g_std )[0-9.]+' || echo "?")
    local rate=$(echo "$cmd" | grep -oP '(?<=--static_input_rate )[0-9.]+' || echo "?")
    local hd_mode=$(echo "$cmd" | grep -oP '(?<=--hd_connection_mode )\w+' || echo "?")

    # Redirect output to clean log files
    local job_log_dir="${LOGDIR}/job_${job_num}"
    mkdir -p "$job_log_dir"

    eval "$cmd" > "${job_log_dir}/stdout.log" 2> "${job_log_dir}/stderr.log"
    local exit_code=$?

    local duration=$(($(date +%s) - start_time))
    local duration_min=$((duration / 60))

    if [ $exit_code -eq 0 ]; then
        # Build param string based on what's available
        local params="Session ${session}"
        if [ "$embed_in" != "?" ] && [ "$embed_out" != "?" ]; then
            params="${params} | embed_in=${embed_in} embed_out=${embed_out} hd_in=${hd_in} hd_out=${hd_out}"
        elif [ "$embed_in" != "?" ]; then
            params="${params} | embed=${embed_in} hd=${hd_in}"
        fi
        if [ "$v_th" != "?" ]; then
            params="${params} | v_th=${v_th} g=${g} rate=${rate}"
        fi
        if [ "$hd_mode" != "?" ]; then
            params="${params} | mode=${hd_mode}"
        fi
        log_message "✓ Job ${job_num}/${TOTAL_JOBS} done | ${params} | ${duration_min}m"
    else
        log_message "✗ Job ${job_num}/${TOTAL_JOBS} FAILED (code ${exit_code}) | Session ${session} | ${duration_min}m"
    fi

    return $exit_code
}

export -f run_and_log_job log_message
export TOTAL_JOBS LOGDIR

# Build parallel command with appropriate resume flag
PARALLEL_CMD="parallel --jobs $NUM_PARALLEL --joblog \"$JOBLOG_FILE\" --line-buffer"

if [ "$IS_RESUME" = true ]; then
    if [ "$RESUME_MODE" = "resume-failed" ]; then
        PARALLEL_CMD="$PARALLEL_CMD --resume-failed"
        log_message ""
        log_message "Using --resume-failed: Will retry failed jobs and run new jobs"
    elif [ "$RESUME_MODE" = "resume" ]; then
        PARALLEL_CMD="$PARALLEL_CMD --resume"
        log_message ""
        log_message "Using --resume: Will skip all logged jobs, only run new ones"
    fi
fi

PARALLEL_CMD="$PARALLEL_CMD \"run_and_log_job {#} {}\" :::: \"$JOBS_FILE\""

# Execute
eval $PARALLEL_CMD
EXIT_CODE=$?

rm -f /tmp/sweep_running_$$
kill $MONITOR_PID 2>/dev/null || true

# ============================================================================
# SUMMARY
# ============================================================================

TOTAL_DURATION=$(($(date +%s) - OVERALL_START))
DURATION_HR=$((TOTAL_DURATION / 3600))
DURATION_MIN=$(((TOTAL_DURATION % 3600) / 60))

log_section "SWEEP COMPLETE"
log_message "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
log_message "Total duration: ${TOTAL_DURATION}s (${DURATION_HR}h ${DURATION_MIN}m)"

if [ -f "$JOBLOG_FILE" ]; then
    echo ""
    log_message "Job Statistics:"
    log_message "---------------"

    TOTAL_RUN=$(awk 'NR>1' "$JOBLOG_FILE" | wc -l)
    SUCCESS=$(awk 'NR>1 && $7 == 0' "$JOBLOG_FILE" | wc -l)
    FAILED=$(awk 'NR>1 && $7 != 0' "$JOBLOG_FILE" | wc -l)

    log_message "  Total jobs: $TOTAL_RUN"
    log_message "  ✓ Successful: $SUCCESS"
    log_message "  ✗ Failed: $FAILED"

    if [ $SUCCESS -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=1; 100 * $SUCCESS / $TOTAL_RUN" | bc 2>/dev/null || echo "N/A")
        log_message "  Success rate: ${SUCCESS_RATE}%"

        SESSIONS_COMPLETE=$((SUCCESS / JOBS_PER_SESSION))
        SESSIONS_PARTIAL=$((SUCCESS % JOBS_PER_SESSION))

        if [ $SESSIONS_PARTIAL -eq 0 ]; then
            log_message "  Sessions completed: ${SESSIONS_COMPLETE}/${N_SESSIONS}"
        else
            log_message "  Sessions completed: ${SESSIONS_COMPLETE}/${N_SESSIONS} (+ ${SESSIONS_PARTIAL} jobs from next)"
        fi
    fi

    echo ""
    log_message "Timing Statistics:"
    log_message "------------------"
    awk 'NR>1 && $7 == 0 {print $4}' "$JOBLOG_FILE" | \
        awk '{sum+=$1; if(NR==1){min=max=$1} if($1<min){min=$1} if($1>max){max=$1}}
             END {if (NR > 0) print "  Mean: " int(sum/NR) "s (" int(sum/NR/60) "m) | Min: " int(min) "s | Max: " int(max) "s"}'

    if [ $FAILED -gt 0 ]; then
        echo ""
        log_message "⚠️  FAILURES DETECTED"
        log_message "Failed Jobs:"
        awk 'NR>1 && $7 != 0 {print "  Job " $1 ": exit code " $7}' "$JOBLOG_FILE" | head -10
        [ $FAILED -gt 10 ] && log_message "  ... and $((FAILED - 10)) more. Check: ${LOGDIR}/"

        echo ""
        log_message "To retry failed jobs only:"
        log_message "  ./run_sweep_engine.sh --jobs_file $JOBS_FILE --task $TASK_TYPE \\"
        log_message "    --num_parallel $NUM_PARALLEL --logdir $LOGDIR --resume resume-failed"
    fi
fi

echo ""
log_message "Results:"
RESULT_DIR="results/${TASK_TYPE}_sweep/data"
if [ -d "$RESULT_DIR" ]; then
    RESULT_COUNT=$(ls -1 "${RESULT_DIR}"/*.pkl 2>/dev/null | wc -l)
    log_message "  ✓ ${RESULT_COUNT} result files in ${RESULT_DIR}/"
else
    log_message "  ! Results directory not found"
fi
log_message "  Job log: $JOBLOG_FILE"

echo ""
if [ $FAILED -eq 0 ] && [ $SUCCESS -eq $TOTAL_JOBS ]; then
    log_section "✓ ALL JOBS COMPLETED SUCCESSFULLY"
    exit 0
elif [ $SUCCESS -gt 0 ]; then
    log_section "⚠ PARTIAL SUCCESS: ${SUCCESS}/${TOTAL_JOBS}"
    if [ $FAILED -gt 0 ]; then
        log_message ""
        log_message "NEXT STEPS:"
        log_message "1. Check failed job logs in ${LOGDIR}/"
        log_message "2. Rerun with: --resume resume-failed"
    fi
    exit 2
else
    log_section "✗ ALL JOBS FAILED"
    exit 1
fi
