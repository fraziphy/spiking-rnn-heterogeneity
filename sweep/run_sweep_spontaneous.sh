#!/bin/bash
# run_sweep_spontaneous.sh
# Spontaneous activity experiment sweep
# Duration: 800ms total (500ms transient + 300ms analysis)

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for job execution
cd "$PROJECT_ROOT"

# Task-specific configuration
TASK_TYPE="spontaneous"
N_SESSIONS=20
NUM_PARALLEL=3  # Spontaneous is fast, can run many in parallel

# Parameter ranges - EDIT THESE!
V_TH_VALUES=(0)      # Threshold heterogeneity
G_VALUES=(0.5 0.75 1.0 1.25 1.5)           # Weight heterogeneity
RATE_VALUES=(28 29 30 31 32)     # Static input rates

# Duration settings
DURATION=800.0  # Total duration in ms (500ms transient + 300ms analysis)

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --v_th_values ${V_TH_VALUES[@]} \
    --g_values ${G_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --duration $DURATION \
    --output "$SCRIPT_DIR/jobs_spontaneous.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

# Run sweep using the common sweep runner
"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_spontaneous.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_spontaneous"
