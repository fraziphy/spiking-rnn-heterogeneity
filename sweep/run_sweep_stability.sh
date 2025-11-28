#!/bin/bash
# sweep/run_sweep_stability.sh
# Network stability experiment - MODIFIED to use cached transient states

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration
TASK_TYPE="stability"
N_SESSIONS=20
NUM_PARALLEL=100

V_TH_VALUES=(0)
G_VALUES=(0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
RATE_VALUES=(30 31 32 33 34 35)

STATIC_INPUT_MODE="common_tonic"
SYNAPTIC_MODE="filter"

# NEW: Enable cached transients
USE_CACHED_TRANSIENTS="--use_cached_transients"
TRANSIENT_CACHE_DIR="results/cached_states"

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --v_th_values ${V_TH_VALUES[@]} \
    --g_values ${G_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --static_input_mode "$STATIC_INPUT_MODE" \
    --synaptic_mode "$SYNAPTIC_MODE" \
    $USE_CACHED_TRANSIENTS \
    --transient_cache_dir "$TRANSIENT_CACHE_DIR" \
    --output "$SCRIPT_DIR/jobs_stability.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_stability.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_stability"
