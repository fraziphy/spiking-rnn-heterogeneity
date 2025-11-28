#!/bin/bash
# sweep/run_sweep_categorical.sh
# Categorical classification task - MODIFIED to use cached spikes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

# Configuration
TASK_TYPE="categorical"
N_PATTERNS=4
N_SESSIONS=20
NUM_PARALLEL=3

V_TH_VALUES=(0)
G_VALUES=(1.0)
RATE_VALUES=(30)
EMBED_DIM_INPUT=(1 2 3 4 5 6 7)

STATIC_INPUT_MODE="common_tonic"
HD_INPUT_MODE="common_tonic"
SYNAPTIC_MODE="filter"
HD_CONNECTION_MODE="overlapping"

# NEW: Enable cached spikes
USE_CACHED_SPIKES="--use_cached_spikes"
SPIKE_CACHE_DIR="results/cached_spikes"

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --n_patterns $N_PATTERNS \
    --sessions $N_SESSIONS \
    --v_th_values ${V_TH_VALUES[@]} \
    --g_values ${G_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --embed_dim_input ${EMBED_DIM_INPUT[@]} \
    --static_input_mode "$STATIC_INPUT_MODE" \
    --hd_input_mode "$HD_INPUT_MODE" \
    --synaptic_mode "$SYNAPTIC_MODE" \
    --hd_connection_mode "$HD_CONNECTION_MODE" \
    $USE_CACHED_SPIKES \
    --spike_cache_dir "$SPIKE_CACHE_DIR" \
    --output "$SCRIPT_DIR/jobs_categorical.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_categorical.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_categorical"
