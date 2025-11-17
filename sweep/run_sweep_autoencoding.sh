#!/bin/bash
# run_sweep_autoencoding.sh
# Autoencoding task (n_patterns=1 automatically)

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for job execution
cd "$PROJECT_ROOT"

# ENABLE multi-threading for few parallel jobs
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

# Task-specific configuration
TASK_TYPE="autoencoding"
N_SESSIONS=20
NUM_PARALLEL=15

# Parameter ranges - EDIT THESE!
V_TH_VALUES=(0)      # Threshold heterogeneity
G_VALUES=(1.0)       # Weight heterogeneity
RATE_VALUES=(30)       # Static input rates
EMBED_DIM_INPUT=(1 2 3 4 5)  # Embedding dimensions (input=output for autoencoding)

STATIC_INPUT_MODE="common_tonic"
HD_INPUT_MODE="common_tonic"
SYNAPTIC_MODE="filter"
HD_CONNECTION_MODE="overlapping"  # "overlapping" or "partitioned"


N_SESSIONS=1
EMBED_DIM_INPUT=(2)  # Input embedding dimensions




# Generate jobs (n_patterns automatically set to 1 for autoencoding)
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --v_th_values ${V_TH_VALUES[@]} \
    --g_values ${G_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --embed_dim_input ${EMBED_DIM_INPUT[@]} \
    --static_input_mode "$STATIC_INPUT_MODE" \
    --hd_input_mode "$HD_INPUT_MODE" \
    --synaptic_mode "$SYNAPTIC_MODE" \
    --hd_connection_mode "$HD_CONNECTION_MODE" \
    --output "$SCRIPT_DIR/jobs_autoencoding.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

# Run sweep using the common sweep runner
"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_autoencoding.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_autoencoding"
