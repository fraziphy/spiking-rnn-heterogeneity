#!/bin/bash
# sweep/run_sweep_transient_cache.sh
# Generate cached transient states

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration
TASK_TYPE="transient_cache"
N_SESSIONS=20
NUM_PARALLEL=100
N_TRIALS=100

G_VALUES=(0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
V_TH_VALUES=(0)
RATE_VALUES=(30 31 32 33 34 35)

STATIC_INPUT_MODE="common_tonic"
SYNAPTIC_MODE="filter"

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --g_values ${G_VALUES[@]} \
    --v_th_values ${V_TH_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --n_trials $N_TRIALS \
    --static_input_mode "$STATIC_INPUT_MODE" \
    --synaptic_mode "$SYNAPTIC_MODE" \
    --output "$SCRIPT_DIR/jobs_transient.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_transient.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_transient"
