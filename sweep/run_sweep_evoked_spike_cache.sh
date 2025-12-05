#!/bin/bash
# sweep/run_sweep_evoked_spike_cache.sh
# Generate cached evoked spikes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration
TASK_TYPE="evoked_spike_cache"
N_SESSIONS=20
NUM_PARALLEL=80

V_TH_VALUES=(0)
G_VALUES=(1.0)
RATE_VALUES=(30)
EMBED_DIMS=(1 2 3 4 5 6 7)
PATTERN_IDS=(0 1 2 3)
HD_CONNECTION_MODES=("overlapping" "partitioned")
HD_CONNECTION_MODES=("overlapping")
SIGNAL_TYPE="hd_input"

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --g_values ${G_VALUES[@]} \
    --v_th_values ${V_TH_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --embed_dims ${EMBED_DIMS[@]} \
    --pattern_ids ${PATTERN_IDS[@]} \
    --hd_connection_modes ${HD_CONNECTION_MODES[@]} \
    --signal_type "$SIGNAL_TYPE" \
    --output "$SCRIPT_DIR/jobs_evoked.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_evoked.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_evoked"
