#!/bin/bash
# sweep/run_sweep_generate_hd_signals.sh
# Generate all HD input and output signals using sweep engine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration
TASK_TYPE="hd_signals"
N_SESSIONS=20
NUM_PARALLEL=50

EMBED_DIMS=(1 2 3 4 5 6 7)
N_PATTERNS=4
SIGNAL_CACHE_DIR="results/hd_signals"

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --embed_dims ${EMBED_DIMS[@]} \
    --n_patterns $N_PATTERNS \
    --signal_cache_dir "$SIGNAL_CACHE_DIR" \
    --output "$SCRIPT_DIR/jobs_hd_signals.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_hd_signals.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_hd_signals"
