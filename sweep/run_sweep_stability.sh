#!/bin/bash
# run_sweep_stability.sh
# Network stability experiment sweep with perturbation analysis
# Duration: 800ms total (500ms pre-perturbation + 300ms post-perturbation)

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for job execution
cd "$PROJECT_ROOT"

# KEEP single-threaded for many parallel jobs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Task-specific configuration
TASK_TYPE="stability"
N_SESSIONS=20
NUM_PARALLEL=100  # Stability runs 100 perturbations, so moderate parallelism

# Parameter ranges - EDIT THESE!
V_TH_VALUES=(0)      # Threshold heterogeneity
G_VALUES=(0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)           # Weight heterogeneity
RATE_VALUES=(30 31 32 33 34 35)     # Static input rates

STATIC_INPUT_MODE="common_tonic"
SYNAPTIC_MODE="filter"

# Note: Duration is hardcoded in stability_experiment.py
# Total: 800ms (500ms pre-perturbation + 300ms post-perturbation)

# Generate jobs
python3 "$SCRIPT_DIR/generate_jobs.py" \
    --task "$TASK_TYPE" \
    --sessions $N_SESSIONS \
    --v_th_values ${V_TH_VALUES[@]} \
    --g_values ${G_VALUES[@]} \
    --rate_values ${RATE_VALUES[@]} \
    --static_input_mode "$STATIC_INPUT_MODE" \
    --synaptic_mode "$SYNAPTIC_MODE" \
    --output "$SCRIPT_DIR/jobs_stability.txt"

if [ $? -ne 0 ]; then
    echo "ERROR: Job generation failed!"
    exit 1
fi

# Run sweep using the common sweep runner
"$SCRIPT_DIR/run_sweep_engine.sh" \
    --jobs_file "$SCRIPT_DIR/jobs_stability.txt" \
    --task "$TASK_TYPE" \
    --num_parallel $NUM_PARALLEL \
    --logdir "$SCRIPT_DIR/logs_stability"
