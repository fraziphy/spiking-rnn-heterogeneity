#!/bin/bash
#
# Temporal transformation task experiment runner
# MODIFIED: New directory structure and filename format
#

set +e

# Source shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiment_utils.sh"

# Default parameters
N_SESSIONS=10
SESSION_START=0
SESSION_END=-1
N_PROCESSES=10

# Network
N_NEURONS=1000
N_INPUT_PATTERNS=10
N_TRIALS_PER_PATTERN=100

# Heterogeneity
V_TH_STD_MIN=0.0
V_TH_STD_MAX=4.0
N_V_TH_STD=5
G_STD_MIN=0.0
G_STD_MAX=4.0
N_G_STD=5

# Static input
STATIC_INPUT_RATE_MIN=50.0
STATIC_INPUT_RATE_MAX=500.0
N_STATIC_INPUT_RATES=3

# HD input
HD_DIM_INPUT_MIN=1
HD_DIM_INPUT_MAX=5
N_HD_DIM_INPUT=1
EMBED_DIM_INPUT=10

# HD output (temporal specific)
HD_DIM_OUTPUT_MIN=1
HD_DIM_OUTPUT_MAX=2
N_HD_DIM_OUTPUT=1
EMBED_DIM_OUTPUT=4

# Network modes
SYNAPTIC_MODE="filter"
STATIC_INPUT_MODE="independent"
HD_INPUT_MODE="independent"
V_TH_DISTRIBUTION="normal"
USE_DISTRIBUTED_CV=""

# Task parameters
STIMULUS_DURATION=300.0
DECISION_WINDOW=50.0
LAMBDA_REG=0.001
TAU_SYN=5.0
DT=0.1

# Paths - MODIFIED: No longer uses /data subdirectory
OUTPUT_DIR="results/temporal"
SIGNAL_CACHE_DIR="hd_signals/temporal"
LOG_DIR="logs/temporal"

# Averaging
AVERAGE_SESSIONS=true

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Temporal transformation task experiment.

Execution:
  --n_sessions N                Sessions to run (default: 10)
  --session_start N             First session (default: 0)
  --session_end N               Last session (auto-calculated)
  --n_processes N               MPI processes (default: 10)

Task:
  --n_input_patterns N          Input patterns (default: 10)
  --n_trials_per_pattern N      Trials per pattern (default: 100)

Heterogeneity:
  --v_th_std_min X              Min threshold std (default: 0.0)
  --v_th_std_max X              Max threshold std (default: 4.0)
  --n_v_th_std N                Points (default: 5)
  --g_std_min X                 Min weight std (default: 0.0)
  --g_std_max X                 Max weight std (default: 4.0)
  --n_g_std N                   Points (default: 5)

Static Input:
  --static_input_rate_min X     Min Hz (default: 50.0)
  --static_input_rate_max X     Max Hz (default: 500.0)
  --n_static_input_rates N      Points (default: 3)

HD Input:
  --hd_dim_input_min N          Min HD dim (default: 1)
  --hd_dim_input_max N          Max HD dim (default: 5)
  --n_hd_dim_input N            Points (default: 1)
  --embed_dim_input N           Embedding dim (default: 10)

HD Output:
  --hd_dim_output_min N         Min HD dim (default: 1)
  --hd_dim_output_max N         Max HD dim (default: 2)
  --n_hd_dim_output N           Points (default: 1)
  --embed_dim_output N          Embedding dim (default: 4)

Network Modes:
  --synaptic_mode MODE          pulse|filter (default: filter)
  --static_input_mode MODE      independent|common_stochastic|common_tonic
  --hd_input_mode MODE          independent|common_stochastic|common_tonic
  --v_th_distribution DIST      normal|uniform (default: normal)
  --use_distributed_cv          Use distributed CV (default: centralized)

Task Parameters:
  --stimulus_duration X         Duration ms (default: 300.0)
  --lambda_reg X                Regularization (default: 0.001)
  --tau_syn X                   Synaptic tau ms (default: 5.0)

Paths:
  --output_dir DIR              Output (default: results/temporal)
  --signal_cache_dir DIR        HD cache (default: hd_signals/temporal)

Options:
  --no_average                  Skip session averaging

  -h, --help                    Show help

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --n_sessions) N_SESSIONS="$2"; shift 2 ;;
        --session_start) SESSION_START="$2"; shift 2 ;;
        --session_end) SESSION_END="$2"; shift 2 ;;
        --n_processes) N_PROCESSES="$2"; shift 2 ;;
        --n_input_patterns) N_INPUT_PATTERNS="$2"; shift 2 ;;
        --n_trials_per_pattern) N_TRIALS_PER_PATTERN="$2"; shift 2 ;;
        --v_th_std_min) V_TH_STD_MIN="$2"; shift 2 ;;
        --v_th_std_max) V_TH_STD_MAX="$2"; shift 2 ;;
        --n_v_th_std) N_V_TH_STD="$2"; shift 2 ;;
        --g_std_min) G_STD_MIN="$2"; shift 2 ;;
        --g_std_max) G_STD_MAX="$2"; shift 2 ;;
        --n_g_std) N_G_STD="$2"; shift 2 ;;
        --static_input_rate_min) STATIC_INPUT_RATE_MIN="$2"; shift 2 ;;
        --static_input_rate_max) STATIC_INPUT_RATE_MAX="$2"; shift 2 ;;
        --n_static_input_rates) N_STATIC_INPUT_RATES="$2"; shift 2 ;;
        --hd_dim_input_min) HD_DIM_INPUT_MIN="$2"; shift 2 ;;
        --hd_dim_input_max) HD_DIM_INPUT_MAX="$2"; shift 2 ;;
        --n_hd_dim_input) N_HD_DIM_INPUT="$2"; shift 2 ;;
        --embed_dim_input) EMBED_DIM_INPUT="$2"; shift 2 ;;
        --hd_dim_output_min) HD_DIM_OUTPUT_MIN="$2"; shift 2 ;;
        --hd_dim_output_max) HD_DIM_OUTPUT_MAX="$2"; shift 2 ;;
        --n_hd_dim_output) N_HD_DIM_OUTPUT="$2"; shift 2 ;;
        --embed_dim_output) EMBED_DIM_OUTPUT="$2"; shift 2 ;;
        --synaptic_mode) SYNAPTIC_MODE="$2"; shift 2 ;;
        --static_input_mode) STATIC_INPUT_MODE="$2"; shift 2 ;;
        --hd_input_mode) HD_INPUT_MODE="$2"; shift 2 ;;
        --v_th_distribution) V_TH_DISTRIBUTION="$2"; shift 2 ;;
        --stimulus_duration) STIMULUS_DURATION="$2"; shift 2 ;;
        --lambda_reg) LAMBDA_REG="$2"; shift 2 ;;
        --tau_syn) TAU_SYN="$2"; shift 2 ;;
        --use_distributed_cv) USE_DISTRIBUTED_CV="--use_distributed_cv"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --signal_cache_dir) SIGNAL_CACHE_DIR="$2"; shift 2 ;;
        --no_average) AVERAGE_SESSIONS=false; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_message "ERROR: Unknown option '$1'"; show_help; exit 1 ;;
    esac
done

if [ $SESSION_END -eq -1 ]; then
    SESSION_END=$((SESSION_START + N_SESSIONS - 1))
fi

# Generate parameter arrays
V_TH_STDS=($(python3 runners/linspace.py $V_TH_STD_MIN $V_TH_STD_MAX $N_V_TH_STD))
G_STDS=($(python3 runners/linspace.py $G_STD_MIN $G_STD_MAX $N_G_STD))
STATIC_RATES=($(python3 runners/linspace.py $STATIC_INPUT_RATE_MIN $STATIC_INPUT_RATE_MAX $N_STATIC_INPUT_RATES))
HD_DIMS_INPUT=($(python3 runners/linspace.py $HD_DIM_INPUT_MIN $HD_DIM_INPUT_MAX $N_HD_DIM_INPUT --int))
HD_DIMS_OUTPUT=($(python3 runners/linspace.py $HD_DIM_OUTPUT_MIN $HD_DIM_OUTPUT_MAX $N_HD_DIM_OUTPUT --int))

TOTAL_COMBOS=$((N_V_TH_STD * N_G_STD * N_STATIC_INPUT_RATES * N_HD_DIM_INPUT * N_HD_DIM_OUTPUT))

log_section "TEMPORAL TASK EXPERIMENT"
log_message "MPI processes: $N_PROCESSES"
log_message "Sessions: ${SESSION_START} to ${SESSION_END}"
log_message "Patterns: ${N_INPUT_PATTERNS}, Trials/pattern: ${N_TRIALS_PER_PATTERN}"
log_message "Parameter grid: ${N_V_TH_STD}×${N_G_STD}×${N_HD_DIM_INPUT}×${N_HD_DIM_OUTPUT}×${N_STATIC_INPUT_RATES} = ${TOTAL_COMBOS} combinations"

# Setup directories
setup_directories "$OUTPUT_DIR" "$LOG_DIR" "$SIGNAL_CACHE_DIR" || exit 1

# Verify files
REQUIRED_FILES=(
    "runners/mpi_task_runner.py"
    "runners/mpi_utils.py"
    "experiments/task_performance_experiment.py"
)
verify_required_files REQUIRED_FILES || exit 1

# Check dependencies
check_python_dependencies "import numpy, scipy, sklearn, mpi4py" || exit 1
check_mpi || exit 1

# Validate modes
validate_mode "$SYNAPTIC_MODE" "pulse filter" "synaptic_mode" || exit 1
validate_mode "$STATIC_INPUT_MODE" "independent common_stochastic common_tonic" "static_input_mode" || exit 1
validate_mode "$HD_INPUT_MODE" "independent common_stochastic common_tonic" "hd_input_mode" || exit 1
validate_mode "$V_TH_DISTRIBUTION" "normal uniform" "v_th_distribution" || exit 1

# Run experiments
log_section "RUNNING EXPERIMENTS"
COMPLETED_SESSIONS=()
FAILED_SESSIONS=()
OVERALL_START=$(date +%s)

for SESSION_ID in $(seq ${SESSION_START} ${SESSION_END}); do
    log_message "Starting temporal session ${SESSION_ID}..."
    SESSION_START_TIME=$(date +%s)
    SESSION_COMPLETED_COMBOS=0
    SESSION_FAILED_COMBOS=0

    # Loop over all parameter combinations
    COMBO_INDEX=0
    for STATIC_RATE in "${STATIC_RATES[@]}"; do
        for HD_OUT in "${HD_DIMS_OUTPUT[@]}"; do
            for HD_IN in "${HD_DIMS_INPUT[@]}"; do
                for V_TH in "${V_TH_STDS[@]}"; do
                    for G in "${G_STDS[@]}"; do

                        COMBO_INDEX=$((COMBO_INDEX + 1))
                        COMBO_START_TIME=$(date +%s)

                        COMBO_LOG="${LOG_DIR}/session_${SESSION_ID}_vth_${V_TH}_g_${G}_rate_${STATIC_RATE}_hdin_${HD_IN}_hdout_${HD_OUT}.log"

                        log_message "  [${COMBO_INDEX}/${TOTAL_COMBOS}] v_th=${V_TH}, g=${G}, rate=${STATIC_RATE}, hd_in=${HD_IN}, hd_out=${HD_OUT}"

                        mpirun -n ${N_PROCESSES} python runners/mpi_task_runner.py \
                            --task_type temporal \
                            --session_id ${SESSION_ID} \
                            --n_input_patterns ${N_INPUT_PATTERNS} \
                            --n_neurons ${N_NEURONS} \
                            --output_dir ${OUTPUT_DIR} \
                            --v_th_std ${V_TH} \
                            --g_std ${G} \
                            --static_input_rate ${STATIC_RATE} \
                            --input_hd_dim ${HD_IN} \
                            --output_hd_dim ${HD_OUT} \
                            --embed_dim_input ${EMBED_DIM_INPUT} \
                            --embed_dim_output ${EMBED_DIM_OUTPUT} \
                            --synaptic_mode ${SYNAPTIC_MODE} \
                            --static_input_mode ${STATIC_INPUT_MODE} \
                            --hd_input_mode ${HD_INPUT_MODE} \
                            --v_th_distribution ${V_TH_DISTRIBUTION} \
                            --signal_cache_dir ${SIGNAL_CACHE_DIR} \
                            --decision_window ${DECISION_WINDOW} \
                            --stimulus_duration ${STIMULUS_DURATION} \
                            --n_trials_per_pattern ${N_TRIALS_PER_PATTERN} \
                            --lambda_reg ${LAMBDA_REG} \
                            --tau_syn ${TAU_SYN} \
                            --dt ${DT} \
                            ${USE_DISTRIBUTED_CV} \
                            >> ${COMBO_LOG} 2>&1

                        if [ $? -eq 0 ]; then
                            ((SESSION_COMPLETED_COMBOS++))
                            COMBO_DURATION=$(($(date +%s) - COMBO_START_TIME))
                            log_message "    ✓ Completed in ${COMBO_DURATION}s"
                        else
                            ((SESSION_FAILED_COMBOS++))
                            log_message "    ✗ FAILED!"
                        fi

                    done
                done
            done
        done
    done

    # Session summary
    SESSION_DURATION=$(($(date +%s) - SESSION_START_TIME))
    log_message ""
    if [ $SESSION_FAILED_COMBOS -eq 0 ]; then
        COMPLETED_SESSIONS+=(${SESSION_ID})
        log_message "✓ Session ${SESSION_ID} completed: ${SESSION_COMPLETED_COMBOS}/${TOTAL_COMBOS}"
    else
        FAILED_SESSIONS+=(${SESSION_ID})
        log_message "✗ Session ${SESSION_ID} FAILED: ${SESSION_COMPLETED_COMBOS}/${TOTAL_COMBOS} successful"
    fi
done

# Session averaging - MODIFIED for new file structure
if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
    log_section "SESSION AVERAGING"

    AVERAGING_SCRIPT=$(mktemp /tmp/average_temporal.XXXXXX.py)
    cat > "$AVERAGING_SCRIPT" << 'EOFPYTHON'
import sys, os, glob, pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from experiments.experiment_utils import save_results

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Get all result files from data/ subdirectory
output_dir = sys.argv[1]
data_dir = os.path.join(output_dir, "data")

all_files = glob.glob(os.path.join(data_dir, "task_temporal_session_*.pkl"))

# Group by parameter combination (everything after session_X_)
combo_files = defaultdict(list)
for filepath in all_files:
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    session_idx = parts.index('session') + 1
    combo_sig = '_'.join(parts[session_idx+1:])
    combo_files[combo_sig].append(filepath)

print(f"Found {len(combo_files)} unique parameter combinations")

# Average each combination across sessions
for combo_sig, files in combo_files.items():
    if len(files) < 2:
        continue

    print(f"Averaging {combo_sig}: {len(files)} sessions")
    session_results = [load_results(f)[0] for f in files]
    averaged = session_results[0].copy()

    # Average CV metrics
    test_rmse = [r['test_rmse_mean'] for r in session_results]
    test_r2 = [r['test_r2_mean'] for r in session_results]
    test_corr = [r['test_correlation_mean'] for r in session_results]

    averaged['test_rmse_mean'] = float(np.mean(test_rmse))
    averaged['test_rmse_std_across_sessions'] = float(np.std(test_rmse))
    averaged['test_r2_mean'] = float(np.mean(test_r2))
    averaged['test_r2_std_across_sessions'] = float(np.std(test_r2))
    averaged['test_correlation_mean'] = float(np.mean(test_corr))
    averaged['test_correlation_std_across_sessions'] = float(np.std(test_corr))

    averaged['n_sessions_averaged'] = len(files)
    averaged['session_ids'] = [r['session_id'] for r in session_results]

    # Save to PARENT directory (results/temporal/, not results/temporal/data/)
    output_file = os.path.join(output_dir, f"task_temporal_averaged_{combo_sig}")
    save_results([averaged], output_file, use_data_subdir=False)

print("Averaging complete")
EOFPYTHON

    python3 "$AVERAGING_SCRIPT" "${OUTPUT_DIR}"
    rm "$AVERAGING_SCRIPT"

    log_message "✓ Session averaging completed - saved to ${OUTPUT_DIR}/"
fi

# Final summary
TOTAL_DURATION=$(($(date +%s) - OVERALL_START))
print_final_summary "temporal" "$TOTAL_DURATION" "${#COMPLETED_SESSIONS[@]}" "$((SESSION_END - SESSION_START + 1))" \
    COMPLETED_SESSIONS FAILED_SESSIONS "$OUTPUT_DIR" "HD signals cached in: ${SIGNAL_CACHE_DIR}/"
exit $?
