#!/bin/bash
#
# Network stability experiment runner
# MODIFIED: New directory structure
#

set -e

# Source shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiment_utils.sh"

# Default parameters
N_SESSIONS=5
SESSION_START=0
SESSION_END=-1
N_PROCESSES=50

# Network
N_NEURONS=1000

# Heterogeneity
V_TH_STD_MIN=0.01
V_TH_STD_MAX=4.0
N_V_TH_STD=10
G_STD_MIN=0.01
G_STD_MAX=4.0
N_G_STD=10

# Static input
STATIC_INPUT_RATE_MIN=50.0
STATIC_INPUT_RATE_MAX=1000.0
N_STATIC_INPUT_RATES=5

# Network modes
SYNAPTIC_MODE="filter"
STATIC_INPUT_MODE="independent"
V_TH_DISTRIBUTION="normal"

# Paths - MODIFIED
OUTPUT_DIR="results/stability"
LOG_DIR="logs/stability"

# Averaging
AVERAGE_SESSIONS=true

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Network stability experiment runner.

Execution:
  --n_sessions N                Sessions to run (default: 5)
  --session_start N             First session (default: 0)
  --session_end N               Last session (auto-calculated)
  --n_processes N               MPI processes (default: 50)

Heterogeneity:
  --v_th_std_min X              Min threshold std (default: 0.01)
  --v_th_std_max X              Max threshold std (default: 4.0)
  --n_v_th_std N                Points (default: 10)
  --g_std_min X                 Min weight std (default: 0.01)
  --g_std_max X                 Max weight std (default: 4.0)
  --n_g_std N                   Points (default: 10)

Static Input:
  --static_input_rate_min X     Min Hz (default: 50.0)
  --static_input_rate_max X     Max Hz (default: 1000.0)
  --n_static_input_rates N      Points (default: 5)

Network Modes:
  --synaptic_mode MODE          pulse|filter (default: filter)
  --static_input_mode MODE      independent|common_stochastic|common_tonic
  --v_th_distribution DIST      normal|uniform (default: normal)

Paths:
  --output_dir DIR              Output (default: results/stability)

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
        --n_neurons) N_NEURONS="$2"; shift 2 ;;
        --v_th_std_min) V_TH_STD_MIN="$2"; shift 2 ;;
        --v_th_std_max) V_TH_STD_MAX="$2"; shift 2 ;;
        --n_v_th_std) N_V_TH_STD="$2"; shift 2 ;;
        --g_std_min) G_STD_MIN="$2"; shift 2 ;;
        --g_std_max) G_STD_MAX="$2"; shift 2 ;;
        --n_g_std) N_G_STD="$2"; shift 2 ;;
        --static_input_rate_min) STATIC_INPUT_RATE_MIN="$2"; shift 2 ;;
        --static_input_rate_max) STATIC_INPUT_RATE_MAX="$2"; shift 2 ;;
        --n_static_input_rates) N_STATIC_INPUT_RATES="$2"; shift 2 ;;
        --synaptic_mode) SYNAPTIC_MODE="$2"; shift 2 ;;
        --static_input_mode) STATIC_INPUT_MODE="$2"; shift 2 ;;
        --v_th_distribution) V_TH_DISTRIBUTION="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no_average) AVERAGE_SESSIONS=false; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_message "ERROR: Unknown option '$1'"; show_help; exit 1 ;;
    esac
done

if [ $SESSION_END -eq -1 ]; then
    SESSION_END=$((SESSION_START + N_SESSIONS - 1))
fi

TOTAL_COMBOS=$((N_V_TH_STD * N_G_STD * N_STATIC_INPUT_RATES))

log_section "NETWORK STABILITY EXPERIMENT"
log_message "MPI processes: $N_PROCESSES"
log_message "Sessions: ${SESSION_START} to ${SESSION_END}"
log_message "Parameter grid: ${N_V_TH_STD}×${N_G_STD}×${N_STATIC_INPUT_RATES} = ${TOTAL_COMBOS} combinations"

# Setup directories
setup_directories "$OUTPUT_DIR" "$LOG_DIR" || exit 1

# Verify files
REQUIRED_FILES=(
    "runners/mpi_stability_runner.py"
    "runners/mpi_utils.py"
    "experiments/stability_experiment.py"
    "analysis/stability_analysis.py"
)
verify_required_files REQUIRED_FILES || exit 1

# Check dependencies
check_python_dependencies "import numpy, scipy, mpi4py" || exit 1
check_mpi || exit 1

# Validate modes
validate_mode "$SYNAPTIC_MODE" "pulse filter" "synaptic_mode" || exit 1
validate_mode "$STATIC_INPUT_MODE" "independent common_stochastic common_tonic" "static_input_mode" || exit 1
validate_mode "$V_TH_DISTRIBUTION" "normal uniform" "v_th_distribution" || exit 1

# Run experiments
log_section "RUNNING EXPERIMENTS"
COMPLETED_SESSIONS=()
FAILED_SESSIONS=()
OVERALL_START=$(date +%s)

for SESSION_ID in $(seq ${SESSION_START} ${SESSION_END}); do
    log_message "Starting stability session ${SESSION_ID}..."
    LOG_FILE="${LOG_DIR}/session_${SESSION_ID}.log"

    mpirun -n ${N_PROCESSES} python runners/mpi_stability_runner.py \
        --session_id ${SESSION_ID} \
        --n_v_th_std ${N_V_TH_STD} \
        --n_g_std ${N_G_STD} \
        --n_neurons ${N_NEURONS} \
        --output_dir ${OUTPUT_DIR} \
        --v_th_std_min ${V_TH_STD_MIN} \
        --v_th_std_max ${V_TH_STD_MAX} \
        --g_std_min ${G_STD_MIN} \
        --g_std_max ${G_STD_MAX} \
        --static_input_rate_min ${STATIC_INPUT_RATE_MIN} \
        --static_input_rate_max ${STATIC_INPUT_RATE_MAX} \
        --n_static_input_rates ${N_STATIC_INPUT_RATES} \
        --synaptic_mode ${SYNAPTIC_MODE} \
        --static_input_mode ${STATIC_INPUT_MODE} \
        --v_th_distribution ${V_TH_DISTRIBUTION} \
        2>&1 | tee ${LOG_FILE}

    if [ $? -eq 0 ]; then
        COMPLETED_SESSIONS+=(${SESSION_ID})
    else
        FAILED_SESSIONS+=(${SESSION_ID})
    fi
done

# Session averaging - MODIFIED for new structure
if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
    log_section "SESSION AVERAGING"

    AVERAGING_SCRIPT=$(mktemp /tmp/average_stability.XXXXXX.py)
    cat > "$AVERAGING_SCRIPT" << 'EOFPYTHON'
import sys, os, glob, pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from experiments.experiment_utils import save_results

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

output_dir = sys.argv[1]
data_dir = os.path.join(output_dir, "data")

all_files = glob.glob(os.path.join(data_dir, "stability_session_*.pkl"))

# Group by parameter combination
combo_files = defaultdict(list)
for filepath in all_files:
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    session_idx = parts.index('session') + 1
    combo_sig = '_'.join(parts[session_idx+1:])
    combo_files[combo_sig].append(filepath)

print(f"Found {len(combo_files)} unique parameter combinations")

for combo_sig, files in combo_files.items():
    if len(files) < 2:
        continue

    print(f"Averaging {combo_sig}: {len(files)} sessions")

    all_results = []
    for f in files:
        all_results.extend(load_results(f))

    # Group by parameter combination within file
    param_groups = defaultdict(list)
    for result in all_results:
        key = (result['v_th_std'], result['g_std'], result['static_input_rate'])
        param_groups[key].append(result)

    averaged_results = []
    for key, results in param_groups.items():
        averaged = results[0].copy()

        # Average metrics
        for metric in ['lz_spatial_patterns_mean', 'lz_column_wise_mean', 'settling_time_ms_mean']:
            if metric in results[0]:
                values = [r[metric] for r in results]
                averaged[metric] = float(np.mean(values))
                averaged[f'{metric}_std_across_sessions'] = float(np.std(values))

        averaged['n_sessions_averaged'] = len(results)
        averaged['session_ids'] = [r['session_id'] for r in results]
        averaged_results.append(averaged)

    # Save to parent directory
    output_file = os.path.join(output_dir, f"stability_averaged_{combo_sig}")
    save_results(averaged_results, output_file, use_data_subdir=False)

print("Averaging complete")
EOFPYTHON

    python3 "$AVERAGING_SCRIPT" "${OUTPUT_DIR}"
    rm "$AVERAGING_SCRIPT"

    log_message "✓ Session averaging completed - saved to ${OUTPUT_DIR}/"
fi

# Final summary
TOTAL_DURATION=$(($(date +%s) - OVERALL_START))
print_final_summary "stability" "$TOTAL_DURATION" "${#COMPLETED_SESSIONS[@]}" "$((SESSION_END - SESSION_START + 1))" \
    COMPLETED_SESSIONS FAILED_SESSIONS "$OUTPUT_DIR" ""
exit $?
