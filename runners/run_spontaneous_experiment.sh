#!/bin/bash
#
# Spontaneous activity experiment runner
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

# Simulation
DURATION=5.0  # seconds

# Paths
OUTPUT_DIR="results/spontaneous"
LOG_DIR="logs/spontaneous"

# Averaging
AVERAGE_SESSIONS=true

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Spontaneous activity experiment runner.

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

Simulation:
  --duration X                  Duration in seconds (default: 5.0)

Paths:
  --output_dir DIR              Output (default: results/spontaneous)

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
        --duration) DURATION="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no_average) AVERAGE_SESSIONS=false; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_message "ERROR: Unknown option '$1'"; show_help; exit 1 ;;
    esac
done

if [ $SESSION_END -eq -1 ]; then
    SESSION_END=$((SESSION_START + N_SESSIONS - 1))
fi

# Convert duration to ms if in seconds
if (( $(echo "$DURATION < 100" | bc -l) )); then
    DURATION_MS=$(echo "$DURATION * 1000" | bc)
else
    DURATION_MS=$DURATION
fi

TOTAL_COMBOS=$((N_V_TH_STD * N_G_STD * N_STATIC_INPUT_RATES))

log_section "SPONTANEOUS ACTIVITY EXPERIMENT"
log_message "MPI processes: $N_PROCESSES"
log_message "Sessions: ${SESSION_START} to ${SESSION_END}"
log_message "Duration: ${DURATION}s"
log_message "Parameter grid: ${N_V_TH_STD}×${N_G_STD}×${N_STATIC_INPUT_RATES} = ${TOTAL_COMBOS} combinations"

# Setup directories
setup_directories "$OUTPUT_DIR" "$LOG_DIR" || exit 1

# Verify files
REQUIRED_FILES=(
    "runners/mpi_spontaneous_runner.py"
    "runners/mpi_utils.py"
    "experiments/spontaneous_experiment.py"
    "analysis/spontaneous_analysis.py"
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
    log_message "Starting spontaneous session ${SESSION_ID}..."
    LOG_FILE="${LOG_DIR}/session_${SESSION_ID}.log"

    mpirun -n ${N_PROCESSES} python runners/mpi_spontaneous_runner.py \
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
        --duration ${DURATION_MS} \
        2>&1 | tee ${LOG_FILE}

    if [ $? -eq 0 ]; then
        COMPLETED_SESSIONS+=(${SESSION_ID})
    else
        FAILED_SESSIONS+=(${SESSION_ID})
    fi
done

# Session averaging
if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
    FILE_PATTERN="spontaneous_session_SESSION_ID_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_${DURATION}s.pkl"
    OUTPUT_PATTERN="spontaneous_averaged_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_${DURATION}s_sessions_SESSION_IDS.pkl"
    average_sessions "$OUTPUT_DIR" "spontaneous" "$FILE_PATTERN" "$OUTPUT_PATTERN" COMPLETED_SESSIONS
fi

# Final summary
TOTAL_DURATION=$(($(date +%s) - OVERALL_START))
print_final_summary "spontaneous" "$TOTAL_DURATION" "${#COMPLETED_SESSIONS[@]}" "$((SESSION_END - SESSION_START + 1))" \
    COMPLETED_SESSIONS FAILED_SESSIONS "$OUTPUT_DIR" ""
exit $?
