#!/bin/bash
# run_stability_experiment.sh - Network stability analysis (refactored)

# Source shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiment_utils.sh"

# Default parameters
N_PROCESSES=50
SESSION_IDS="1 2 3 4 5"
N_V_TH=10
N_G=10
N_NEURONS=1000
OUTPUT_DIR="results"
V_TH_STD_MIN=0.01
V_TH_STD_MAX=4.0
G_STD_MIN=0.01
G_STD_MAX=4.0
INPUT_RATE_MIN=50.0
INPUT_RATE_MAX=1000.0
N_INPUT_RATES=5
SYNAPTIC_MODE="filter"
STATIC_INPUT_MODE="independent"
V_TH_DISTRIBUTION="normal"
AVERAGE_SESSIONS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nproc) N_PROCESSES="$2"; shift 2 ;;
        -s|--session_ids) SESSION_IDS="$2"; shift 2 ;;
        --n_v_th) N_V_TH="$2"; shift 2 ;;
        --n_g) N_G="$2"; shift 2 ;;
        --n_neurons) N_NEURONS="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --v_th_std_min) V_TH_STD_MIN="$2"; shift 2 ;;
        --v_th_std_max) V_TH_STD_MAX="$2"; shift 2 ;;
        --g_std_min) G_STD_MIN="$2"; shift 2 ;;
        --g_std_max) G_STD_MAX="$2"; shift 2 ;;
        --input_rate_min) INPUT_RATE_MIN="$2"; shift 2 ;;
        --input_rate_max) INPUT_RATE_MAX="$2"; shift 2 ;;
        --n_input_rates) N_INPUT_RATES="$2"; shift 2 ;;
        --synaptic_mode) SYNAPTIC_MODE="$2"; shift 2 ;;
        --static_input_mode) STATIC_INPUT_MODE="$2"; shift 2 ;;
        --v_th_distribution) V_TH_DISTRIBUTION="$2"; shift 2 ;;
        --no_average) AVERAGE_SESSIONS=false; shift ;;
        -h|--help)
            echo "Network Stability Experiment"
            echo "Usage: $0 [OPTIONS]"
            echo "See experiment_utils.sh for detailed options"
            exit 0
            ;;
        *) log_message "ERROR: Unknown option '$1'"; exit 1 ;;
    esac
done

# Parse session IDs
parse_session_ids "$SESSION_IDS" SESSION_ID_ARRAY
N_SESSIONS=${#SESSION_ID_ARRAY[@]}
TOTAL_COMBINATIONS=$((N_V_TH * N_G * N_INPUT_RATES))

log_section "NETWORK STABILITY EXPERIMENT"
log_message "MPI processes: $N_PROCESSES"
log_message "Sessions: ${SESSION_IDS} (${N_SESSIONS} sessions)"
log_message "Parameter grid: ${N_V_TH} × ${N_G} × ${N_INPUT_RATES} = ${TOTAL_COMBINATIONS} combinations"

# Setup directories
setup_directories "$OUTPUT_DIR" || exit 1

# Verify required files
REQUIRED_FILES=(
    "runners/mpi_stability_runner.py"
    "runners/mpi_utils.py"
    "experiments/stability_experiment.py"
    "analysis/stability_analysis.py"
    "src/spiking_network.py"
)
verify_required_files REQUIRED_FILES || exit 1

# Check dependencies
check_python_dependencies "import numpy, scipy, mpi4py, psutil" || exit 1
check_mpi || exit 1

# Validate modes
validate_mode "$SYNAPTIC_MODE" "pulse filter" "synaptic mode" || exit 1
validate_mode "$STATIC_INPUT_MODE" "independent common_stochastic common_tonic" "static input mode" || exit 1
validate_mode "$V_TH_DISTRIBUTION" "normal uniform" "threshold distribution" || exit 1

# Run experiments
log_section "RUNNING EXPERIMENTS"
COMPLETED_SESSIONS=()
FAILED_SESSIONS=()
OVERALL_START_TIME=$(date +%s)

for SESSION_ID in "${SESSION_ID_ARRAY[@]}"; do
    log_message "Starting session ${SESSION_ID}..."

    mpirun -n ${N_PROCESSES} python runners/mpi_stability_runner.py \
        --session_id ${SESSION_ID} \
        --n_v_th ${N_V_TH} \
        --n_g ${N_G} \
        --n_neurons ${N_NEURONS} \
        --output_dir ${OUTPUT_DIR} \
        --v_th_std_min ${V_TH_STD_MIN} \
        --v_th_std_max ${V_TH_STD_MAX} \
        --g_std_min ${G_STD_MIN} \
        --g_std_max ${G_STD_MAX} \
        --input_rate_min ${INPUT_RATE_MIN} \
        --input_rate_max ${INPUT_RATE_MAX} \
        --n_input_rates ${N_INPUT_RATES} \
        --synaptic_mode ${SYNAPTIC_MODE} \
        --static_input_mode ${STATIC_INPUT_MODE} \
        --v_th_distribution ${V_TH_DISTRIBUTION}

    if [ $? -eq 0 ]; then
        COMPLETED_SESSIONS+=(${SESSION_ID})
    else
        FAILED_SESSIONS+=(${SESSION_ID})
    fi
done

# Session averaging
if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
    FILE_PATTERN="stability_session_SESSION_ID_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}.pkl"
    OUTPUT_PATTERN="stability_averaged_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_sessions_SESSION_IDS.pkl"
    average_sessions "$OUTPUT_DIR" "stability" "$FILE_PATTERN" "$OUTPUT_PATTERN" COMPLETED_SESSIONS
fi

# Final summary
OVERALL_END_TIME=$(date +%s)
TOTAL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))
print_final_summary "stability" "$TOTAL_DURATION" "${#COMPLETED_SESSIONS[@]}" "$N_SESSIONS" \
    COMPLETED_SESSIONS FAILED_SESSIONS "$OUTPUT_DIR" "Analysis: LZ complexity, Shannon entropy, settling time, coincidence"
exit $?
