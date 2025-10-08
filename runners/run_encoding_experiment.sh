#!/bin/bash
# run_encoding_experiment.sh - HD input encoding experiment

# Source shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiment_utils.sh"

# Default parameters
N_PROCESSES=50
SESSION_IDS="1 2 3 4 5 6 7 8 9 10"
N_V_TH=5
N_G=5
N_HD=10
N_NEURONS=1000
OUTPUT_DIR="results"
V_TH_STD_MIN=0.03
V_TH_STD_MAX=3.0
G_STD_MIN=0.03
G_STD_MAX=3.0
HD_DIM_MIN=1
HD_DIM_MAX=13
INPUT_RATE_MIN=100.0
INPUT_RATE_MAX=500.0
N_INPUT_RATES=3
SYNAPTIC_MODE="filter"
STATIC_INPUT_MODE="independent"
HD_INPUT_MODE="independent"
V_TH_DISTRIBUTION="normal"
EMBED_DIM=13
SIGNAL_CACHE_DIR="hd_signals"
AVERAGE_SESSIONS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nproc) N_PROCESSES="$2"; shift 2 ;;
        -s|--session_ids) SESSION_IDS="$2"; shift 2 ;;
        --n_v_th) N_V_TH="$2"; shift 2 ;;
        --n_g) N_G="$2"; shift 2 ;;
        --n_hd) N_HD="$2"; shift 2 ;;
        --n_neurons) N_NEURONS="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --v_th_std_min) V_TH_STD_MIN="$2"; shift 2 ;;
        --v_th_std_max) V_TH_STD_MAX="$2"; shift 2 ;;
        --g_std_min) G_STD_MIN="$2"; shift 2 ;;
        --g_std_max) G_STD_MAX="$2"; shift 2 ;;
        --hd_dim_min) HD_DIM_MIN="$2"; shift 2 ;;
        --hd_dim_max) HD_DIM_MAX="$2"; shift 2 ;;
        --input_rate_min) INPUT_RATE_MIN="$2"; shift 2 ;;
        --input_rate_max) INPUT_RATE_MAX="$2"; shift 2 ;;
        --n_input_rates) N_INPUT_RATES="$2"; shift 2 ;;
        --synaptic_mode) SYNAPTIC_MODE="$2"; shift 2 ;;
        --static_input_mode) STATIC_INPUT_MODE="$2"; shift 2 ;;
        --hd_input_mode) HD_INPUT_MODE="$2"; shift 2 ;;
        --v_th_distribution) V_TH_DISTRIBUTION="$2"; shift 2 ;;
        --embed_dim) EMBED_DIM="$2"; shift 2 ;;
        --signal_cache_dir) SIGNAL_CACHE_DIR="$2"; shift 2 ;;
        --no_average) AVERAGE_SESSIONS=false; shift ;;
        -h|--help)
            echo "Encoding Experiment - HD Input Decoding Analysis"
            echo "Usage: $0 [OPTIONS]"
            exit 0
            ;;
        *) log_message "ERROR: Unknown option '$1'"; exit 1 ;;
    esac
done

# Parse session IDs
parse_session_ids "$SESSION_IDS" SESSION_ID_ARRAY
N_SESSIONS=${#SESSION_ID_ARRAY[@]}
TOTAL_COMBINATIONS=$((N_V_TH * N_G * N_HD * N_INPUT_RATES))

log_section "ENCODING EXPERIMENT"
log_message "MPI processes: $N_PROCESSES"
log_message "Sessions: $N_SESSIONS | Combinations: $TOTAL_COMBINATIONS"
log_message "HD dim range: ${HD_DIM_MIN}-${HD_DIM_MAX} | Embed dim: ${EMBED_DIM}"

# Setup directories
setup_directories "$OUTPUT_DIR" "$SIGNAL_CACHE_DIR" || exit 1

# Verify files
REQUIRED_FILES=(
    "runners/mpi_encoding_runner.py"
    "runners/mpi_utils.py"
    "experiments/encoding_experiment.py"
    "experiments/base_experiment.py"
    "experiments/experiment_utils.py"
    "analysis/encoding_analysis.py"
    "analysis/common_utils.py"
    "analysis/statistics_utils.py"
    "src/hd_input.py"
)
verify_required_files REQUIRED_FILES || exit 1

# Check dependencies
check_python_dependencies "import numpy, scipy, mpi4py, sklearn, psutil" || exit 1
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

for SESSION_ID in "${SESSION_ID_ARRAY[@]}"; do
    log_message "Starting encoding session ${SESSION_ID}..."

    mpirun -n ${N_PROCESSES} python runners/mpi_encoding_runner.py \
        --session_id ${SESSION_ID} --n_v_th ${N_V_TH} --n_g ${N_G} --n_hd ${N_HD} \
        --n_neurons ${N_NEURONS} --output_dir ${OUTPUT_DIR} \
        --v_th_std_min ${V_TH_STD_MIN} --v_th_std_max ${V_TH_STD_MAX} \
        --g_std_min ${G_STD_MIN} --g_std_max ${G_STD_MAX} \
        --hd_dim_min ${HD_DIM_MIN} --hd_dim_max ${HD_DIM_MAX} \
        --input_rate_min ${INPUT_RATE_MIN} --input_rate_max ${INPUT_RATE_MAX} \
        --n_input_rates ${N_INPUT_RATES} --synaptic_mode ${SYNAPTIC_MODE} \
        --static_input_mode ${STATIC_INPUT_MODE} --hd_input_mode ${HD_INPUT_MODE} \
        --v_th_distribution ${V_TH_DISTRIBUTION} --embed_dim ${EMBED_DIM} \
        --signal_cache_dir ${SIGNAL_CACHE_DIR}

    if [ $? -eq 0 ]; then
        COMPLETED_SESSIONS+=(${SESSION_ID})
    else
        FAILED_SESSIONS+=(${SESSION_ID})
    fi
done

# Session averaging
if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
    FILE_PATTERN="encoding_session_SESSION_ID_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${HD_INPUT_MODE}_${V_TH_DISTRIBUTION}_k${EMBED_DIM}.pkl"
    OUTPUT_PATTERN="encoding_averaged_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${HD_INPUT_MODE}_${V_TH_DISTRIBUTION}_k${EMBED_DIM}_sessions_SESSION_IDS.pkl"
    average_sessions "$OUTPUT_DIR" "encoding" "$FILE_PATTERN" "$OUTPUT_PATTERN" COMPLETED_SESSIONS
fi

# Final summary
TOTAL_DURATION=$(($(date +%s) - OVERALL_START))
EXTRA_INFO="HD signals cached in: ${SIGNAL_CACHE_DIR}/"
print_final_summary "encoding" "$TOTAL_DURATION" "${#COMPLETED_SESSIONS[@]}" "$N_SESSIONS" \
    COMPLETED_SESSIONS FAILED_SESSIONS "$OUTPUT_DIR" "$EXTRA_INFO"
exit $?
