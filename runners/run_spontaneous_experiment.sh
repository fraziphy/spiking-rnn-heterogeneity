#!/bin/bash
# run_spontaneous_experiment.sh - Spontaneous activity analysis with pulse/filter and static input modes

# Default parameters
N_PROCESSES=50
SESSION_IDS="1 2 3 4 5 6 7 8 9 10"
N_V_TH=20
N_G=20
N_NEURONS=1000
OUTPUT_DIR="results"
V_TH_STD_MIN=0.01
V_TH_STD_MAX=4.0
G_STD_MIN=0.01
G_STD_MAX=4.0
INPUT_RATE_MIN=1.0
INPUT_RATE_MAX=50.0
N_INPUT_RATES=15
SYNAPTIC_MODE="filter"
STATIC_INPUT_MODE="independent"
V_TH_DISTRIBUTION="normal"
DURATION=2.0  # Default 2 seconds
AVERAGE_SESSIONS=true

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_section() {
    echo ""
    log_message "========================================="
    log_message "$1"
    log_message "========================================="
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nproc)
            N_PROCESSES="$2"
            shift 2
            ;;
        -s|--session_ids)
            SESSION_IDS="$2"
            shift 2
            ;;
        --n_v_th)
            N_V_TH="$2"
            shift 2
            ;;
        --n_g)
            N_G="$2"
            shift 2
            ;;
        --n_neurons)
            N_NEURONS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --v_th_std_min)
            V_TH_STD_MIN="$2"
            shift 2
            ;;
        --v_th_std_max)
            V_TH_STD_MAX="$2"
            shift 2
            ;;
        --g_std_min)
            G_STD_MIN="$2"
            shift 2
            ;;
        --g_std_max)
            G_STD_MAX="$2"
            shift 2
            ;;
        --input_rate_min)
            INPUT_RATE_MIN="$2"
            shift 2
            ;;
        --input_rate_max)
            INPUT_RATE_MAX="$2"
            shift 2
            ;;
        --n_input_rates)
            N_INPUT_RATES="$2"
            shift 2
            ;;
        --synaptic_mode)
            SYNAPTIC_MODE="$2"
            shift 2
            ;;
        --static_input_mode)
            STATIC_INPUT_MODE="$2"
            shift 2
            ;;
        --v_th_distribution)
            V_TH_DISTRIBUTION="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --no_average)
            AVERAGE_SESSIONS=false
            shift
            ;;
        -h|--help)
            echo "Spontaneous Activity Experiment - Firing Rates & Dimensionality Analysis"
            echo ""
            echo "FEATURES:"
            echo "  • Firing rate statistics (mean, std, silent neurons %)"
            echo "  • Extended dimensionality analysis (6 bin sizes):"
            echo "    - 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms"
            echo "  • Participation ratio and variance analysis"
            echo "  • Poisson analysis (CV ISI, Fano factor)"
            echo "  • Randomized job distribution for CPU load balancing"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --nproc N_PROCESSES      Number of MPI processes (default: $N_PROCESSES)"
            echo "  -s, --session_ids IDS        Session IDs to run (default: '$SESSION_IDS')"
            echo "  --n_v_th N_V_TH             Number of v_th_std values (default: $N_V_TH)"
            echo "  --n_g N_G                   Number of g_std values (default: $N_G)"
            echo "  --n_neurons N_NEURONS       Number of neurons (default: $N_NEURONS)"
            echo "  -o, --output OUTPUT_DIR      Output directory (default: $OUTPUT_DIR)"
            echo "  --v_th_std_min MIN          Minimum v_th_std (default: $V_TH_STD_MIN)"
            echo "  --v_th_std_max MAX          Maximum v_th_std (default: $V_TH_STD_MAX)"
            echo "  --g_std_min MIN             Minimum g_std (default: $G_STD_MIN)"
            echo "  --g_std_max MAX             Maximum g_std (default: $G_STD_MAX)"
            echo "  --input_rate_min RATE       Min input rate Hz (default: $INPUT_RATE_MIN)"
            echo "  --input_rate_max RATE       Max input rate Hz (default: $INPUT_RATE_MAX)"
            echo "  --n_input_rates N_RATES     Number of input rates (default: $N_INPUT_RATES)"
            echo "  --synaptic_mode MODE        'pulse' or 'filter' (default: $SYNAPTIC_MODE)"
            echo "  --static_input_mode MODE    'independent', 'common_stochastic', or 'common_tonic' (default: $STATIC_INPUT_MODE)"
            echo "  --v_th_distribution DIST    'normal' or 'uniform' (default: $V_TH_DISTRIBUTION)"
            echo "  --duration SECONDS          Simulation duration in seconds (default: $DURATION)"
            echo "  --no_average                Skip automatic session averaging"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  # Short 2-second spontaneous activity analysis:"
            echo "  $0 --duration 2 --session_ids '1 2' --n_v_th 5 --n_g 5"
            echo ""
            echo "  # Long 10-second analysis for detailed statistics:"
            echo "  $0 --duration 10 --session_ids '1 2 3 4 5' --n_v_th 20 --n_g 20"
            echo ""
            echo "  # Test pulse vs filter synapses:"
            echo "  $0 --synaptic_mode pulse --duration 5"
            echo "  $0 --synaptic_mode filter --duration 5"
            echo ""
            echo "  # Test different static input modes:"
            echo "  $0 --static_input_mode independent --duration 5"
            echo "  $0 --static_input_mode common_stochastic --duration 5"
            echo "  $0 --static_input_mode common_tonic --duration 5"
            echo ""
            echo "  # High input rate range study with uniform distribution:"
            echo "  $0 --duration 3 --input_rate_min 200 --input_rate_max 1000 --v_th_distribution uniform"
            echo ""
            exit 0
            ;;
        *)
            log_message "ERROR: Unknown option '$1'"
            log_message "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Convert session IDs to array
IFS=' ' read -r -a SESSION_ID_ARRAY <<< "$SESSION_IDS"
N_SESSIONS=${#SESSION_ID_ARRAY[@]}

# Calculate total combinations
TOTAL_COMBINATIONS=$((N_V_TH * N_G * N_INPUT_RATES))

log_section "SPONTANEOUS ACTIVITY EXPERIMENT CONFIGURATION"
log_message "MPI processes: $N_PROCESSES"
log_message "Sessions to run: ${SESSION_IDS} (${N_SESSIONS} sessions)"
log_message "Parameter grid: ${N_V_TH} × ${N_G} × ${N_INPUT_RATES} = ${TOTAL_COMBINATIONS} combinations"
log_message "Network size: $N_NEURONS neurons"
log_message "Simulation duration: ${DURATION} seconds"
log_message ""
log_message "SPONTANEOUS ACTIVITY FEATURES:"
log_message "  v_th_std range: ${V_TH_STD_MIN}-${V_TH_STD_MAX}"
log_message "  g_std range: ${G_STD_MIN}-${G_STD_MAX}"
log_message "  Threshold distribution: ${V_TH_DISTRIBUTION}"
log_message "  Trials per combination: 10"
log_message ""
log_message "ANALYSIS MEASURES:"
log_message "  • Firing rate statistics (mean, std, min, max)"
log_message "  • Silent/active neuron percentages"
log_message "  • Dimensionality analysis with 6 bin sizes:"
log_message "    - 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms"
log_message "  • Intrinsic/effective dimensionality"
log_message "  • Participation ratio and total variance"
log_message "  • Poisson analysis (CV ISI, Fano factor)"
log_message ""
log_message "SYNAPTIC MODE:"
log_message "  Mode: ${SYNAPTIC_MODE}"
log_message ""
log_message "STATIC INPUT MODE:"
log_message "  Mode: ${STATIC_INPUT_MODE}"
log_message ""
log_message "EXECUTION STRATEGY:"
log_message "  Single session runs with RANDOMIZED job distribution"
if [ "$AVERAGE_SESSIONS" = true ]; then
    log_message "  Session averaging: Automatic after all sessions complete"
else
    log_message "  Session averaging: Disabled (--no_average)"
fi

# Time estimation
DURATION_INT=${DURATION%.*}  # Remove decimal part
ESTIMATED_MINUTES_PER_SESSION=$((TOTAL_COMBINATIONS * DURATION_INT / 60))  # Rough estimate
TOTAL_ESTIMATED_MINUTES=$((ESTIMATED_MINUTES_PER_SESSION * N_SESSIONS))
ESTIMATED_HOURS=$((TOTAL_ESTIMATED_MINUTES / 60))

log_message ""
log_message "Estimated duration: ~${ESTIMATED_HOURS} hours total"
log_message "Input rate range: ${INPUT_RATE_MIN}-${INPUT_RATE_MAX} Hz"
log_message "Output directory: ${OUTPUT_DIR}/data/"

# Directory setup
log_section "DIRECTORY SETUP"
mkdir -p "${OUTPUT_DIR}/data"
if [ $? -eq 0 ]; then
    log_message "Created directory structure: ${OUTPUT_DIR}/data/"
else
    log_message "ERROR: Could not create output directory"
    exit 1
fi

# File verification
log_section "FILE VERIFICATION"
REQUIRED_FILES=(
    "runners/mpi_spontaneous_runner.py"
    "experiments/spontaneous_experiment.py"
    "analysis/spontaneous_analysis.py"
    "src/spiking_network.py"
    "src/lif_neuron.py"
    "src/synaptic_model.py"
    "src/rng_utils.py"
)

ALL_FILES_EXIST=true
for file_path in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file_path" ]; then
        log_message "✓ Found: $file_path"
    else
        log_message "✗ Missing: $file_path"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    log_message "ERROR: Missing required files. Cannot proceed."
    exit 1
fi

# Python dependencies check
log_section "PYTHON DEPENDENCIES CHECK"
python3 -c "
import sys
try:
    import numpy, scipy, mpi4py, psutil
    print('✓ Core dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    log_message "ERROR: Python dependencies not satisfied"
    exit 1
fi

# MPI availability check
log_section "MPI SETUP CHECK"
if command -v mpirun &> /dev/null; then
    log_message "✓ mpirun found: $(which mpirun)"
    mpirun -n 2 python3 -c "from mpi4py import MPI; print('MPI test OK')" &> /dev/null
    if [ $? -eq 0 ]; then
        log_message "✓ MPI test successful"
    else
        log_message "✗ MPI test failed"
        exit 1
    fi
else
    log_message "ERROR: mpirun not found"
    exit 1
fi

# Synaptic mode validation
if [[ "$SYNAPTIC_MODE" != "pulse" && "$SYNAPTIC_MODE" != "filter" ]]; then
    log_message "ERROR: Invalid synaptic mode '$SYNAPTIC_MODE'. Use 'pulse' or 'filter'"
    exit 1
fi

# Static input mode validation
if [[ "$STATIC_INPUT_MODE" != "independent" && "$STATIC_INPUT_MODE" != "common_stochastic" && "$STATIC_INPUT_MODE" != "common_tonic" ]]; then
    log_message "ERROR: Invalid static input mode '$STATIC_INPUT_MODE'"
    log_message "Use 'independent', 'common_stochastic', or 'common_tonic'"
    exit 1
fi

# Distribution validation
if [[ "$V_TH_DISTRIBUTION" != "normal" && "$V_TH_DISTRIBUTION" != "uniform" ]]; then
    log_message "ERROR: Invalid threshold distribution '$V_TH_DISTRIBUTION'. Use 'normal' or 'uniform'"
    exit 1
fi

# Duration validation
if [ "$(python3 -c "print(1 if $DURATION <= 0 else 0)")" -eq 1 ]; then
    log_message "ERROR: Duration must be positive: $DURATION"
    exit 1
fi

# Run experiments for each session
log_section "RUNNING SPONTANEOUS ACTIVITY EXPERIMENTS"

COMPLETED_SESSIONS=()
FAILED_SESSIONS=()
OVERALL_START_TIME=$(date +%s)

for SESSION_ID in "${SESSION_ID_ARRAY[@]}"; do
    log_message "Starting spontaneous activity session ${SESSION_ID} (${DURATION}s duration)..."
    SESSION_START_TIME=$(date +%s)

    mpirun -n ${N_PROCESSES} python runners/mpi_spontaneous_runner.py \
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
        --v_th_distribution ${V_TH_DISTRIBUTION} \
        --duration ${DURATION}

    SESSION_EXIT_CODE=$?
    SESSION_END_TIME=$(date +%s)
    SESSION_DURATION=$((SESSION_END_TIME - SESSION_START_TIME))

    if [ $SESSION_EXIT_CODE -eq 0 ]; then
        log_message "✓ Spontaneous activity session ${SESSION_ID} completed successfully (${SESSION_DURATION}s)"
        COMPLETED_SESSIONS+=(${SESSION_ID})
    else
        log_message "✗ Spontaneous activity session ${SESSION_ID} failed with exit code ${SESSION_EXIT_CODE}"
        FAILED_SESSIONS+=(${SESSION_ID})
    fi
done

# Summary of session execution
log_section "SPONTANEOUS ACTIVITY SESSION EXECUTION SUMMARY"
log_message "Completed sessions: ${#COMPLETED_SESSIONS[@]}/${N_SESSIONS}"
if [ ${#COMPLETED_SESSIONS[@]} -gt 0 ]; then
    log_message "Successful: [${COMPLETED_SESSIONS[*]}]"
fi
if [ ${#FAILED_SESSIONS[@]} -gt 0 ]; then
    log_message "Failed: [${FAILED_SESSIONS[*]}]"
fi

# Session averaging
if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
    log_section "SESSION AVERAGING"
    log_message "Averaging spontaneous activity results across ${#COMPLETED_SESSIONS[@]} sessions..."

    # Create Python script for averaging
    AVERAGING_SCRIPT=$(cat << 'EOF'
import sys
import os
sys.path.insert(0, 'experiments')

from spontaneous_experiment import average_across_sessions, save_results

# Get command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--result_files', nargs='+', required=True)
parser.add_argument('--output_file', required=True)
args = parser.parse_args()

try:
    averaged_results = average_across_sessions(args.result_files)
    save_results(averaged_results, args.output_file, use_data_subdir=False)
    print(f'Spontaneous activity session averaging completed: {args.output_file}')
except Exception as e:
    print(f'Spontaneous activity session averaging failed: {e}')
    sys.exit(1)
EOF
)

    # Prepare result files list
    RESULT_FILES=()
    for SESSION_ID in "${COMPLETED_SESSIONS[@]}"; do
        RESULT_FILE="${OUTPUT_DIR}/data/spontaneous_session_${SESSION_ID}_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_${DURATION}s.pkl"
        if [ -f "$RESULT_FILE" ]; then
            RESULT_FILES+=("$RESULT_FILE")
        fi
    done

    if [ ${#RESULT_FILES[@]} -gt 1 ]; then
        # Create temporary averaging script
        TEMP_SCRIPT=$(mktemp /tmp/average_spontaneous_sessions.XXXXXX.py)
        echo "$AVERAGING_SCRIPT" > "$TEMP_SCRIPT"

        # Run averaging
        AVERAGED_FILE="$(pwd)/${OUTPUT_DIR}/data/spontaneous_averaged_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_${DURATION}s_sessions_$(IFS=_; echo "${COMPLETED_SESSIONS[*]}").pkl"
        python3 "$TEMP_SCRIPT" --result_files "${RESULT_FILES[@]}" --output_file "$AVERAGED_FILE"
        AVERAGING_EXIT_CODE=$?

        # Cleanup
        rm "$TEMP_SCRIPT"

        if [ $AVERAGING_EXIT_CODE -eq 0 ]; then
            log_message "✓ Spontaneous activity session averaging completed successfully"
            log_message "Averaged file: $(basename "$AVERAGED_FILE")"
        else
            log_message "✗ Spontaneous activity session averaging failed"
        fi
    else
        log_message "Only one result file found, skipping averaging"
    fi
elif [ "$AVERAGE_SESSIONS" = false ]; then
    log_message "Spontaneous activity session averaging skipped (--no_average)"
else
    log_message "Spontaneous activity session averaging skipped (insufficient successful sessions)"
fi

# Final summary and exit
OVERALL_END_TIME=$(date +%s)
TOTAL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))

log_section "SPONTANEOUS ACTIVITY EXPERIMENT COMPLETED"
if [ ${#COMPLETED_SESSIONS[@]} -eq $N_SESSIONS ]; then
    log_message "✓ ALL SPONTANEOUS ACTIVITY SESSIONS COMPLETED SUCCESSFULLY"
    log_message "Total duration: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60)) minutes)"
    log_message "Results saved in: ${OUTPUT_DIR}/data/"
    log_message "Individual files: spontaneous_session_*_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_${DURATION}s.pkl"

    if [ "$AVERAGE_SESSIONS" = true ] && [ ${#COMPLETED_SESSIONS[@]} -gt 1 ]; then
        log_message "Averaged file: spontaneous_averaged_${SYNAPTIC_MODE}_${STATIC_INPUT_MODE}_${V_TH_DISTRIBUTION}_${DURATION}s_*.pkl"
    fi

    # Suggest comparison experiments
    log_message ""
    log_message "To compare with different configurations:"
    if [ "$SYNAPTIC_MODE" = "pulse" ]; then
        log_message "  Filter synapses: $0 --synaptic_mode filter --static_input_mode ${STATIC_INPUT_MODE} --duration ${DURATION} --session_ids '${SESSION_IDS}'"
    else
        log_message "  Pulse synapses: $0 --synaptic_mode pulse --static_input_mode ${STATIC_INPUT_MODE} --duration ${DURATION} --session_ids '${SESSION_IDS}'"
    fi

    if [ "$STATIC_INPUT_MODE" = "independent" ]; then
        log_message "  Common stochastic: $0 --synaptic_mode ${SYNAPTIC_MODE} --static_input_mode common_stochastic --duration ${DURATION} --session_ids '${SESSION_IDS}'"
        log_message "  Common tonic: $0 --synaptic_mode ${SYNAPTIC_MODE} --static_input_mode common_tonic --duration ${DURATION} --session_ids '${SESSION_IDS}'"
    fi

    log_message ""
    log_message "To run network stability analysis:"
    log_message "./runners/run_stability_experiment.sh --synaptic_mode ${SYNAPTIC_MODE} --static_input_mode ${STATIC_INPUT_MODE} --session_ids '${SESSION_IDS}'"

    EXIT_CODE=0
elif [ ${#COMPLETED_SESSIONS[@]} -gt 0 ]; then
    log_message "⚠ PARTIAL SUCCESS"
    log_message "Completed: ${#COMPLETED_SESSIONS[@]}/${N_SESSIONS} sessions"
    log_message "Check logs for failed session details"
    EXIT_CODE=2
else
    log_message "✗ ALL SPONTANEOUS ACTIVITY SESSIONS FAILED"
    log_message "Check system requirements and file permissions"
    EXIT_CODE=1
fi

exit $EXIT_CODE
