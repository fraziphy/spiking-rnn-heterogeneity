#!/bin/bash
# run_chaos_experiment.sh - Updated with input rate sweep support

# Default parameters
N_PROCESSES=50
SESSION_ID=1
N_V_TH=10
N_G=10
N_NEURONS=1000
OUTPUT_DIR="results"
INPUT_RATE_MIN=100.0
INPUT_RATE_MAX=200.0
N_INPUT_RATES=5

# Function to print messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to print section headers
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
        -s|--session)
            SESSION_ID="$2"
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
        -h|--help)
            echo "Spiking RNN Chaos Experiment Runner with Input Rate Sweep"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --nproc N_PROCESSES      Number of MPI processes (default: $N_PROCESSES)"
            echo "  -s, --session SESSION_ID     Session ID for reproducibility (default: $SESSION_ID)"
            echo "  --n_v_th N_V_TH             Number of v_th_std values (default: $N_V_TH)"
            echo "  --n_g N_G                   Number of g_std values (default: $N_G)"
            echo "  --n_neurons N_NEURONS       Number of neurons in network (default: $N_NEURONS)"
            echo "  -o, --output OUTPUT_DIR      Output directory (default: $OUTPUT_DIR)"
            echo "  --input_rate_min RATE       Minimum static input rate Hz (default: $INPUT_RATE_MIN)"
            echo "  --input_rate_max RATE       Maximum static input rate Hz (default: $INPUT_RATE_MAX)"
            echo "  --n_input_rates N_RATES     Number of input rate values (default: $N_INPUT_RATES)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Test with different input rates:"
            echo "  $0 --session 1 --n_v_th 3 --n_g 3 --input_rate_min 100 --input_rate_max 300 --n_input_rates 3"
            echo ""
            echo "  # Full experiment with input rate sweep:"
            echo "  $0 --session 1 --n_v_th 10 --n_g 10 --n_input_rates 5 --nproc 20"
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

# Calculate total combinations including input rates
TOTAL_COMBINATIONS=$((N_V_TH * N_G * N_INPUT_RATES))

# Print experiment configuration
log_section "CHAOS EXPERIMENT WITH INPUT RATE SWEEP"
log_message "MPI processes: $N_PROCESSES"
log_message "Session ID: $SESSION_ID"
log_message "Parameter grid: ${N_V_TH} × ${N_G} × ${N_INPUT_RATES} = ${TOTAL_COMBINATIONS} combinations"
log_message "Network size: $N_NEURONS neurons"
log_message "Input rate range: ${INPUT_RATE_MIN}-${INPUT_RATE_MAX} Hz (${N_INPUT_RATES} values)"
log_message "Output directory: $OUTPUT_DIR"
log_message "Data will be saved to: ${OUTPUT_DIR}/data/"

# Calculate estimated time (longer per combo due to input rate sweep)
ESTIMATED_MINUTES=$((TOTAL_COMBINATIONS * 2))  # 2 minutes per combination
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))

log_message ""
log_message "Estimated experiment duration: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"

if [ $ESTIMATED_HOURS -gt 24 ]; then
    log_message "WARNING: Very long experiment detected (>${ESTIMATED_HOURS}h)"
    log_message "Consider reducing parameter grid or input rate range"
fi

# Create output directory structure
log_section "DIRECTORY SETUP"
mkdir -p "${OUTPUT_DIR}/data"
if [ $? -eq 0 ]; then
    log_message "Created directory structure: ${OUTPUT_DIR}/data/"
    AVAILABLE_SPACE=$(df -BG "${OUTPUT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_message "Available disk space: ${AVAILABLE_SPACE}GB"

    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        log_message "WARNING: Low disk space (<5GB available) - input rate experiments generate more data"
    fi
else
    log_message "ERROR: Could not create output directory: ${OUTPUT_DIR}/data/"
    exit 1
fi

# Verify required files exist
log_section "FILE VERIFICATION"
REQUIRED_FILES_AND_PATHS=(
    "runners/mpi_chaos_runner.py"
    "experiments/chaos_experiment.py"
    "src/spiking_network.py"
    "src/lif_neuron.py"
    "src/synaptic_model.py"
    "analysis/spike_analysis.py"
    "src/rng_utils.py"
)

log_message "Checking required Python files..."
ALL_FILES_EXIST=true

for file_path in "${REQUIRED_FILES_AND_PATHS[@]}"; do
    if [ -f "$file_path" ]; then
        log_message "✓ Found: $file_path"
    else
        log_message "✗ Missing: $file_path"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    log_message "ERROR: Missing required files - cannot proceed"
    exit 1
fi

# Check MPI installation
log_section "MPI VERIFICATION"
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version 2>/dev/null | head -n1)
    log_message "MPI found: $MPI_VERSION"
else
    log_message "ERROR: mpirun not found"
    exit 1
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export MPI_UNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD/analysis:$PWD/experiments:$PWD/runners:$PYTHONPATH"

# Execute the experiment
log_section "STARTING EXPERIMENT"
log_message "Command to execute:"
log_message "mpirun -n $N_PROCESSES python runners/mpi_chaos_runner.py \\"
log_message "    --session_id $SESSION_ID \\"
log_message "    --n_v_th $N_V_TH \\"
log_message "    --n_g $N_G \\"
log_message "    --n_neurons $N_NEURONS \\"
log_message "    --output_dir '$OUTPUT_DIR' \\"
log_message "    --input_rate_min $INPUT_RATE_MIN \\"
log_message "    --input_rate_max $INPUT_RATE_MAX \\"
log_message "    --n_input_rates $N_INPUT_RATES"

EXPERIMENT_START=$(date '+%s')
log_message "Experiment started at: $(date)"

# Execute the MPI experiment with input rate parameters
mpirun -n $N_PROCESSES python runners/mpi_chaos_runner.py \
    --session_id $SESSION_ID \
    --n_v_th $N_V_TH \
    --n_g $N_G \
    --n_neurons $N_NEURONS \
    --output_dir "$OUTPUT_DIR" \
    --input_rate_min $INPUT_RATE_MIN \
    --input_rate_max $INPUT_RATE_MAX \
    --n_input_rates $N_INPUT_RATES

EXIT_STATUS=$?
EXPERIMENT_END=$(date '+%s')
DURATION=$((EXPERIMENT_END - EXPERIMENT_START))
DURATION_HOURS=$((DURATION / 3600))
DURATION_MINUTES=$(((DURATION % 3600) / 60))

# Report results
log_section "EXPERIMENT COMPLETED"
log_message "Exit status: $EXIT_STATUS"
log_message "Total runtime: ${DURATION_HOURS}h ${DURATION_MINUTES}m"
log_message "Finished at: $(date)"

if [ $EXIT_STATUS -eq 0 ]; then
    log_message "🎉 INPUT RATE EXPERIMENT COMPLETED SUCCESSFULLY!"

    log_message ""
    log_message "Generated files in ${OUTPUT_DIR}/data/:"
    if [ -d "${OUTPUT_DIR}/data" ]; then
        find "${OUTPUT_DIR}/data" -name "*.pkl" -o -name "*.txt" | while read file; do
            if [ -f "$file" ]; then
                FILE_SIZE=$(du -h "$file" | cut -f1)
                log_message "  $(basename "$file") (${FILE_SIZE})"
            fi
        done
    fi

    log_message ""
    log_message "Analysis suggestions:"
    log_message "  1. Load results and analyze input rate effects:"
    log_message "     import pickle"
    log_message "     with open('${OUTPUT_DIR}/data/chaos_with_input_rates_session_${SESSION_ID}.pkl', 'rb') as f:"
    log_message "         results = pickle.load(f)"
    log_message "  2. Plot chaos measures vs input rate"
    log_message "  3. Identify optimal input rate ranges for different network parameters"

else
    log_message "⚠ EXPERIMENT FAILED"
    log_message "Exit code: $EXIT_STATUS"
    log_message ""
    log_message "Try with smaller parameter space first:"
    log_message "  $0 --session $SESSION_ID --n_v_th 2 --n_g 2 --n_input_rates 3 --nproc 4"
fi

exit $EXIT_STATUS
