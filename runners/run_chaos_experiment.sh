#!/bin/bash
# run_chaos_experiment.sh - Updated with multiplier parameters and fixed structure

# Default parameters
N_PROCESSES=50
SESSION_ID=1
N_V_TH=10
N_G=10
N_NEURONS=1000
OUTPUT_DIR="results"
MULTIPLIER_MIN=1.0
MULTIPLIER_MAX=100.0
INPUT_RATE_MIN=50.0
INPUT_RATE_MAX=500.0
N_INPUT_RATES=5

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
        --multiplier_min)
            MULTIPLIER_MIN="$2"
            shift 2
            ;;
        --multiplier_max)
            MULTIPLIER_MAX="$2"
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
            echo "Fixed-Structure Spiking RNN Chaos Experiment with Multiplier Scaling"
            echo ""
            echo "FIXED STRUCTURE ARCHITECTURE:"
            echo "  • Network topology depends ONLY on session_id"
            echo "  • Base heterogeneities: v_th_std=0.01, g_std=0.01 (fixed)"
            echo "  • Multiplier scaling: 1-100 → actual heterogeneity 0.01-1.0"
            echo "  • Exact mean preservation: -55mV thresholds, 0 weights"
            echo "  • Same connectivity, perturbation targets across all combinations"
            echo "  • 100 trials per combination for robust statistics"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --nproc N_PROCESSES      Number of MPI processes (default: $N_PROCESSES)"
            echo "  -s, --session SESSION_ID     Session ID for fixed structure (default: $SESSION_ID)"
            echo "  --n_v_th N_V_TH             Number of v_th multiplier values (default: $N_V_TH)"
            echo "  --n_g N_G                   Number of g multiplier values (default: $N_G)"
            echo "  --n_neurons N_NEURONS       Number of neurons (default: $N_NEURONS)"
            echo "  -o, --output OUTPUT_DIR      Output directory (default: $OUTPUT_DIR)"
            echo "  --multiplier_min MIN        Minimum multiplier (default: $MULTIPLIER_MIN)"
            echo "  --multiplier_max MAX        Maximum multiplier (default: $MULTIPLIER_MAX)"
            echo "  --input_rate_min RATE       Min input rate Hz (default: $INPUT_RATE_MIN)"
            echo "  --input_rate_max RATE       Max input rate Hz (default: $INPUT_RATE_MAX)"
            echo "  --n_input_rates N_RATES     Number of input rates (default: $N_INPUT_RATES)"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  # Test fixed structure with small grid:"
            echo "  $0 --session 1 --n_v_th 3 --n_g 3 --multiplier_min 1 --multiplier_max 10 --nproc 4"
            echo ""
            echo "  # Full experiment with systematic scaling:"
            echo "  $0 --session 1 --n_v_th 20 --n_g 20 --multiplier_min 1 --multiplier_max 100 --nproc 50"
            echo ""
            echo "  # Custom heterogeneity range:"
            echo "  $0 --session 2 --multiplier_min 10 --multiplier_max 50 --n_v_th 15 --n_g 15"
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

# Calculate total combinations
TOTAL_COMBINATIONS=$((N_V_TH * N_G * N_INPUT_RATES))

# Calculate actual heterogeneity ranges
ACTUAL_MIN=$(python3 -c "print(f'{0.01 * $MULTIPLIER_MIN:.3f}')")
ACTUAL_MAX=$(python3 -c "print(f'{0.01 * $MULTIPLIER_MAX:.2f}')")

log_section "FIXED-STRUCTURE CHAOS EXPERIMENT CONFIGURATION"
log_message "MPI processes: $N_PROCESSES"
log_message "Session ID: $SESSION_ID (determines ALL network structure)"
log_message "Parameter grid: ${N_V_TH} × ${N_G} × ${N_INPUT_RATES} = ${TOTAL_COMBINATIONS} combinations"
log_message "Network size: $N_NEURONS neurons"
log_message ""
log_message "FIXED STRUCTURE ARCHITECTURE:"
log_message "  Base heterogeneities: v_th_std=0.01, g_std=0.01 (exact means: -55mV, 0)"
log_message "  Multiplier range: ${MULTIPLIER_MIN}-${MULTIPLIER_MAX}"
log_message "  Actual heterogeneity range: ${ACTUAL_MIN}-${ACTUAL_MAX}"
log_message "  Network topology: IDENTICAL across ALL parameter combinations"
log_message "  Connectivity patterns: Fixed by session_id=${SESSION_ID}"
log_message "  Perturbation targets: Same 100 neurons for all combinations"
log_message "  Trials per combination: 20 (faster testing)"
log_message ""
log_message "SYSTEMATIC SCALING BENEFITS:"
log_message "  • Pure heterogeneity effects (no topology confounds)"
log_message "  • Preserved relative network structure at all scales"
log_message "  • Perfect experimental control and reproducibility"
log_message "  • Reduced trial count for quicker experiments"
log_message ""
log_message "Input rate range: ${INPUT_RATE_MIN}-${INPUT_RATE_MAX} Hz (${N_INPUT_RATES} values)"
log_message "Output directory: ${OUTPUT_DIR}/data/"

# Enhanced time estimation for 100 trials
ESTIMATED_MINUTES=$((TOTAL_COMBINATIONS * 1))  # 1 minutes per combination (20 trials)
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))

log_message ""
log_message "Estimated duration: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"
log_message "Note: 20 trials per combination for faster computation"

if [ $ESTIMATED_HOURS -gt 48 ]; then
    log_message "WARNING: Very long experiment (>${ESTIMATED_HOURS}h)"
    log_message "Consider reducing grid size or multiplier range for testing"
fi

# Directory setup
log_section "DIRECTORY SETUP"
mkdir -p "${OUTPUT_DIR}/data"
if [ $? -eq 0 ]; then
    log_message "Created directory structure: ${OUTPUT_DIR}/data/"
    AVAILABLE_SPACE=$(df -BG "${OUTPUT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_message "Available disk space: ${AVAILABLE_SPACE}GB"

    # 100 trials generate much larger files
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        log_message "WARNING: <20GB available - 100-trial experiments generate large files"
    fi
else
    log_message "ERROR: Could not create output directory"
    exit 1
fi

# File verification
log_section "FILE VERIFICATION"
REQUIRED_FILES_AND_PATHS=(
    "runners/mpi_chaos_runner.py"
    "experiments/chaos_experiment.py"
    "analysis/spike_analysis.py"
    "src/spiking_network.py"
    "src/lif_neuron.py"
    "src/synaptic_model.py"
    "src/rng_utils.py"
)
