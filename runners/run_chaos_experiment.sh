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
            echo "  • 20 trials per combination for efficient computation"
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
log_message "  Trials per combination: 20 (efficient computation)"
log_message ""
log_message "SYSTEMATIC SCALING BENEFITS:"
log_message "  • Pure heterogeneity effects (no topology confounds)"
log_message "  • Preserved relative network structure at all scales"
log_message "  • Perfect experimental control and reproducibility"
log_message "  • Efficient computation with 20 trials"
log_message ""
log_message "Input rate range: ${INPUT_RATE_MIN}-${INPUT_RATE_MAX} Hz (${N_INPUT_RATES} values)"
log_message "Output directory: ${OUTPUT_DIR}/data/"

# Enhanced time estimation for 20 trials
ESTIMATED_MINUTES=$((TOTAL_COMBINATIONS * 1))  # 1 minute per combination (20 trials)
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))

log_message ""
log_message "Estimated duration: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"
log_message "Note: 20 trials per combination for efficient computation"

if [ $ESTIMATED_HOURS -gt 12 ]; then
    log_message "WARNING: Long experiment (>${ESTIMATED_HOURS}h)"
    log_message "Consider reducing grid size or multiplier range for testing"
fi

# Directory setup
log_section "DIRECTORY SETUP"
mkdir -p "${OUTPUT_DIR}/data"
if [ $? -eq 0 ]; then
    log_message "Created directory structure: ${OUTPUT_DIR}/data/"
    AVAILABLE_SPACE=$(df -BG "${OUTPUT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_message "Available disk space: ${AVAILABLE_SPACE}GB"

    # 20 trials generate smaller files
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        log_message "WARNING: <5GB available - may need more space for results"
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
"

if [ $? -ne 0 ]; then
    log_message "ERROR: Python dependencies not satisfied"
    exit 1
fi

# MPI availability check
log_section "MPI SETUP CHECK"
if command -v mpirun &> /dev/null; then
    log_message "✓ mpirun found: $(which mpirun)"

    # Test MPI with a simple command
    mpirun -n 2 python3 -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}/{MPI.COMM_WORLD.Get_size()}')" &> /dev/null
    if [ $? -eq 0 ]; then
        log_message "✓ MPI test successful"
    else
        log_message "✗ MPI test failed"
        exit 1
    fi
else
    log_message "ERROR: mpirun not found. Install MPI implementation."
    exit 1
fi

# Final parameter summary
log_section "LAUNCHING EXPERIMENT"
log_message "Command: mpirun -n ${N_PROCESSES} python runners/mpi_chaos_runner.py"
log_message "Parameters:"
log_message "  --session_id ${SESSION_ID}"
log_message "  --n_v_th ${N_V_TH}"
log_message "  --n_g ${N_G}"
log_message "  --n_neurons ${N_NEURONS}"
log_message "  --output_dir ${OUTPUT_DIR}"
log_message "  --multiplier_min ${MULTIPLIER_MIN}"
log_message "  --multiplier_max ${MULTIPLIER_MAX}"
log_message "  --input_rate_min ${INPUT_RATE_MIN}"
log_message "  --input_rate_max ${INPUT_RATE_MAX}"
log_message "  --n_input_rates ${N_INPUT_RATES}"

# Launch the experiment
log_message "Starting MPI chaos experiment..."
log_message "Progress will be reported by individual MPI ranks"

mpirun -n ${N_PROCESSES} python runners/mpi_chaos_runner.py \
    --session_id ${SESSION_ID} \
    --n_v_th ${N_V_TH} \
    --n_g ${N_G} \
    --n_neurons ${N_NEURONS} \
    --output_dir ${OUTPUT_DIR} \
    --multiplier_min ${MULTIPLIER_MIN} \
    --multiplier_max ${MULTIPLIER_MAX} \
    --input_rate_min ${INPUT_RATE_MIN} \
    --input_rate_max ${INPUT_RATE_MAX} \
    --n_input_rates ${N_INPUT_RATES}

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    log_section "EXPERIMENT COMPLETED SUCCESSFULLY"
    log_message "Results saved in: ${OUTPUT_DIR}/data/"
    log_message "Check for file: chaos_fixed_structure_session_${SESSION_ID}.pkl"
else
    log_section "EXPERIMENT FAILED"
    log_message "Exit code: $EXIT_CODE"
    log_message "Check logs above for error details"
fi

exit $EXIT_CODE
