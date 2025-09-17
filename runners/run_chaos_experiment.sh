#!/bin/bash
# run_chaos_experiment.sh - Enhanced with new analysis metrics

# Default parameters
N_PROCESSES=50
SESSION_ID=1
N_V_TH=10
N_G=10
N_NEURONS=1000
OUTPUT_DIR="results"
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

# Parse command line arguments (same as before)
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
            echo "Enhanced Spiking RNN Chaos Experiment Runner"
            echo ""
            echo "Enhanced Analysis Includes:"
            echo "  • Network activity dimensionality (PCA-based)"
            echo "  • Spike train difference magnitudes"
            echo "  • Normalized gamma coincidence metrics"
            echo "  • Updated parameter ranges (0.01-1.0)"
            echo "  • Extended post-perturbation analysis (300ms)"
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
            echo "  # Test enhanced analysis:"
            echo "  $0 --session 1 --n_v_th 3 --n_g 3 --n_input_rates 2 --nproc 4"
            echo ""
            echo "  # Full enhanced experiment:"
            echo "  $0 --session 1 --n_v_th 20 --n_g 20 --n_input_rates 5 --nproc 50"
            echo ""
            exit 0
            ;;
        *)
            log_message "ERROR: Unknown option '$1'"
            exit 1
            ;;
    esac
done

TOTAL_COMBINATIONS=$((N_V_TH * N_G * N_INPUT_RATES))

log_section "ENHANCED CHAOS EXPERIMENT CONFIGURATION"
log_message "MPI processes: $N_PROCESSES"
log_message "Session ID: $SESSION_ID"
log_message "Parameter grid: ${N_V_TH} × ${N_G} × ${N_INPUT_RATES} = ${TOTAL_COMBINATIONS} combinations"
log_message "Network size: $N_NEURONS neurons"
log_message "Input rate range: ${INPUT_RATE_MIN}-${INPUT_RATE_MAX} Hz (${N_INPUT_RATES} values)"
log_message "Output directory: $OUTPUT_DIR"
log_message ""
log_message "ENHANCED ANALYSIS FEATURES:"
log_message "  • Parameter ranges: v_th_std & g_std from 0.01-1.0 (higher heterogeneity)"
log_message "  • Post-perturbation duration: 300ms (extended analysis window)"
log_message "  • Network activity dimensionality (intrinsic & effective dimensions)"
log_message "  • Spike difference magnitude quantification"
log_message "  • Normalized gamma coincidence metrics (5ms window)"
log_message "  • All original chaos measures (LZ complexity, Hamming slopes)"

# Estimate longer time for enhanced analysis
ESTIMATED_MINUTES=$((TOTAL_COMBINATIONS * 3))  # 3 minutes per combination (more analysis)
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))

log_message ""
log_message "Estimated duration: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"
log_message "Note: Enhanced analysis takes ~50% longer due to additional computations"

# Directory setup
log_section "DIRECTORY SETUP"
mkdir -p "${OUTPUT_DIR}/data"
if [ $? -eq 0 ]; then
    log_message "Created directory structure: ${OUTPUT_DIR}/data/"
    AVAILABLE_SPACE=$(df -BG "${OUTPUT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_message "Available disk space: ${AVAILABLE_SPACE}GB"

    # Enhanced analysis generates larger files
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        log_message "WARNING: <10GB available - enhanced analysis generates larger result files"
    fi
else
    log_message "ERROR: Could not create output directory"
    exit 1
fi

# File verification (same as before but mention enhanced versions)
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

log_message "Checking required files (enhanced versions)..."
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
    log_message "ERROR: Missing required files"
    log_message "Ensure you have the enhanced versions with:"
    log_message "  • analyze_perturbation_response_enhanced() function"
    log_message "  • compute_activity_dimensionality() function"
    log_message "  • gamma_coincidence() functions"
    log_message "  • Updated parameter ranges (0.01-1.0)"
    exit 1
fi

# MPI and Python checks (same as before)
log_section "SYSTEM VERIFICATION"
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version 2>/dev/null | head -n1)
    log_message "MPI found: $MPI_VERSION"
else
    log_message "ERROR: mpirun not found"
    exit 1
fi

python3 -c "import numpy, scipy, mpi4py" 2>/dev/null
if [ $? -eq 0 ]; then
    log_message "✓ Python dependencies available"
else
    log_message "WARNING: Some dependencies may be missing"
fi

# Environment setup
export PYTHONUNBUFFERED=1
export MPI_UNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD/analysis:$PWD/experiments:$PWD/runners:$PYTHONPATH"

# Execute enhanced experiment
log_section "STARTING ENHANCED EXPERIMENT"
log_message "Launching enhanced analysis with relaxed health monitoring..."
log_message ""
log_message "Command: mpirun -n $N_PROCESSES python runners/mpi_chaos_runner.py"
log_message "  --session_id $SESSION_ID"
log_message "  --n_v_th $N_V_TH --n_g $N_G --n_neurons $N_NEURONS"
log_message "  --input_rate_min $INPUT_RATE_MIN --input_rate_max $INPUT_RATE_MAX"
log_message "  --n_input_rates $N_INPUT_RATES --output_dir '$OUTPUT_DIR'"

EXPERIMENT_START=$(date '+%s')
log_message "Enhanced experiment started at: $(date)"

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

# Results reporting
log_section "ENHANCED EXPERIMENT COMPLETED"
log_message "Exit status: $EXIT_STATUS"
log_message "Total runtime: ${DURATION_HOURS}h ${DURATION_MINUTES}m"
log_message "Finished at: $(date)"

if [ $EXIT_STATUS -eq 0 ]; then
    log_message "🎉 ENHANCED EXPERIMENT COMPLETED SUCCESSFULLY!"

    log_message ""
    log_message "Generated enhanced analysis files:"
    if [ -d "${OUTPUT_DIR}/data" ]; then
        find "${OUTPUT_DIR}/data" -name "*.pkl" -o -name "*.txt" | while read file; do
            if [ -f "$file" ]; then
                FILE_SIZE=$(du -h "$file" | cut -f1)
                log_message "  $(basename "$file") (${FILE_SIZE})"
            fi
        done
    fi

    log_message ""
    log_message "Enhanced analysis results include:"
    log_message "  📊 Original chaos measures (LZ complexity, Hamming slopes)"
    log_message "  📐 Network activity dimensionality (intrinsic & effective)"
    log_message "  📏 Spike difference magnitudes (total divergence)"
    log_message "  🎯 Gamma coincidence metrics (temporal precision)"
    log_message ""
    log_message "Python analysis example:"
    log_message "  import pickle"
    log_message "  with open('${OUTPUT_DIR}/data/chaos_relaxed_health_session_${SESSION_ID}.pkl', 'rb') as f:"
    log_message "      results = pickle.load(f)"
    log_message "  # Access enhanced metrics:"
    log_message "  # results[0]['effective_dim_mean']  # Network dimensionality"
    log_message "  # results[0]['spike_diff_mean']     # Spike differences"
    log_message "  # results[0]['gamma_coincidence_mean'] # Coincidence metrics"

else
    log_message "❌ ENHANCED EXPERIMENT FAILED"
    log_message "Exit code: $EXIT_STATUS"
    log_message ""
    log_message "For testing, try smaller parameter space:"
    log_message "  $0 --session $SESSION_ID --n_v_th 2 --n_g 2 --n_input_rates 2 --nproc 4"
fi

exit $EXIT_STATUS
