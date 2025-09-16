#!/bin/bash
# run_chaos_experiment.sh
# MPI chaos experiment runner with nohup compatibility and data directory management

# Default parameters
N_PROCESSES=50
SESSION_ID=1
N_V_TH=20
N_G=20
N_NEURONS=1000
MAX_CORES=50
OUTPUT_DIR="results"

# Function to print messages with timestamps (essential for nohup logs)
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
        --max_cores)
            MAX_CORES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Spiking RNN Chaos Experiment Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --nproc N_PROCESSES    Number of MPI processes (default: $N_PROCESSES)"
            echo "  -s, --session SESSION_ID   Session ID for reproducibility (default: $SESSION_ID)"
            echo "  --n_v_th N_V_TH           Number of v_th_std values (default: $N_V_TH)"
            echo "  --n_g N_G                 Number of g_std values (default: $N_G)"
            echo "  --n_neurons N_NEURONS     Number of neurons in network (default: $N_NEURONS)"
            echo "  --max_cores MAX_CORES     Maximum CPU cores to use (default: $MAX_CORES)"
            echo "  -o, --output OUTPUT_DIR    Output directory (default: $OUTPUT_DIR)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Quick test:"
            echo "  $0 --session 1 --n_v_th 3 --n_g 3 --nproc 4"
            echo ""
            echo "  # Full experiment:"
            echo "  $0 --session 1 --n_v_th 20 --n_g 20 --nproc 50"
            echo ""
            echo "  # With nohup (recommended for long runs):"
            echo "  nohup $0 --session 1 --n_v_th 20 --n_g 20 > experiment.log 2>&1 &"
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

# Print experiment configuration
log_section "CHAOS EXPERIMENT CONFIGURATION"
log_message "MPI processes: $N_PROCESSES"
log_message "Session ID: $SESSION_ID"
log_message "Parameter grid: ${N_V_TH} × ${N_G} = $((N_V_TH * N_G)) combinations"
log_message "Network size: $N_NEURONS neurons"
log_message "Max CPU cores: $MAX_CORES"
log_message "Output directory: $OUTPUT_DIR"
log_message "Data will be saved to: ${OUTPUT_DIR}/data/"

# Calculate estimated time
TOTAL_COMBINATIONS=$((N_V_TH * N_G))
ESTIMATED_MINUTES=$((TOTAL_COMBINATIONS * 1))  # Rough estimate: 1 minute per combination
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))

log_message ""
log_message "Estimated experiment duration: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"

if [ $ESTIMATED_HOURS -gt 12 ]; then
    log_message "WARNING: Long experiment detected (>${ESTIMATED_HOURS}h)"
    log_message "Consider using smaller parameter grid for testing first"
fi

# Create output directory structure
log_section "DIRECTORY SETUP"
mkdir -p "${OUTPUT_DIR}/data"
if [ $? -eq 0 ]; then
    log_message "Created directory structure: ${OUTPUT_DIR}/data/"

    # Check available disk space
    AVAILABLE_SPACE=$(df -BG "${OUTPUT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_message "Available disk space: ${AVAILABLE_SPACE}GB"

    if [ "$AVAILABLE_SPACE" -lt 2 ]; then
        log_message "WARNING: Low disk space (<2GB available)"
        log_message "Large experiments may fail due to insufficient storage"
    fi
else
    log_message "ERROR: Could not create output directory: ${OUTPUT_DIR}/data/"
    exit 1
fi

# Verify all required files exist - FIXED to check correct paths
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
    log_message ""
    log_message "Make sure you are running this script from the project root directory:"
    log_message "  cd spiking_rnn_heterogeneity/"
    log_message "  runners/run_chaos_experiment.sh --n_v_th 2 --n_g 2 --nproc 2"
    exit 1
fi

log_message "All required files found successfully"

# Check MPI installation
log_section "MPI VERIFICATION"
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version 2>/dev/null | head -n1)
    log_message "MPI found: $MPI_VERSION"
else
    log_message "ERROR: mpirun not found"
    log_message "Please install MPI: sudo apt-get install openmpi-bin openmpi-dev"
    exit 1
fi

# Check Python dependencies
log_message "Checking Python dependencies..."
python3 -c "import numpy, scipy, mpi4py" 2>/dev/null
if [ $? -eq 0 ]; then
    log_message "✓ Core Python dependencies available"
else
    log_message "WARNING: Some Python dependencies may be missing"
    log_message "Run: pip install -r requirements.txt"
fi

# Set environment for nohup compatibility
export PYTHONUNBUFFERED=1  # Ensure immediate output flushing
export MPI_UNBUFFERED=1     # Unbuffer MPI output

# Add project directories to Python path
export PYTHONPATH="$PWD/src:$PWD/analysis:$PWD/experiments:$PWD/runners:$PYTHONPATH"

# Final confirmation before starting
log_section "STARTING EXPERIMENT"
log_message "All checks passed - initializing MPI chaos experiment"
log_message ""
log_message "Command to execute:"
log_message "mpirun -n $N_PROCESSES python runners/mpi_chaos_runner.py \\"
log_message "    --session_id $SESSION_ID \\"
log_message "    --n_v_th $N_V_TH \\"
log_message "    --n_g $N_G \\"
log_message "    --n_neurons $N_NEURONS \\"
log_message "    --max_cores $MAX_CORES \\"
log_message "    --output_dir '$OUTPUT_DIR'"
log_message ""

# Record start time
EXPERIMENT_START=$(date '+%s')
log_message "Experiment started at: $(date)"

# Execute the MPI chaos experiment - FIXED to use correct path
mpirun -n $N_PROCESSES python runners/mpi_chaos_runner.py \
    --session_id $SESSION_ID \
    --n_v_th $N_V_TH \
    --n_g $N_G \
    --n_neurons $N_NEURONS \
    --max_cores $MAX_CORES \
    --output_dir "$OUTPUT_DIR"

# Capture exit status
EXIT_STATUS=$?
EXPERIMENT_END=$(date '+%s')
DURATION=$((EXPERIMENT_END - EXPERIMENT_START))
DURATION_HOURS=$((DURATION / 3600))
DURATION_MINUTES=$(((DURATION % 3600) / 60))

# Report final results
log_section "EXPERIMENT COMPLETED"
log_message "Exit status: $EXIT_STATUS"
log_message "Total runtime: ${DURATION_HOURS}h ${DURATION_MINUTES}m"
log_message "Finished at: $(date)"

if [ $EXIT_STATUS -eq 0 ]; then
    log_message "🎉 EXPERIMENT COMPLETED SUCCESSFULLY!"

    # List generated files
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

    # Provide next steps
    log_message ""
    log_message "Next steps:"
    log_message "  1. Check results: ls -la ${OUTPUT_DIR}/data/"
    log_message "  2. Load results in Python:"
    log_message "     import pickle"
    log_message "     with open('${OUTPUT_DIR}/data/chaos_results_session_${SESSION_ID}.pkl', 'rb') as f:"
    log_message "         results = pickle.load(f)"
    log_message "  3. Analyze chaos patterns across parameter space"

else
    log_message "⚠ EXPERIMENT FAILED"
    log_message "Exit code: $EXIT_STATUS"
    log_message ""
    log_message "Troubleshooting:"
    log_message "  1. Check system resources (CPU, memory, disk space)"
    log_message "  2. Verify all Python dependencies are installed"
    log_message "  3. Try with smaller parameter grid first:"
    log_message "     $0 --session $SESSION_ID --n_v_th 3 --n_g 3 --nproc 4"
    log_message "  4. Check for any error messages above"
fi

log_message ""
log_message "Experiment runner finished"

exit $EXIT_STATUS
