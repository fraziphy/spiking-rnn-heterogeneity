#!/bin/bash
# runners/experiment_utils.sh - Shared utilities for experiment shell scripts
# Source this file in other scripts: source "$(dirname "$0")/experiment_utils.sh"

# Logging functions
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_section() {
    echo ""
    log_message "========================================="
    log_message "$1"
    log_message "========================================="
}

# Directory setup
setup_directories() {
    local output_dir="$1"
    local additional_dirs="${@:2}"

    log_section "DIRECTORY SETUP"
    mkdir -p "${output_dir}/data"

    for dir in $additional_dirs; do
        mkdir -p "$dir"
    done

    if [ $? -eq 0 ]; then
        log_message "Created directory structure:"
        log_message "  Output: ${output_dir}/data/"
        for dir in $additional_dirs; do
            log_message "  Additional: ${dir}/"
        done
        return 0
    else
        log_message "ERROR: Could not create directories"
        return 1
    fi
}

# File verification
verify_required_files() {
    local -n files_array=$1

    log_section "FILE VERIFICATION"
    local all_exist=true

    for file_path in "${files_array[@]}"; do
        if [ -f "$file_path" ]; then
            log_message "✓ Found: $file_path"
        else
            log_message "✗ Missing: $file_path"
            all_exist=false
        fi
    done

    if [ "$all_exist" = false ]; then
        log_message "ERROR: Missing required files. Cannot proceed."
        return 1
    fi
    return 0
}

# Python dependency check
check_python_dependencies() {
    local dependencies="$1"

    log_section "PYTHON DEPENDENCIES CHECK"
    python3 -c "
import sys
try:
    $dependencies
    print('✓ Core dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
" 2>/dev/null

    if [ $? -ne 0 ]; then
        log_message "ERROR: Python dependencies not satisfied"
        return 1
    fi
    return 0
}

# MPI availability check
check_mpi() {
    log_section "MPI SETUP CHECK"
    if command -v mpirun &> /dev/null; then
        log_message "✓ mpirun found: $(which mpirun)"
        mpirun -n 2 python3 -c "from mpi4py import MPI; print('MPI test OK')" &> /dev/null
        if [ $? -eq 0 ]; then
            log_message "✓ MPI test successful"
            return 0
        else
            log_message "✗ MPI test failed"
            return 1
        fi
    else
        log_message "ERROR: mpirun not found"
        return 1
    fi
}

# Validate mode parameter
validate_mode() {
    local mode="$1"
    local valid_modes="$2"
    local param_name="$3"

    for valid_mode in $valid_modes; do
        if [ "$mode" = "$valid_mode" ]; then
            return 0
        fi
    done

    log_message "ERROR: Invalid $param_name '$mode'"
    log_message "Valid options: $valid_modes"
    return 1
}

# Session averaging
average_sessions() {
    local output_dir="$1"
    local experiment_type="$2"  # spontaneous, stability, or encoding
    local file_pattern="$3"     # Pattern to match result files
    local output_pattern="$4"   # Pattern for averaged output file
    local -n session_array=$5   # Array of completed session IDs

    log_section "SESSION AVERAGING"
    log_message "Averaging ${experiment_type} results across ${#session_array[@]} sessions..."

    # Create Python averaging script
    local averaging_script=$(cat << EOF
import sys
import os
sys.path.insert(0, 'experiments')

from ${experiment_type}_experiment import average_across_sessions, save_results

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--result_files', nargs='+', required=True)
parser.add_argument('--output_file', required=True)
args = parser.parse_args()

try:
    averaged_results = average_across_sessions(args.result_files)
    save_results(averaged_results, args.output_file, use_data_subdir=False)
    print(f'${experiment_type} session averaging completed: {args.output_file}')
except Exception as e:
    print(f'${experiment_type} session averaging failed: {e}')
    sys.exit(1)
EOF
)

    # Prepare result files list
    local result_files=()
    for session_id in "${session_array[@]}"; do
        local result_file="${output_dir}/data/${file_pattern/SESSION_ID/$session_id}"
        if [ -f "$result_file" ]; then
            result_files+=("$result_file")
        fi
    done

    if [ ${#result_files[@]} -gt 1 ]; then
        local temp_script=$(mktemp /tmp/average_sessions.XXXXXX.py)
        echo "$averaging_script" > "$temp_script"

        local averaged_file="$(pwd)/${output_dir}/data/${output_pattern/SESSION_IDS/$(IFS=_; echo "${session_array[*]}")}"
        python3 "$temp_script" --result_files "${result_files[@]}" --output_file "$averaged_file"
        local exit_code=$?

        rm "$temp_script"

        if [ $exit_code -eq 0 ]; then
            log_message "✓ ${experiment_type} session averaging completed successfully"
            log_message "Averaged file: $(basename "$averaged_file")"
            return 0
        else
            log_message "✗ ${experiment_type} session averaging failed"
            return 1
        fi
    else
        log_message "Only one result file found, skipping averaging"
        return 0
    fi
}

# Print final summary
print_final_summary() {
    local experiment_type="$1"
    local total_duration="$2"
    local n_completed="$3"
    local n_total="$4"
    local -n completed_array=$5
    local -n failed_array=$6
    local output_dir="$7"
    local extra_info="$8"

    log_section "${experiment_type^^} EXPERIMENT COMPLETED"

    if [ $n_completed -eq $n_total ]; then
        log_message "✓ ALL ${experiment_type^^} SESSIONS COMPLETED SUCCESSFULLY"
        log_message "Total duration: ${total_duration}s ($((total_duration / 60)) minutes)"
        log_message "Results saved in: ${output_dir}/data/"
        if [ -n "$extra_info" ]; then
            log_message "$extra_info"
        fi
        return 0
    elif [ $n_completed -gt 0 ]; then
        log_message "⚠ PARTIAL SUCCESS"
        log_message "Completed: ${n_completed}/${n_total} sessions"
        log_message "Successful: [${completed_array[*]}]"
        log_message "Failed: [${failed_array[*]}]"
        log_message "Check logs for failed session details"
        return 2
    else
        log_message "✗ ALL ${experiment_type^^} SESSIONS FAILED"
        log_message "Check system requirements and file permissions"
        return 1
    fi
}

# Convert session IDs string to array
parse_session_ids() {
    local session_string="$1"
    local -n output_array=$2
    IFS=' ' read -r -a output_array <<< "$session_string"
}
