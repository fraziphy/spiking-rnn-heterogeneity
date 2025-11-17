#!/usr/bin/env python3
"""
Generate job commands for parameter sweep with support for all experiment types
Supports: categorical, temporal, autoencoding, spontaneous, stability tasks

Usage:
  python3 generate_jobs.py --task categorical --n_patterns 4 --sessions 20
  python3 generate_jobs.py --task temporal --n_patterns 1 --sessions 20
  python3 generate_jobs.py --task autoencoding --sessions 20
  python3 generate_jobs.py --task spontaneous --sessions 20
  python3 generate_jobs.py --task stability --sessions 20
"""

import argparse
import itertools
import numpy as np

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Generate parameter sweep jobs')
parser.add_argument('--task', required=True,
                    choices=['categorical', 'temporal', 'autoencoding', 'spontaneous', 'stability'],
                    help='Task type')
parser.add_argument('--n_patterns', type=int, default=None,
                    help='Number of input patterns (for task experiments only)')
parser.add_argument('--sessions', type=int, default=20,
                    help='Number of sessions to run')
parser.add_argument('--session_start', type=int, default=0,
                    help='First session ID')
parser.add_argument('--output', default='jobs.txt',
                    help='Output file for jobs')

# Optional parameter ranges
parser.add_argument('--v_th_values', nargs='+', type=float, default=[0, 0.5, 1.0],
                    help='Threshold heterogeneity values')
parser.add_argument('--g_values', nargs='+', type=float, default=[0.5, 1.0, 1.5],
                    help='Weight heterogeneity values')
parser.add_argument('--rate_values', nargs='+', type=float, default=[20, 30, 40],
                    help='Static input rate values')
parser.add_argument('--synaptic_mode', type=str, default=None,
                    help='Synaptic mode: pulse or filter')
parser.add_argument('--static_input_mode', type=str, default=None,
                    help='Static input mode: independent, common_stochastic, common_tonic')
parser.add_argument('--hd_input_mode', type=str, default=None,
                    help='HD input mode: independent, common_tonic (task experiments only)')
parser.add_argument('--hd_connection_mode', type=str, default=None,
                    choices=[None, 'overlapping', 'partitioned'],
                    help='HD connection mode: overlapping (30%% random) or partitioned (equal division)')

# Task-specific parameters
parser.add_argument('--embed_dim_input', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                    help='Input embedding dimensions (task experiments only)')
parser.add_argument('--embed_dim_output', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                    help='Output embedding dimensions (temporal task only)')
parser.add_argument('--duration', type=float, default=800.0,
                    help='Simulation duration in ms (spontaneous/stability only)')

args = parser.parse_args()

# ============================================================================
# TASK CONFIGURATION
# ============================================================================

TASK_TYPE = args.task

# ============================================================================
# SESSION CONFIGURATION
# ============================================================================

N_SESSIONS = args.sessions
SESSION_START = args.session_start
SESSION_END = SESSION_START + N_SESSIONS - 1

# ============================================================================
# PARAMETER GRID
# ============================================================================

v_th_values = args.v_th_values
g_values = args.g_values
rate_values = args.rate_values

# ============================================================================
# TASK-SPECIFIC CONFIGURATION
# ============================================================================

if TASK_TYPE in ["categorical", "temporal", "autoencoding"]:
    # Task experiments with HD inputs

    if TASK_TYPE == "autoencoding":
        n_input_patterns = 1  # MUST be 1 for autoencoding
    elif args.n_patterns is not None:
        n_input_patterns = args.n_patterns
    else:
        # Default values
        if TASK_TYPE == "categorical":
            n_input_patterns = 4
        else:  # temporal
            n_input_patterns = 2

    embed_dim_input_all = np.array(args.embed_dim_input)
    embed_dim_output_all = np.array(args.embed_dim_output)

    # Task-specific overrides
    if TASK_TYPE == "categorical":
        embed_dim_output_all = np.array([0])
    elif TASK_TYPE == "autoencoding":
        embed_dim_output_all = embed_dim_input_all

    # Determine HD connection mode (default: overlapping for backward compatibility)
    hd_connection_mode = args.hd_connection_mode if args.hd_connection_mode else 'overlapping'
    
    # Adjust output directory based on connection mode
    mode_suffix = f"_{hd_connection_mode}" if hd_connection_mode != "overlapping" else ""

    FIXED_PARAMS = {
        'n_input_patterns': n_input_patterns,
        'n_neurons': 1000,
        'n_trials_per_pattern': 100,
        'output_dir': f'results/{TASK_TYPE}_sweep{mode_suffix}',
        'signal_cache_dir': f'hd_signals/{TASK_TYPE}_sweep',
        'stimulus_duration': 300.0,
        'decision_window': 300.0,
        'synaptic_mode': args.synaptic_mode if args.synaptic_mode else 'filter',
        'static_input_mode': args.static_input_mode if args.static_input_mode else 'common_tonic',
        'hd_input_mode': args.hd_input_mode if args.hd_input_mode else 'common_tonic',
        'hd_connection_mode': hd_connection_mode,
        'v_th_distribution': 'normal',
        'lambda_reg': 0.001,
        'tau_syn': 5.0,
        'dt': 0.1
    }

elif TASK_TYPE in ["spontaneous", "stability"]:
    # Spontaneous and stability experiments

    duration = args.duration

    FIXED_PARAMS = {
        'n_neurons': 1000,
        'output_dir': f'results/{TASK_TYPE}_sweep',
        'synaptic_mode': args.synaptic_mode if args.synaptic_mode else 'filter',
        'static_input_mode': args.static_input_mode if args.static_input_mode else 'independent',
        'v_th_distribution': 'normal',
    }

    if TASK_TYPE == "spontaneous":
        FIXED_PARAMS['duration'] = duration

# ============================================================================
# GENERATE JOBS
# ============================================================================

jobs = []

if TASK_TYPE == "autoencoding":
    # AUTOENCODING: Special handling
    for session_id in range(SESSION_START, SESSION_END + 1):
        for embed_dim in embed_dim_input_all:
            hd_dims = list(range(1, int(embed_dim) + 1))

            for v_th, g, rate in itertools.product(v_th_values, g_values, rate_values):
                for hd_dim in hd_dims:
                    cmd = "python3 runners/mpi_autoencoding_runner.py"
                    cmd += f" --session_id {session_id}"
                    cmd += f" --v_th_std {v_th}"
                    cmd += f" --g_std {g}"
                    cmd += f" --static_input_rate {rate}"
                    cmd += f" --input_hd_dim {hd_dim}"
                    cmd += f" --embed_dim_input {int(embed_dim)}"

                    for key, value in FIXED_PARAMS.items():
                        if isinstance(value, bool):
                            if value:
                                cmd += f" --{key}"
                        else:
                            cmd += f" --{key} {value}"

                    jobs.append(cmd)

elif TASK_TYPE in ["categorical", "temporal"]:
    # CATEGORICAL AND TEMPORAL
    for session_id in range(SESSION_START, SESSION_END + 1):
        for embed_in in embed_dim_input_all:
            for embed_out in embed_dim_output_all:

                input_hd_dims = list(range(1, int(embed_in) + 1))

                if TASK_TYPE == "categorical":
                    output_hd_dims = [n_input_patterns]
                elif TASK_TYPE == "temporal":
                    output_hd_dims = list(range(1, int(embed_out) + 1)) if embed_out > 0 else [1]

                for v_th, g, rate in itertools.product(v_th_values, g_values, rate_values):
                    for hd_in, hd_out in itertools.product(input_hd_dims, output_hd_dims):

                        cmd = f"python3 runners/mpi_task_runner.py --task_type {TASK_TYPE}"
                        cmd += f" --session_id {session_id}"
                        cmd += f" --v_th_std {v_th}"
                        cmd += f" --g_std {g}"
                        cmd += f" --static_input_rate {rate}"
                        cmd += f" --input_hd_dim {hd_in}"
                        cmd += f" --output_hd_dim {hd_out}"
                        cmd += f" --embed_dim_input {int(embed_in)}"
                        cmd += f" --embed_dim_output {int(embed_out)}"

                        for key, value in FIXED_PARAMS.items():
                            if isinstance(value, bool):
                                if value:
                                    cmd += f" --{key}"
                            else:
                                cmd += f" --{key} {value}"

                        jobs.append(cmd)

elif TASK_TYPE == "spontaneous":
    # SPONTANEOUS EXPERIMENT
    for session_id in range(SESSION_START, SESSION_END + 1):
        for v_th, g, rate in itertools.product(v_th_values, g_values, rate_values):
            cmd = "python3 runners/mpi_spontaneous_runner.py"
            cmd += f" --session_id {session_id}"
            cmd += f" --v_th_std {v_th}"
            cmd += f" --g_std {g}"
            cmd += f" --static_input_rate {rate}"

            for key, value in FIXED_PARAMS.items():
                if isinstance(value, bool):
                    if value:
                        cmd += f" --{key}"
                else:
                    cmd += f" --{key} {value}"

            jobs.append(cmd)

elif TASK_TYPE == "stability":
    # STABILITY EXPERIMENT
    for session_id in range(SESSION_START, SESSION_END + 1):
        for v_th, g, rate in itertools.product(v_th_values, g_values, rate_values):
            cmd = "python3 runners/mpi_stability_runner.py"
            cmd += f" --session_id {session_id}"
            cmd += f" --v_th_std {v_th}"
            cmd += f" --g_std {g}"
            cmd += f" --static_input_rate {rate}"

            for key, value in FIXED_PARAMS.items():
                if isinstance(value, bool):
                    if value:
                        cmd += f" --{key}"
                else:
                    cmd += f" --{key} {value}"

            jobs.append(cmd)

# ============================================================================
# SAVE TO FILE
# ============================================================================

with open(args.output, 'w') as f:
    for job in jobs:
        f.write(job + '\n')

print("=" * 80)
print("JOB FILE GENERATED")
print("=" * 80)
print(f"Task type: {TASK_TYPE}")
print(f"Sessions: {SESSION_START} to {SESSION_END} ({N_SESSIONS} total)")
print(f"Total combinations: {len(jobs)}")
print(f"Jobs per session: {len(jobs) // N_SESSIONS if N_SESSIONS > 0 else len(jobs)}")
print(f"Output file: {args.output}")
print()

if TASK_TYPE in ["categorical", "temporal", "autoencoding"]:
    print("Task-specific settings:")
    print(f"  n_input_patterns: {n_input_patterns}")
    if TASK_TYPE == "autoencoding":
        print("    (autoencoding REQUIRES n_input_patterns=1)")
    print(f"  decision_window: {FIXED_PARAMS['decision_window']} ms")
    print()
    print("Embedding dimensions:")
    print(f"  embed_dim_input: {embed_dim_input_all.tolist()}")
    print(f"  embed_dim_output: {embed_dim_output_all.tolist()}")
elif TASK_TYPE in ["spontaneous", "stability"]:
    print("Experiment settings:")
    if TASK_TYPE == "spontaneous":
        print(f"  Total duration: {FIXED_PARAMS['duration']} ms")
        print(f"  Transient time: 500 ms (analysis on remaining 300 ms)")
    else:  # stability
        print(f"  Pre-perturbation: 500 ms")
        print(f"  Post-perturbation: 300 ms")
        print(f"  Total duration: 800 ms")

print()
print("Parameter ranges:")
print(f"  v_th_std: {v_th_values}")
print(f"  g_std: {g_values}")
print(f"  static_input_rate: {rate_values}")
print("=" * 80)
