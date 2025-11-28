#!/usr/bin/env python3
# sweep/generate_jobs.py
"""
Generate MPI job commands for parameter sweeps.
MODIFIED: Supports --use_cached_spikes and --use_cached_transients flags.
"""

import argparse
import itertools
from typing import List, Dict, Any


def generate_task_jobs(task_type: str, sessions: List[int], v_th_values: List[float],
                      g_values: List[float], rate_values: List[float],
                      embed_dim_input: List[int], embed_dim_output: List[int],
                      n_patterns: int, static_input_mode: str, hd_input_mode: str,
                      synaptic_mode: str, hd_connection_mode: str,
                      use_cached_spikes: bool, spike_cache_dir: str) -> List[str]:
    """Generate job commands for task experiments."""
    jobs = []

    # ALL tasks use mpi_task_runner.py
    runner = 'runners/mpi_task_runner.py'

    for session_id in sessions:
        for v_th, g, rate in itertools.product(v_th_values, g_values, rate_values):

            if task_type == 'temporal':
                # TEMPORAL: Loop over ALL input × output combinations
                for embed_in in embed_dim_input:
                    for embed_out in embed_dim_output:
                        input_hd_dims = list(range(1, embed_in + 1))
                        output_hd_dims = list(range(1, embed_out + 1))
                        for hd_in in input_hd_dims:
                            for hd_out in output_hd_dims:
                                cmd = f"python3 {runner} --task_type {task_type}"
                                cmd += f" --session_id {session_id}"
                                cmd += f" --v_th_std {v_th}"
                                cmd += f" --g_std {g}"
                                cmd += f" --static_input_rate {rate}"
                                cmd += f" --input_hd_dim {hd_in}"
                                cmd += f" --output_hd_dim {hd_out}"
                                cmd += f" --embed_dim_input {embed_in}"
                                cmd += f" --embed_dim_output {embed_out}"
                                cmd += f" --n_input_patterns {n_patterns}"
                                cmd += f" --synaptic_mode {synaptic_mode}"
                                cmd += f" --static_input_mode {static_input_mode}"
                                cmd += f" --hd_input_mode {hd_input_mode}"
                                cmd += f" --hd_connection_mode {hd_connection_mode}"

                                if use_cached_spikes:
                                    cmd += f" --use_cached_spikes"
                                    cmd += f" --spike_cache_dir {spike_cache_dir}"

                                jobs.append(cmd)

            else:
                # CATEGORICAL or AUTOENCODING: Loop only over input
                for embed_in in embed_dim_input:
                    if task_type == 'categorical':
                        embed_out = n_patterns
                    else:  # autoencoding
                        embed_out = embed_in

                    input_hd_dims = list(range(1, embed_in + 1))
                    for hd_in in input_hd_dims:
                        hd_out = hd_in if task_type == 'autoencoding' else n_patterns

                        cmd = f"python3 {runner} --task_type {task_type}"
                        cmd += f" --session_id {session_id}"
                        cmd += f" --v_th_std {v_th}"
                        cmd += f" --g_std {g}"
                        cmd += f" --static_input_rate {rate}"
                        cmd += f" --input_hd_dim {hd_in}"
                        cmd += f" --output_hd_dim {hd_out}"
                        cmd += f" --embed_dim_input {embed_in}"
                        cmd += f" --embed_dim_output {embed_out}"
                        cmd += f" --n_input_patterns {n_patterns}"
                        cmd += f" --synaptic_mode {synaptic_mode}"
                        cmd += f" --static_input_mode {static_input_mode}"
                        cmd += f" --hd_input_mode {hd_input_mode}"
                        cmd += f" --hd_connection_mode {hd_connection_mode}"

                        if use_cached_spikes:
                            cmd += f" --use_cached_spikes"
                            cmd += f" --spike_cache_dir {spike_cache_dir}"

                        jobs.append(cmd)

    return jobs


def generate_stability_jobs(sessions: List[int], v_th_values: List[float],
                           g_values: List[float], rate_values: List[float],
                           static_input_mode: str, synaptic_mode: str,
                           use_cached_transients: bool,
                           transient_cache_dir: str) -> List[str]:
    """Generate job commands for stability experiments."""
    jobs = []
    runner = 'runners/mpi_stability_runner.py'

    for session in sessions:
        for v_th in v_th_values:
            for g in g_values:
                for rate in rate_values:
                    cmd = (f"python3 {runner} "
                          f"--session_id {session} "
                          f"--v_th_std {v_th} "
                          f"--g_std {g} "
                          f"--static_input_rate {rate} "
                          f"--static_input_mode {static_input_mode} "
                          f"--synaptic_mode {synaptic_mode}")

                    if use_cached_transients:
                        cmd += f" --use_cached_transients --transient_cache_dir {transient_cache_dir}"
                    else:
                        cmd += " --no_cached_transients"

                    jobs.append(cmd)

    return jobs


def generate_transient_jobs(sessions: List[int], g_values: List[float],
                           v_th_values: List[float], rate_values: List[float],
                           n_trials: int, static_input_mode: str,
                           synaptic_mode: str) -> List[str]:
    """Generate transient cache job commands."""

    jobs = []

    for session in sessions:
        for g in g_values:
            for vth in v_th_values:
                for rate in rate_values:
                    cmd = (
                        f"python3 experiments/transient_cache_experiment.py "
                        f"--session-start {session} "
                        f"--session-end {session + 1} "
                        f"--g-std {g} "
                        f"--v-th-std {vth} "
                        f"--static-rate {rate} "
                        f"--n-trials {n_trials} "
                        f"--static-input-mode {static_input_mode} "
                        f"--synaptic-mode {synaptic_mode} "
                        f"--cache-dir results/cached_states"
                    )
                    jobs.append(cmd)

    return jobs


def generate_evoked_jobs(sessions: List[int], g_values: List[float],
                        v_th_values: List[float], rate_values: List[float],
                        embed_dims: List[int], pattern_ids: List[int],
                        hd_connection_modes: List[str], signal_type: str) -> List[str]:
    """Generate evoked spike cache job commands."""

    jobs = []

    for session in sessions:
        for g in g_values:
            for vth in v_th_values:
                for rate in rate_values:
                    for k in embed_dims:
                        for d in range(1, k + 1):
                            for pattern_id in pattern_ids:
                                for mode in hd_connection_modes:
                                    cmd = (
                                        f"python3 experiments/evoked_spike_to_hd_input_cache_experiment.py "
                                        f"--session-start {session} "
                                        f"--session-end {session + 1} "
                                        f"--g-std {g} "
                                        f"--v-th-std {vth} "
                                        f"--static-rate {rate} "
                                        f"--hd-dims {d} "
                                        f"--embed-dims {k} "
                                        f"--pattern-ids {pattern_id} "
                                        f"--modes {mode} "
                                        f"--signal-type {signal_type}"
                                    )
                                    jobs.append(cmd)

    return jobs


def generate_hd_signal_jobs(sessions: List[int], embed_dims: List[int],
                           n_patterns: int, signal_cache_dir: str) -> List[str]:
    """Generate HD signal generation job commands."""

    jobs = []

    # Create one job per (session, embed_dim) combination
    for session in sessions:
        for k in embed_dims:
            cmd = (
                f"python3 experiments/generate_hd_signals.py "
                f"--sessions {session} "
                f"--embed-dims {k} "
                f"--n-patterns {n_patterns} "
                f"--cache-dir {signal_cache_dir}"
            )
            jobs.append(cmd)

    return jobs


def print_job_summary(task_type: str, jobs: List[str], sessions: int, args):
    """Print formatted job generation summary."""
    print("=" * 80)
    print("JOB FILE GENERATED")
    print("=" * 80)
    print(f"Task type: {task_type}")
    print(f"Sessions: 0 to {sessions - 1} ({sessions} total)")
    print(f"Total combinations: {len(jobs)}")
    print(f"Jobs per session: {len(jobs) // sessions}")
    print(f"Output file: {args.output}")
    print()

    # Task-specific settings
    if task_type in ['categorical', 'temporal', 'autoencoding']:
        print("Task-specific settings:")
        if hasattr(args, 'n_patterns'):
            print(f"  n_patterns: {args.n_patterns}")
        print()
        print("Embedding dimensions:")
        print(f"  embed_dim_input: {args.embed_dim_input}")
        if task_type == 'temporal' and hasattr(args, 'embed_dim_output'):
            print(f"  embed_dim_output: {args.embed_dim_output}")
        print()
    elif task_type == 'transient_cache':
        print("Task-specific settings:")
        print(f"  n_trials: {args.n_trials}")
        print()
    elif task_type == 'evoked_spike_cache':
        print("Task-specific settings:")
        print(f"  embed_dims: {args.embed_dims}")
        print(f"  pattern_ids: {args.pattern_ids}")
        print(f"  hd_connection_modes: {args.hd_connection_modes}")
        print(f"  signal_type: {args.signal_type}")
        print()

    print("Parameter ranges:")
    if hasattr(args, 'v_th_values'):
        print(f"  v_th_std: {args.v_th_values}")
    if hasattr(args, 'g_values'):
        print(f"  g_std: {args.g_values}")
    if hasattr(args, 'rate_values'):
        print(f"  static_input_rate: {args.rate_values}")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPI job commands for parameter sweeps")

    # Task type
    parser.add_argument('--task', type=str, required=True,
                       choices=['categorical', 'temporal', 'autoencoding',
                               'stability', 'transient_cache', 'evoked_spike_cache', 'hd_signals'],
                       help='Type of experiment')

    # Common parameters
    parser.add_argument('--sessions', type=int, required=True,
                       help='Number of sessions')
    parser.add_argument('--v_th_values', type=float, nargs='+',
                       help='Threshold heterogeneity values')
    parser.add_argument('--g_values', type=float, nargs='+',
                       help='Weight heterogeneity values')
    parser.add_argument('--rate_values', type=float, nargs='+',
                       help='Static input rate values')
    parser.add_argument('--static_input_mode', type=str, default='common_tonic',
                       help='Static input mode')
    parser.add_argument('--synaptic_mode', type=str, default='filter',
                       help='Synaptic dynamics mode')

    # Task-specific parameters
    parser.add_argument('--n_patterns', type=int, default=1,
                       help='Number of patterns (for task experiments)')
    parser.add_argument('--embed_dim_input', type=int, nargs='+', default=[10],
                       help='Input embedding dimensions')
    parser.add_argument('--embed_dim_output', type=int, nargs='+', default=[10],
                       help='Output embedding dimensions (for temporal)')
    parser.add_argument('--hd_input_mode', type=str, default='common_tonic',
                       help='HD input mode')
    parser.add_argument('--hd_connection_mode', type=str, default='overlapping',
                       choices=['overlapping', 'partitioned'],
                       help='HD connection mode')

    # Transient cache parameters
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of trials (for transient cache)')

    # Evoked spike cache parameters
    parser.add_argument('--embed_dims', type=int, nargs='+',
                       help='Embedding dimensions (for evoked cache)')
    parser.add_argument('--pattern_ids', type=int, nargs='+',
                       help='Pattern IDs (for evoked cache)')
    parser.add_argument('--hd_connection_modes', type=str, nargs='+',
                       help='HD connection modes (for evoked cache)')
    parser.add_argument('--signal_type', type=str, default='hd_input',
                       help='Signal type (for evoked cache)')

    # HD signal generation parameters
    parser.add_argument('--signal_cache_dir', type=str, default='results/hd_signals',
                       help='Signal cache directory (for HD signals)')

    # Caching parameters
    parser.add_argument('--use_cached_spikes', action='store_true',
                       help='Use cached evoked spikes (for task experiments)')
    parser.add_argument('--spike_cache_dir', type=str, default='results/cached_spikes',
                       help='Directory with cached spikes')
    parser.add_argument('--use_cached_transients', action='store_true',
                       help='Use cached transient states (for stability)')
    parser.add_argument('--transient_cache_dir', type=str, default='results/cached_states',
                       help='Directory with cached transient states')

    # Output
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for job list')

    args = parser.parse_args()

    # Generate session list
    sessions = list(range(args.sessions))

    # Generate jobs based on task type
    if args.task in ['categorical', 'temporal', 'autoencoding']:
        jobs = generate_task_jobs(
            task_type=args.task,
            sessions=sessions,
            v_th_values=args.v_th_values,
            g_values=args.g_values,
            rate_values=args.rate_values,
            embed_dim_input=args.embed_dim_input,
            embed_dim_output=args.embed_dim_output,
            n_patterns=args.n_patterns,
            static_input_mode=args.static_input_mode,
            hd_input_mode=args.hd_input_mode,
            synaptic_mode=args.synaptic_mode,
            hd_connection_mode=args.hd_connection_mode,
            use_cached_spikes=args.use_cached_spikes,
            spike_cache_dir=args.spike_cache_dir
        )
    elif args.task == 'stability':
        jobs = generate_stability_jobs(
            sessions=sessions,
            v_th_values=args.v_th_values,
            g_values=args.g_values,
            rate_values=args.rate_values,
            static_input_mode=args.static_input_mode,
            synaptic_mode=args.synaptic_mode,
            use_cached_transients=args.use_cached_transients,
            transient_cache_dir=args.transient_cache_dir
        )
    elif args.task == 'transient_cache':
        jobs = generate_transient_jobs(
            sessions=sessions,
            g_values=args.g_values,
            v_th_values=args.v_th_values,
            rate_values=args.rate_values,
            n_trials=args.n_trials,
            static_input_mode=args.static_input_mode,
            synaptic_mode=args.synaptic_mode
        )
    elif args.task == 'evoked_spike_cache':
        jobs = generate_evoked_jobs(
            sessions=sessions,
            g_values=args.g_values,
            v_th_values=args.v_th_values,
            rate_values=args.rate_values,
            embed_dims=args.embed_dims,
            pattern_ids=args.pattern_ids,
            hd_connection_modes=args.hd_connection_modes,
            signal_type=args.signal_type
        )
    elif args.task == 'hd_signals':
        jobs = generate_hd_signal_jobs(
            sessions=sessions,
            embed_dims=args.embed_dims,
            n_patterns=args.n_patterns,
            signal_cache_dir=args.signal_cache_dir
        )
    else:
        raise ValueError(f"Unknown task type: {args.task}")

    # Write jobs to file
    with open(args.output, 'w') as f:
        for job in jobs:
            f.write(job + '\n')

    # Print summary
    print_job_summary(args.task, jobs, args.sessions, args)


if __name__ == "__main__":
    main()
