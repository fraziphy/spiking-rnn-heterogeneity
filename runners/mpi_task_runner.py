# runners/mpi_task_runner.py
"""
MPI-parallelized task-performance experiment runner.
MODIFIED: Supports loading cached evoked spikes.
DIAGNOSTIC: Added memory tracking.
"""

import numpy as np
import os
import sys
import time
import argparse
from mpi4py import MPI
from typing import Dict, Any
import psutil
import gc

from mpi_utils import monitor_system_health, recovery_break

try:
    from experiments.task_performance_experiment import TaskPerformanceExperiment
    from experiments.experiment_utils import save_results
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    from experiments.task_performance_experiment import TaskPerformanceExperiment
    from experiments.experiment_utils import save_results


def print_memory_usage(label, rank=0):
    """Print current memory usage"""
    if rank == 0:
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024**3
        print(f"[MEMORY] {label}: {mem_gb:.2f} GB")


def run_mpi_task_experiment(args):
    """
    Run task-performance experiment with MPI parallelization.

    Can operate in two modes:
    1. use_cached_spikes=True: Load pre-computed evoked spikes
    2. use_cached_spikes=False: Simulate trials from scratch
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print_memory_usage("START", rank)

    # Extract parameters
    task_type = args.task_type
    session_id = args.session_id
    n_neurons = args.n_neurons
    output_dir = args.output_dir
    signal_cache_dir = args.signal_cache_dir
    spike_cache_dir = args.spike_cache_dir
    use_cached_spikes = args.use_cached_spikes

    v_th_std = args.v_th_std
    g_std = args.g_std
    input_hd_dim = args.input_hd_dim
    output_hd_dim = args.output_hd_dim
    static_input_rate = args.static_input_rate

    synaptic_mode = args.synaptic_mode
    static_input_mode = args.static_input_mode
    hd_input_mode = args.hd_input_mode
    v_th_distribution = args.v_th_distribution
    hd_connection_mode = args.hd_connection_mode

    n_input_patterns = args.n_input_patterns
    n_trials_per_pattern = args.n_trials_per_pattern
    n_cv_folds = 20

    if rank == 0:
        print("=" * 80)
        print(f"{task_type.upper()} TASK - SESSION {session_id}")
        print(f"Parameters: v_th={v_th_std:.3f}, g={g_std:.3f}, rate={static_input_rate:.0f}")
        print(f"            hd_in={input_hd_dim}, hd_out={output_hd_dim}")
        print(f"            mode={hd_connection_mode}, cached={use_cached_spikes}")
        print("=" * 80)
        print(f"MPI processes: {size}")
        print(f"Total trials: {n_input_patterns} × {n_trials_per_pattern} = {n_input_patterns * n_trials_per_pattern}")
        print(f"CV folds: {n_cv_folds}")

        # Setup output directory with organized structure
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        else:
            output_dir = os.path.join(output_dir, "data")

        # Add subdirectories: data/{mode}/{task_type}/
        output_dir = os.path.join(output_dir, hd_connection_mode, task_type)
        os.makedirs(output_dir, exist_ok=True)

        # Setup signal cache directory
        if not os.path.isabs(signal_cache_dir):
            signal_cache_dir = os.path.abspath(signal_cache_dir)
        os.makedirs(signal_cache_dir, exist_ok=True)

    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    signal_cache_dir = comm.bcast(signal_cache_dir if rank == 0 else None, root=0)
    comm.Barrier()

    print_memory_usage("After setup", rank)

    # Create experiment
    experiment = TaskPerformanceExperiment(
        task_type=task_type,
        n_neurons=n_neurons,
        n_input_patterns=n_input_patterns,
        input_dim_intrinsic=input_hd_dim,
        input_dim_embedding=args.embed_dim_input,
        output_dim_intrinsic=output_hd_dim,
        output_dim_embedding=args.embed_dim_output,
        dt=args.dt,
        tau_syn=args.tau_syn,
        synaptic_mode=synaptic_mode,
        static_input_mode=static_input_mode,
        hd_input_mode=hd_input_mode,
        hd_connection_mode=hd_connection_mode,
        signal_cache_dir=signal_cache_dir,
        decision_window=args.decision_window,
        stimulus_duration=args.stimulus_duration,
        n_trials_per_pattern=n_trials_per_pattern,
        lambda_reg=args.lambda_reg,
        use_distributed_cv=args.use_distributed_cv
    )

    print_memory_usage("After creating experiment", rank)

    if rank == 0:
        print("\nStep 1: Generating HD patterns...")

    # All ranks generate patterns (deterministic)
    input_patterns = experiment.input_generator.initialize_and_get_patterns(
        session_id, input_hd_dim, n_input_patterns
    )

    if task_type == 'autoencoding':
        output_patterns = input_patterns  # Use same patterns
    else:
        output_patterns = experiment.generate_output_patterns(session_id)

    print_memory_usage("After generating patterns", rank)

    if rank == 0:
        print(f"Step 2: Distributing {n_input_patterns * n_trials_per_pattern} trials across {size} ranks...")

    # Distribute trials across ranks
    total_trials = n_input_patterns * n_trials_per_pattern
    trials_per_rank = total_trials // size
    remainder = total_trials % size

    if rank < remainder:
        start_trial = rank * (trials_per_rank + 1)
        end_trial = start_trial + trials_per_rank + 1
    else:
        start_trial = rank * trials_per_rank + remainder
        end_trial = start_trial + trials_per_rank

    my_trials = list(range(start_trial, end_trial))

    if rank == 0:
        print(f"   Rank 0: trials {my_trials[0]}-{my_trials[-1]} ({len(my_trials)} trials)")

    # Load cached spikes or simulate
    local_spike_times = experiment.simulate_trials_parallel(
        session_id=session_id,
        v_th_std=v_th_std,
        g_std=g_std,
        v_th_distribution=v_th_distribution,
        static_input_rate=static_input_rate,
        my_trial_indices=my_trials,
        input_patterns=input_patterns,
        rank=rank,
        use_cached_spikes=use_cached_spikes,
        hd_connection_mode=hd_connection_mode,
        spike_cache_dir=spike_cache_dir
    )

    print_memory_usage("After simulate_trials_parallel", rank)

    # Convert to NumPy immediately to save memory (Python tuples use 10x more!)
    local_spike_arrays = []
    for trial in local_spike_times:
        spikes = trial['spike_times']
        if len(spikes) > 0:
            spike_array = np.array(spikes, dtype=[('time', 'f8'), ('neuron_id', 'i4')])
        else:
            spike_array = np.array([], dtype=[('time', 'f8'), ('neuron_id', 'i4')])
        local_spike_arrays.append({
            'pattern_id': trial['pattern_id'],
            'trial_id': trial['trial_id'],
            'global_trial_idx': trial['global_trial_idx'],
            'spike_array': spike_array
        })

    print_memory_usage("After converting to NumPy arrays", rank)

    # Free Python list memory
    del local_spike_times
    gc.collect()

    print_memory_usage("After deleting local_spike_times", rank)

    if rank == 0:
        print(f"Step 3: Gathering all spike data to rank 0...")

    # Keep arrays efficient - convert only when calling convert_spikes_to_traces
    if size == 1:
        # Single process: use data directly, no gather needed
        all_spike_arrays_flat = local_spike_arrays
        all_spike_arrays_flat.sort(key=lambda x: x['global_trial_idx'])
    else:
        # Multi-process: do MPI gather
        all_spike_arrays = comm.gather(local_spike_arrays, root=0)

        if rank == 0:
            all_spike_arrays_flat = []
            for rank_arrays in all_spike_arrays:
                all_spike_arrays_flat.extend(rank_arrays)
            all_spike_arrays_flat.sort(key=lambda x: x['global_trial_idx'])

    print_memory_usage("After gathering/sorting", rank)

    # Convert to spike_times format ONLY on rank 0, right before use
    if rank == 0:
        all_spike_times_flat = []
        for trial_data in all_spike_arrays_flat:
            spike_array = trial_data['spike_array']
            if len(spike_array) > 0:
                spike_list = [(float(s['time']), int(s['neuron_id'])) for s in spike_array]
            else:
                spike_list = []
            all_spike_times_flat.append({
                'pattern_id': trial_data['pattern_id'],
                'trial_id': trial_data['trial_id'],
                'global_trial_idx': trial_data['global_trial_idx'],
                'spike_times': spike_list
            })

        print_memory_usage("After creating all_spike_times_flat", rank)

        # FREE the NumPy arrays immediately
        del all_spike_arrays_flat
        if size > 1:
            del all_spike_arrays
        gc.collect()

        print_memory_usage("After deleting all_spike_arrays_flat", rank)

        print(f"Step 4: Converting spikes to traces on rank 0...")
        traces_all, ground_truth_all, pattern_ids = experiment.convert_spikes_to_traces(
            all_spike_times_flat,
            output_patterns,
            n_input_patterns,
            n_trials_per_pattern
        )

        print_memory_usage("After convert_spikes_to_traces", rank)

        print(f"[DEBUG] traces_all dtype: {traces_all.dtype}")
        print(f"[DEBUG] ground_truth_all dtype: {ground_truth_all.dtype}")

        print(f"Step 5: Cross-validation on rank 0...")
        cv_results = experiment.cross_validate(
            traces_all=traces_all,
            ground_truth_all=ground_truth_all,
            pattern_ids=pattern_ids,
            session_id=session_id,
            n_folds=n_cv_folds,
            rank=rank,
            size=size,
            comm=comm
        )

        print_memory_usage("After cross_validate", rank)

        del traces_all
        del ground_truth_all
        del pattern_ids
    else:
        cv_results = {}

    gc.collect()

    print_memory_usage("After final cleanup", rank)

    # Save results
    if rank == 0:
        print(f"Step 6: Saving results...")

        from experiments.task_performance_experiment import compute_pattern_dimensionalities

        input_empirical_dims = compute_pattern_dimensionalities(input_patterns)

        if task_type == 'temporal':
            output_empirical_dims = compute_pattern_dimensionalities(output_patterns)
        else:
            output_empirical_dims = None

        result = {
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'input_hd_dim': input_hd_dim,
            'output_hd_dim': output_hd_dim,
            'static_input_rate': static_input_rate,
            'v_th_distribution': v_th_distribution,
            'task_type': task_type,
            'synaptic_mode': synaptic_mode,
            'static_input_mode': static_input_mode,
            'hd_input_mode': hd_input_mode,
            'hd_connection_mode': hd_connection_mode,
            'n_input_patterns': n_input_patterns,
            'n_trials_per_pattern': n_trials_per_pattern,
            'n_cv_folds': n_cv_folds,
            'tau_syn': args.tau_syn,
            'decision_window': args.decision_window,
            'embed_dim_input': args.embed_dim_input,
            'embed_dim_output': args.embed_dim_output,
            'input_empirical_dims': input_empirical_dims,
            'used_cached_spikes': use_cached_spikes,
            **cv_results
        }

        if output_empirical_dims is not None:
            result['output_empirical_dims'] = output_empirical_dims

        if task_type == 'temporal':
            filename = (f"session_{session_id}_"
                    f"vth_{v_th_std:.3f}_g_{g_std:.3f}_rate_{static_input_rate:.0f}_"
                    f"hdin_{input_hd_dim}_embdin_{args.embed_dim_input}_"
                    f"hdout_{output_hd_dim}_embdout_{args.embed_dim_output}_"
                    f"npat_{n_input_patterns}.pkl")
        elif task_type == 'categorical':
            filename = (f"session_{session_id}_"
                    f"vth_{v_th_std:.3f}_g_{g_std:.3f}_rate_{static_input_rate:.0f}_"
                    f"hdin_{input_hd_dim}_embdin_{args.embed_dim_input}_"
                    f"npat_{n_input_patterns}.pkl")
        elif task_type == 'autoencoding':
            filename = (f"session_{session_id}_"
                    f"vth_{v_th_std:.3f}_g_{g_std:.3f}_rate_{static_input_rate:.0f}_"
                    f"hdin_{input_hd_dim}_embdin_{args.embed_dim_input}_"
                    f"npat_{n_input_patterns}.pkl")

        output_file = os.path.join(output_dir, filename)
        save_results([result], output_file, use_data_subdir=False)
        print(f"Results saved: {output_file}")
        print("=" * 80)

    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPI task-performance experiment")

    # Task configuration
    parser.add_argument("--task_type", type=str, required=True,
                       choices=["categorical", "temporal", "autoencoding"])
    parser.add_argument("--session_id", type=int, required=True)
    parser.add_argument("--n_input_patterns", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=1000)

    # Single parameter combination
    parser.add_argument("--v_th_std", type=float, required=True)
    parser.add_argument("--g_std", type=float, required=True)
    parser.add_argument("--static_input_rate", type=float, required=True)
    parser.add_argument("--input_hd_dim", type=int, required=True)
    parser.add_argument("--output_hd_dim", type=int, required=True)

    # HD embedding dimensions
    parser.add_argument("--embed_dim_input", type=int, default=10)
    parser.add_argument("--embed_dim_output", type=int, default=10)

    # Network modes
    parser.add_argument("--synaptic_mode", type=str, default="filter",
                       choices=["pulse", "filter"])
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--hd_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--hd_connection_mode", type=str, default="overlapping",
                       choices=["overlapping", "partitioned"])
    parser.add_argument("--v_th_distribution", type=str, default="normal",
                       choices=["normal", "uniform"])
    parser.add_argument("--use_distributed_cv", action="store_true")

    # Task parameters
    parser.add_argument("--signal_cache_dir", type=str, default="results/hd_signals")
    parser.add_argument("--decision_window", type=float, default=50.0)
    parser.add_argument("--stimulus_duration", type=float, default=300.0)
    parser.add_argument("--n_trials_per_pattern", type=int, default=100)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--tau_syn", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="results")

    # NEW: Caching arguments
    parser.add_argument("--use_cached_spikes", action="store_true",
                       help="Load cached evoked spikes instead of simulating")
    parser.add_argument("--spike_cache_dir", type=str, default="results/cached_spikes",
                       help="Directory with cached spike data")

    args = parser.parse_args()

    try:
        run_mpi_task_experiment(args)
    finally:
        comm = MPI.COMM_WORLD
        comm.Barrier()
        MPI.Finalize()
