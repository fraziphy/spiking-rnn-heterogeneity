# runners/mpi_autoencoding_runner.py
"""
MPI-parallelized auto-encoding experiment runner.
Same as temporal task but reconstructs INPUT instead of separate output.
"""

import numpy as np
import os
import sys
import time
import argparse
from mpi4py import MPI
from typing import Dict, Any

# Import shared MPI utilities
from mpi_utils import (
    monitor_system_health,
    recovery_break
)

# Import experiment modules
try:
    from experiments.task_performance_experiment import TaskPerformanceExperiment
    from experiments.experiment_utils import save_results
    from analysis.common_utils import spikes_to_binary, compute_dimensionality_svd, apply_exponential_filter
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    from experiments.task_performance_experiment import TaskPerformanceExperiment
    from experiments.experiment_utils import save_results
    from analysis.common_utils import spikes_to_binary, compute_dimensionality_svd, apply_exponential_filter


def run_mpi_autoencoding_experiment(args):
    """Run auto-encoding experiment with distributed trials and CV."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Extract parameters (single combination now)
    session_id = args.session_id
    n_neurons = args.n_neurons
    output_dir = args.output_dir
    signal_cache_dir = args.signal_cache_dir

    # Single parameter combination
    v_th_std = args.v_th_std
    g_std = args.g_std
    input_hd_dim = args.input_hd_dim
    static_input_rate = args.static_input_rate

    # Task parameters
    synaptic_mode = args.synaptic_mode
    static_input_mode = args.static_input_mode
    hd_input_mode = args.hd_input_mode
    v_th_distribution = args.v_th_distribution

    n_input_patterns = args.n_input_patterns
    n_trials_per_pattern = args.n_trials_per_pattern
    n_cv_folds = 20  # FIXED at 20

    if rank == 0:
        print("=" * 80)
        print(f"AUTO-ENCODING TASK - SESSION {session_id}")
        print(f"Parameters: v_th={v_th_std:.3f}, g={g_std:.3f}, rate={static_input_rate:.0f}")
        print(f"            hd_dim={input_hd_dim} (output=input)")
        print("=" * 80)
        print(f"MPI processes: {size}")
        print(f"Total trials: {n_input_patterns} × {n_trials_per_pattern} = {n_input_patterns * n_trials_per_pattern}")
        print(f"CV folds: {n_cv_folds}")
        print(f"CV iterations per rank: {n_cv_folds // size}")

        # Setup directories
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isabs(signal_cache_dir):
            signal_cache_dir = os.path.abspath(signal_cache_dir)
        os.makedirs(signal_cache_dir, exist_ok=True)

    # Broadcast directories
    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    signal_cache_dir = comm.bcast(signal_cache_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create experiment instance (use temporal infrastructure)
    experiment = TaskPerformanceExperiment(
        task_type='temporal',  # Use temporal infrastructure
        n_neurons=n_neurons,
        n_input_patterns=n_input_patterns,
        input_dim_intrinsic=input_hd_dim,
        input_dim_embedding=args.embed_dim_input,
        output_dim_intrinsic=input_hd_dim,  # Same as input!
        output_dim_embedding=args.embed_dim_input,  # Same as input!
        dt=args.dt,
        tau_syn=args.tau_syn,
        synaptic_mode=synaptic_mode,
        static_input_mode=static_input_mode,
        hd_input_mode=hd_input_mode,
        signal_cache_dir=signal_cache_dir,
        decision_window=args.decision_window,
        stimulus_duration=args.stimulus_duration,
        n_trials_per_pattern=n_trials_per_pattern,
        lambda_reg=args.lambda_reg,
        use_distributed_cv=args.use_distributed_cv
    )

    if rank == 0:
        print("\nStep 1: Generating HD patterns...")

    # All ranks generate/load the same patterns (deterministic)
    input_patterns = experiment.input_generator.initialize_and_get_patterns(
        session_id, input_hd_dim, n_input_patterns
    )

    # KEY CHANGE: Auto-encoding uses input as output!
    output_patterns = input_patterns

    if rank == 0:
        print(f"Step 2: Distributing {n_input_patterns * n_trials_per_pattern} trials across {size} ranks...")

    # Distribute trials
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

    # Step 2: Simulate trials in parallel
    local_spike_times = experiment.simulate_trials_parallel(
        session_id=session_id,
        v_th_std=v_th_std,
        g_std=g_std,
        v_th_distribution=v_th_distribution,
        static_input_rate=static_input_rate,
        my_trial_indices=my_trials,
        input_patterns=input_patterns,
        rank=rank
    )

    if rank == 0:
        print(f"Step 3: Each rank computing dimensionality for its trials...")

    # Define bin sizes once
    bin_sizes = [2.0, 10.0, 20.0]  # ms

    # Each rank computes dimensionality for ITS OWN trials only (PARALLEL!)
    local_trial_dimensionality = []

    for trial_result in local_spike_times:
        trial_dim = {
            'trial_id': trial_result['trial_id'],
            'pattern_id': trial_result['pattern_id'],
            'global_trial_idx': trial_result['global_trial_idx']
        }

        for bin_size in bin_sizes:
            # Convert spikes to binary matrix
            binary_matrix = spikes_to_binary(
                trial_result['spike_times'],
                n_neurons,
                experiment.stimulus_duration,
                bin_size
            )

            # Compute dimensionality metrics
            dim_metrics = compute_dimensionality_svd(
                binary_matrix,
                variance_threshold=0.95
            )

            # Store with bin size label
            trial_dim[f'bin_{bin_size}ms'] = dim_metrics

        local_trial_dimensionality.append(trial_dim)

    # Gather dimensionality metrics to rank 0 (much smaller than spike times!)
    all_trial_dimensionality = comm.gather(local_trial_dimensionality, root=0)

    if rank == 0:
        # Flatten gathered dimensionality
        trial_dimensionality = []
        for rank_dims in all_trial_dimensionality:
            trial_dimensionality.extend(rank_dims)

        print(f"   Dimensionality computed for {len(trial_dimensionality)} trials")

    # Step 4: Different data flow for distributed vs centralized CV
    if experiment.use_distributed_cv:
        # === DISTRIBUTED CV PATH ===
        if rank == 0:
            print(f"Step 4: Broadcasting spike times to all ranks (distributed CV)...")

        # Simple allgather of raw spike data
        all_spike_times = comm.allgather(local_spike_times)

        # Flatten list of lists
        complete_spike_times = []
        for rank_spikes in all_spike_times:
            complete_spike_times.extend(rank_spikes)

        if rank == 0:
            print(f"   Collected {len(complete_spike_times)} trial spike times")
            print(f"Step 5: Each rank converts spike times to traces locally...")

        # ALL ranks convert to traces
        traces_all, ground_truth_all, pattern_ids = experiment.convert_spikes_to_traces(
            complete_spike_times,
            output_patterns,
            n_input_patterns,
            n_trials_per_pattern
        )

        if rank == 0:
            print(f"   Traces shape: {traces_all.shape}")
            print(f"Step 6: Distributed CV training...")

        # Clean up
        del local_spike_times
        del all_spike_times
        del complete_spike_times
        import gc
        gc.collect()

        # ALL ranks do CV
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

        # ALL ranks clean up
        del traces_all
        del ground_truth_all
        del pattern_ids
        gc.collect()

    else:
        # === CENTRALIZED CV PATH ===
        if rank == 0:
            print(f"Step 4: Rank 0 gathering spike times from all ranks...")

        # Convert to NumPy arrays
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

        # ONLY rank 0 gets data
        all_spike_arrays = comm.gather(local_spike_arrays, root=0)

        # ONLY rank 0 processes
        if rank == 0:
            complete_spike_times = []
            for rank_arrays in all_spike_arrays:
                for trial_data in rank_arrays:
                    spike_array = trial_data['spike_array']
                    spike_list = [(float(s['time']), int(s['neuron_id'])) for s in spike_array]
                    complete_spike_times.append({
                        'pattern_id': trial_data['pattern_id'],
                        'trial_id': trial_data['trial_id'],
                        'global_trial_idx': trial_data['global_trial_idx'],
                        'spike_times': spike_list
                    })

            print(f"   Collected {len(complete_spike_times)} trial spike times")
            print(f"Step 5: Rank 0 converts spike times to traces...")

            # Rank 0 converts to traces
            traces_all, ground_truth_all, pattern_ids = experiment.convert_spikes_to_traces(
                complete_spike_times,
                output_patterns,
                n_input_patterns,
                n_trials_per_pattern
            )

            print(f"   Traces shape: {traces_all.shape}")
            print(f"Step 6: Centralized CV training...")

            # Clean up
            del complete_spike_times
            del all_spike_arrays
        else:
            traces_all = None
            ground_truth_all = None
            pattern_ids = None

        # Clean up
        del local_spike_times
        del local_spike_arrays
        import gc
        gc.collect()

        # ONLY rank 0 does CV
        if rank == 0:
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

            # Clean up
            del traces_all
            del ground_truth_all
            del pattern_ids
        else:
            cv_results = {}

        gc.collect()

    # Only rank 0 compiles and saves results
    if rank == 0:
        print(f"Step 7: Saving results...")

        # Aggregate dimensionality statistics (mean and std across trials)
        dim_summary = {}
        for bin_size in bin_sizes:
            bin_key = f'bin_{bin_size}ms'

            # Extract metrics across all trials
            pr_values = [trial[bin_key]['participation_ratio'] for trial in trial_dimensionality]
            eff_dim_values = [trial[bin_key]['effective_dimensionality'] for trial in trial_dimensionality]
            intr_dim_values = [trial[bin_key]['intrinsic_dimensionality'] for trial in trial_dimensionality]

            dim_summary[bin_key] = {
                'participation_ratio_mean': float(np.mean(pr_values)),
                'participation_ratio_std': float(np.std(pr_values)),
                'effective_dimensionality_mean': float(np.mean(eff_dim_values)),
                'effective_dimensionality_std': float(np.std(eff_dim_values)),
                'intrinsic_dimensionality_mean': float(np.mean(intr_dim_values)),
                'intrinsic_dimensionality_std': float(np.std(intr_dim_values)),
                'n_trials': len(pr_values)
            }

        result = {
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'input_hd_dim': input_hd_dim,
            'output_hd_dim': input_hd_dim,
            'static_input_rate': static_input_rate,
            'v_th_distribution': v_th_distribution,
            'task_type': 'auto_encoding',
            'synaptic_mode': synaptic_mode,
            'static_input_mode': static_input_mode,
            'hd_input_mode': hd_input_mode,
            'n_input_patterns': n_input_patterns,
            'n_trials_per_pattern': n_trials_per_pattern,
            'n_cv_folds': n_cv_folds,
            'tau_syn': args.tau_syn,
            'decision_window': args.decision_window,
            'dimensionality_summary': dim_summary,  # <-- ADD THIS
            **cv_results
        }

        # Filename: one file per session per parameter combination
        filename = (f"task_autoencoding_session_{session_id}_"
                   f"vth_{v_th_std:.3f}_g_{g_std:.3f}_rate_{static_input_rate:.0f}_"
                   f"hd_{input_hd_dim}.pkl")

        output_file = os.path.join(output_dir, filename)
        save_results([result], output_file, use_data_subdir=False)
        print(f"Results saved: {output_file}")
        print("=" * 80)

    # Synchronize all ranks before exit
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPI auto-encoding experiment")

    # Task configuration
    parser.add_argument("--session_id", type=int, required=True)
    parser.add_argument("--n_input_patterns", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=1000)

    # Single parameter combination (not grid!)
    parser.add_argument("--v_th_std", type=float, required=True)
    parser.add_argument("--g_std", type=float, required=True)
    parser.add_argument("--static_input_rate", type=float, required=True)
    parser.add_argument("--input_hd_dim", type=int, required=True)

    # HD embedding dimensions (output = input)
    parser.add_argument("--embed_dim_input", type=int, default=10)

    # Network modes
    parser.add_argument("--synaptic_mode", type=str, default="filter", choices=["pulse", "filter"])
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--hd_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--v_th_distribution", type=str, default="normal", choices=["normal", "uniform"])
    parser.add_argument("--use_distributed_cv", action="store_true",
                   help="Use distributed CV (default: centralized to save RAM)")

    # Task parameters
    parser.add_argument("--signal_cache_dir", type=str, default="hd_signals")
    parser.add_argument("--decision_window", type=float, default=50.0)
    parser.add_argument("--stimulus_duration", type=float, default=300.0)
    parser.add_argument("--n_trials_per_pattern", type=int, default=100)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--tau_syn", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    try:
        run_mpi_autoencoding_experiment(args)
    finally:
        # Ensure clean MPI shutdown
        comm = MPI.COMM_WORLD
        comm.Barrier()  # Wait for all ranks
        MPI.Finalize()  # Clean MPI shutdown
