# runners/mpi_stability_runner.py
"""
MPI-parallelized stability experiment runner.
MODIFIED: Uses cached transient states by default.
"""

import numpy as np
import os
import sys
import argparse
from mpi4py import MPI
from typing import Dict, Any

from mpi_utils import monitor_system_health, recovery_break

try:
    from experiments.stability_experiment import StabilityExperiment
    from experiments.experiment_utils import save_results
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    from experiments.stability_experiment import StabilityExperiment
    from experiments.experiment_utils import save_results


def run_mpi_stability_experiment(args):
    """
    Run stability experiment with MPI parallelization.

    Uses cached transient states by default (use_cached_transients=True).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Extract parameters
    session_id = args.session_id
    n_neurons = args.n_neurons
    output_dir = args.output_dir
    transient_cache_dir = args.transient_cache_dir
    use_cached_transients = args.use_cached_transients

    v_th_std = args.v_th_std
    g_std = args.g_std
    static_input_rate = args.static_input_rate

    synaptic_mode = args.synaptic_mode
    static_input_mode = args.static_input_mode
    v_th_distribution = args.v_th_distribution

    n_perturbation_trials = args.n_perturbation_trials

    if rank == 0:
        print("=" * 80)
        print(f"STABILITY EXPERIMENT - SESSION {session_id}")
        print(f"Parameters: v_th={v_th_std:.3f}, g={g_std:.3f}, rate={static_input_rate:.0f}")
        print(f"            cached_transients={use_cached_transients}")
        print("=" * 80)
        print(f"MPI processes: {size}")
        print(f"Total perturbation trials: {n_perturbation_trials}")

        # Setup directories
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        else:
            output_dir = os.path.join(output_dir, "data")
        os.makedirs(output_dir, exist_ok=True)

    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create experiment
    experiment = StabilityExperiment(
        n_neurons=n_neurons,
        dt=args.dt,
        synaptic_mode=synaptic_mode,
        static_input_mode=static_input_mode,
        transient_cache_dir=transient_cache_dir,
        use_cached_transients=use_cached_transients
    )

    if rank == 0:
        print(f"\nDistributing {n_perturbation_trials} trials across {size} ranks...")

    # Distribute trials across ranks
    trials_per_rank = n_perturbation_trials // size
    remainder = n_perturbation_trials % size

    if rank < remainder:
        start_trial = rank * (trials_per_rank + 1)
        end_trial = start_trial + trials_per_rank + 1
    else:
        start_trial = rank * trials_per_rank + remainder
        end_trial = start_trial + trials_per_rank

    my_trials = list(range(start_trial, end_trial))

    if rank == 0:
        print(f"   Rank 0: trials {my_trials[0]}-{my_trials[-1]} ({len(my_trials)} trials)")

    # Run assigned trials
    local_results = []
    for i, trial_id in enumerate(my_trials):
        result = experiment.run_single_perturbation(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            v_th_distribution=v_th_distribution,
            perturbation_neuron_idx=trial_id,
            static_input_rate=static_input_rate
        )
        local_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"   Rank {rank}: completed {i+1}/{len(my_trials)} trials")

    # Gather all results to rank 0
    if rank == 0:
        print(f"\nGathering all results to rank 0...")

    all_results = comm.gather(local_results, root=0)

    # Aggregate and save results
    if rank == 0:
        # Flatten results
        results_all_trials = [result for rank_results in all_results for result in rank_results]

        print(f"Aggregating {len(results_all_trials)} trial results...")

        # Aggregate
        aggregated = experiment._aggregate_trial_results(results_all_trials)
        aggregated['session_id'] = session_id
        aggregated['v_th_std'] = v_th_std
        aggregated['g_std'] = g_std
        aggregated['static_input_rate'] = static_input_rate
        aggregated['v_th_distribution'] = v_th_distribution
        aggregated['synaptic_mode'] = synaptic_mode
        aggregated['static_input_mode'] = static_input_mode
        aggregated['n_trials'] = n_perturbation_trials
        aggregated['used_cached_transients'] = use_cached_transients

        # Save results
        filename = (f"stability_session_{session_id}_"
                   f"vth_{v_th_std:.3f}_g_{g_std:.3f}_"
                   f"rate_{static_input_rate:.0f}.pkl")

        output_file = os.path.join(output_dir, filename)
        save_results([aggregated], output_file, use_data_subdir=False)
        print(f"Results saved: {output_file}")
        print("=" * 80)

    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPI stability experiment")

    # Experiment configuration
    parser.add_argument("--session_id", type=int, required=True)
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--n_perturbation_trials", type=int, default=100)

    # Single parameter combination
    parser.add_argument("--v_th_std", type=float, required=True)
    parser.add_argument("--g_std", type=float, required=True)
    parser.add_argument("--static_input_rate", type=float, required=True)

    # Network modes
    parser.add_argument("--synaptic_mode", type=str, default="filter",
                       choices=["pulse", "filter"])
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--v_th_distribution", type=str, default="normal",
                       choices=["normal", "uniform"])

    # Timing
    parser.add_argument("--dt", type=float, default=0.1)

    # Directories
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--transient_cache_dir", type=str, default="results/cached_states",
                       help="Directory with cached transient states")

    # Caching
    parser.add_argument("--use_cached_transients", action="store_true", default=True,
                       help="Use cached transient states (default: True)")
    parser.add_argument("--no_cached_transients", dest="use_cached_transients",
                       action="store_false",
                       help="Simulate transients from scratch (legacy mode)")

    args = parser.parse_args()

    try:
        run_mpi_stability_experiment(args)
    finally:
        comm = MPI.COMM_WORLD
        comm.Barrier()
        MPI.Finalize()
