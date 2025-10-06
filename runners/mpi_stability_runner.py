# runners/mpi_stability_runner.py
"""
MPI-parallelized network stability experiment runner.
"""

import numpy as np
import os
import sys
import time
import argparse
from mpi4py import MPI
from typing import Dict, Any

# Import shared MPI utilities
from .mpi_utils import (
    distribute_work,
    monitor_system_health,
    recovery_break,
    print_work_distribution,
    estimate_computation_time
)

# Import experiment modules
try:
    from experiments.stability_experiment import StabilityExperiment
    from experiments.base_experiment import BaseExperiment
    from experiments.experiment_utils import save_results
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, os.path.join(project_root, 'experiments'))
    from stability_experiment import StabilityExperiment
    from base_experiment import BaseExperiment
    from experiment_utils import save_results


def execute_combination_with_recovery(experiment: StabilityExperiment, rank: int,
                                    session_id: int, v_th_std: float, g_std: float,
                                    v_th_distribution: str, static_input_rate: float,
                                    combination_index: int) -> Dict[str, Any]:
    """Execute single parameter combination with recovery and monitoring."""
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        healthy, status = monitor_system_health()
        if not healthy:
            print(f"[Rank {rank}] Health issue (attempt {attempt}): {status}")
            recovery_break(rank, 300, status)
            continue

        if attempt > 1:
            print(f"[Rank {rank}] Retry attempt {attempt}")

        try:
            start_time = time.time()

            result = experiment.run_parameter_combination(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                v_th_distribution=v_th_distribution,
                static_input_rate=static_input_rate
            )

            result.update({
                'rank': rank,
                'combination_index': combination_index,
                'attempt_count': attempt,
                'computation_time': time.time() - start_time,
                'successful_completion': True
            })

            print(f"[Rank {rank}] Success:")
            print(f"    LZ (spatial): {result['lz_spatial_patterns_mean']:.2f}")
            print(f"    LZ (column): {result['lz_column_wise_mean']:.2f}")
            print(f"    Settling: {result['settling_time_ms_mean']:.1f} ms")

            return result

        except Exception as e:
            print(f"[Rank {rank}] Error (attempt {attempt}): {str(e)}")
            if "memory" in str(e).lower():
                recovery_break(rank, 600, "memory_error")
            else:
                recovery_break(rank, 300, "general_error")
            continue

    # Return failure result
    print(f"[Rank {rank}] FAILED after {max_attempts} attempts")
    return {
        'session_id': session_id,
        'v_th_std': v_th_std,
        'g_std': g_std,
        'v_th_distribution': v_th_distribution,
        'static_input_rate': static_input_rate,
        'synaptic_mode': experiment.synaptic_mode,
        'static_input_mode': experiment.static_input_mode,
        'rank': rank,
        'combination_index': combination_index,
        'lz_spatial_patterns_mean': np.nan,
        'computation_time': 0.0,
        'attempt_count': max_attempts,
        'successful_completion': False,
        'failure_reason': "Exceeded maximum attempts"
    }


def run_mpi_stability_experiment(session_id: int = 1,
                              n_v_th: int = 10, n_g: int = 10,
                              n_neurons: int = 1000, output_dir: str = "results",
                              v_th_std_min: float = 0.0, v_th_std_max: float = 4.0,
                              g_std_min: float = 0.0, g_std_max: float = 4.0,
                              input_rate_min: float = 50.0, input_rate_max: float = 1000.0,
                              n_input_rates: int = 5, synaptic_mode: str = "filter",
                              static_input_mode: str = "independent",
                              v_th_distribution: str = "normal"):
    """Run network stability experiment for single session."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 80)
        print("NETWORK STABILITY EXPERIMENT - SINGLE SESSION")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  MPI processes: {size}")
        print(f"  Session ID: {session_id}")
        print(f"  Parameter grid: {n_v_th} × {n_g} × {n_input_rates}")

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create parameter grids using base class method
    v_th_stds, g_stds, static_input_rates = BaseExperiment.create_parameter_grid(
        n_v_th_points=n_v_th,
        n_g_points=n_g,
        v_th_std_range=(v_th_std_min, v_th_std_max),
        g_std_range=(g_std_min, g_std_max),
        input_rate_range=(input_rate_min, input_rate_max),
        n_input_rates=n_input_rates
    )

    # Initialize experiment
    experiment = StabilityExperiment(
        n_neurons=n_neurons,
        synaptic_mode=synaptic_mode,
        static_input_mode=static_input_mode
    )

    # Generate parameter combinations
    all_combinations = experiment.create_parameter_combinations(
        session_id=session_id,
        v_th_stds=v_th_stds,
        g_stds=g_stds,
        static_input_rates=static_input_rates,
        v_th_distribution=v_th_distribution
    )

    total_combinations = len(all_combinations)

    if rank == 0:
        print_work_distribution(total_combinations, size)
        expected_time = estimate_computation_time(total_combinations, size, 90, 100)
        print(f"\nEstimated total time: {expected_time:.1f} hours")

    start_idx, end_idx = distribute_work(total_combinations, comm)
    my_combinations = all_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Processing {len(my_combinations)} combinations")

    local_results = []
    rank_start_time = time.time()

    for i, combo in enumerate(my_combinations):
        progress = (i + 1) / len(my_combinations) * 100
        print(f"[Rank {rank}] [{i+1}/{len(my_combinations)} - {progress:.1f}%]:")
        print(f"    v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}, rate={combo['static_input_rate']:.0f}")

        result = execute_combination_with_recovery(
            experiment=experiment,
            rank=rank,
            session_id=combo['session_id'],
            v_th_std=combo['v_th_std'],
            g_std=combo['g_std'],
            v_th_distribution=combo['v_th_distribution'],
            static_input_rate=combo['static_input_rate'],
            combination_index=start_idx + i
        )

        result['original_combination_index'] = combo['combo_idx']
        local_results.append(result)

    rank_total_time = time.time() - rank_start_time
    successful_local = [r for r in local_results if r.get('successful_completion', False)]
    print(f"[Rank {rank}] COMPLETED: {len(successful_local)}/{len(local_results)} successful ({rank_total_time/3600:.2f}h)")

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        final_results = []
        for proc_results in all_results:
            final_results.extend(proc_results)
        final_results.sort(key=lambda x: x['original_combination_index'])

        output_file = os.path.join(output_dir,
                                   f"stability_session_{session_id}_{synaptic_mode}_{static_input_mode}_{v_th_distribution}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)
        print(f"\nStability results saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPI network stability experiment")
    parser.add_argument("--session_id", type=int, default=1)
    parser.add_argument("--n_v_th", type=int, default=10)
    parser.add_argument("--n_g", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--v_th_std_min", type=float, default=0.0)
    parser.add_argument("--v_th_std_max", type=float, default=4.0)
    parser.add_argument("--g_std_min", type=float, default=0.0)
    parser.add_argument("--g_std_max", type=float, default=4.0)
    parser.add_argument("--input_rate_min", type=float, default=50.0)
    parser.add_argument("--input_rate_max", type=float, default=1000.0)
    parser.add_argument("--n_input_rates", type=int, default=5)
    parser.add_argument("--synaptic_mode", type=str, default="filter", choices=["pulse", "filter"])
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--v_th_distribution", type=str, default="normal", choices=["normal", "uniform"])

    args = parser.parse_args()
    run_mpi_stability_experiment(**vars(args))
