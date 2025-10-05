# runners/mpi_spontaneous_runner.py - MPI runner (refactored with shared utilities)
"""
MPI-parallelized spontaneous activity experiment runner with extended dimensionality analysis.
"""

import numpy as np
import os
import sys
import time
import pickle
import argparse
from mpi4py import MPI
from typing import List, Dict, Any, Tuple

# Import shared MPI utilities
from mpi_utils import (
    distribute_work,
    distribute_work_for_rank,
    monitor_system_health,
    recovery_break,
    print_work_distribution,
    estimate_computation_time
)

# Import modules with path handling
try:
    from experiments.spontaneous_experiment import SpontaneousExperiment, create_parameter_grid, save_results
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    experiments_dir = os.path.join(project_root, 'experiments')
    sys.path.insert(0, experiments_dir)
    from spontaneous_experiment import SpontaneousExperiment, create_parameter_grid, save_results


def execute_combination_with_recovery(experiment: SpontaneousExperiment, rank: int,
                                    session_id: int, v_th_std: float, g_std: float,
                                    v_th_distribution: str, static_input_rate: float,
                                    combination_index: int, duration: float) -> Dict[str, Any]:
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
                static_input_rate=static_input_rate,
                duration=duration
            )

            result.update({
                'rank': rank,
                'combination_index': combination_index,
                'attempt_count': attempt,
                'computation_time': time.time() - start_time,
                'successful_completion': True
            })

            # Log spontaneous activity results
            print(f"[Rank {rank}] Success:")
            print(f"    Mean firing rate: {result['mean_firing_rate_mean']:.2f} Hz")
            print(f"    Silent neurons: {result['percent_silent_mean']:.1f}%")
            print(f"    Dimensionality (5ms): {result['effective_dimensionality_bin_5.0ms_mean']:.1f}")

            return result

        except Exception as e:
            print(f"[Rank {rank}] Error (attempt {attempt}): {str(e)}")
            if "memory" in str(e).lower():
                recovery_break(rank, 600, "memory_error")
            elif "dimensionality" in str(e).lower() or "firing" in str(e).lower():
                recovery_break(rank, 300, "analysis_error")
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
        'duration': duration,
        'synaptic_mode': experiment.synaptic_mode,
        'static_input_mode': experiment.static_input_mode,
        'rank': rank,
        'combination_index': combination_index,
        'mean_firing_rate_mean': np.nan,
        'percent_silent_mean': np.nan,
        'computation_time': 0.0,
        'attempt_count': max_attempts,
        'successful_completion': False,
        'failure_reason': "Exceeded maximum attempts"
    }


def run_mpi_spontaneous_experiment(session_id: int = 1,
                                 n_v_th: int = 10, n_g: int = 10,
                                 n_neurons: int = 1000, output_dir: str = "results",
                                 v_th_std_min: float = 0.0, v_th_std_max: float = 4.0,
                                 g_std_min: float = 0.0, g_std_max: float = 4.0,
                                 input_rate_min: float = 50.0, input_rate_max: float = 1000.0,
                                 n_input_rates: int = 5, synaptic_mode: str = "filter",
                                 static_input_mode: str = "independent",
                                 v_th_distribution: str = "normal",
                                 duration: float = 5000.0):
    """Run spontaneous activity experiment for single session."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert duration from seconds to ms if needed
    if duration < 100:
        duration_ms = duration * 1000.0
        print(f"[Rank {rank}] Converting duration from {duration}s to {duration_ms}ms")
        duration = duration_ms

    if rank == 0:
        print("=" * 80)
        print("SPONTANEOUS ACTIVITY EXPERIMENT - SINGLE SESSION")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  MPI processes: {size}")
        print(f"  Session ID: {session_id}")
        print(f"  Parameter grid: {n_v_th} × {n_g} × {n_input_rates}")
        print(f"  Network size: {n_neurons} neurons")
        print(f"  Simulation duration: {duration:.0f} ms ({duration/1000:.1f} s)")

        # Setup output directory
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    # Broadcast output directory to all ranks
    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create parameter grids
    v_th_stds, g_stds, static_input_rates = create_parameter_grid(
        n_v_th_points=n_v_th,
        n_g_points=n_g,
        v_th_std_range=(v_th_std_min, v_th_std_max),
        g_std_range=(g_std_min, g_std_max),
        input_rate_range=(input_rate_min, input_rate_max),
        n_input_rates=n_input_rates
    )

    # Generate all parameter combinations
    import random
    param_combinations = []
    combo_idx = 0
    for input_rate in static_input_rates:
        for v_th_std in v_th_stds:
            for g_std in g_stds:
                param_combinations.append({
                    'combo_idx': combo_idx,
                    'session_id': session_id,
                    'v_th_std': v_th_std,
                    'g_std': g_std,
                    'v_th_distribution': v_th_distribution,
                    'static_input_rate': input_rate,
                    'duration': duration
                })
                combo_idx += 1

    # RANDOMIZE job order
    random.shuffle(param_combinations)

    total_combinations = len(param_combinations)

    if rank == 0:
        print_work_distribution(total_combinations, size)
        expected_time = estimate_computation_time(total_combinations, size, duration/1000.0 * 0.5, 10)
        print(f"\nEstimated total time: {expected_time:.1f} hours")

    # Distribute work among ranks
    start_idx, end_idx = distribute_work(total_combinations, comm)
    my_combinations = param_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Processing {len(my_combinations)} combinations")

    # Initialize spontaneous experiment
    experiment = SpontaneousExperiment(n_neurons=n_neurons, synaptic_mode=synaptic_mode,
                                      static_input_mode=static_input_mode)

    # Execute assigned combinations
    local_results = []
    rank_start_time = time.time()

    for i, combo in enumerate(my_combinations):
        progress = (i + 1) / len(my_combinations) * 100
        elapsed_hours = (time.time() - rank_start_time) / 3600

        print(f"[Rank {rank}] [{i+1}/{len(my_combinations)} - {progress:.1f}%]:")
        print(f"    v_th_std={combo['v_th_std']:.3f}, g_std={combo['g_std']:.3f}")

        result = execute_combination_with_recovery(
            experiment=experiment,
            rank=rank,
            session_id=combo['session_id'],
            v_th_std=combo['v_th_std'],
            g_std=combo['g_std'],
            v_th_distribution=combo['v_th_distribution'],
            static_input_rate=combo['static_input_rate'],
            combination_index=start_idx + i,
            duration=combo['duration']
        )

        result['original_combination_index'] = combo['combo_idx']
        local_results.append(result)

    # Final local report
    rank_total_time = time.time() - rank_start_time
    successful_local = [r for r in local_results if r.get('successful_completion', False)]

    print(f"[Rank {rank}] COMPLETED: {len(successful_local)}/{len(local_results)} successful ({rank_total_time/3600:.2f}h)")

    # Gather results at root
    all_results = comm.gather(local_results, root=0)

    # Process and save results (root rank only)
    if rank == 0:
        print(f"\nProcessing spontaneous activity results from all {size} ranks...")

        # Combine results from all ranks
        final_results = []
        for proc_results in all_results:
            final_results.extend(proc_results)

        # Sort by combination index
        final_results.sort(key=lambda x: x['original_combination_index'])

        # Save results
        duration_sec = duration / 1000.0
        output_file = os.path.join(output_dir,
                                   f"spontaneous_session_{session_id}_{synaptic_mode}_{static_input_mode}_{v_th_distribution}_{duration_sec:.1f}s.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        print(f"\nSpontaneous activity results saved: {output_file}")
        print("Spontaneous activity single session experiment completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MPI spontaneous activity experiment"
    )

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
    parser.add_argument("--duration", type=float, default=5.0)

    args = parser.parse_args()

    run_mpi_spontaneous_experiment(
        session_id=args.session_id,
        n_v_th=args.n_v_th,
        n_g=args.n_g,
        n_neurons=args.n_neurons,
        output_dir=args.output_dir,
        v_th_std_min=args.v_th_std_min,
        v_th_std_max=args.v_th_std_max,
        g_std_min=args.g_std_min,
        g_std_max=args.g_std_max,
        input_rate_min=args.input_rate_min,
        input_rate_max=args.input_rate_max,
        n_input_rates=args.n_input_rates,
        synaptic_mode=args.synaptic_mode,
        static_input_mode=args.static_input_mode,
        v_th_distribution=args.v_th_distribution,
        duration=args.duration
    )
