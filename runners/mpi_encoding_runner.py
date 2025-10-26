# runners/mpi_encoding_runner.py
"""
MPI-parallelized encoding experiment runner.
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
    distribute_work,
    monitor_system_health,
    recovery_break,
    print_work_distribution,
    estimate_computation_time
)

# Import experiment modules
try:
    from experiments.encoding_experiment import EncodingExperiment
    from experiments.base_experiment import BaseExperiment
    from experiments.experiment_utils import save_results
    from analysis.statistics_utils import get_extreme_combinations
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    from experiments.encoding_experiment import EncodingExperiment
    from experiments.base_experiment import BaseExperiment
    from experiments.experiment_utils import save_results
    from analysis.statistics_utils import get_extreme_combinations


def execute_combination_with_recovery(experiment: EncodingExperiment, rank: int,
                                    session_id: int, v_th_std: float, g_std: float,
                                    hd_dim: int, v_th_distribution: str,
                                    static_input_rate: float,
                                    combination_index: int,
                                    extreme_combos: list) -> Dict[str, Any]:
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
                hd_dim=hd_dim,
                v_th_distribution=v_th_distribution,
                static_input_rate=static_input_rate,
                extreme_combos=extreme_combos
            )

            result.update({
                'rank': rank,
                'combination_index': combination_index,
                'attempt_count': attempt,
                'computation_time': time.time() - start_time,
                'successful_completion': True
            })

            print(f"[Rank {rank}] Success:")
            print(f"    Test RMSE: {result['decoding']['test_rmse_mean']:.4f}")
            print(f"    Test R²: {result['decoding']['test_r2_mean']:.4f}")

            return result

        except Exception as e:
            print(f"[Rank {rank}] Error (attempt {attempt}): {str(e)}")
            import traceback
            traceback.print_exc()

            if "memory" in str(e).lower():
                recovery_break(rank, 600, "memory_error")
            elif "decoding" in str(e).lower() or "svd" in str(e).lower():
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
        'hd_dim': hd_dim,
        'embed_dim': experiment.embed_dim,
        'rank': rank,
        'combination_index': combination_index,
        'decoding': {'test_rmse_mean': np.nan, 'test_r2_mean': np.nan},
        'computation_time': 0.0,
        'attempt_count': max_attempts,
        'successful_completion': False,
        'failure_reason': "Exceeded maximum attempts"
    }


def run_mpi_encoding_experiment(args):
    """Run encoding experiment with MPI parallelization."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    session_id = args.session_id
    n_neurons = args.n_neurons
    output_dir = args.output_dir
    signal_cache_dir = args.signal_cache_dir
    synaptic_mode = args.synaptic_mode
    static_input_mode = args.static_input_mode
    hd_input_mode = args.hd_input_mode
    v_th_distribution = args.v_th_distribution
    embed_dim = args.embed_dim_input

    if rank == 0:
        print("=" * 80)
        print(f"ENCODING EXPERIMENT - SESSION {session_id}")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  MPI processes: {size}")
        print(f"  Session ID: {session_id}")
        print(f"  Embed dim: {embed_dim}")

        # NEW DIRECTORY STRUCTURE
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        else:
            output_dir = os.path.join(output_dir, "data")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isabs(signal_cache_dir):
            signal_cache_dir = os.path.abspath(signal_cache_dir)
        os.makedirs(signal_cache_dir, exist_ok=True)

        print(f"  Output directory: {output_dir}")
        print(f"  Signal cache directory: {signal_cache_dir}")

    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    signal_cache_dir = comm.bcast(signal_cache_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create parameter grids
    v_th_stds, g_stds, hd_dims, static_input_rates = BaseExperiment.create_parameter_grid(
        n_v_th_points=args.n_v_th_std,
        n_g_points=args.n_g_std,
        v_th_std_range=(args.v_th_std_min, args.v_th_std_max),
        g_std_range=(args.g_std_min, args.g_std_max),
        input_rate_range=(args.static_input_rate_min, args.static_input_rate_max),
        n_input_rates=args.n_static_input_rates,
        n_hd_input_points=args.n_hd_dim_input,
        hd_input_dim_range=(args.hd_dim_input_min, args.hd_dim_input_max)
    )

    # Get extreme combinations for smart storage
    extreme_combos = get_extreme_combinations(v_th_stds, g_stds)

    # Initialize experiment
    experiment = EncodingExperiment(
        n_neurons=n_neurons,
        synaptic_mode=synaptic_mode,
        static_input_mode=static_input_mode,
        hd_input_mode=hd_input_mode,
        embed_dim=embed_dim,
        signal_cache_dir=signal_cache_dir
    )

    # Pre-generate HD signals (rank 0 only)
    if rank == 0:
        print(f"\nPre-generating HD signals...")
        for hd_dim in hd_dims:
            experiment.hd_generator.initialize_base_input(session_id, int(hd_dim))
        print("HD signal pre-generation complete.\n")

    comm.Barrier()

    # Generate parameter combinations
    all_combinations = experiment.create_parameter_combinations(
        session_id=session_id,
        v_th_stds=v_th_stds,
        g_stds=g_stds,
        static_input_rates=static_input_rates,
        v_th_distribution=v_th_distribution,
        hd_dims=hd_dims
    )

    total_combinations = len(all_combinations)

    if rank == 0:
        print(f"  Parameter grid: {args.n_v_th_std} × {args.n_g_std} × {args.n_hd_dim_input} × {args.n_static_input_rates}")
        print(f"  Total combinations: {total_combinations}")
        print(f"  HD dims: {args.hd_dim_input_min} to {args.hd_dim_input_max} ({args.n_hd_dim_input} points)")
        print(f"  Static rates: {args.static_input_rate_min} to {args.static_input_rate_max} ({args.n_static_input_rates} points)")
        print_work_distribution(total_combinations, size)

        expected_time = estimate_computation_time(total_combinations, size, 180, 20)
        print(f"\nEstimated time: {expected_time:.1f} hours")

    start_idx, end_idx = distribute_work(total_combinations, comm)
    my_combinations = all_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Processing {len(my_combinations)} combinations")

    local_results = []
    rank_start_time = time.time()

    for i, combo in enumerate(my_combinations):
        progress = (i + 1) / len(my_combinations) * 100
        print(f"[Rank {rank}] [{i+1}/{len(my_combinations)} - {progress:.1f}%]:")
        print(f"    v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}, hd={combo['hd_dim']}, rate={combo['static_input_rate']:.0f}")

        result = execute_combination_with_recovery(
            experiment=experiment,
            rank=rank,
            session_id=combo['session_id'],
            v_th_std=combo['v_th_std'],
            g_std=combo['g_std'],
            hd_dim=combo['hd_dim'],
            v_th_distribution=combo['v_th_distribution'],
            static_input_rate=combo['static_input_rate'],
            combination_index=start_idx + i,
            extreme_combos=extreme_combos
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
                                   f"encoding_session_{session_id}_{synaptic_mode}_{static_input_mode}_{hd_input_mode}_{v_th_distribution}_k{embed_dim}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)
        print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPI encoding experiment")

    parser.add_argument("--session_id", type=int, required=True)
    parser.add_argument("--n_v_th_std", type=int, default=5)
    parser.add_argument("--n_g_std", type=int, default=5)
    parser.add_argument("--n_hd_dim_input", type=int, default=4)
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--v_th_std_min", type=float, default=0.0)
    parser.add_argument("--v_th_std_max", type=float, default=4.0)
    parser.add_argument("--g_std_min", type=float, default=0.0)
    parser.add_argument("--g_std_max", type=float, default=4.0)
    parser.add_argument("--hd_dim_input_min", type=int, default=1)
    parser.add_argument("--hd_dim_input_max", type=int, default=10)
    parser.add_argument("--static_input_rate_min", type=float, default=100.0)
    parser.add_argument("--static_input_rate_max", type=float, default=500.0)
    parser.add_argument("--n_static_input_rates", type=int, default=3)
    parser.add_argument("--synaptic_mode", type=str, default="filter", choices=["pulse", "filter"])
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--hd_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--v_th_distribution", type=str, default="normal", choices=["normal", "uniform"])
    parser.add_argument("--embed_dim_input", type=int, default=10)
    parser.add_argument("--signal_cache_dir", type=str, default="hd_signals")

    args = parser.parse_args()
    run_mpi_encoding_experiment(args)
