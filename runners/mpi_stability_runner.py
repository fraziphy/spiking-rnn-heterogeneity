# runners/mpi_stability_runner.py
"""
MPI-parallelized network stability experiment runner.
MODIFIED: Simplified for sweep usage - single parameter combination per invocation.
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
    from experiments.stability_experiment import StabilityExperiment
    from experiments.experiment_utils import save_results
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    from experiments.stability_experiment import StabilityExperiment
    from experiments.experiment_utils import save_results


def execute_with_recovery(experiment: StabilityExperiment, rank: int,
                          session_id: int, v_th_std: float, g_std: float,
                          v_th_distribution: str, static_input_rate: float) -> Dict[str, Any]:
    """Execute single parameter combination with recovery and monitoring."""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        healthy, status = monitor_system_health()
        if not healthy:
            print(f"[Rank {rank}] Health issue (attempt {attempt}): {status}")
            recovery_break(rank, 180, status)
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
                recovery_break(rank, 300, "memory_error")
            else:
                recovery_break(rank, 180, "general_error")
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
        'lz_spatial_patterns_mean': np.nan,
        'computation_time': 0.0,
        'attempt_count': max_attempts,
        'successful_completion': False,
        'failure_reason': "Exceeded maximum attempts"
    }


def run_mpi_stability_experiment(args):
    """Run network stability experiment - single parameter combination."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Single parameter combination (not a grid!)
    session_id = args.session_id
    v_th_std = args.v_th_std
    g_std = args.g_std
    static_input_rate = args.static_input_rate
    
    n_neurons = args.n_neurons
    output_dir = args.output_dir
    synaptic_mode = args.synaptic_mode
    static_input_mode = args.static_input_mode
    v_th_distribution = args.v_th_distribution

    if rank == 0:
        print("=" * 80)
        print(f"NETWORK STABILITY - SESSION {session_id}")
        print(f"Parameters: v_th={v_th_std:.3f}, g={g_std:.3f}, rate={static_input_rate:.0f}")
        print("=" * 80)
        print(f"MPI processes: {size}")
        print(f"Duration: 800 ms (500ms pre-perturbation + 300ms post-perturbation)")

        # Setup directories
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        else:
            output_dir = os.path.join(output_dir, "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Initialize experiment
    experiment = StabilityExperiment(
        n_neurons=n_neurons,
        synaptic_mode=synaptic_mode,
        static_input_mode=static_input_mode
    )

    # Run the single parameter combination (rank 0 only for simplicity)
    if rank == 0:
        result = execute_with_recovery(
            experiment=experiment,
            rank=rank,
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            v_th_distribution=v_th_distribution,
            static_input_rate=static_input_rate
        )

        # Save result
        output_file = os.path.join(
            output_dir,
            f"stability_session_{session_id}_vth_{v_th_std:.3f}_g_{g_std:.3f}_rate_{static_input_rate:.0f}.pkl"
        )
        save_results([result], output_file, use_data_subdir=False)
        print(f"\nResults saved: {output_file}")
        print("=" * 80)

    # Synchronize all ranks before exit
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPI network stability experiment")

    # Single parameter combination (not grid!)
    parser.add_argument("--session_id", type=int, required=True)
    parser.add_argument("--v_th_std", type=float, required=True)
    parser.add_argument("--g_std", type=float, required=True)
    parser.add_argument("--static_input_rate", type=float, required=True)

    # Network configuration
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/stability_sweep")
    parser.add_argument("--synaptic_mode", type=str, default="filter", choices=["pulse", "filter"])
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"])
    parser.add_argument("--v_th_distribution", type=str, default="normal", choices=["normal", "uniform"])

    args = parser.parse_args()
    run_mpi_stability_experiment(args)
