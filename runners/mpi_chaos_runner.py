# runners/mpi_chaos_runner.py - Modified version with input rate support
"""
MPI-parallelized chaos experiment runner with configurable static input rates.
"""

import numpy as np
import os
import sys
import time
import pickle
import argparse
from mpi4py import MPI
from typing import List, Dict, Any, Tuple

# Import modules with flexible handling
try:
    from experiments.chaos_experiment import ChaosExperiment, create_parameter_grid_with_input_rates, save_results
except ImportError:
    try:
        from ..experiments.chaos_experiment import ChaosExperiment, create_parameter_grid_with_input_rates, save_results
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        experiments_dir = os.path.join(project_root, 'experiments')
        sys.path.insert(0, experiments_dir)
        from chaos_experiment import ChaosExperiment, create_parameter_grid_with_input_rates, save_results

def run_mpi_chaos_experiment_with_input_rates(session_id: int = 1, n_v_th: int = 10, n_g: int = 10,
                                            n_neurons: int = 1000, output_dir: str = "results",
                                            input_rate_min: float = 50.0, input_rate_max: float = 500.0,
                                            n_input_rates: int = 5):
    """
    Run chaos experiment with MPI parallelization across input rates and heterogeneity parameters.

    Args:
        session_id: Session ID for random number generation
        n_v_th: Number of v_th_std values to test
        n_g: Number of g_std values to test
        n_neurons: Number of neurons in network
        output_dir: Output directory for results
        input_rate_min: Minimum static input rate (Hz)
        input_rate_max: Maximum static input rate (Hz)
        n_input_rates: Number of input rate values to test
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"🧠 Spiking RNN Chaos Experiment - MPI Parallel with Input Rate Sweep")
        print(f"=" * 70)
        print(f"MPI processes: {size}")
        print(f"Session ID: {session_id}")
        print(f"Parameter grid: {n_v_th} × {n_g} × {n_input_rates} = {n_v_th * n_g * n_input_rates} combinations")
        print(f"Network size: {n_neurons} neurons")
        print(f"Input rate range: {input_rate_min}-{input_rate_max} Hz ({n_input_rates} values)")

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Estimate experiment duration
        total_combinations = n_v_th * n_g * n_input_rates
        estimated_duration = estimate_experiment_duration(total_combinations)
        if estimated_duration < 0:
            print("Experiment cancelled by user - exiting safely")
            return

    # Synchronize output directory across all processes
    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    if rank == 0:
        print(f"\n🚀 Initializing parallel experiment execution...")

    # Create parameter grids with input rates
    v_th_std_values, g_std_values, static_input_rates = create_parameter_grid_with_input_rates(
        n_points=max(n_v_th, n_g),
        input_rate_range=(input_rate_min, input_rate_max),
        n_input_rates=n_input_rates
    )

    # Adjust grid sizes
    if len(v_th_std_values) != n_v_th:
        v_th_std_values = np.linspace(0.05, 0.5, n_v_th)
    if len(g_std_values) != n_g:
        g_std_values = np.linspace(0.05, 0.5, n_g)

    # Generate all parameter combinations including input rates
    param_combinations = []
    block_id = 0
    for input_rate in static_input_rates:
        for v_th_std in v_th_std_values:
            for g_std in g_std_values:
                param_combinations.append((block_id, v_th_std, g_std, input_rate))
                block_id += 1

    total_jobs = len(param_combinations)

    # Display work distribution
    if rank == 0:
        print(f"\n📋 Work distribution across {size} processes:")
        for r in range(size):
            s, e = distribute_work_for_rank(total_jobs, r, size)
            n_jobs = e - s
            print(f"   Process {r:2d}: {n_jobs:3d} combinations (indices {s:3d}-{e-1:3d})")

    # Distribute work among processes
    start_idx, end_idx = distribute_work(total_jobs, comm)
    my_combinations = param_combinations[start_idx:end_idx]

    if rank == 0:
        print(f"\n🔬 Starting parallel execution with input rate analysis...")

    # Initialize chaos experiment
    experiment = ChaosExperiment(n_neurons=n_neurons)

    # Execute assigned parameter combinations
    local_results = []
    last_health_check = time.time()
    experiment_start_time = time.time()

    for i, (block_id, v_th_std, g_std, input_rate) in enumerate(my_combinations):
        # Progress reporting
        if rank == 0 or i % max(1, len(my_combinations)//5) == 0:
            elapsed_hours = (time.time() - experiment_start_time) / 3600
            progress = (i + 1) / len(my_combinations) * 100
            print(f"Process {rank}: [{i+1}/{len(my_combinations)} - {progress:.1f}%] "
                  f"input_rate={input_rate:.0f}Hz, v_th={v_th_std:.3f}, g_std={g_std:.3f} - {elapsed_hours:.1f}h elapsed")

        # System health monitoring
        if time.time() - last_health_check > 300:  # Every 5 minutes
            healthy, status = monitor_system_health()
            if rank == 0:
                print(f"🌡️  Health check: {status}")
            if not healthy:
                if rank == 0:
                    print(f"⚠ EMERGENCY STOP: {status}")
                break
            last_health_check = time.time()

        # Execute parameter combination with input rate
        computation_start = time.time()

        try:
            result = experiment.run_parameter_combination(
                session_id=session_id,
                block_id=block_id,
                v_th_std=v_th_std,
                g_std=g_std,
                static_input_rate=input_rate  # Pass input rate to experiment
            )

            # Add metadata
            result['block_id'] = block_id
            result['rank'] = rank
            result['combination_index'] = start_idx + i

            local_results.append(result)

            # Save intermediate results periodically
            if len(local_results) % 10 == 0 and rank == 0:
                intermediate_file = os.path.join(output_dir, f"intermediate_session_{session_id}_rank_{rank}.pkl")
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(local_results, f)
                print(f"💾 Intermediate backup saved: {len(local_results)} combinations")

        except Exception as e:
            error_msg = f"Process {rank}: ERROR in combination {i} (block_id={block_id}): {str(e)}"
            print(error_msg)
            continue

    # Gather all results from all processes
    if rank == 0:
        print(f"📊 Gathering results from all {size} processes...")

    all_results = comm.gather(local_results, root=0)

    # Process and save final results (only root process)
    if rank == 0:
        # Flatten results from all processes
        final_results = []
        for proc_results in all_results:
            final_results.extend(proc_results)

        # Sort by block_id for consistent ordering
        final_results.sort(key=lambda x: x['block_id'])

        # Save complete results
        output_file = os.path.join(output_dir, f"chaos_with_input_rates_session_{session_id}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        # Calculate experiment statistics
        total_experiment_time = time.time() - experiment_start_time

        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"=" * 70)
        print(f"Results Summary:")
        print(f"   Total combinations processed: {len(final_results)}/{total_jobs}")
        print(f"   Success rate: {100*len(final_results)/total_jobs:.1f}%")
        print(f"   Total experiment time: {total_experiment_time/3600:.1f} hours")

        if final_results:
            # Timing statistics
            total_compute_time = sum(r['computation_time'] for r in final_results)
            avg_time = total_compute_time / len(final_results)
            print(f"   Average time per combination: {avg_time:.1f}s")

            # Scientific results statistics
            lz_values = [r['lz_mean'] for r in final_results]
            hamming_values = [r['hamming_mean'] for r in final_results]
            input_rates = list(set(r['static_input_rate'] for r in final_results))

            print(f"   Input rates tested: {sorted(input_rates)}")
            print(f"   LZ complexity range: {np.min(lz_values):.1f} - {np.max(lz_values):.1f}")
            print(f"   Hamming slope range: {np.min(hamming_values):.4f} - {np.max(hamming_values):.4f}")

        # Save comprehensive experiment summary
        summary_file = os.path.join(output_dir, f"input_rate_experiment_summary_session_{session_id}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Spiking RNN Chaos Experiment with Input Rate Sweep\n")
            f.write(f"===================================================\n\n")
            f.write(f"Experiment Configuration:\n")
            f.write(f"  Session ID: {session_id}\n")
            f.write(f"  MPI processes: {size}\n")
            f.write(f"  Network size: {n_neurons} neurons\n")
            f.write(f"  Parameter combinations: {total_jobs}\n")
            f.write(f"  v_th_std values: {n_v_th}\n")
            f.write(f"  g_std values: {n_g}\n")
            f.write(f"  Input rates: {n_input_rates} ({input_rate_min}-{input_rate_max} Hz)\n\n")

            if final_results:
                f.write(f"Results by Input Rate:\n")
                for rate in sorted(input_rates):
                    rate_results = [r for r in final_results if r['static_input_rate'] == rate]
                    if rate_results:
                        lz_vals = [r['lz_mean'] for r in rate_results]
                        hamm_vals = [r['hamming_mean'] for r in rate_results]
                        f.write(f"  {rate:.0f} Hz: LZ={np.mean(lz_vals):.1f}±{np.std(lz_vals):.1f}, "
                               f"Hamming={np.mean(hamm_vals):.4f}±{np.std(hamm_vals):.4f}\n")

        print(f"\n📄 Detailed summary saved: {summary_file}")
        print(f"✅ All files saved successfully in: {output_dir}")

# Utility functions (same as original but with input rate support)
def distribute_work_for_rank(total_jobs: int, rank: int, size: int) -> Tuple[int, int]:
    """Calculate work distribution for a specific MPI rank."""
    jobs_per_proc = total_jobs // size
    remainder = total_jobs % size
    if rank < remainder:
        start_idx = rank * (jobs_per_proc + 1)
        end_idx = start_idx + jobs_per_proc + 1
    else:
        start_idx = rank * jobs_per_proc + remainder
        end_idx = start_idx + jobs_per_proc
    return start_idx, end_idx

def distribute_work(total_jobs: int, comm: MPI.Comm) -> Tuple[int, int]:
    """Distribute work among MPI processes for current process."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    return distribute_work_for_rank(total_jobs, rank, size)

def monitor_system_health():
    """Monitor system health including temperature, CPU usage, and memory."""
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        max_temp = 0
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp:
                        max_temp = entry.current

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        if max_temp > 85:
            return False, f"CRITICAL: CPU temperature {max_temp:.1f}°C exceeds 85°C"
        elif cpu_percent > 95:
            return False, f"CRITICAL: CPU usage {cpu_percent:.1f}% exceeds 95%"
        elif memory_percent > 90:
            return False, f"CRITICAL: Memory usage {memory_percent:.1f}% exceeds 90%"
        else:
            return True, f"Healthy - Temp: {max_temp:.1f}°C, CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%"
    except Exception as e:
        return True, f"Health monitoring unavailable: {str(e)}"

def estimate_experiment_duration(n_combinations: int, avg_time_per_combo: float = 120.0):
    """Estimate total experiment duration with warnings for long experiments."""
    total_seconds = n_combinations * avg_time_per_combo
    hours = total_seconds / 3600
    days = hours / 24

    print(f"\n⏱️  EXPERIMENT DURATION ESTIMATE:")
    print(f"   Parameter combinations: {n_combinations}")
    print(f"   Estimated time per combination: {avg_time_per_combo:.1f}s")
    print(f"   Total estimated time: {hours:.1f} hours ({days:.1f} days)")

    if days > 7:
        print(f"   ⚠️  WARNING: >7 day experiment detected!")
        response = input("   Continue with this long experiment anyway? (type 'yes' to continue): ")
        if response.lower() != 'yes':
            return -1
    elif days > 2:
        print(f"   ⚠️  CAUTION: >2 day experiment")
    else:
        print(f"   ✅ Reasonable experiment duration")

    return total_seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MPI-parallelized chaos experiment with input rate sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--session_id", type=int, default=1)
    parser.add_argument("--n_v_th", type=int, default=10)
    parser.add_argument("--n_g", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--input_rate_min", type=float, default=50.0,
                       help="Minimum static input rate (Hz)")
    parser.add_argument("--input_rate_max", type=float, default=500.0,
                       help="Maximum static input rate (Hz)")
    parser.add_argument("--n_input_rates", type=int, default=5,
                       help="Number of input rate values to test")

    args = parser.parse_args()

    run_mpi_chaos_experiment_with_input_rates(
        session_id=args.session_id,
        n_v_th=args.n_v_th,
        n_g=args.n_g,
        n_neurons=args.n_neurons,
        output_dir=args.output_dir,
        input_rate_min=args.input_rate_min,
        input_rate_max=args.input_rate_max,
        n_input_rates=args.n_input_rates
    )
