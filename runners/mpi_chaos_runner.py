# mpi_chaos_runner.py
"""
MPI-parallelized chaos experiment runner for studying spiking RNN dynamics.
Complete implementation with safety monitoring and cluster support.
"""

import numpy as np
import os
import sys
import time
import pickle
import argparse
from mpi4py import MPI
from typing import List, Dict, Any, Tuple

# Import our modules with flexible import handling
try:
    # Try package-style imports (if installed with pip install -e .)
    from experiments.chaos_experiment import ChaosExperiment, create_parameter_grid, save_results
except ImportError:
    try:
        # Try relative imports within package
        from ..experiments.chaos_experiment import ChaosExperiment, create_parameter_grid, save_results
    except ImportError:
        # Fallback: add directories to path
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        experiments_dir = os.path.join(project_root, 'experiments')

        sys.path.insert(0, experiments_dir)

        from chaos_experiment import ChaosExperiment, create_parameter_grid, save_results

def distribute_work_for_rank(total_jobs: int, rank: int, size: int) -> Tuple[int, int]:
    """
    Calculate work distribution for a specific MPI rank.

    Args:
        total_jobs: Total number of jobs to distribute
        rank: Specific process rank to calculate for
        size: Total number of MPI processes

    Returns:
        Tuple of (start_idx, end_idx) for the specified rank
    """
    jobs_per_proc = total_jobs // size
    remainder = total_jobs % size

    # Distribute remainder among first few processes
    if rank < remainder:
        start_idx = rank * (jobs_per_proc + 1)
        end_idx = start_idx + jobs_per_proc + 1
    else:
        start_idx = rank * jobs_per_proc + remainder
        end_idx = start_idx + jobs_per_proc

    return start_idx, end_idx

def distribute_work(total_jobs: int, comm: MPI.Comm) -> Tuple[int, int]:
    """
    Distribute work among MPI processes for current process.

    Args:
        total_jobs: Total number of jobs to distribute
        comm: MPI communicator

    Returns:
        Tuple of (start_idx, end_idx) for current process
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    return distribute_work_for_rank(total_jobs, rank, size)

def setup_cpu_affinity(max_cores: int = 50):
    """
    Set CPU affinity to prevent overheating and be colleague-friendly.

    Args:
        max_cores: Maximum number of CPU cores to use
    """
    try:
        import psutil

        # Get current process
        current_process = psutil.Process()

        # Get available CPUs
        available_cpus = list(range(psutil.cpu_count()))

        # Limit to max_cores to leave resources for colleagues
        if len(available_cpus) > max_cores:
            available_cpus = available_cpus[:max_cores]

        # Set CPU affinity
        current_process.cpu_affinity(available_cpus)

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print(f"CPU affinity set to {len(available_cpus)} cores: {available_cpus[:5]}{'...' if len(available_cpus) > 5 else ''}")

    except ImportError:
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print("psutil not available - CPU affinity not set")
    except Exception as e:
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print(f"Warning: Could not set CPU affinity: {e}")

def monitor_system_health():
    """
    Monitor system health including temperature, CPU usage, and memory.

    Returns:
        Tuple of (is_healthy, status_message)
    """
    try:
        import psutil

        # Check CPU temperature (if sensors available)
        temps = psutil.sensors_temperatures()
        max_temp = 0

        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp:
                        max_temp = entry.current

        # Check CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Safety thresholds
        TEMP_WARNING = 75    # °C
        TEMP_CRITICAL = 85   # °C
        CPU_WARNING = 95     # %
        MEMORY_WARNING = 90  # %

        # Check for critical conditions
        if max_temp > TEMP_CRITICAL:
            return False, f"CRITICAL: CPU temperature {max_temp:.1f}°C exceeds {TEMP_CRITICAL}°C"
        elif cpu_percent > CPU_WARNING:
            return False, f"CRITICAL: CPU usage {cpu_percent:.1f}% exceeds {CPU_WARNING}%"
        elif memory_percent > MEMORY_WARNING:
            return False, f"CRITICAL: Memory usage {memory_percent:.1f}% exceeds {MEMORY_WARNING}%"
        elif max_temp > TEMP_WARNING:
            return True, f"CAUTION: Temperature {max_temp:.1f}°C (warm but acceptable)"
        else:
            return True, f"Healthy - Temp: {max_temp:.1f}°C, CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%"

    except Exception as e:
        return True, f"Health monitoring unavailable: {str(e)}"

def implement_cooling_breaks(break_duration: int = 300):
    """
    Implement mandatory cooling breaks during long experiments.

    Args:
        break_duration: Break duration in seconds (default: 5 minutes)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(f"🌡️  Implementing cooling break: {break_duration//60} minutes")
        print("   Allowing system to cool down and prevent overheating...")

        # Monitor temperature during break
        start_break = time.time()
        while time.time() - start_break < break_duration:
            healthy, status = monitor_system_health()
            elapsed_min = (time.time() - start_break) / 60
            remaining_min = break_duration/60 - elapsed_min
            print(f"   Break: {elapsed_min:.1f}/{break_duration/60:.1f} min remaining ({remaining_min:.1f} min left) - {status}")
            time.sleep(60)  # Check every minute during break

    # Synchronize all processes after break
    comm.Barrier()

    if rank == 0:
        print("🚀 Cooling break complete - resuming experiment")

def estimate_experiment_duration(n_combinations: int, avg_time_per_combo: float = 60.0):
    """
    Estimate total experiment duration and provide warnings for long experiments.

    Args:
        n_combinations: Number of parameter combinations
        avg_time_per_combo: Average time per combination in seconds

    Returns:
        Estimated duration in seconds, or -1 if user cancels
    """
    total_seconds = n_combinations * avg_time_per_combo
    hours = total_seconds / 3600
    days = hours / 24

    print(f"\n⏱️  EXPERIMENT DURATION ESTIMATE:")
    print(f"   Parameter combinations: {n_combinations}")
    print(f"   Estimated time per combination: {avg_time_per_combo:.1f}s")
    print(f"   Total estimated time: {hours:.1f} hours ({days:.1f} days)")

    # Provide warnings and suggestions for long experiments
    if days > 7:
        print(f"   ⚠️  WARNING: >7 day experiment detected!")
        print(f"   🔥 High risk of system overheating over such long periods")
        print(f"   💡 Strong recommendations:")
        print(f"      - Reduce parameter grid: use fewer --n_v_th or --n_g points")
        print(f"      - Use smaller network: reduce --n_neurons")
        print(f"      - Split into multiple smaller experiments")
        print(f"      - Consider running on dedicated cluster nodes")

        response = input("   Continue with this long experiment anyway? (type 'yes' to continue): ")
        if response.lower() != 'yes':
            print("   Experiment cancelled - wise choice for system safety!")
            return -1

    elif days > 2:
        print(f"   ⚠️  CAUTION: >2 day experiment")
        print(f"   🌡️  Will implement automatic cooling breaks every hour")
        print(f"   📊 Intermediate results will be saved regularly")

    elif hours > 6:
        print(f"   ℹ️  Long experiment (>6 hours)")
        print(f"   🛡️  Safety monitoring will be active")
    else:
        print(f"   ✅ Reasonable experiment duration")

    return total_seconds

def run_mpi_chaos_experiment(session_id: int = 1, n_v_th: int = 20,
                           n_g: int = 20, n_neurons: int = 1000,
                           output_dir: str = "results"):
    """
    Run chaos experiment with MPI parallelization and comprehensive safety monitoring.

    Args:
        session_id: Session ID for random number generation
        n_v_th: Number of v_th_std values to test
        n_g: Number of g_std values to test
        n_neurons: Number of neurons in network
        output_dir: Output directory for results
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"🧠 Spiking RNN Chaos Experiment - MPI Parallel Execution")
        print(f"=" * 60)
        print(f"MPI processes: {size}")
        print(f"Session ID: {session_id}")
        print(f"Parameter grid: {n_v_th} × {n_g} = {n_v_th * n_g} combinations")
        print(f"Network size: {n_neurons} neurons")

        # Setup output directory with data subdirectory
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")

        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Check available storage space
        try:
            import shutil
            total, used, free = shutil.disk_usage(output_dir)
            free_gb = free / (1024**3)
            print(f"Available storage: {free_gb:.1f} GB")

            if free_gb < 1.0:
                print("⚠️  WARNING: Less than 1GB free storage space!")
                print("   Large experiments may fail due to insufficient storage")
        except Exception:
            print("Could not check storage space")

        # Estimate experiment duration with user confirmation for long runs
        total_combinations = n_v_th * n_g
        estimated_duration = estimate_experiment_duration(total_combinations)
        if estimated_duration < 0:  # User cancelled
            print("Experiment cancelled by user - exiting safely")
            return

    # Synchronize output directory and user decisions across all processes
    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)

    # Wait for all processes after user input
    comm.Barrier()

    if rank == 0:
        print(f"\n🚀 Initializing parallel experiment execution...")

    # Create parameter grids (identical on all processes)
    v_th_std_values, g_std_values = create_parameter_grid(n_v_th)

    # Generate all parameter combinations
    param_combinations = []
    block_id = 0
    for v_th_std in v_th_std_values:
        for g_std in g_std_values:
            param_combinations.append((block_id, v_th_std, g_std))
            block_id += 1

    total_jobs = len(param_combinations)

    # Display work distribution (only rank 0)
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
        print(f"\n🔬 Starting parallel execution with safety monitoring...")

    # Initialize chaos experiment
    experiment = ChaosExperiment(n_neurons=n_neurons)

    # Execute assigned parameter combinations with comprehensive monitoring
    local_results = []
    last_health_check = time.time()
    last_cooling_break = time.time()
    experiment_start_time = time.time()

    for i, (block_id, v_th_std, g_std) in enumerate(my_combinations):
        # Progress reporting
        if rank == 0 or i % max(1, len(my_combinations)//5) == 0:
            elapsed_hours = (time.time() - experiment_start_time) / 3600
            progress = (i + 1) / len(my_combinations) * 100
            print(f"Process {rank}: [{i+1}/{len(my_combinations)} - {progress:.1f}%] "
                  f"v_th={v_th_std:.3f}, g_std={g_std:.3f} - {elapsed_hours:.1f}h elapsed")

        # System health monitoring (every 5 minutes)
        if time.time() - last_health_check > 300:
            healthy, status = monitor_system_health()
            if rank == 0:
                print(f"🌡️  Health check: {status}")

            if not healthy:
                if rank == 0:
                    print(f"❌ EMERGENCY STOP: {status}")
                    print("   System safety threshold exceeded - terminating experiment")
                break

            last_health_check = time.time()

        # Automatic cooling breaks for long experiments (every hour after 2 hours)
        experiment_time = time.time() - experiment_start_time
        if (time.time() - last_cooling_break > 3600 and experiment_time > 7200):
            implement_cooling_breaks(break_duration=300)  # 5 minute cooling break
            last_cooling_break = time.time()

        # Execute parameter combination
        computation_start = time.time()

        try:
            result = experiment.run_parameter_combination(
                session_id=session_id,
                block_id=block_id,
                v_th_std=v_th_std,
                g_std=g_std
            )

            # Add metadata
            result['block_id'] = block_id
            result['rank'] = rank
            result['combination_index'] = start_idx + i

            local_results.append(result)

            # Save intermediate results periodically (every 10 combinations)
            if len(local_results) % 10 == 0 and rank == 0:
                intermediate_file = os.path.join(output_dir, f"intermediate_session_{session_id}_rank_{rank}.pkl")
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(local_results, f)
                print(f"💾 Intermediate backup saved: {len(local_results)} combinations")

        except Exception as e:
            error_msg = f"Process {rank}: ERROR in combination {i} (block_id={block_id}): {str(e)}"
            print(error_msg)
            # Continue with remaining combinations rather than crashing
            continue

    # Final health check
    if rank == 0:
        healthy, status = monitor_system_health()
        print(f"🏁 Final system health check: {status}")

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
        output_file = os.path.join(output_dir, f"chaos_results_session_{session_id}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        # Calculate experiment statistics
        total_experiment_time = time.time() - experiment_start_time

        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"=" * 60)
        print(f"Results Summary:")
        print(f"   Total combinations processed: {len(final_results)}/{total_jobs}")
        print(f"   Success rate: {100*len(final_results)/total_jobs:.1f}%")
        print(f"   Total experiment time: {total_experiment_time/3600:.1f} hours")

        if final_results:
            # Timing statistics
            total_compute_time = sum(r['computation_time'] for r in final_results)
            avg_time = total_compute_time / len(final_results)
            print(f"   Average time per combination: {avg_time:.1f}s")
            print(f"   Total computation time: {total_compute_time/3600:.1f} hours")
            print(f"   Parallelization efficiency: {total_compute_time/total_experiment_time/size*100:.1f}%")

            # Scientific results statistics
            lz_values = [r['lz_mean'] for r in final_results]
            hamming_values = [r['hamming_mean'] for r in final_results]

            print(f"   LZ complexity range: {np.min(lz_values):.1f} - {np.max(lz_values):.1f}")
            print(f"   Hamming slope range: {np.min(hamming_values):.4f} - {np.max(hamming_values):.4f}")

        # Save comprehensive experiment summary
        summary_file = os.path.join(output_dir, f"experiment_summary_session_{session_id}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Spiking RNN Chaos Experiment Summary\n")
            f.write(f"====================================\n\n")
            f.write(f"Experiment Configuration:\n")
            f.write(f"  Session ID: {session_id}\n")
            f.write(f"  MPI processes: {size}\n")
            f.write(f"  Network size: {n_neurons} neurons\n")
            f.write(f"  Parameter grid: {n_v_th} × {n_g} = {total_jobs} combinations\n")
            f.write(f"  v_th_std range: {np.min(v_th_std_values):.3f} - {np.max(v_th_std_values):.3f} mV\n")
            f.write(f"  g_std range: {np.min(g_std_values):.3f} - {np.max(g_std_values):.3f}\n\n")

            f.write(f"Execution Results:\n")
            f.write(f"  Combinations processed: {len(final_results)}/{total_jobs}\n")
            f.write(f"  Success rate: {100*len(final_results)/total_jobs:.1f}%\n")
            f.write(f"  Total experiment time: {total_experiment_time/3600:.1f} hours\n")
            f.write(f"  Output directory: {output_dir}\n")

            if final_results:
                f.write(f"  Total computation time: {total_compute_time/3600:.1f} hours\n")
                f.write(f"  Average time per combination: {avg_time:.1f}s\n")
                f.write(f"  Parallelization efficiency: {total_compute_time/total_experiment_time/size*100:.1f}%\n\n")

                f.write(f"Scientific Results:\n")
                f.write(f"  LZ complexity range: {np.min(lz_values):.1f} - {np.max(lz_values):.1f}\n")
                f.write(f"  Hamming slope range: {np.min(hamming_values):.4f} - {np.max(hamming_values):.4f}\n")

        print(f"\n📝 Detailed summary saved: {summary_file}")

        # Final system status
        healthy, status = monitor_system_health()
        print(f"🌡️  Final system status: {status}")
        print(f"\n✅ All files saved successfully in: {output_dir}")

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run MPI-parallelized chaos experiment for spiking RNNs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--session_id", type=int, default=1,
                       help="Session ID for reproducible random number generation")
    parser.add_argument("--n_v_th", type=int, default=20,
                       help="Number of spike threshold heterogeneity values to test")
    parser.add_argument("--n_g", type=int, default=20,
                       help="Number of synaptic weight heterogeneity values to test")
    parser.add_argument("--n_neurons", type=int, default=1000,
                       help="Number of neurons in the spiking RNN")
    parser.add_argument("--max_cores", type=int, default=50,
                       help="Maximum CPU cores to use (for colleague-friendly execution)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for all results and logs")

    args = parser.parse_args()

    # Set CPU affinity to be colleague-friendly
    setup_cpu_affinity(args.max_cores)

    # Run the complete MPI chaos experiment
    run_mpi_chaos_experiment(
        session_id=args.session_id,
        n_v_th=args.n_v_th,
        n_g=args.n_g,
        n_neurons=args.n_neurons,
        output_dir=args.output_dir
    )
