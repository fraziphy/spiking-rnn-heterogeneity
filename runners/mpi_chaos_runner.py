# runners/mpi_chaos_runner.py - Complete rewrite with relaxed health tolerance
"""
MPI-parallelized chaos experiment runner with individual rank recovery.
Higher tolerance for system stress - allows intensive computational workloads.
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

def monitor_system_health_relaxed() -> Tuple[bool, str]:
    """
    Monitor system health with higher tolerance for computational workloads.
    More permissive thresholds allowing intensive computation.
    """
    try:
        import psutil

        # Check CPU temperature (if available)
        temps = psutil.sensors_temperatures()
        max_temp = 0
        temp_available = False

        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp:
                        max_temp = entry.current
                        temp_available = True

        # Check CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval for responsiveness
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Relaxed thresholds for intensive computational workloads
        TEMP_CRITICAL = 90      # Celsius (increased from 80)
        CPU_CRITICAL = 98       # Percent (increased from 90)
        MEMORY_CRITICAL = 95    # Percent (increased from 85)

        # Warning thresholds (for informational purposes)
        TEMP_WARNING = 85
        CPU_WARNING = 95
        MEMORY_WARNING = 90

        # Check for critical conditions
        critical_issues = []
        warnings = []

        if temp_available and max_temp > TEMP_CRITICAL:
            critical_issues.append(f"Temperature {max_temp:.1f}°C > {TEMP_CRITICAL}°C")
        elif temp_available and max_temp > TEMP_WARNING:
            warnings.append(f"Temperature {max_temp:.1f}°C approaching limit")

        if cpu_percent > CPU_CRITICAL:
            critical_issues.append(f"CPU {cpu_percent:.1f}% > {CPU_CRITICAL}%")
        elif cpu_percent > CPU_WARNING:
            warnings.append(f"CPU {cpu_percent:.1f}% high usage")

        if memory_percent > MEMORY_CRITICAL:
            critical_issues.append(f"Memory {memory_percent:.1f}% > {MEMORY_CRITICAL}%")
        elif memory_percent > MEMORY_WARNING:
            warnings.append(f"Memory {memory_percent:.1f}% high usage")

        # Determine overall health status
        if critical_issues:
            status = f"CRITICAL: {'; '.join(critical_issues)}"
            return False, status
        else:
            # Build status message
            status_parts = [f"CPU: {cpu_percent:.1f}%", f"Memory: {memory_percent:.1f}%"]
            if temp_available:
                status_parts.insert(0, f"Temp: {max_temp:.1f}°C")

            if warnings:
                status = f"WARNING - {'; '.join(warnings)} | " + " | ".join(status_parts)
            else:
                status = "HEALTHY - " + " | ".join(status_parts)

            return True, status

    except ImportError:
        return True, "Health monitoring unavailable (psutil not installed)"
    except Exception as e:
        return True, f"Health monitoring error: {str(e)}"

def implement_recovery_break(rank: int, break_duration: int = 300, reason: str = "system_stress"):
    """
    Implement recovery break for individual rank with adaptive duration.
    """
    print(f"[Rank {rank}] RECOVERY BREAK INITIATED")
    print(f"[Rank {rank}] Reason: {reason}")
    print(f"[Rank {rank}] Planned duration: {break_duration//60} minutes")
    print(f"[Rank {rank}] Other ranks continue working normally...")

    start_break = time.time()
    check_interval = 30  # Check every 30 seconds for responsiveness
    min_break_time = 60  # Minimum 1 minute break (reduced from 2 minutes)

    # Adaptive break duration based on issue type
    if "memory" in reason.lower():
        break_duration = max(break_duration, 600)  # Minimum 10 minutes for memory issues
    elif "temperature" in reason.lower():
        break_duration = max(break_duration, 900)  # Minimum 15 minutes for overheating

    consecutive_healthy_checks = 0
    required_healthy_checks = 3  # Need 3 consecutive healthy checks to end early

    while time.time() - start_break < break_duration:
        elapsed = time.time() - start_break
        remaining = break_duration - elapsed

        # Monitor this rank's system health during break
        healthy, status = monitor_system_health_relaxed()

        # Log break progress every minute
        if int(elapsed) % 60 == 0 or elapsed < 60:
            print(f"[Rank {rank}] Break: {elapsed/60:.1f}/{break_duration/60:.1f} min | {status}")

        # Track consecutive healthy checks for early termination
        if healthy and "HEALTHY" in status:
            consecutive_healthy_checks += 1
        else:
            consecutive_healthy_checks = 0

        # Allow early termination if system is consistently healthy
        if (consecutive_healthy_checks >= required_healthy_checks and
            elapsed >= min_break_time):
            print(f"[Rank {rank}] System stable for {consecutive_healthy_checks} checks - ending break early")
            break

        # Sleep with remaining time consideration
        sleep_time = min(check_interval, remaining)
        if sleep_time > 0:
            time.sleep(sleep_time)

    final_elapsed = time.time() - start_break
    print(f"[Rank {rank}] Recovery complete after {final_elapsed/60:.1f} minutes")

    # Final health check
    healthy, status = monitor_system_health_relaxed()
    print(f"[Rank {rank}] Post-break status: {status}")

def execute_single_combination(experiment: ChaosExperiment, rank: int, session_id: int,
                             block_id: int, v_th_std: float, g_std: float,
                             input_rate: float, combination_index: int) -> Dict[str, Any]:
    """
    Execute a single parameter combination with robust error handling and recovery.
    Uses individual rank recovery with higher system tolerance.
    """
    attempt = 0
    max_attempts = 15  # Increased from 10 to allow for more recovery attempts

    while attempt < max_attempts:
        attempt += 1

        # Pre-execution health check with relaxed thresholds
        healthy, status = monitor_system_health_relaxed()
        if not healthy:
            print(f"[Rank {rank}] Pre-execution health issue (attempt {attempt}): {status}")

            # Determine recovery duration based on issue severity
            if "temperature" in status.lower():
                recovery_duration = 900  # 15 minutes for temperature issues
            elif "memory" in status.lower():
                recovery_duration = 600  # 10 minutes for memory issues
            else:
                recovery_duration = 300  # 5 minutes for other issues

            implement_recovery_break(rank, recovery_duration, f"pre_execution: {status}")
            continue

        # Log attempt if not first try
        if attempt > 1:
            print(f"[Rank {rank}] Retry attempt {attempt}: v_th={v_th_std:.3f}, g_std={g_std:.3f}, rate={input_rate:.0f}Hz")

        computation_start = time.time()

        try:
            # Execute the parameter combination
            result = experiment.run_parameter_combination(
                session_id=session_id,
                block_id=block_id,
                v_th_std=v_th_std,
                g_std=g_std,
                static_input_rate=input_rate
            )

            # Add execution metadata
            result.update({
                'block_id': block_id,
                'rank': rank,
                'combination_index': combination_index,
                'attempt_count': attempt,
                'computation_time': time.time() - computation_start,
                'successful_completion': True
            })

            # Log success for retried combinations
            if attempt > 1:
                print(f"[Rank {rank}] SUCCESS after {attempt} attempts")

            return result

        except MemoryError as e:
            error_details = f"Memory exhaustion: {str(e)}"
            print(f"[Rank {rank}] {error_details} (attempt {attempt})")

            # Longer recovery for memory issues
            implement_recovery_break(rank, 900, f"memory_error: {error_details}")
            continue

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                error_details = f"Runtime memory error: {str(e)}"
                print(f"[Rank {rank}] {error_details} (attempt {attempt})")
                implement_recovery_break(rank, 600, f"runtime_memory: {error_details}")
            else:
                error_details = f"Runtime error: {str(e)}"
                print(f"[Rank {rank}] {error_details} (attempt {attempt})")
                implement_recovery_break(rank, 300, f"runtime_error: {error_details}")
            continue

        except Exception as e:
            error_details = f"General error: {type(e).__name__}: {str(e)}"
            print(f"[Rank {rank}] {error_details} (attempt {attempt})")
            implement_recovery_break(rank, 300, f"general_error: {error_details}")
            continue

    # Create failure placeholder after exhausting all attempts
    print(f"[Rank {rank}] FAILURE: Combination failed after {max_attempts} attempts")
    print(f"[Rank {rank}] Creating placeholder result for parameter combination")

    return {
        'block_id': block_id,
        'v_th_std': v_th_std,
        'g_std': g_std,
        'static_input_rate': input_rate,
        'rank': rank,
        'combination_index': combination_index,
        'lz_complexities': np.array([]),
        'hamming_slopes': np.array([]),
        'lz_mean': np.nan,
        'lz_std': np.nan,
        'hamming_mean': np.nan,
        'hamming_std': np.nan,
        'n_trials': 0,
        'computation_time': 0.0,
        'attempt_count': max_attempts,
        'failed_after_max_attempts': True,
        'failure_reason': "Exceeded maximum recovery attempts",
        'successful_completion': False
    }

def run_mpi_chaos_experiment_relaxed_health(session_id: int = 1, n_v_th: int = 10, n_g: int = 10,
                                          n_neurons: int = 1000, output_dir: str = "results",
                                          input_rate_min: float = 50.0, input_rate_max: float = 500.0,
                                          n_input_rates: int = 5):
    """
    Run chaos experiment with relaxed health monitoring and individual rank recovery.
    Higher tolerance for intensive computational workloads.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 85)
        print("SPIKING RNN CHAOS EXPERIMENT - RELAXED HEALTH MONITORING")
        print("=" * 85)
        print(f"Configuration:")
        print(f"  MPI processes: {size}")
        print(f"  Session ID: {session_id}")
        print(f"  Parameter grid: {n_v_th} × {n_g} × {n_input_rates} = {n_v_th * n_g * n_input_rates} combinations")
        print(f"  Network size: {n_neurons} neurons")
        print(f"  Input rate range: {input_rate_min}-{input_rate_max} Hz ({n_input_rates} values)")
        print(f"Health Policy:")
        print(f"  Temperature limit: 90°C (relaxed)")
        print(f"  CPU usage limit: 98% (relaxed)")
        print(f"  Memory usage limit: 95% (relaxed)")
        print(f"  Individual rank recovery (no coordinated breaks)")

        # Setup output directory
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    # Synchronize output directory
    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create parameter grids
    v_th_std_values, g_std_values, static_input_rates = create_parameter_grid_with_input_rates(
        n_points=max(n_v_th, n_g),
        input_rate_range=(input_rate_min, input_rate_max),
        n_input_rates=n_input_rates
    )

    # Adjust grid sizes to exact specifications
    if len(v_th_std_values) != n_v_th:
        v_th_std_values = np.linspace(0.05, 0.5, n_v_th)
    if len(g_std_values) != n_g:
        g_std_values = np.linspace(0.05, 0.5, n_g)

    # Generate complete parameter combinations
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
        print(f"\nWork Distribution:")
        for r in range(min(size, 10)):  # Show first 10 ranks
            s, e = distribute_work_for_rank(total_jobs, r, size)
            n_jobs = e - s
            print(f"  Rank {r:2d}: {n_jobs:3d} combinations (indices {s:3d}-{e-1:3d})")
        if size > 10:
            print(f"  ... and {size-10} more ranks")
        print(f"\nStarting experiment with relaxed health monitoring...")

    # Distribute work to this rank
    start_idx, end_idx = distribute_work(total_jobs, comm)
    my_combinations = param_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Assigned {len(my_combinations)} combinations (indices {start_idx}-{end_idx-1})")

    # Initialize chaos experiment
    experiment = ChaosExperiment(n_neurons=n_neurons)

    # Execute parameter combinations with individual recovery
    local_results = []
    rank_start_time = time.time()

    for i, (block_id, v_th_std, g_std, input_rate) in enumerate(my_combinations):
        # Progress reporting
        progress = (i + 1) / len(my_combinations) * 100
        elapsed_hours = (time.time() - rank_start_time) / 3600

        print(f"[Rank {rank}] Starting [{i+1}/{len(my_combinations)} - {progress:.1f}%]: "
              f"v_th={v_th_std:.3f}, g_std={g_std:.3f}, rate={input_rate:.0f}Hz "
              f"(elapsed: {elapsed_hours:.1f}h)")

        # Execute combination with recovery handling
        result = execute_single_combination(
            experiment=experiment,
            rank=rank,
            session_id=session_id,
            block_id=block_id,
            v_th_std=v_th_std,
            g_std=g_std,
            input_rate=input_rate,
            combination_index=start_idx + i
        )

        local_results.append(result)

        # Periodic intermediate saves
        if (i + 1) % 5 == 0:
            intermediate_file = os.path.join(output_dir, f"intermediate_session_{session_id}_rank_{rank}.pkl")
            with open(intermediate_file, 'wb') as f:
                pickle.dump(local_results, f)
            print(f"[Rank {rank}] Intermediate save: {len(local_results)} combinations completed")

    # Final local completion report
    rank_total_time = time.time() - rank_start_time
    successful_local = [r for r in local_results if r.get('successful_completion', False)]
    failed_local = [r for r in local_results if not r.get('successful_completion', False)]

    print(f"[Rank {rank}] LOCAL COMPLETION REPORT:")
    print(f"[Rank {rank}] Total time: {rank_total_time/3600:.1f} hours")
    print(f"[Rank {rank}] Successful: {len(successful_local)}/{len(local_results)}")
    print(f"[Rank {rank}] Failed: {len(failed_local)}")

    if successful_local:
        attempts = [r.get('attempt_count', 1) for r in successful_local]
        print(f"[Rank {rank}] Average attempts: {np.mean(attempts):.1f}")
        print(f"[Rank {rank}] Max attempts: {np.max(attempts)}")

    # Gather all results from all ranks
    print(f"[Rank {rank}] Waiting for other ranks to complete...")
    all_results = comm.gather(local_results, root=0)

    # Final processing and reporting (root process only)
    if rank == 0:
        print(f"\nProcessing results from all {size} ranks...")

        # Combine all results
        final_results = []
        for proc_results in all_results:
            final_results.extend(proc_results)

        # Sort by block_id for consistent ordering
        final_results.sort(key=lambda x: x['block_id'])

        # Calculate overall statistics
        total_experiment_time = time.time() - rank_start_time
        successful_results = [r for r in final_results if r.get('successful_completion', False)]
        failed_results = [r for r in final_results if not r.get('successful_completion', False)]

        print(f"\n" + "=" * 85)
        print("EXPERIMENT COMPLETED - RELAXED HEALTH MONITORING")
        print("=" * 85)
        print(f"Final Results:")
        print(f"  Total combinations: {len(final_results)}/{total_jobs}")
        print(f"  Successful: {len(successful_results)} ({100*len(successful_results)/total_jobs:.1f}%)")
        print(f"  Failed after max attempts: {len(failed_results)} ({100*len(failed_results)/total_jobs:.1f}%)")
        print(f"  Total experiment time: {total_experiment_time/3600:.1f} hours")

        if successful_results:
            # Attempt statistics
            attempts = [r.get('attempt_count', 1) for r in successful_results]
            computation_times = [r.get('computation_time', 0) for r in successful_results]

            print(f"Performance Statistics:")
            print(f"  Average attempts per combination: {np.mean(attempts):.1f}")
            print(f"  Maximum attempts needed: {np.max(attempts)}")
            print(f"  Average computation time: {np.mean(computation_times):.1f}s")

            # Scientific results
            lz_values = [r['lz_mean'] for r in successful_results if not np.isnan(r.get('lz_mean', np.nan))]
            hamming_values = [r['hamming_mean'] for r in successful_results if not np.isnan(r.get('hamming_mean', np.nan))]

            if lz_values and hamming_values:
                print(f"Scientific Results:")
                print(f"  LZ complexity range: {np.min(lz_values):.1f} - {np.max(lz_values):.1f}")
                print(f"  Hamming slope range: {np.min(hamming_values):.4f} - {np.max(hamming_values):.4f}")

        # Save complete results
        output_file = os.path.join(output_dir, f"chaos_relaxed_health_session_{session_id}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        # Save detailed summary
        summary_file = os.path.join(output_dir, f"relaxed_health_summary_session_{session_id}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Spiking RNN Chaos Experiment - Relaxed Health Monitoring\n")
            f.write(f"=======================================================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Session ID: {session_id}\n")
            f.write(f"  MPI processes: {size}\n")
            f.write(f"  Network size: {n_neurons} neurons\n")
            f.write(f"  Parameter combinations: {total_jobs}\n")
            f.write(f"  Health thresholds: 90°C, 98% CPU, 95% Memory\n\n")

            f.write(f"Results:\n")
            f.write(f"  Successful combinations: {len(successful_results)}\n")
            f.write(f"  Failed combinations: {len(failed_results)}\n")
            f.write(f"  Success rate: {100*len(successful_results)/total_jobs:.1f}%\n")
            f.write(f"  Total experiment time: {total_experiment_time/3600:.1f} hours\n")

        print(f"\nFiles Generated:")
        print(f"  Results: {output_file}")
        print(f"  Summary: {summary_file}")
        print(f"\nRelaxed health monitoring experiment completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MPI chaos experiment with relaxed health monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--session_id", type=int, default=1,
                       help="Session ID for reproducible random number generation")
    parser.add_argument("--n_v_th", type=int, default=10,
                       help="Number of spike threshold heterogeneity values")
    parser.add_argument("--n_g", type=int, default=10,
                       help="Number of synaptic weight heterogeneity values")
    parser.add_argument("--n_neurons", type=int, default=1000,
                       help="Number of neurons in the network")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--input_rate_min", type=float, default=50.0,
                       help="Minimum static input rate (Hz)")
    parser.add_argument("--input_rate_max", type=float, default=500.0,
                       help="Maximum static input rate (Hz)")
    parser.add_argument("--n_input_rates", type=int, default=5,
                       help="Number of input rate values to test")

    args = parser.parse_args()

    run_mpi_chaos_experiment_relaxed_health(
        session_id=args.session_id,
        n_v_th=args.n_v_th,
        n_g=args.n_g,
        n_neurons=args.n_neurons,
        output_dir=args.output_dir,
        input_rate_min=args.input_rate_min,
        input_rate_max=args.input_rate_max,
        n_input_rates=args.n_input_rates
    )
