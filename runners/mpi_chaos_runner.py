# runners/mpi_chaos_runner.py - Updated with multiplier parameters
"""
MPI-parallelized chaos experiment runner with fixed network structure and multiplier scaling.
"""

import numpy as np
import os
import sys
import time
import pickle
import argparse
from mpi4py import MPI
from typing import List, Dict, Any, Tuple

# Import modules
try:
    from experiments.chaos_experiment import ChaosExperiment, create_parameter_grid_with_multipliers, save_results
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    experiments_dir = os.path.join(project_root, 'experiments')
    sys.path.insert(0, experiments_dir)
    from chaos_experiment import ChaosExperiment, create_parameter_grid_with_multipliers, save_results

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
    """Monitor system health with relaxed thresholds."""
    try:
        import psutil

        temps = psutil.sensors_temperatures()
        max_temp = 0
        temp_available = False

        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp:
                        max_temp = entry.current
                        temp_available = True

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Relaxed thresholds
        TEMP_CRITICAL = 90
        CPU_CRITICAL = 98
        MEMORY_CRITICAL = 95

        critical_issues = []

        if temp_available and max_temp > TEMP_CRITICAL:
            critical_issues.append(f"Temperature {max_temp:.1f}°C > {TEMP_CRITICAL}°C")
        if cpu_percent > CPU_CRITICAL:
            critical_issues.append(f"CPU {cpu_percent:.1f}% > {CPU_CRITICAL}%")
        if memory_percent > MEMORY_CRITICAL:
            critical_issues.append(f"Memory {memory_percent:.1f}% > {MEMORY_CRITICAL}%")

        if critical_issues:
            status = f"CRITICAL: {'; '.join(critical_issues)}"
            return False, status
        else:
            status_parts = [f"CPU: {cpu_percent:.1f}%", f"Memory: {memory_percent:.1f}%"]
            if temp_available:
                status_parts.insert(0, f"Temp: {max_temp:.1f}°C")
            status = "HEALTHY - " + " | ".join(status_parts)
            return True, status

    except ImportError:
        return True, "Health monitoring unavailable (psutil not installed)"
    except Exception as e:
        return True, f"Health monitoring error: {str(e)}"

def implement_recovery_break(rank: int, break_duration: int = 300, reason: str = "system_stress"):
    """Implement recovery break for individual rank."""
    print(f"[Rank {rank}] RECOVERY BREAK INITIATED")
    print(f"[Rank {rank}] Reason: {reason}")
    print(f"[Rank {rank}] Duration: {break_duration//60} minutes")

    start_break = time.time()
    check_interval = 30
    min_break_time = 60

    if "memory" in reason.lower():
        break_duration = max(break_duration, 600)
    elif "temperature" in reason.lower():
        break_duration = max(break_duration, 900)

    consecutive_healthy_checks = 0
    required_healthy_checks = 3

    while time.time() - start_break < break_duration:
        elapsed = time.time() - start_break
        remaining = break_duration - elapsed

        healthy, status = monitor_system_health_relaxed()

        if int(elapsed) % 60 == 0 or elapsed < 60:
            print(f"[Rank {rank}] Break: {elapsed/60:.1f}/{break_duration/60:.1f} min | {status}")

        if healthy and "HEALTHY" in status:
            consecutive_healthy_checks += 1
        else:
            consecutive_healthy_checks = 0

        if (consecutive_healthy_checks >= required_healthy_checks and
            elapsed >= min_break_time):
            print(f"[Rank {rank}] System stable - ending break early")
            break

        sleep_time = min(check_interval, remaining)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[Rank {rank}] Recovery complete")

def execute_single_combination(experiment: ChaosExperiment, rank: int, session_id: int,
                             block_id: int, v_th_multiplier: float, g_multiplier: float,
                             static_input_rate: float, combination_index: int) -> Dict[str, Any]:
    """Execute single parameter combination with recovery."""
    attempt = 0
    max_attempts = 15

    while attempt < max_attempts:
        attempt += 1

        healthy, status = monitor_system_health_relaxed()
        if not healthy:
            print(f"[Rank {rank}] Pre-execution health issue (attempt {attempt}): {status}")

            if "temperature" in status.lower():
                recovery_duration = 900
            elif "memory" in status.lower():
                recovery_duration = 600
            else:
                recovery_duration = 300

            implement_recovery_break(rank, recovery_duration, f"pre_execution: {status}")
            continue

        if attempt > 1:
            print(f"[Rank {rank}] Retry attempt {attempt}: v_th_mult={v_th_multiplier:.1f}, g_mult={g_multiplier:.1f}")

        computation_start = time.time()

        try:
            result = experiment.run_parameter_combination(
                session_id=session_id,
                block_id=block_id,
                v_th_multiplier=v_th_multiplier,
                g_multiplier=g_multiplier,
                static_input_rate=static_input_rate
            )

            result.update({
                'block_id': block_id,
                'rank': rank,
                'combination_index': combination_index,
                'attempt_count': attempt,
                'computation_time': time.time() - computation_start,
                'successful_completion': True
            })

            if attempt > 1:
                print(f"[Rank {rank}] SUCCESS after {attempt} attempts")

            return result

        except MemoryError as e:
            print(f"[Rank {rank}] Memory error (attempt {attempt}): {str(e)}")
            implement_recovery_break(rank, 900, f"memory_error: {str(e)}")
            continue

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[Rank {rank}] Runtime memory error (attempt {attempt}): {str(e)}")
                implement_recovery_break(rank, 600, f"runtime_memory: {str(e)}")
            else:
                print(f"[Rank {rank}] Runtime error (attempt {attempt}): {str(e)}")
                implement_recovery_break(rank, 300, f"runtime_error: {str(e)}")
            continue

        except Exception as e:
            print(f"[Rank {rank}] General error (attempt {attempt}): {str(e)}")
            implement_recovery_break(rank, 300, f"general_error: {type(e).__name__}")
            continue

    # Create failure placeholder
    print(f"[Rank {rank}] FAILURE: Combination failed after {max_attempts} attempts")

    return {
        'block_id': block_id,
        'v_th_multiplier': v_th_multiplier,
        'g_multiplier': g_multiplier,
        'v_th_std': 0.01 * v_th_multiplier,
        'g_std': 0.01 * g_multiplier,
        'static_input_rate': static_input_rate,
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

def run_mpi_chaos_experiment_multipliers(session_id: int = 1, n_v_th: int = 10, n_g: int = 10,
                                       n_neurons: int = 1000, output_dir: str = "results",
                                       multiplier_min: float = 1.0, multiplier_max: float = 100.0,
                                       input_rate_min: float = 50.0, input_rate_max: float = 500.0,
                                       n_input_rates: int = 5):
    """
    Run chaos experiment with multiplier scaling and fixed structure.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 85)
        print("SPIKING RNN CHAOS EXPERIMENT - FIXED STRUCTURE WITH MULTIPLIERS")
        print("=" * 85)
        print(f"Configuration:")
        print(f"  MPI processes: {size}")
        print(f"  Session ID: {session_id}")
        print(f"  Parameter grid: {n_v_th} × {n_g} × {n_input_rates} = {n_v_th * n_g * n_input_rates} combinations")
        print(f"  Network size: {n_neurons} neurons")
        print(f"  Multiplier range: {multiplier_min}-{multiplier_max}")
        print(f"  Actual heterogeneity range: {0.01*multiplier_min:.3f}-{0.01*multiplier_max:.2f}")
        print(f"  Input rate range: {input_rate_min}-{input_rate_max} Hz ({n_input_rates} values)")
        print(f"  Trials per combination: 20")
        print(f"Fixed Structure Policy:")
        print(f"  Network topology depends only on session_id={session_id}")
        print(f"  Base heterogeneities: v_th_std=0.01, g_std=0.01")
        print(f"  Exact mean preservation: -55mV thresholds, 0 weights")

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create parameter grids with multipliers
    v_th_multipliers, g_multipliers, static_input_rates = create_parameter_grid_with_multipliers(
        n_points=max(n_v_th, n_g),
        multiplier_range=(multiplier_min, multiplier_max),
        input_rate_range=(input_rate_min, input_rate_max),
        n_input_rates=n_input_rates
    )

    # Adjust grid sizes
    if len(v_th_multipliers) != n_v_th:
        v_th_multipliers = np.linspace(multiplier_min, multiplier_max, n_v_th)
    if len(g_multipliers) != n_g:
        g_multipliers = np.linspace(multiplier_min, multiplier_max, n_g)

    # Generate all parameter combinations
    param_combinations = []
    block_id = 0
    for input_rate in static_input_rates:
        for v_th_mult in v_th_multipliers:
            for g_mult in g_multipliers:
                param_combinations.append((block_id, v_th_mult, g_mult, input_rate))
                block_id += 1

    total_jobs = len(param_combinations)

    if rank == 0:
        print(f"\nWork Distribution:")
        for r in range(min(size, 10)):
            s, e = distribute_work_for_rank(total_jobs, r, size)
            n_jobs = e - s
            print(f"  Rank {r:2d}: {n_jobs:3d} combinations (indices {s:3d}-{e-1:3d})")
        if size > 10:
            print(f"  ... and {size-10} more ranks")
        print(f"\nStarting fixed-structure experiment with multiplier scaling...")

    # Distribute work
    start_idx, end_idx = distribute_work(total_jobs, comm)
    my_combinations = param_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Assigned {len(my_combinations)} combinations")

    # Initialize experiment
    experiment = ChaosExperiment(n_neurons=n_neurons)

    # Execute combinations
    local_results = []
    rank_start_time = time.time()

    for i, (block_id, v_th_mult, g_mult, input_rate) in enumerate(my_combinations):
        progress = (i + 1) / len(my_combinations) * 100
        elapsed_hours = (time.time() - rank_start_time) / 3600

        actual_v_th_std = 0.01 * v_th_mult
        actual_g_std = 0.01 * g_mult

        print(f"[Rank {rank}] Starting [{i+1}/{len(my_combinations)} - {progress:.1f}%]: "
              f"v_th_mult={v_th_mult:.1f}→{actual_v_th_std:.3f}, g_mult={g_mult:.1f}→{actual_g_std:.3f}, "
              f"rate={input_rate:.0f}Hz (elapsed: {elapsed_hours:.1f}h)")

        result = execute_single_combination(
            experiment=experiment,
            rank=rank,
            session_id=session_id,
            block_id=block_id,
            v_th_multiplier=v_th_mult,
            g_multiplier=g_mult,
            static_input_rate=input_rate,
            combination_index=start_idx + i
        )

        local_results.append(result)

        if (i + 1) % 5 == 0:
            intermediate_file = os.path.join(output_dir, f"intermediate_session_{session_id}_rank_{rank}.pkl")
            with open(intermediate_file, 'wb') as f:
                pickle.dump(local_results, f)
            print(f"[Rank {rank}] Intermediate save: {len(local_results)} combinations completed")

    # Final local report
    rank_total_time = time.time() - rank_start_time
    successful_local = [r for r in local_results if r.get('successful_completion', False)]
    failed_local = [r for r in local_results if not r.get('successful_completion', False)]

    print(f"[Rank {rank}] LOCAL COMPLETION: {len(successful_local)}/{len(local_results)} successful ({rank_total_time/3600:.1f}h)")

    # Gather results
    all_results = comm.gather(local_results, root=0)

    # Process and save (root only)
    if rank == 0:
        print(f"\nProcessing results from all {size} ranks...")

        final_results = []
        for proc_results in all_results:
            final_results.extend(proc_results)

        final_results.sort(key=lambda x: x['block_id'])

        total_experiment_time = time.time() - rank_start_time
        successful_results = [r for r in final_results if r.get('successful_completion', False)]
        failed_results = [r for r in final_results if not r.get('successful_completion', False)]

        print(f"\n" + "=" * 85)
        print("FIXED STRUCTURE EXPERIMENT COMPLETED")
        print("=" * 85)
        print(f"Results:")
        print(f"  Total combinations: {len(final_results)}/{total_jobs}")
        print(f"  Successful: {len(successful_results)} ({100*len(successful_results)/total_jobs:.1f}%)")
        print(f"  Failed: {len(failed_results)} ({100*len(failed_results)/total_jobs:.1f}%)")
        print(f"  Total time: {total_experiment_time/3600:.1f} hours")

        if successful_results:
            attempts = [r.get('attempt_count', 1) for r in successful_results]
            print(f"  Average attempts: {np.mean(attempts):.1f}")
            print(f"  Max attempts: {np.max(attempts)}")

            lz_values = [r['lz_mean'] for r in successful_results if not np.isnan(r.get('lz_mean', np.nan))]
            hamming_values = [r['hamming_mean'] for r in successful_results if not np.isnan(r.get('hamming_mean', np.nan))]

            if lz_values and hamming_values:
                print(f"  LZ complexity range: {np.min(lz_values):.1f} - {np.max(lz_values):.1f}")
                print(f"  Hamming slope range: {np.min(hamming_values):.4f} - {np.max(hamming_values):.4f}")

        # Save results
        output_file = os.path.join(output_dir, f"chaos_fixed_structure_session_{session_id}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        print(f"\nResults saved: {output_file}")
        print(f"Fixed-structure experiment with multiplier scaling completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MPI chaos experiment with fixed structure and multiplier scaling"
    )

    parser.add_argument("--session_id", type=int, default=1,
                       help="Session ID for fixed network structure")
    parser.add_argument("--n_v_th", type=int, default=10,
                       help="Number of v_th multiplier values")
    parser.add_argument("--n_g", type=int, default=10,
                       help="Number of g multiplier values")
    parser.add_argument("--n_neurons", type=int, default=1000,
                       help="Number of neurons in network")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--multiplier_min", type=float, default=1.0,
                       help="Minimum multiplier value")
    parser.add_argument("--multiplier_max", type=float, default=100.0,
                       help="Maximum multiplier value")
    parser.add_argument("--input_rate_min", type=float, default=50.0,
                       help="Minimum static input rate (Hz)")
    parser.add_argument("--input_rate_max", type=float, default=500.0,
                       help="Maximum static input rate (Hz)")
    parser.add_argument("--n_input_rates", type=int, default=5,
                       help="Number of input rate values")

    args = parser.parse_args()

    run_mpi_chaos_experiment_multipliers(
        session_id=args.session_id,
        n_v_th=args.n_v_th,
        n_g=args.n_g,
        n_neurons=args.n_neurons,
        output_dir=args.output_dir,
        multiplier_min=args.multiplier_min,
        multiplier_max=args.multiplier_max,
        input_rate_min=args.input_rate_min,
        input_rate_max=args.input_rate_max,
        n_input_rates=args.n_input_rates
    )
