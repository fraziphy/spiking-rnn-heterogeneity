# runners/mpi_spontaneous_runner.py - MPI runner with pulse/filter and static_input_mode
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

# Import modules with path handling
try:
    from experiments.spontaneous_experiment import SpontaneousExperiment, create_parameter_grid, save_results
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    experiments_dir = os.path.join(project_root, 'experiments')
    sys.path.insert(0, experiments_dir)
    from spontaneous_experiment import SpontaneousExperiment, create_parameter_grid, save_results

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

def monitor_system_health() -> Tuple[bool, str]:
    """Monitor system health with relaxed thresholds."""
    try:
        import psutil

        # Temperature monitoring
        temps = psutil.sensors_temperatures()
        max_temp = 0
        temp_available = False

        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp:
                        max_temp = entry.current
                        temp_available = True

        # CPU and memory monitoring
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Relaxed thresholds
        critical_issues = []
        if temp_available and max_temp > 90:
            critical_issues.append(f"Temperature {max_temp:.1f}°C")
        if cpu_percent > 98:
            critical_issues.append(f"CPU {cpu_percent:.1f}%")
        if memory_percent > 95:
            critical_issues.append(f"Memory {memory_percent:.1f}%")

        if critical_issues:
            return False, f"CRITICAL: {'; '.join(critical_issues)}"
        else:
            status_parts = [f"CPU: {cpu_percent:.1f}%", f"Memory: {memory_percent:.1f}%"]
            if temp_available:
                status_parts.insert(0, f"Temp: {max_temp:.1f}°C")
            return True, "HEALTHY - " + " | ".join(status_parts)

    except ImportError:
        return True, "Health monitoring unavailable"
    except Exception as e:
        return True, f"Health monitoring error: {str(e)}"

def recovery_break(rank: int, duration: int = 300, reason: str = "system_stress"):
    """Implement recovery break for individual rank."""
    print(f"[Rank {rank}] RECOVERY BREAK: {reason} ({duration//60} min)")

    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        if int(elapsed) % 60 == 0:
            print(f"[Rank {rank}] Break: {elapsed//60}/{duration//60} min")

        healthy, status = monitor_system_health()
        if healthy and "HEALTHY" in status:
            print(f"[Rank {rank}] System recovered early")
            break

        time.sleep(30)

    print(f"[Rank {rank}] Recovery complete")

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
            print(f"    CV ISI: {result.get('mean_cv_isi_mean', 'N/A')}")
            print(f"    Fano Factor: {result.get('mean_fano_factor_mean', 'N/A')}")
            print(f"    Poisson-like (ISI): {result.get('poisson_isi_fraction_mean', 'N/A'):.1%}")
            print(f"    Total spikes: {result['total_spikes_mean']:.0f}")

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
        'effective_dimensionality_bin_5.0ms_mean': np.nan,
        'total_spikes_mean': np.nan,
        'mean_cv_isi_mean': np.nan,
        'mean_fano_factor_mean': np.nan,
        'poisson_isi_fraction_mean': np.nan,
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

    # Convert duration from seconds to milliseconds if needed
    if duration < 100:  # Assume seconds if < 100, convert to ms
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
        print(f"  v_th_std range: {v_th_std_min}-{v_th_std_max}")
        print(f"  g_std range: {g_std_min}-{g_std_max}")
        print(f"  Input rate range: {input_rate_min}-{input_rate_max} Hz")
        print(f"  Synaptic mode: {synaptic_mode}")
        print(f"  Static input mode: {static_input_mode}")
        print(f"  Threshold distribution: {v_th_distribution}")
        print(f"  Trials per combination: 10")

        print(f"\nSpontaneous Activity Analysis Features:")
        print(f"  • Firing rate statistics (mean, std, silent %)")
        print(f"  • Dimensionality analysis with 6 bin sizes:")
        print(f"    - 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms")
        print(f"  • Participation ratio and variance analysis")

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
    param_combinations = []
    combo_id = 0
    for input_rate in static_input_rates:
        for v_th_std in v_th_stds:
            for g_std in g_stds:
                param_combinations.append((combo_id, v_th_std, g_std, v_th_distribution, input_rate))
                combo_id += 1

    total_jobs = len(param_combinations)

    if rank == 0:
        print(f"\nWork Distribution:")
        for r in range(min(size, 8)):
            s, e = distribute_work_for_rank(total_jobs, r, size)
            n_jobs = e - s
            print(f"  Rank {r:2d}: {n_jobs:3d} combinations")
        if size > 8:
            print(f"  ... and {size-8} more ranks")

        # Estimate computation time
        trials_per_combo = 10
        expected_time_per_combo = duration / 1000.0 * 0.5  # Rough estimate
        total_expected_time = (total_jobs * expected_time_per_combo * trials_per_combo) / (size * 3600)
        print(f"\nEstimated total time: {total_expected_time:.1f} hours")

    # Distribute work among ranks
    start_idx, end_idx = distribute_work(total_jobs, comm)
    my_combinations = param_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Processing {len(my_combinations)} combinations")
    if len(my_combinations) > 0:
        print(f"[Rank {rank}] Rate range: {my_combinations[0][4]:.0f}-{my_combinations[-1][4]:.0f}Hz")

    # Initialize spontaneous experiment
    experiment = SpontaneousExperiment(n_neurons=n_neurons, synaptic_mode=synaptic_mode,
                                      static_input_mode=static_input_mode)

    # Execute assigned combinations
    local_results = []
    rank_start_time = time.time()

    for i, (combo_id, v_th_std, g_std, v_th_dist, input_rate) in enumerate(my_combinations):
        progress = (i + 1) / len(my_combinations) * 100
        elapsed_hours = (time.time() - rank_start_time) / 3600

        print(f"[Rank {rank}] [{i+1}/{len(my_combinations)} - {progress:.1f}%]: "
              f"v_th_std={v_th_std:.3f}, g_std={g_std:.3f}, dist={v_th_dist}, "
              f"rate={input_rate:.0f}Hz (elapsed: {elapsed_hours:.2f}h)")

        result = execute_combination_with_recovery(
            experiment=experiment,
            rank=rank,
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            v_th_distribution=v_th_dist,
            static_input_rate=input_rate,
            combination_index=start_idx + i,
            duration=duration
        )

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
        final_results.sort(key=lambda x: x['combination_index'])

        # Calculate statistics
        total_experiment_time = time.time() - rank_start_time
        successful_results = [r for r in final_results if r.get('successful_completion', False)]
        failed_results = [r for r in final_results if not r.get('successful_completion', False)]

        print(f"\n" + "=" * 80)
        print("SPONTANEOUS ACTIVITY EXPERIMENT COMPLETED")
        print("=" * 80)
        print(f"Session ID: {session_id}")
        print(f"Synaptic mode: {synaptic_mode}")
        print(f"Static input mode: {static_input_mode}")
        print(f"Threshold distribution: {v_th_distribution}")
        print(f"Duration: {duration:.0f} ms")
        print(f"Total combinations: {len(final_results)}")
        print(f"Successful: {len(successful_results)} ({100*len(successful_results)/len(final_results):.1f}%)")
        print(f"Failed: {len(failed_results)} ({100*len(failed_results)/len(final_results):.1f}%)")
        print(f"Total time: {total_experiment_time/3600:.2f} hours")

        if successful_results:
            attempts = [r.get('attempt_count', 1) for r in successful_results]
            print(f"Average attempts: {np.mean(attempts):.1f}")

            # Spontaneous activity measure ranges
            firing_rate_values = [r['mean_firing_rate_mean'] for r in successful_results if not np.isnan(r.get('mean_firing_rate_mean', np.nan))]
            silent_values = [r['percent_silent_mean'] for r in successful_results if not np.isnan(r.get('percent_silent_mean', np.nan))]
            dim_values = [r['effective_dimensionality_bin_5.0ms_mean'] for r in successful_results if not np.isnan(r.get('effective_dimensionality_bin_5.0ms_mean', np.nan))]

            if firing_rate_values:
                print(f"Firing rates: {np.min(firing_rate_values):.1f} - {np.max(firing_rate_values):.1f} Hz")
            if silent_values:
                print(f"Silent neurons: {np.min(silent_values):.1f} - {np.max(silent_values):.1f}%")
            if dim_values:
                print(f"Dimensionality (5ms): {np.min(dim_values):.1f} - {np.max(dim_values):.1f}")

        # Save spontaneous activity results with updated filename
        duration_sec = duration / 1000.0
        output_file = os.path.join(output_dir,
                                   f"spontaneous_session_{session_id}_{synaptic_mode}_{static_input_mode}_{v_th_distribution}_{duration_sec:.1f}s.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        print(f"\nSpontaneous activity results saved: {output_file}")
        print("Spontaneous activity single session experiment completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MPI spontaneous activity experiment with extended dimensionality analysis"
    )

    parser.add_argument("--session_id", type=int, default=1,
                       help="Session ID for this run")
    parser.add_argument("--n_v_th", type=int, default=10,
                       help="Number of v_th_std values")
    parser.add_argument("--n_g", type=int, default=10,
                       help="Number of g_std values")
    parser.add_argument("--n_neurons", type=int, default=1000,
                       help="Number of neurons in network")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--v_th_std_min", type=float, default=0.0,
                       help="Minimum v_th_std value")
    parser.add_argument("--v_th_std_max", type=float, default=4.0,
                       help="Maximum v_th_std value")
    parser.add_argument("--g_std_min", type=float, default=0.0,
                       help="Minimum g_std value")
    parser.add_argument("--g_std_max", type=float, default=4.0,
                       help="Maximum g_std value")
    parser.add_argument("--input_rate_min", type=float, default=50.0,
                       help="Minimum static input rate (Hz)")
    parser.add_argument("--input_rate_max", type=float, default=1000.0,
                       help="Maximum static input rate (Hz)")
    parser.add_argument("--n_input_rates", type=int, default=5,
                       help="Number of input rate values")
    parser.add_argument("--synaptic_mode", type=str, default="filter",
                       choices=["pulse", "filter"],
                       help="Synaptic mode: pulse or filter")
    parser.add_argument("--static_input_mode", type=str, default="independent",
                       choices=["independent", "common_stochastic", "common_tonic"],
                       help="Static input mode: independent, common_stochastic, or common_tonic")
    parser.add_argument("--v_th_distribution", type=str, default="normal",
                       choices=["normal", "uniform"],
                       help="Threshold distribution: normal or uniform")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Simulation duration in seconds (will be converted to ms)")

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
