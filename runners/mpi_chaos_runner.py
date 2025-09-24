# runners/mpi_chaos_runner.py - Updated MPI runner with enhanced analysis
"""
MPI-parallelized chaos experiment runner with enhanced complexity measures and firing rate analysis.
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
    from experiments.chaos_experiment import ChaosExperiment, create_parameter_grid, save_results
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    experiments_dir = os.path.join(project_root, 'experiments')
    sys.path.insert(0, experiments_dir)
    from chaos_experiment import ChaosExperiment, create_parameter_grid, save_results

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

def execute_combination_with_recovery(experiment: ChaosExperiment, rank: int,
                                    session_id: int, v_th_std: float, g_std: float,
                                    v_th_distribution: str, static_input_rate: float,
                                    combination_index: int) -> Dict[str, Any]:
    """Execute single parameter combination with recovery and enhanced monitoring."""
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

            # Log enhanced results
            print(f"[Rank {rank}] Success:")
            print(f"    LZ (flattened): {result['lz_matrix_flattened_mean']:.2f}")
            print(f"    LZ (spatial): {result['lz_spatial_patterns_mean']:.2f}")
            print(f"    PCI (normalized): {result['pci_normalized_mean']:.3f}")
            print(f"    Kistler 2ms: {result['kistler_delta_2ms_mean']:.3f}")
            print(f"    Silent neurons: {result['control_percent_silent_mean']:.1f}%")
            print(f"    Stable patterns: {result['stable_pattern_fraction']:.2f}")

            return result

        except Exception as e:
            print(f"[Rank {rank}] Error (attempt {attempt}): {str(e)}")
            if "memory" in str(e).lower():
                recovery_break(rank, 600, "memory_error")
            elif "firing" in str(e).lower() or "dimensionality" in str(e).lower():
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
        'synaptic_mode': experiment.synaptic_mode,
        'rank': rank,
        'combination_index': combination_index,
        'lz_matrix_flattened_mean': np.nan,
        'lz_spatial_patterns_mean': np.nan,
        'pci_normalized_mean': np.nan,
        'hamming_mean': np.nan,
        'kistler_delta_2ms_mean': np.nan,
        'gamma_window_5ms_mean': np.nan,
        'control_percent_silent_mean': np.nan,
        'computation_time': 0.0,
        'attempt_count': max_attempts,
        'successful_completion': False,
        'failure_reason': "Exceeded maximum attempts"
    }

def run_mpi_chaos_experiment(session_id: int = 1,
                           n_v_th: int = 10, n_g: int = 10,
                           n_neurons: int = 1000, output_dir: str = "results",
                           v_th_std_min: float = 0.0, v_th_std_max: float = 4.0,
                           g_std_min: float = 0.0, g_std_max: float = 4.0,
                           input_rate_min: float = 50.0, input_rate_max: float = 1000.0,
                           n_input_rates: int = 5, synaptic_mode: str = "dynamic",
                           v_th_distributions: List[str] = ["normal"]):
    """Run enhanced chaos experiment for single session with extended analysis."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 80)
        print("ENHANCED SPIKING RNN CHAOS EXPERIMENT - SINGLE SESSION")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  MPI processes: {size}")
        print(f"  Session ID: {session_id}")
        print(f"  Parameter grid: {n_v_th} × {n_g} × {len(v_th_distributions)} × {n_input_rates}")
        print(f"  Network size: {n_neurons} neurons")
        print(f"  v_th_std range: {v_th_std_min}-{v_th_std_max}")
        print(f"  g_std range: {g_std_min}-{g_std_max}")
        print(f"  Input rate range: {input_rate_min}-{input_rate_max} Hz")
        print(f"  Synaptic mode: {synaptic_mode}")
        print(f"  Threshold distributions: {v_th_distributions}")
        print(f"  Trials per combination: 100")

        print(f"\nEnhanced Analysis Features:")
        print(f"  • 4 complexity measures (LZ flattened, LZ spatial, PCI variants)")
        print(f"  • Kistler coincidence (2ms, 5ms) + Gamma coincidence (5ms, 10ms)")
        print(f"  • Dimensionality analysis (2ms, 5ms, 20ms bins)")
        print(f"  • Firing rate statistics and silent neuron analysis")
        print(f"  • Pattern stability detection")

        # Setup output directory
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.abspath(output_dir), "data")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    # Broadcast output directory to all ranks
    output_dir = comm.bcast(output_dir if rank == 0 else None, root=0)
    comm.Barrier()

    # Create parameter grids with extended ranges
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
        for v_th_dist in v_th_distributions:
            for v_th_std in v_th_stds:
                for g_std in g_stds:
                    param_combinations.append((combo_id, v_th_std, g_std, v_th_dist, input_rate))
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
        trials_per_combo = 100
        expected_time_per_combo = 120 if synaptic_mode == "dynamic" else 60  # seconds
        total_expected_time = (total_jobs * expected_time_per_combo * trials_per_combo) / (size * 3600)
        print(f"\nEstimated total time: {total_expected_time:.1f} hours")

    # Distribute work among ranks
    start_idx, end_idx = distribute_work(total_jobs, comm)
    my_combinations = param_combinations[start_idx:end_idx]

    print(f"[Rank {rank}] Processing {len(my_combinations)} combinations")
    if len(my_combinations) > 0:
        print(f"[Rank {rank}] Rate range: {my_combinations[0][4]:.0f}-{my_combinations[-1][4]:.0f}Hz")

    # Initialize enhanced experiment
    experiment = ChaosExperiment(n_neurons=n_neurons, synaptic_mode=synaptic_mode)

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
            combination_index=start_idx + i
        )

        local_results.append(result)

        # Progress update for high-rate combinations
        if input_rate >= 500.0:
            print(f"[Rank {rank}] High-rate analysis ({input_rate:.0f}Hz) completed")

    # Final local report
    rank_total_time = time.time() - rank_start_time
    successful_local = [r for r in local_results if r.get('successful_completion', False)]

    print(f"[Rank {rank}] COMPLETED: {len(successful_local)}/{len(local_results)} successful ({rank_total_time/3600:.2f}h)")

    # Gather results at root
    all_results = comm.gather(local_results, root=0)

    # Process and save results (root rank only)
    if rank == 0:
        print(f"\nProcessing enhanced results from all {size} ranks...")

        # Combine results from all ranks
        final_results = []
        for proc_results in all_results:
            final_results.extend(proc_results)

        # Sort by combination index
        final_results.sort(key=lambda x: x['combination_index'])

        # Calculate enhanced statistics
        total_experiment_time = time.time() - rank_start_time
        successful_results = [r for r in final_results if r.get('successful_completion', False)]
        failed_results = [r for r in final_results if not r.get('successful_completion', False)]

        print(f"\n" + "=" * 80)
        print("ENHANCED CHAOS EXPERIMENT COMPLETED")
        print("=" * 80)
        print(f"Session ID: {session_id}")
        print(f"Synaptic mode: {synaptic_mode}")
        print(f"Total combinations: {len(final_results)}")
        print(f"Successful: {len(successful_results)} ({100*len(successful_results)/len(final_results):.1f}%)")
        print(f"Failed: {len(failed_results)} ({100*len(failed_results)/len(final_results):.1f}%)")
        print(f"Total time: {total_experiment_time/3600:.2f} hours")

        if successful_results:
            attempts = [r.get('attempt_count', 1) for r in successful_results]
            print(f"Average attempts: {np.mean(attempts):.1f}")

            # Enhanced measure ranges
            lz_flat_values = [r['lz_matrix_flattened_mean'] for r in successful_results if not np.isnan(r.get('lz_matrix_flattened_mean', np.nan))]
            lz_spatial_values = [r['lz_spatial_patterns_mean'] for r in successful_results if not np.isnan(r.get('lz_spatial_patterns_mean', np.nan))]
            pci_values = [r['pci_normalized_mean'] for r in successful_results if not np.isnan(r.get('pci_normalized_mean', np.nan))]
            kistler_values = [r['kistler_delta_2ms_mean'] for r in successful_results if not np.isnan(r.get('kistler_delta_2ms_mean', np.nan))]
            silent_values = [r['control_percent_silent_mean'] for r in successful_results if not np.isnan(r.get('control_percent_silent_mean', np.nan))]

            if lz_flat_values:
                print(f"LZ complexity (flattened): {np.min(lz_flat_values):.1f} - {np.max(lz_flat_values):.1f}")
            if lz_spatial_values:
                print(f"LZ complexity (spatial): {np.min(lz_spatial_values):.1f} - {np.max(lz_spatial_values):.1f}")
            if pci_values:
                print(f"PCI (normalized): {np.min(pci_values):.3f} - {np.max(pci_values):.3f}")
            if kistler_values:
                print(f"Kistler coincidence (2ms): {np.min(kistler_values):.3f} - {np.max(kistler_values):.3f}")
            if silent_values:
                print(f"Silent neurons: {np.min(silent_values):.1f}% - {np.max(silent_values):.1f}%")

        # Save enhanced results
        output_file = os.path.join(output_dir, f"chaos_enhanced_session_{session_id}_{synaptic_mode}.pkl")
        save_results(final_results, output_file, use_data_subdir=False)

        print(f"\nEnhanced results saved: {output_file}")
        print("Enhanced single session experiment completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run enhanced MPI chaos experiment with comprehensive analysis"
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
    parser.add_argument("--synaptic_mode", type=str, default="dynamic",
                       choices=["immediate", "dynamic"],
                       help="Synaptic mode: immediate or dynamic")
    parser.add_argument("--v_th_distributions", type=str, nargs='+',
                       default=["normal"], choices=["normal", "uniform"],
                       help="Threshold distributions to test")

    args = parser.parse_args()

    run_mpi_chaos_experiment(
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
        v_th_distributions=args.v_th_distributions
    )
