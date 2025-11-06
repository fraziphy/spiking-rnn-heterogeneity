# runners/mpi_utils.py - Shared utilities for MPI runners
"""
Common utilities for MPI-parallelized experiments.
Includes work distribution, system monitoring, and recovery mechanisms.

IMPORTANT: Adjust thresholds based on YOUR system's normal operating conditions!
"""

import time
import os
from typing import Tuple
from mpi4py import MPI


# ============================================================================
# CONFIGURABLE THRESHOLDS - ADJUST THESE FOR YOUR SYSTEM!
# ============================================================================

# Temperature threshold (°C)
# - Check your system's normal temp: Run `sensors` or check with psutil
# - Set this 5-10°C ABOVE your normal maximum
# - Example: Normal max = 92°C → Set to 100°C
TEMP_THRESHOLD = float(os.getenv('TEMP_THRESHOLD', '100'))  # Default: 100°C

# CPU threshold (%)
# - Triggers when CPU usage exceeds this
# - 98% is usually fine (allows full utilization)
CPU_THRESHOLD = float(os.getenv('CPU_THRESHOLD', '98'))  # Default: 98%

# Memory threshold (%)
# - Triggers when RAM usage exceeds this
# - 95% is usually safe (leaves 5% buffer)
MEMORY_THRESHOLD = float(os.getenv('MEMORY_THRESHOLD', '95'))  # Default: 95%

# ============================================================================


def distribute_work_for_rank(total_jobs: int, rank: int, size: int) -> Tuple[int, int]:
    """
    Calculate work distribution for a specific MPI rank.

    Args:
        total_jobs: Total number of jobs to distribute
        rank: Rank ID
        size: Total number of ranks

    Returns:
        Tuple of (start_idx, end_idx) for this rank
    """
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
    """
    Distribute work among MPI processes for current process.

    Args:
        total_jobs: Total number of jobs
        comm: MPI communicator

    Returns:
        Tuple of (start_idx, end_idx) for current rank
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    return distribute_work_for_rank(total_jobs, rank, size)


def monitor_system_health() -> Tuple[bool, str]:
    """
    Monitor system health with CONFIGURABLE thresholds.
    
    Thresholds can be set via environment variables or by editing this file:
    - TEMP_THRESHOLD: Temperature threshold in °C (default: 100)
    - CPU_THRESHOLD: CPU usage threshold in % (default: 98)
    - MEMORY_THRESHOLD: Memory usage threshold in % (default: 95)

    Returns:
        Tuple of (is_healthy, status_message)
    """
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

        # Check against configurable thresholds
        critical_issues = []
        if temp_available and max_temp > TEMP_THRESHOLD:
            critical_issues.append(f"Temperature {max_temp:.1f}°C > {TEMP_THRESHOLD:.0f}°C")
        if cpu_percent > CPU_THRESHOLD:
            critical_issues.append(f"CPU {cpu_percent:.1f}% > {CPU_THRESHOLD:.0f}%")
        if memory_percent > MEMORY_THRESHOLD:
            critical_issues.append(f"Memory {memory_percent:.1f}% > {MEMORY_THRESHOLD:.0f}%")

        if critical_issues:
            return False, f"CRITICAL: {'; '.join(critical_issues)}"
        else:
            status_parts = [f"CPU: {cpu_percent:.1f}%", f"Memory: {memory_percent:.1f}%"]
            if temp_available:
                status_parts.insert(0, f"Temp: {max_temp:.1f}°C")
            return True, "HEALTHY - " + " | ".join(status_parts)

    except ImportError:
        return True, "Health monitoring unavailable (psutil not installed)"
    except Exception as e:
        return True, f"Health monitoring error: {str(e)}"


def recovery_break(rank: int, duration: int = 300, reason: str = "system_stress"):
    """
    Implement recovery break for individual rank.

    Args:
        rank: Rank ID
        duration: Break duration in seconds (default: 300 = 5 min)
        reason: Reason for break
    """
    print(f"[Rank {rank}] RECOVERY BREAK: {reason} ({duration//60} min)")

    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        if int(elapsed) % 60 == 0:
            remaining = duration - elapsed
            print(f"[Rank {rank}] Break: {int(elapsed//60)}/{duration//60} min remaining: {int(remaining//60)} min")

        healthy, status = monitor_system_health()
        if healthy and "HEALTHY" in status:
            print(f"[Rank {rank}] System recovered early - resuming work")
            break

        time.sleep(30)

    print(f"[Rank {rank}] Recovery complete")


def print_work_distribution(total_jobs: int, size: int, max_display: int = 8):
    """
    Print work distribution across ranks.

    Args:
        total_jobs: Total number of jobs
        size: Number of MPI ranks
        max_display: Maximum number of ranks to display details for
    """
    print(f"\nWork Distribution:")
    for r in range(min(size, max_display)):
        s, e = distribute_work_for_rank(total_jobs, r, size)
        n_jobs = e - s
        print(f"  Rank {r:2d}: {n_jobs:3d} combinations")
    if size > max_display:
        print(f"  ... and {size-max_display} more ranks")


def estimate_computation_time(total_combinations: int, size: int,
                             time_per_combo: float, trials_per_combo: int = 1) -> float:
    """
    Estimate total computation time.

    Args:
        total_combinations: Total number of parameter combinations
        size: Number of MPI processes
        time_per_combo: Expected time per combination (seconds)
        trials_per_combo: Number of trials per combination

    Returns:
        Estimated time in hours
    """
    return (total_combinations * time_per_combo * trials_per_combo) / (size * 3600)


def print_threshold_info():
    """Print current threshold configuration."""
    print("="*70)
    print("SYSTEM HEALTH MONITORING - CURRENT THRESHOLDS")
    print("="*70)
    print(f"🌡️  Temperature: > {TEMP_THRESHOLD:.0f}°C")
    print(f"💻 CPU Usage:   > {CPU_THRESHOLD:.0f}%")
    print(f"🧠 Memory Usage: > {MEMORY_THRESHOLD:.0f}%")
    print()
    print("To change thresholds:")
    print("  1. Edit TEMP_THRESHOLD, CPU_THRESHOLD, MEMORY_THRESHOLD in mpi_utils.py")
    print("  2. Or set environment variables before running:")
    print(f"     export TEMP_THRESHOLD=105")
    print(f"     export CPU_THRESHOLD=98")
    print(f"     export MEMORY_THRESHOLD=95")
    print("="*70)
