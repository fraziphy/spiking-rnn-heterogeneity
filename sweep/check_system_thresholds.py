#!/usr/bin/env python3
"""
Check your system's current stats and recommend appropriate thresholds.
Run this BEFORE deploying the sweep to determine safe threshold values.

Usage:
  python3 check_system_thresholds.py
  python3 check_system_thresholds.py --stress-test  # Run under load
"""

import argparse
import time
import sys

def check_system():
    """Check current system stats and recommend thresholds."""
    try:
        import psutil
    except ImportError:
        print("ERROR: psutil not installed!")
        print("Install with: pip install psutil")
        sys.exit(1)
    
    print("="*70)
    print("SYSTEM HEALTH CHECK")
    print("="*70)
    print()
    
    # Temperature
    print("🌡️  TEMPERATURE")
    print("-" * 70)
    temps = psutil.sensors_temperatures()
    max_temp = 0
    temp_available = False
    
    if temps:
        for name, entries in temps.items():
            print(f"\n  Sensor: {name}")
            for entry in entries:
                if entry.current:
                    label = entry.label or "N/A"
                    high = f"{entry.high:.0f}°C" if entry.high else "N/A"
                    critical = f"{entry.critical:.0f}°C" if entry.critical else "N/A"
                    print(f"    {label:20s}: {entry.current:5.1f}°C (high: {high:>6s}, critical: {critical:>6s})")
                    if entry.current > max_temp:
                        max_temp = entry.current
                        temp_available = True
        
        if temp_available:
            print(f"\n  📊 MAXIMUM: {max_temp:.1f}°C")
            recommended_temp = max_temp + 10
            print(f"  ✅ RECOMMENDED THRESHOLD: {recommended_temp:.0f}°C (current + 10°C buffer)")
        else:
            print("  ⚠️  No temperature readings available")
    else:
        print("  ⚠️  Temperature monitoring not available on this system")
        print("  ℹ️  The recovery mechanism will skip temperature checks")
    
    # CPU
    print("\n" + "="*70)
    print("💻 CPU USAGE")
    print("-" * 70)
    cpu_percent = psutil.cpu_percent(interval=1.0)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    
    print(f"  Cores: {cpu_count_physical} physical, {cpu_count_logical} logical")
    print(f"  Current usage: {cpu_percent:.1f}%")
    
    # Check per-core
    per_cpu = psutil.cpu_percent(interval=0.5, percpu=True)
    max_cpu = max(per_cpu)
    print(f"  Max single-core usage: {max_cpu:.1f}%")
    
    print(f"\n  ✅ RECOMMENDED THRESHOLD: 98% (standard)")
    
    # Memory
    print("\n" + "="*70)
    print("🧠 MEMORY USAGE")
    print("-" * 70)
    memory = psutil.virtual_memory()
    
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Free: {memory.free / (1024**3):.1f} GB")
    
    print(f"\n  ✅ RECOMMENDED THRESHOLD: 95% (standard)")
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print()
    print("Current system status:")
    if temp_available:
        print(f"  🌡️  Temperature: {max_temp:.1f}°C")
        print(f"     ⚠️  WARNING: Normal operating temp is HIGH!" if max_temp > 80 else "     ✓  Normal operating temperature")
    else:
        print(f"  🌡️  Temperature: Not available")
    print(f"  💻 CPU: {cpu_percent:.1f}%")
    print(f"  🧠 Memory: {memory.percent:.1f}%")
    
    print("\n" + "-"*70)
    print("Recommended thresholds for mpi_utils.py:")
    print("-"*70)
    
    if temp_available:
        recommended_temp = int(max_temp + 10)
        print(f"  TEMP_THRESHOLD = {recommended_temp}  # Your max: {max_temp:.1f}°C + 10°C buffer")
        if max_temp > 90:
            print(f"     ⚠️  NOTE: Default is 90°C, but your system runs HOT!")
            print(f"     ⚠️  You MUST change this or jobs will constantly fail!")
    else:
        print(f"  TEMP_THRESHOLD = 100  # Temperature monitoring not available (won't be checked)")
    
    print(f"  CPU_THRESHOLD = 98   # Standard threshold")
    print(f"  MEMORY_THRESHOLD = 95  # Standard threshold")
    
    print("\n" + "-"*70)
    print("How to apply these thresholds:")
    print("-"*70)
    print("  Option 1: Edit mpi_utils.py directly")
    print("    Open: runners/mpi_utils.py")
    print("    Change lines at top of file:")
    if temp_available:
        print(f"      TEMP_THRESHOLD = {int(max_temp + 10)}")
    print("      CPU_THRESHOLD = 98")
    print("      MEMORY_THRESHOLD = 95")
    print()
    print("  Option 2: Set environment variables")
    if temp_available:
        print(f"    export TEMP_THRESHOLD={int(max_temp + 10)}")
    print("    export CPU_THRESHOLD=98")
    print("    export MEMORY_THRESHOLD=95")
    print("    ./sweep/run_sweep_spontaneous.sh")
    
    print("\n" + "="*70)
    
    # Health check with current thresholds
    print("\nTesting with DEFAULT thresholds (Temp: 90°C, CPU: 98%, Mem: 95%):")
    print("-" * 70)
    
    issues = []
    if temp_available and max_temp > 90:
        issues.append(f"❌ Temperature {max_temp:.1f}°C > 90°C (DEFAULT will FAIL!)")
    else:
        if temp_available:
            issues.append(f"✅ Temperature {max_temp:.1f}°C ≤ 90°C (OK)")
    
    if cpu_percent > 98:
        issues.append(f"❌ CPU {cpu_percent:.1f}% > 98% (Currently under stress)")
    else:
        issues.append(f"✅ CPU {cpu_percent:.1f}% ≤ 98% (OK)")
    
    if memory.percent > 95:
        issues.append(f"❌ Memory {memory.percent:.1f}% > 95% (Running low!)")
    else:
        issues.append(f"✅ Memory {memory.percent:.1f}% ≤ 95% (OK)")
    
    for issue in issues:
        print(f"  {issue}")
    
    # Warning about temp
    if temp_available and max_temp > 90:
        print("\n" + "⚠️ "*35)
        print("⚠️  CRITICAL WARNING: YOUR SYSTEM RUNS HOT!")
        print("⚠️ "*35)
        print(f"⚠️  Current temperature: {max_temp:.1f}°C")
        print(f"⚠️  Default threshold: 90°C")
        print(f"⚠️  ")
        print(f"⚠️  If you don't change TEMP_THRESHOLD, your jobs will:")
        print(f"⚠️    - Constantly trigger 'CRITICAL' warnings")
        print(f"⚠️    - Pause for 3-5 minute recovery breaks")
        print(f"⚠️    - Many jobs will fail after 3 retry attempts")
        print(f"⚠️  ")
        print(f"⚠️  YOU MUST CHANGE TEMP_THRESHOLD TO AT LEAST {int(max_temp + 10)}°C")
        print("⚠️ "*35)
    
    print("\n" + "="*70)


def stress_test():
    """Run a brief stress test to see max values under load."""
    try:
        import psutil
    except ImportError:
        print("ERROR: psutil not installed!")
        sys.exit(1)
    
    print("="*70)
    print("STRESS TEST")
    print("="*70)
    print("\nRunning 30-second CPU stress test...")
    print("(This will spike CPU to check maximum temperatures)")
    print()
    
    # Baseline
    temps_before = psutil.sensors_temperatures()
    cpu_before = psutil.cpu_percent(interval=1)
    
    # Stress for 30 seconds
    import multiprocessing
    import math
    
    def cpu_stress():
        """Stress single CPU core."""
        end_time = time.time() + 30
        while time.time() < end_time:
            math.factorial(10000)
    
    print("Starting stress on all cores...")
    n_cores = psutil.cpu_count(logical=True)
    processes = []
    for _ in range(n_cores):
        p = multiprocessing.Process(target=cpu_stress)
        p.start()
        processes.append(p)
    
    # Monitor during stress
    print("\nMonitoring (30 seconds)...")
    max_temp_stress = 0
    max_cpu_stress = 0
    
    for i in range(6):  # Check every 5 seconds
        time.sleep(5)
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp_stress:
                        max_temp_stress = entry.current
        
        cpu = psutil.cpu_percent(interval=0.1)
        if cpu > max_cpu_stress:
            max_cpu_stress = cpu
        
        print(f"  {(i+1)*5}s: Temp={max_temp_stress:.1f}°C, CPU={cpu:.0f}%")
    
    # Wait for processes
    for p in processes:
        p.join()
    
    print("\nStress test complete!")
    print("-" * 70)
    print(f"Maximum temperature under load: {max_temp_stress:.1f}°C")
    print(f"Maximum CPU under load: {max_cpu_stress:.0f}%")
    print()
    print(f"✅ RECOMMENDED THRESHOLD: {int(max_temp_stress + 5)}°C (max under load + 5°C)")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check system and recommend thresholds")
    parser.add_argument("--stress-test", action="store_true", 
                       help="Run CPU stress test to find max temperature under load")
    
    args = parser.parse_args()
    
    if args.stress_test:
        stress_test()
    else:
        check_system()
