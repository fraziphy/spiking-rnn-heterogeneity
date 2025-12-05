# data_curation/data_curation_autoencoding.py
"""
Data curation for autoencoding experiments (Main Figure 2)
Processes autoencoding task data, saves to autoencoding_data.pkl

Computes participation ratio from evoked spikes using the same method
as stability experiments (compute_activity_dimensionality_multi_bin).

R² values are read from saved result files (no simulation).
"""

import numpy as np
import pickle
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change working directory to script directory so files save here
os.chdir(script_dir)
# Add project root to path (one level up from script directory)
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import analysis utilities
from analysis.spontaneous_analysis import compute_activity_dimensionality_multi_bin

# =============================================================================
# PARAMETERS
# =============================================================================

START_SESSION = 0
END_SESSION = 20  # Exclusive (will process sessions START_SESSION to END_SESSION-1)

V_TH_VALUES = [0.0]
G_VALUES = [1.0]
RATE_VALUES = [30.0]
EMBED_DIM_INPUT = [1, 2, 3, 4, 5, 6, 7]  # k values
N_PATTERNS = 1  # For autoencoding
N_TRIALS_PER_PATTERN = 100

# Timing parameters (must match experiment)
DT = 0.1  # ms
STIMULUS_DURATION = 300.0  # ms
N_NEURONS = 1000
BIN_SIZE = 2.0  # ms (same as stability experiment)

# Data types to process
DATA_TYPES = ["overlapping", "partitioned"]

# Output file (saves in script_dir)
OUTPUT_FILE = 'autoencoding_data.pkl'

print("="*80)
print("DATA CURATION FOR AUTOENCODING EXPERIMENTS")
print("="*80)
print(f"Sessions: {START_SESSION} to {END_SESSION-1} (inclusive)")
print(f"Embedding dimensions (k): {EMBED_DIM_INPUT}")
print(f"V_th values: {V_TH_VALUES}")
print(f"G values: {G_VALUES}")
print(f"Static input rates: {RATE_VALUES}")
print(f"Data types: {DATA_TYPES}")
print(f"Bin size for PR: {BIN_SIZE} ms")
print()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_results_dir(data_type):
    """Get the results directory for a given data type."""
    return os.path.join(project_root, 'results', 'data', data_type, 'autoencoding')


def get_spike_cache_dir(data_type):
    """Get the spike cache directory for a given data type.

    Structure: results/cached_spikes/{data_type}/
    """
    return os.path.join(project_root, 'results', 'cached_spikes', data_type)


def get_hd_signals_dir():
    """Get the HD signals directory (shared across data types)."""
    return os.path.join(project_root, 'results', 'hd_signals')


def get_encoding_filename(results_dir, session, v_th, g, rate, d, k):
    """Get the filename for an encoding result file."""
    return os.path.join(
        results_dir,
        f'session_{session}_vth_{v_th:.3f}_g_{g:.3f}_rate_{int(rate)}_hdin_{d}_embdin_{k}_npat_1.pkl'
    )


def get_spike_cache_filename(spike_cache_dir, session, v_th, g, rate, d, k, pattern_id=0):
    """
    Get the filename for a cached spike file.

    Format: session_{s}_g_{g:.3f}_vth_{v:.3f}_rate_{r:.1f}_h_{d}_d_{k}_pattern_{p}_spikes.pkl

    Note: In the filename, h={d} (intrinsic dim) and d={k} (embedding dim)
    """
    return os.path.join(
        spike_cache_dir,
        f'session_{session}_g_{g:.3f}_vth_{v_th:.3f}_rate_{rate:.1f}_h_{d}_d_{k}_pattern_{pattern_id}_spikes.pkl'
    )


def get_hd_signal_filename(hd_signals_dir, session, d, k, pattern_id=0):
    """
    Get the filename for an HD signal file.

    Format: hd_hd_input_session_{s}_hd_{d}_k_{k}_pattern_{p}.pkl
    """
    return os.path.join(
        hd_signals_dir,
        f'hd_hd_input_session_{session}_hd_{d}_k_{k}_pattern_{pattern_id}.pkl'
    )


def compute_pr_from_spikes(spike_cache_file: str, n_neurons: int = 1000,
                           stimulus_duration: float = 300.0,
                           bin_size: float = 2.0) -> dict:
    """
    Compute participation ratio from cached spikes using same method as stability experiment.

    For each trial:
    1. Get spike list
    2. Call compute_activity_dimensionality_multi_bin
    3. Extract participation_ratio

    Then compute mean/std across trials.

    Args:
        spike_cache_file: Path to cached spike file
        n_neurons: Number of neurons
        stimulus_duration: Duration of stimulus (ms)
        bin_size: Bin size for dimensionality computation (ms)

    Returns:
        Dictionary with 'mean', 'std', and 'values' (per-trial PR values)
    """
    # Load spikes
    with open(spike_cache_file, 'rb') as f:
        spike_data = pickle.load(f)

    trial_spikes = spike_data['trial_spikes']  # dict: trial_id -> list of (time, neuron_id)
    n_trials = len(trial_spikes)

    pr_values = []

    for trial_id in range(n_trials):
        spike_list = trial_spikes[trial_id]

        # Compute dimensionality using same function as stability experiment
        dimensionality_metrics = compute_activity_dimensionality_multi_bin(
            spikes=spike_list,
            num_neurons=n_neurons,
            duration=stimulus_duration,
            bin_sizes=[bin_size]
        )

        # Extract PR for this bin size
        bin_key = f'bin_{bin_size}ms'
        if bin_key in dimensionality_metrics:
            pr = dimensionality_metrics[bin_key].get('participation_ratio', np.nan)
        else:
            pr = np.nan

        pr_values.append(pr)

    pr_values = np.array(pr_values)
    valid_values = pr_values[~np.isnan(pr_values)]

    return {
        'mean': float(np.mean(valid_values)) if len(valid_values) > 0 else np.nan,
        'std': float(np.std(valid_values)) if len(valid_values) > 0 else np.nan,
        'values': pr_values.tolist(),
        'n_trials': n_trials,
        'n_valid': len(valid_values)
    }


# =============================================================================
# CHECK WHICH DATA TYPES ARE AVAILABLE
# =============================================================================

print("="*80)
print("CHECKING AVAILABLE DATA TYPES")
print("="*80)

available_data_types = []
for data_type in DATA_TYPES:
    results_dir = get_results_dir(data_type)
    spike_cache_dir = get_spike_cache_dir(data_type)

    has_results = False
    has_spikes = False

    # Check if results directory exists and has files
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
        if len(files) > 0:
            has_results = True
            print(f"  {data_type}: Found {len(files)} result files in {results_dir}")

    # Check if spike cache exists and has files
    if os.path.exists(spike_cache_dir):
        files = [f for f in os.listdir(spike_cache_dir) if f.endswith('.pkl')]
        if len(files) > 0:
            has_spikes = True
            print(f"  {data_type}: Found {len(files)} spike cache files in {spike_cache_dir}")

    if has_results or has_spikes:
        available_data_types.append(data_type)
    else:
        print(f"  {data_type}: No data found, skipping")

if len(available_data_types) == 0:
    print("\nERROR: No data found for any data type!")
    print("Expected directories:")
    for data_type in DATA_TYPES:
        print(f"  Results: {get_results_dir(data_type)}")
        print(f"  Spikes:  {get_spike_cache_dir(data_type)}")
    sys.exit(1)

print(f"\nWill process: {available_data_types}")
print()

# =============================================================================
# LOAD EXAMPLE PATTERNS FOR VISUALIZATION (Panel A)
# =============================================================================

print("="*80)
print("LOADING EXAMPLE PATTERNS FOR PANEL A")
print("="*80)

hd_signals_dir = get_hd_signals_dir()

# Load example patterns: d=1 and d=2 with k=4
k_example = 4
pattern_d1 = None
pattern_d2 = None

for d_val, var_name in [(1, 'pattern_d1'), (2, 'pattern_d2')]:
    filename = get_hd_signal_filename(hd_signals_dir, session=0, d=d_val, k=k_example, pattern_id=0)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            signal_data = pickle.load(f)
        if var_name == 'pattern_d1':
            pattern_d1 = signal_data['Y_base']
        else:
            pattern_d2 = signal_data['Y_base']
        print(f"  Loaded {var_name}: shape {signal_data['Y_base'].shape}")
    else:
        print(f"  WARNING: Could not find {filename}")

print()

# =============================================================================
# PROCESS EACH DATA TYPE
# =============================================================================

print("="*80)
print("PROCESSING ENCODING DATA")
print("="*80)

all_results = {}

for data_type in available_data_types:
    print(f"\nProcessing: {data_type}")
    print("-" * 40)

    results_dir = get_results_dir(data_type)
    spike_cache_dir = get_spike_cache_dir(data_type)

    # Storage for R² (from result files)
    r2_vs_d = {
        'k_values': EMBED_DIM_INPUT,
        'd_values': {},
        'mean': {},
        'std': {},
        'per_session': {}
    }

    # Storage for PR (computed from spikes)
    pr_vs_d = {
        'k_values': EMBED_DIM_INPUT,
        'd_values': {},
        'mean': {},
        'std': {},
        'per_session': {}
    }

    empirical_dims_list = []
    theoretical_dims_list = []

    missing_results_count = 0
    missing_spikes_count = 0
    processed_spikes_count = 0

    for k_val in EMBED_DIM_INPUT:
        r2_vs_d['d_values'][k_val] = list(range(1, k_val + 1))
        r2_vs_d['mean'][k_val] = {}
        r2_vs_d['std'][k_val] = {}
        r2_vs_d['per_session'][k_val] = {}

        pr_vs_d['d_values'][k_val] = list(range(1, k_val + 1))
        pr_vs_d['mean'][k_val] = {}
        pr_vs_d['std'][k_val] = {}
        pr_vs_d['per_session'][k_val] = {}

        for d in range(1, k_val + 1):
            r2_session_values = []  # Mean R² per session
            pr_session_values = []  # Mean PR per session
            session_r2 = {}
            session_pr = {}

            for session in range(START_SESSION, END_SESSION):
                for v_th in V_TH_VALUES:
                    for g in G_VALUES:
                        for rate in RATE_VALUES:
                            # === Get R² from results file ===
                            results_filename = get_encoding_filename(
                                results_dir, session, v_th, g, rate, d, k_val
                            )

                            if os.path.exists(results_filename):
                                with open(results_filename, 'rb') as f:
                                    results = pickle.load(f)

                                result = results[0]

                                # Get R²
                                r2_val = result.get('test_r2_mean', np.nan)
                                if np.isnan(r2_val) and 'decoding' in result:
                                    r2_val = result['decoding'].get('test_r2_mean', np.nan)

                                if not np.isnan(r2_val):
                                    r2_session_values.append(r2_val)
                                    session_r2[session] = r2_val

                                # Collect empirical dims
                                if 'input_empirical_dims' in result:
                                    emp_dim = result['input_empirical_dims'][0]
                                else:
                                    emp_dim = d
                                empirical_dims_list.append(emp_dim)
                                theoretical_dims_list.append(d)
                            else:
                                missing_results_count += 1

                            # === Compute PR from cached spikes ===
                            spike_filename = get_spike_cache_filename(
                                spike_cache_dir, session, v_th, g, rate, d, k_val, pattern_id=0
                            )

                            if os.path.exists(spike_filename):
                                try:
                                    pr_result = compute_pr_from_spikes(
                                        spike_filename,
                                        n_neurons=N_NEURONS,
                                        stimulus_duration=STIMULUS_DURATION,
                                        bin_size=BIN_SIZE
                                    )
                                    pr_mean = pr_result['mean']

                                    if not np.isnan(pr_mean):
                                        pr_session_values.append(pr_mean)
                                        session_pr[session] = pr_mean
                                        processed_spikes_count += 1

                                except Exception as e:
                                    print(f"    Warning: Could not compute PR for session {session}, k={k_val}, d={d}: {e}")
                            else:
                                missing_spikes_count += 1

            # Compute mean and std across sessions for R²
            if len(r2_session_values) > 0:
                r2_vs_d['mean'][k_val][d] = float(np.mean(r2_session_values))
                r2_vs_d['std'][k_val][d] = float(np.std(r2_session_values))
                r2_vs_d['per_session'][k_val][d] = session_r2
            else:
                r2_vs_d['mean'][k_val][d] = np.nan
                r2_vs_d['std'][k_val][d] = np.nan
                r2_vs_d['per_session'][k_val][d] = {}

            # Compute mean and std across sessions for PR
            if len(pr_session_values) > 0:
                pr_vs_d['mean'][k_val][d] = float(np.mean(pr_session_values))
                pr_vs_d['std'][k_val][d] = float(np.std(pr_session_values))
                pr_vs_d['per_session'][k_val][d] = session_pr
            else:
                pr_vs_d['mean'][k_val][d] = np.nan
                pr_vs_d['std'][k_val][d] = np.nan
                pr_vs_d['per_session'][k_val][d] = {}

        print(f"  k={k_val}: processed d=1..{k_val}")

    # Summary
    r2_count = sum(1 for k in r2_vs_d['mean'] for d in r2_vs_d['mean'][k]
                   if not np.isnan(r2_vs_d['mean'][k][d]))
    pr_count = sum(1 for k in pr_vs_d['mean'] for d in pr_vs_d['mean'][k]
                   if not np.isnan(pr_vs_d['mean'][k][d]))

    print(f"\n  Summary for {data_type}:")
    print(f"    R² values: {r2_count} (k,d) combinations")
    print(f"    PR values: {pr_count} (k,d) combinations")
    print(f"    Spike files processed: {processed_spikes_count}")

    if missing_results_count > 0:
        print(f"    ⚠ {missing_results_count} missing result files")
    if missing_spikes_count > 0:
        print(f"    ⚠ {missing_spikes_count} missing spike cache files")

    all_results[data_type] = {
        'r2_vs_d': r2_vs_d,
        'pr_vs_d': pr_vs_d,
        'empirical_dims': np.array(empirical_dims_list),
        'theoretical_dims': np.array(theoretical_dims_list)
    }

print()

# =============================================================================
# SAVE ALL DATA
# =============================================================================

print("="*80)
print("SAVING DATA TO PICKLE FILE")
print("="*80)

data = {
    # Example patterns for visualization (Panel A)
    'pattern_d1': pattern_d1,
    'pattern_d2': pattern_d2,
    'pattern_k': k_example,
    'pattern_dt': 0.1,

    # Results per data type
    'available_data_types': available_data_types,
    'results_by_type': all_results,

    # For backward compatibility, also store first available type at top level
    'r2_vs_d': all_results[available_data_types[0]]['r2_vs_d'],
    'pr_vs_d': all_results[available_data_types[0]]['pr_vs_d'],
    'empirical_dims': all_results[available_data_types[0]]['empirical_dims'],
    'theoretical_dims': all_results[available_data_types[0]]['theoretical_dims'],

    # Metadata
    'start_session': START_SESSION,
    'end_session': END_SESSION,
    'n_sessions': END_SESSION - START_SESSION,
    'embed_dim_input': EMBED_DIM_INPUT,
    'v_th_values': V_TH_VALUES,
    'g_values': G_VALUES,
    'rate_values': RATE_VALUES,
    'bin_size_ms': BIN_SIZE,
    'stimulus_duration_ms': STIMULUS_DURATION,
    'n_neurons': N_NEURONS,
    'n_trials_per_pattern': N_TRIALS_PER_PATTERN,
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f)

print(f"✓ Data saved to '{OUTPUT_FILE}'")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("DATA CURATION COMPLETE!")
print("="*80)
print(f"\nData types processed: {available_data_types}")

for dt in available_data_types:
    print(f"\n  {dt}:")
    r2_count = sum(1 for k in all_results[dt]['r2_vs_d']['mean']
                   for d in all_results[dt]['r2_vs_d']['mean'][k]
                   if not np.isnan(all_results[dt]['r2_vs_d']['mean'][k][d]))
    pr_count = sum(1 for k in all_results[dt]['pr_vs_d']['mean']
                   for d in all_results[dt]['pr_vs_d']['mean'][k]
                   if not np.isnan(all_results[dt]['pr_vs_d']['mean'][k][d]))
    print(f"    - R² vs d: {r2_count} (k,d) combinations with data")
    print(f"    - PR vs d: {pr_count} (k,d) combinations with data")
    print(f"    - Per-session values stored for scatter plots")

print(f"\nOutput file: {OUTPUT_FILE}")
print("="*80)
