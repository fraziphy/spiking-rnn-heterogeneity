# data_curation_network_classification.py
"""
Data curation for network classification experiments (Main Figure 3)
Processes classification task data, saves to network_classification_data.pkl

Updated for v7.0.0:
- Supports overlapping/partitioned data types
- Independent of encoding data (no longer reads encoding file)
- Stores per-session values for scatter plots
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

# =============================================================================
# PARAMETERS
# =============================================================================

START_SESSION = 0
END_SESSION = 20  # Exclusive (will process sessions START_SESSION to END_SESSION-1)

V_TH_VALUES = [0.0]
G_VALUES = [1.0]
RATE_VALUES = [30]
EMBED_DIM_INPUT = [1, 2, 3, 4, 5, 6, 7]
N_PATTERNS = 4  # Number of patterns for classification

# NEW: Data types to process (can be ["overlapping"], ["partitioned"], or both)
DATA_TYPES = ["overlapping", "partitioned"]

# Output file (saves in script_dir)
OUTPUT_FILE = 'network_classification_data.pkl'

print("="*80)
print("DATA CURATION FOR NETWORK CLASSIFICATION EXPERIMENTS")
print("="*80)
print(f"Sessions: {START_SESSION} to {END_SESSION-1} (inclusive)")
print(f"Embedding dimensions: {EMBED_DIM_INPUT}")
print(f"V_th values: {V_TH_VALUES}")
print(f"G values: {G_VALUES}")
print(f"Static input rates: {RATE_VALUES}")
print(f"Number of patterns: {N_PATTERNS}")
print(f"Data types: {DATA_TYPES}")
print()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_results_dir(data_type):
    """Get the results directory for a given data type."""
    # Updated path structure for v7.0.0
    return os.path.join(project_root, 'results', 'data', data_type, 'categorical')


def get_classification_filename(results_dir, session, v_th, g, rate, d, k, n_patterns):
    """Get the filename for a classification result file."""
    return os.path.join(
        results_dir,
        f'session_{session}_vth_{v_th:.3f}_g_{g:.3f}_rate_{rate}_hdin_{d}_embdin_{k}_npat_{n_patterns}.pkl'
    )

# =============================================================================
# CHECK WHICH DATA TYPES ARE AVAILABLE
# =============================================================================

print("="*80)
print("CHECKING AVAILABLE DATA TYPES")
print("="*80)

available_data_types = []
for data_type in DATA_TYPES:
    results_dir = get_results_dir(data_type)
    if os.path.exists(results_dir):
        # Check if there's at least one file
        files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
        if len(files) > 0:
            available_data_types.append(data_type)
            print(f"✓ {data_type}: Found {len(files)} files in {results_dir}")
        else:
            print(f"✗ {data_type}: Directory exists but no .pkl files found")
    else:
        print(f"✗ {data_type}: Directory not found ({results_dir})")

if not available_data_types:
    raise FileNotFoundError("No data types available. Please run experiments first.")

print(f"\nProceeding with data types: {available_data_types}")
print()

# =============================================================================
# COMPUTE CLASSIFICATION ACCURACY VS D FOR EACH DATA TYPE
# =============================================================================

print("="*80)
print("COMPUTING CLASSIFICATION ACCURACY VS INTRINSIC DIMENSIONALITY")
print("="*80)

# Store results per data type
all_results = {}

for data_type in available_data_types:
    print(f"\nProcessing {data_type}...")
    results_dir = get_results_dir(data_type)

    accuracy_vs_d = {
        'k_values': EMBED_DIM_INPUT,
        'd_values': {},
        'mean': {},
        'std': {},
        'per_session': {}  # NEW: Store per-session values for scatter plot
    }

    missing_files = []

    for k_val in EMBED_DIM_INPUT:
        accuracy_vs_d['d_values'][k_val] = list(range(1, k_val + 1))
        accuracy_vs_d['mean'][k_val] = {}
        accuracy_vs_d['std'][k_val] = {}
        accuracy_vs_d['per_session'][k_val] = {}

        for d in range(1, k_val + 1):
            accuracy_values = []
            session_accuracy = {}  # Store per-session

            for session in range(START_SESSION, END_SESSION):
                for v_th in V_TH_VALUES:
                    for g in G_VALUES:
                        for rate in RATE_VALUES:
                            filename = get_classification_filename(
                                results_dir, session, v_th, g, rate, d, k_val, N_PATTERNS
                            )

                            if not os.path.exists(filename):
                                missing_files.append(filename)
                                continue

                            with open(filename, 'rb') as f:
                                results = pickle.load(f)

                            # Extract accuracy
                            acc = results[0].get('test_accuracy_bayesian_mean', np.nan)

                            if not np.isnan(acc):
                                accuracy_values.append(acc)
                                session_accuracy[session] = acc

            # Store mean, std, and per-session values
            if len(accuracy_values) > 0:
                accuracy_vs_d['mean'][k_val][d] = np.mean(accuracy_values)
                accuracy_vs_d['std'][k_val][d] = np.std(accuracy_values)
                accuracy_vs_d['per_session'][k_val][d] = session_accuracy
            else:
                accuracy_vs_d['mean'][k_val][d] = np.nan
                accuracy_vs_d['std'][k_val][d] = np.nan
                accuracy_vs_d['per_session'][k_val][d] = {}

    if missing_files:
        print(f"  ⚠ {len(missing_files)} missing files for {data_type}")
        if len(missing_files) <= 5:
            for mf in missing_files:
                print(f"    - {os.path.basename(mf)}")
        else:
            print(f"    (showing first 5)")
            for mf in missing_files[:5]:
                print(f"    - {os.path.basename(mf)}")

    all_results[data_type] = {
        'accuracy_vs_d': accuracy_vs_d
    }

    print(f"  ✓ Processed {data_type}: {len(EMBED_DIM_INPUT)} k values")

print()

# =============================================================================
# SAVE ALL DATA
# =============================================================================

print("="*80)
print("SAVING DATA TO PICKLE FILE")
print("="*80)

data = {
    # Results per data type
    'available_data_types': available_data_types,
    'results_by_type': all_results,

    # For backward compatibility, also store first available type at top level
    'accuracy_vs_d': all_results[available_data_types[0]]['accuracy_vs_d'],

    # Metadata
    'start_session': START_SESSION,
    'end_session': END_SESSION,
    'embed_dim_input': EMBED_DIM_INPUT,
    'n_patterns': N_PATTERNS,
    'v_th_values': V_TH_VALUES,
    'g_values': G_VALUES,
    'rate_values': RATE_VALUES,
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
    print(f"    - Accuracy vs d: {len(EMBED_DIM_INPUT)} k values")
    print(f"    - Per-session values stored for scatter plots")
print(f"\nOutput file: {OUTPUT_FILE}")
print()
print("NOTE: Scatter plot data (encoding vs classification) is now created")
print("      in the plotting script by combining encoding and classification data.")
print("="*80)
