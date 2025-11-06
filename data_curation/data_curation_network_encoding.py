# data_curation_network_encoding.py
"""
Data curation for network encoding experiments (Main Figure 2 & Supplementary Figure 2)
Processes encoding task data and HD input signals, saves to network_encoding_data.pkl
"""

import numpy as np
import pickle
import os
import sys
from scipy.signal import welch

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change working directory to script directory so files save here
os.chdir(script_dir)
# Add project root to path (one level up from script directory)
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.hd_input import HDInputGenerator

# =============================================================================
# PARAMETERS
# =============================================================================

START_SESSION = 0
END_SESSION = 20  # Exclusive (will process sessions START_SESSION to END_SESSION-1)

V_TH_VALUES = [0.0]
G_VALUES = [1.0]
RATE_VALUES = [30]
EMBED_DIM_INPUT = [1, 2, 3, 4, 5]

# Paths (relative to project root)
HD_SIGNALS_DIR = os.path.join(project_root, 'hd_signals', 'autoencoding_sweep', 'inputs')
RESULTS_DIR = os.path.join(project_root, 'results', 'autoencoding_sweep', 'data')
OUTPUT_FILE = 'network_encoding_data.pkl'  # Saves in script_dir (data_curation/)

print("="*80)
print("DATA CURATION FOR NETWORK ENCODING EXPERIMENTS")
print("="*80)
print(f"Sessions: {START_SESSION} to {END_SESSION-1} (inclusive)")
print(f"Embedding dimensions: {EMBED_DIM_INPUT}")
print(f"V_th values: {V_TH_VALUES}")
print(f"G values: {G_VALUES}")
print(f"Static input rates: {RATE_VALUES}")
print()

# =============================================================================
# HELPER FUNCTION: PARTICIPATION RATIO
# =============================================================================

def compute_participation_ratio(X):
    """
    Compute participation ratio from data matrix X (time x features).
    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
    """
    # Handle case where X is 1D or has only 1 feature
    if X.ndim == 1 or X.shape[1] == 1:
        # For 1D data, participation ratio is 1.0 by definition
        return 1.0

    cov_matrix = np.cov(X, rowvar=False)

    # Handle scalar covariance (should not happen after above check, but be safe)
    if cov_matrix.ndim == 0:
        return 1.0

    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Keep only positive eigenvalues

    if len(eigenvalues) == 0:
        return 0.0

    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr

# =============================================================================
# PART 1: GENERATE HD INPUT EXAMPLES FOR PANEL A (k=4, d=1 and d=4)
# =============================================================================

print("="*80)
print("PART 1: GENERATING HD INPUT EXAMPLES FOR PANEL A")
print("="*80)

session_id = 120
pattern_id = 0
k = 4  # Embedding dimension
d1 = 1  # Low intrinsic dimensionality
d2 = 4  # High intrinsic dimensionality
dt = 0.1
stimulus_duration = 300.0

# Create generators (no caching - these are just temporary for illustration)
gen1 = HDInputGenerator(embed_dim=k, dt=dt, signal_cache_dir=None)
gen2 = HDInputGenerator(embed_dim=k, dt=dt, signal_cache_dir=None)

print(f"\nGenerating pattern with k={k}, d={d1}...")
gen1.initialize_base_input(
    session_id=session_id,
    hd_dim=d1,
    pattern_id=pattern_id,
    rate_rnn_params={
        'n_neurons': 1000,
        'T': 200.0 + stimulus_duration,
        'g': 2.0
    }
)
pattern_d1 = gen1.Y_base.copy() + 2

print(f"Generating pattern with k={k}, d={d2}...")
gen2.initialize_base_input(
    session_id=session_id,
    hd_dim=d2,
    pattern_id=pattern_id,
    rate_rnn_params={
        'n_neurons': 1000,
        'T': 200.0 + stimulus_duration,
        'g': 2.0
    }
)
pattern_d2 = gen2.Y_base.copy() + 2

print(f"Pattern d={d1}: shape = {pattern_d1.shape}")
print(f"Pattern d={d2}: shape = {pattern_d2.shape}")
print("✓ HD input examples generated")
print()

# =============================================================================
# PART 2: PANEL B - EMPIRICAL VS THEORETICAL DIMENSIONALITY
# =============================================================================

print("="*80)
print("PART 2: COMPUTING EMPIRICAL VS THEORETICAL DIMENSIONALITY")
print("="*80)
print("Generating HD signals on-the-fly and computing empirical dimensionality...")
print()

empirical_dims = []
theoretical_dims = []
sessions_list = []
embed_dims_list = []

total_expected = 0

for session in range(START_SESSION, END_SESSION):
    print(f"Processing session {session}...")
    for embed_dim in EMBED_DIM_INPUT:
        for intrinsic_dim in range(1, embed_dim + 1):
            total_expected += 1

            # Generate HD signal on-the-fly
            gen_temp = HDInputGenerator(
                embed_dim=embed_dim,
                dt=0.1,
                signal_cache_dir=HD_SIGNALS_DIR
            )

            gen_temp.initialize_base_input(
                session_id=session,
                hd_dim=intrinsic_dim,
                pattern_id=0,
                rate_rnn_params={
                    'n_neurons': 1000,
                    'T': 500.0,
                    'g': 2.0
                }
            )

            Y_base = gen_temp.Y_base

            # Compute empirical dimensionality using participation ratio
            empirical_dim = compute_participation_ratio(Y_base)

            empirical_dims.append(empirical_dim)
            theoretical_dims.append(intrinsic_dim)
            sessions_list.append(session)
            embed_dims_list.append(embed_dim)

print(f"\nProcessed {len(empirical_dims)} / {total_expected} HD signals")
print("✓ Empirical dimensionality computed for all signals")
print()

# =============================================================================
# PART 3: PANEL D - R² VS D (FOR EACH K)
# =============================================================================

print("="*80)
print("PART 3: COMPUTING R² VS INTRINSIC DIMENSIONALITY")
print("="*80)

r2_vs_d = {
    'k_values': EMBED_DIM_INPUT,
    'd_values': {},
    'mean': {},
    'std': {}
}

missing_result_files = []
total_expected_results = 0

for k_val in EMBED_DIM_INPUT:
    r2_vs_d['d_values'][k_val] = list(range(1, k_val + 1))
    r2_vs_d['mean'][k_val] = {}
    r2_vs_d['std'][k_val] = {}

    for d in range(1, k_val + 1):
        r2_values = []

        for session in range(START_SESSION, END_SESSION):
            for v_th in V_TH_VALUES:
                for g in G_VALUES:
                    for rate in RATE_VALUES:
                        total_expected_results += 1

                        filename = os.path.join(
                            RESULTS_DIR,
                            f'task_autoencoding_session_{session}_vth_{v_th:.3f}_g_{g:.3f}_rate_{rate}_hdin_{d}_embd_{k_val}_npat_1.pkl'
                        )

                        if not os.path.exists(filename):
                            missing_result_files.append(filename)
                            continue

                        with open(filename, 'rb') as f:
                            results = pickle.load(f)

                        r2_values.append(results[0]['test_r2_mean'])

        if len(r2_values) > 0:
            r2_vs_d['mean'][k_val][d] = np.mean(r2_values)
            r2_vs_d['std'][k_val][d] = np.std(r2_values)
        else:
            r2_vs_d['mean'][k_val][d] = np.nan
            r2_vs_d['std'][k_val][d] = np.nan

print(f"Processed R² data for {len(EMBED_DIM_INPUT)} embedding dimensions")

if missing_result_files:
    print(f"\n⚠ WARNING: {len(missing_result_files)} result files are missing:")
    for mf in missing_result_files[:10]:
        print(f"  - {mf}")
    if len(missing_result_files) > 10:
        print(f"  ... and {len(missing_result_files) - 10} more")
    print()
    raise FileNotFoundError(f"Missing {len(missing_result_files)} result files. Cannot proceed.")

print("✓ R² vs d computed for all k values")
print()

# =============================================================================
# PART 4: PANEL E - NETWORK DIMENSIONALITY VS D (FOR EACH K)
# =============================================================================

print("="*80)
print("PART 4: COMPUTING NETWORK DIMENSIONALITY VS INTRINSIC DIMENSIONALITY")
print("="*80)

pr_vs_d = {
    'k_values': EMBED_DIM_INPUT,
    'd_values': {},
    'mean': {},
    'std': {}
}

for k_val in EMBED_DIM_INPUT:
    pr_vs_d['d_values'][k_val] = list(range(1, k_val + 1))
    pr_vs_d['mean'][k_val] = {}
    pr_vs_d['std'][k_val] = {}

    for d in range(1, k_val + 1):
        pr_values = []

        for session in range(START_SESSION, END_SESSION):
            for v_th in V_TH_VALUES:
                for g in G_VALUES:
                    for rate in RATE_VALUES:
                        filename = os.path.join(
                            RESULTS_DIR,
                            f'task_autoencoding_session_{session}_vth_{v_th:.3f}_g_{g:.3f}_rate_{rate}_hdin_{d}_embd_{k_val}_npat_1.pkl'
                        )

                        if not os.path.exists(filename):
                            continue

                        with open(filename, 'rb') as f:
                            results = pickle.load(f)

                        # Extract participation ratio
                        pr = results[0]['dimensionality_summary']['bin_2.0ms']['participation_ratio_mean']
                        pr_values.append(pr)

        if len(pr_values) > 0:
            pr_vs_d['mean'][k_val][d] = np.mean(pr_values)
            pr_vs_d['std'][k_val][d] = np.std(pr_values)
        else:
            pr_vs_d['mean'][k_val][d] = np.nan
            pr_vs_d['std'][k_val][d] = np.nan

print("✓ Network dimensionality vs d computed for all k values")
print()

# =============================================================================
# PART 5: SUPPLEMENTARY FIGURE - GENERATE RATE RNN DATA
# =============================================================================

print("="*80)
print("PART 5: GENERATING RATE RNN DATA FOR SUPPLEMENTARY FIGURE")
print("="*80)

from src.hd_input import run_rate_rnn

# Generate rate RNN activity
n_neurons_rnn = 1000
T_rnn = 500.0  # 200ms transient + 300ms recording
dt_rnn = 0.1
g_rnn = 2.0
session_rnn = 42  # Fixed session for reproducibility
hd_dim_rnn = 20
embed_dim_rnn = 20

print(f"Running rate RNN (n={n_neurons_rnn}, T={T_rnn}ms, g={g_rnn})...")
rates, time_rnn = run_rate_rnn(
    n_neurons=n_neurons_rnn,
    T=T_rnn,
    dt=dt_rnn,
    g=g_rnn,
    session_id=session_rnn,
    hd_dim=hd_dim_rnn,
    embed_dim=embed_dim_rnn,
    pattern_id=0
)

print(f"Rate RNN output shape: {rates.shape}")
print("✓ Rate RNN data generated")
print()

# =============================================================================
# PART 5B: SUPPLEMENTARY FIGURE - GENERATE HD SIGNAL FOR subplot_a ROW 2
# =============================================================================

print("="*80)
print("PART 5B: GENERATING HD SIGNAL (k=5, d=5) FOR SUPPLEMENTARY FIGURE")
print("="*80)

session_hd = 42
k_hd = 5  # Changed from 20 to 5 to match first two rows
d_hd = 5
pattern_id_hd = 0

print(f"Generating HD signal (k={k_hd}, d={d_hd})...")
gen_hd = HDInputGenerator(embed_dim=k_hd, dt=dt_rnn, signal_cache_dir=None)
gen_hd.initialize_base_input(
    session_id=session_hd,
    hd_dim=d_hd,
    pattern_id=pattern_id_hd,
    rate_rnn_params={
        'n_neurons': 1000,
        'T': 500.0,
        'g': 2.0
    }
)
Y_hd_signal = gen_hd.Y_base.copy()

print(f"HD signal shape: {Y_hd_signal.shape}")
print("✓ HD signal generated")
print()

# =============================================================================
# PART 6: SUPPLEMENTARY FIGURE - FREQUENCY ANALYSIS WITH CV
# =============================================================================

print("="*80)
print("PART 6: FREQUENCY ANALYSIS FOR ALL HD SIGNALS")
print("="*80)

# Define frequency bands (Hz) - adjusted for 300ms signal duration
# With fs=10000 Hz and nperseg=len(signal), frequency resolution ≈ 3.3 Hz
frequency_bands = {
    'Theta': (4, 8),      # 4-8 Hz
    'Alpha': (8, 13),     # 8-13 Hz
    'Beta': (13, 30),     # 13-30 Hz
    'Gamma': (30, 100),   # 30-100 Hz
    'High-Gamma': (100, 200)  # 100-200 Hz
}

all_signals = []
all_relative_powers = {band: [] for band in frequency_bands.keys()}

freq_max = 200  # Hz

print("Computing frequency content for all HD signals...")
n_signals = 0

for session in range(START_SESSION, END_SESSION):
    for embed_dim in EMBED_DIM_INPUT:
        for intrinsic_dim in range(1, embed_dim + 1):
            filename = os.path.join(
                HD_SIGNALS_DIR,
                f'hd_signal_session_{session}_hd_{intrinsic_dim}_k_{embed_dim}_pattern_0.pkl'
            )

            if not os.path.exists(filename):
                continue

            # Load the cached signal
            with open(filename, 'rb') as f:
                signal_data = pickle.load(f)

            Y_base = signal_data['Y_base']

            # For each channel in the signal, compute frequency content
            # We'll use the first and last channel as representative
            channels_to_analyze = [0]
            if Y_base.shape[1] > 1:
                channels_to_analyze.append(Y_base.shape[1] - 1)

            for ch_idx in channels_to_analyze:
                signal = Y_base[:, ch_idx]

                # Welch's method for PSD
                # Use full signal length for nperseg to get best frequency resolution (~3.3 Hz)
                fs = 1000.0 / 0.1  # Sampling frequency (dt=0.1 for HD signals)
                f, Pxx = welch(signal, fs=fs, nperseg=len(signal))

                # Limit to freq_max
                freq_lim = np.searchsorted(f, freq_max)
                f = f[:freq_lim]
                Pxx = Pxx[:freq_lim]

                # Calculate power in each frequency band
                total_power = np.sum(Pxx)

                if total_power > 0:  # Avoid division by zero
                    for band_name, (f_low, f_high) in frequency_bands.items():
                        idx_band = np.where((f >= f_low) & (f < f_high))[0]
                        band_power = np.sum(Pxx[idx_band])
                        relative_power = (band_power / total_power) * 100.0  # Percentage
                        all_relative_powers[band_name].append(relative_power)

                    n_signals += 1

print(f"Processed {n_signals} signal channels")

# Compute CV for each frequency band
cv_per_band = {}
for band_name in frequency_bands.keys():
    powers = np.array(all_relative_powers[band_name])
    mean_power = np.mean(powers)
    std_power = np.std(powers)

    # Debug output
    print(f"\n{band_name} band:")
    print(f"  Number of values: {len(powers)}")
    print(f"  Mean power: {mean_power:.4f}%")
    print(f"  Std power: {std_power:.4f}%")
    print(f"  Min power: {np.min(powers):.4f}%")
    print(f"  Max power: {np.max(powers):.4f}%")

    # Handle zero mean (band is completely empty across all signals)
    if mean_power > 0:
        cv = (std_power / mean_power) * 100.0  # CV as percentage
    else:
        cv = 0.0  # No variance if mean is zero

    cv_per_band[band_name] = cv
    print(f"  CV = {cv:.2f}%")

print("✓ Frequency analysis completed with CV computation")
print()

# =============================================================================
# PART 7: SUPPLEMENTARY FIGURE - R² VS RMSE SCATTER
# =============================================================================

print("="*80)
print("PART 7: COLLECTING R² VS RMSE DATA FOR SCATTER PLOT")
print("="*80)

r2_folds = []
rmse_folds = []

for session in range(START_SESSION, END_SESSION):
    for embed_dim in EMBED_DIM_INPUT:
        for intrinsic_dim in range(1, embed_dim + 1):
            for v_th in V_TH_VALUES:
                for g in G_VALUES:
                    for rate in RATE_VALUES:
                        filename = os.path.join(
                            RESULTS_DIR,
                            f'task_autoencoding_session_{session}_vth_{v_th:.3f}_g_{g:.3f}_rate_{rate}_hdin_{intrinsic_dim}_embd_{embed_dim}_npat_1.pkl'
                        )

                        if not os.path.exists(filename):
                            continue

                        with open(filename, 'rb') as f:
                            results = pickle.load(f)

                        # Each has 20 folds
                        r2_per_fold = results[0]['cv_r2_per_fold']
                        rmse_per_fold = results[0]['cv_rmse_per_fold']

                        r2_folds.extend(r2_per_fold)
                        rmse_folds.extend(rmse_per_fold)

print(f"Collected {len(r2_folds)} data points from CV folds")

# Compute correlations
from scipy.stats import spearmanr, pearsonr
r2_array = np.array(r2_folds)
rmse_array = np.array(rmse_folds)
spearman_corr, spearman_p = spearmanr(r2_array, rmse_array)
pearson_corr, pearson_p = pearsonr(r2_array, rmse_array)

print(f"Spearman correlation: ρ = {spearman_corr:.3f} (p = {spearman_p:.2e})")
print(f"Pearson correlation: r = {pearson_corr:.3f} (p = {pearson_p:.2e})")
print("✓ R² vs RMSE data collected")
print()

# =============================================================================
# SAVE ALL DATA
# =============================================================================

print("="*80)
print("SAVING DATA TO PICKLE FILE")
print("="*80)

data = {
    # Panel a: HD input examples
    'pattern_d1': pattern_d1,
    'pattern_d2': pattern_d2,
    'pattern_k': k,
    'pattern_dt': dt,

    # Panel b: Empirical vs theoretical
    'empirical_dims': np.array(empirical_dims),
    'theoretical_dims': np.array(theoretical_dims),
    'sessions': np.array(sessions_list),
    'embed_dims': np.array(embed_dims_list),

    # Panel d: R² vs d
    'r2_vs_d': r2_vs_d,

    # Panel e: Network PR vs d
    'pr_vs_d': pr_vs_d,

    # Supplementary: Rate RNN
    'rate_rnn_rates': rates,
    'rate_rnn_time': time_rnn,
    'rate_rnn_dt': dt_rnn,

    # Supplementary: HD signal for subplot_a row 2
    'hd_signal': Y_hd_signal,
    'hd_signal_k': k_hd,
    'hd_signal_d': d_hd,

    # Supplementary: Frequency analysis
    'frequency_bands': frequency_bands,
    'all_relative_powers': all_relative_powers,
    'cv_per_band': cv_per_band,

    # Supplementary: R² vs RMSE
    'r2_folds': np.array(r2_folds),
    'rmse_folds': np.array(rmse_folds),
    'spearman_corr': spearman_corr,
    'spearman_p': spearman_p,
    'pearson_corr': pearson_corr,
    'pearson_p': pearson_p,

    # Metadata
    'start_session': START_SESSION,
    'end_session': END_SESSION,
    'embed_dim_input': EMBED_DIM_INPUT,
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
print("\nData summary:")
print(f"  - HD input examples: 2 patterns (k={k}, d={d1} and d={d2})")
print(f"  - Empirical vs theoretical dims: {len(empirical_dims)} points")
print(f"  - R² vs d: {len(EMBED_DIM_INPUT)} k values, mean±std computed")
print(f"  - Network PR vs d: {len(EMBED_DIM_INPUT)} k values, mean±std computed")
print(f"  - Rate RNN data: shape {rates.shape}")
print(f"  - Frequency analysis: {n_signals} signals, CV computed for {len(frequency_bands)} bands")
print(f"  - R² vs RMSE: {len(r2_folds)} CV fold points")
print(f"\nOutput file: {OUTPUT_FILE}")
print("="*80)
