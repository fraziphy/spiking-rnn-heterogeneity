# plot_supplementary_figure_2.py
"""
Supplementary Figure 2: HD input generation and frequency analysis
Follows Nature Neuroscience specifications
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
from scipy.signal import welch
from sklearn.decomposition import PCA

# Set matplotlib parameters for Nature specifications
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 2.5
plt.rcParams['ytick.major.size'] = 2.5

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(script_dir)  # Save outputs in plots directory

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("PLOTTING SUPPLEMENTARY FIGURE 2")
print("="*80)

data_file = os.path.join(project_root, 'data_curation', 'network_encoding_data.pkl')
with open(data_file, 'rb') as f:
    data = pickle.load(f)

# Extract data
Rates = data['rate_rnn_rates']
time_rnn = data['rate_rnn_time']
dt = data['rate_rnn_dt']

# HD signal for subplot_a row 2
Y_hd = data['hd_signal']
k_hd = data['hd_signal_k']
d_hd = data['hd_signal_d']

frequency_bands = data['frequency_bands']
all_relative_powers = data['all_relative_powers']
cv_per_band = data['cv_per_band']

r2_folds = data['r2_folds']
rmse_folds = data['rmse_folds']
spearman_corr = data['spearman_corr']
spearman_p = data['spearman_p']
pearson_corr = data['pearson_corr']  # Load but don't display
pearson_p = data['pearson_p']

print("Data loaded successfully!")
print(f"Rate RNN shape: {Rates.shape}")
print(f"HD signal shape: {Y_hd.shape}")
print(f"Spearman correlation: ρ = {spearman_corr:.3f}")
print(f"Pearson correlation: r = {pearson_corr:.3f}")
print()

# =============================================================================
# HELPER FUNCTION: PARTICIPATION RATIO
# =============================================================================

def compute_participation_ratio(X):
    """Compute participation ratio from data matrix X (time x features)."""
    # Handle case where X is 1D or has only 1 feature
    if X.ndim == 1 or X.shape[1] == 1:
        # For 1D data, participation ratio is 1.0 by definition
        return 1.0

    cov_matrix = np.cov(X, rowvar=False)

    # Handle scalar covariance (should not happen after above check, but be safe)
    if cov_matrix.ndim == 0:
        return 1.0

    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr

# =============================================================================
# CREATE FIGURE WITH 2x3 LAYOUT
# =============================================================================

fig = plt.figure(figsize=(7.2, 5.5))

# Main grid: 2 rows x 3 columns
main_gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.5,
                            height_ratios=[2, 1])

# Merge ax[0,0], ax[0,1], ax[1,0], ax[1,1] into subplot_a
subplot_a_spec = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=main_gs[0:2, 0:2],
    hspace=0.35, wspace=0.35
)

# Create subplot_a as 3x3 internal grid across the merged area
subplot_a = gridspec.GridSpecFromSubplotSpec(
    3, 3, subplot_spec=subplot_a_spec[:, :],
    hspace=0.4, wspace=0.4
)

# Create the 9 subplots within subplot_a
ax_a00 = fig.add_subplot(subplot_a[0, 0])
ax_a01 = fig.add_subplot(subplot_a[0, 1])
ax_a02 = fig.add_subplot(subplot_a[0, 2])

ax_a10 = fig.add_subplot(subplot_a[1, 0])
ax_a11 = fig.add_subplot(subplot_a[1, 1])
ax_a12 = fig.add_subplot(subplot_a[1, 2])

ax_a20 = fig.add_subplot(subplot_a[2, 0])
ax_a21 = fig.add_subplot(subplot_a[2, 1])
ax_a22 = fig.add_subplot(subplot_a[2, 2])

# Remaining two panels
ax_02 = fig.add_subplot(main_gs[0, 2])
ax_12 = fig.add_subplot(main_gs[1, 2])

# =============================================================================
# SUBPLOT_A ROW 0: RNN ACTIVITY ANALYSIS
# =============================================================================

print("Plotting subplot_a Row 0: RNN activity analysis...")

# Select 2 random neurons
np.random.seed(42)
neuron_indices = np.random.choice(Rates.shape[1], 2, replace=False)
colors = ['#e74c3c', '#3498db']

time_array = np.arange(Rates.shape[0]) * dt

# [0,0]: Time traces of 2 neurons
for i, idx in enumerate(neuron_indices):
    ax_a00.plot(time_array, Rates[:, idx], label=f'N. {idx}',
                color=colors[i], linewidth=0.8)
ax_a00.set_xlabel('Time (ms)', fontsize=6)
ax_a00.set_ylabel('Firing rate (Hz)', fontsize=6)
ax_a00.tick_params(labelsize=5)
ax_a00.legend(fontsize=5, frameon=False, loc='best')

# [0,1]: Frequency content of 2 neurons
freq_max = 200  # Hz
fs = 1000.0 / dt

for i, idx in enumerate(neuron_indices):
    f, Pxx = welch(Rates[:, idx], fs=fs, nperseg=min(512, len(Rates)))
    freq_lim = np.searchsorted(f, freq_max)
    ax_a01.semilogy(f[:freq_lim], Pxx[:freq_lim],
                    label=f'N. {idx}', color=colors[i], linewidth=0.8)

ax_a01.set_xlabel('Frequency (Hz)', fontsize=6)
ax_a01.set_ylabel('Power', fontsize=6)
ax_a01.tick_params(labelsize=5)
ax_a01.legend(fontsize=5, frameon=False, loc='best')

# [0,2]: Participation ratio distribution for 200 random selections of 5 neurons
k_sample = 5
n_samples = 200
part_ratios = []

for _ in range(n_samples):
    idx = np.random.choice(Rates.shape[1], k_sample, replace=False)
    X = Rates[:, idx]
    pr = compute_participation_ratio(X)
    part_ratios.append(pr)

ax_a02.hist(part_ratios, bins=15, color='grey', alpha=0.8, edgecolor='black', linewidth=0.5)
ax_a02.set_xlabel('Participation ratio', fontsize=6)
ax_a02.set_ylabel('Count', fontsize=6)
ax_a02.tick_params(labelsize=5)

# =============================================================================
# SUBPLOT_A ROW 1: PCA ANALYSIS
# =============================================================================

print("Plotting subplot_a Row 1: PCA analysis...")

# Perform PCA on Rates
pca = PCA(n_components=k_sample)
S = pca.fit_transform(Rates)
S_norm = S  # Already normalized by sklearn

# [1,0]: Time traces of PC1 and PC5
pc_indices = [0, k_sample-1]  # PC1 and PC5 (0-indexed)
pc_colors = ['#9b59b6', '#e67e22']

for pc_idx_pos, pc in enumerate(pc_indices):
    ax_a10.plot(time_array, S_norm[:, pc],
                label=f'PC {pc+1}', color=pc_colors[pc_idx_pos], linewidth=0.8)

ax_a10.set_xlabel('Time (ms)', fontsize=6)
ax_a10.set_ylabel('PC activity (a.u.)', fontsize=6)
ax_a10.tick_params(labelsize=5)
ax_a10.legend(fontsize=5, frameon=False, loc='best')

# [1,1]: Frequency content of PC1 and PC5
for pc_idx_pos, pc in enumerate(pc_indices):
    f, Pxx = welch(S_norm[:, pc], fs=fs, nperseg=min(512, len(Rates)))
    freq_lim = np.searchsorted(f, freq_max)
    ax_a11.semilogy(f[:freq_lim], Pxx[:freq_lim],
                    label=f'PC {pc+1}', color=pc_colors[pc_idx_pos], linewidth=0.8)

ax_a11.set_xlabel('Frequency (Hz)', fontsize=6)
ax_a11.set_ylabel('Power', fontsize=6)
ax_a11.tick_params(labelsize=5)
ax_a11.legend(fontsize=5, frameon=False, loc='best')

# [1,2]: Participation ratio bar plot for 5 PCA components
eig_pca = np.linalg.eigvalsh(np.cov(S_norm, rowvar=False))
part_ratio_pca = (np.sum(eig_pca) ** 2) / np.sum(eig_pca ** 2)

ax_a12.bar([0], [part_ratio_pca], color='grey', alpha=0.9, width=0.5)
ax_a12.set_ylabel('Participation ratio', fontsize=6)
ax_a12.set_xticks([0])
ax_a12.set_xticklabels([f'{k_sample} PCs'], fontsize=5)
ax_a12.tick_params(labelsize=5)
ax_a12.set_ylim([0, max(part_ratio_pca * 1.2, 1)])

# =============================================================================
# SUBPLOT_A ROW 2: HD SIGNAL ANALYSIS (k=5, d=5)
# =============================================================================

print("Plotting subplot_a Row 2: HD signal analysis...")

# [2,0]: Time traces of first and last channel
hd_colors = ['#16a085', '#c0392b']
ch_indices = [0, Y_hd.shape[1] - 1]

time_array_hd = np.arange(Y_hd.shape[0]) * dt

for ch_idx_pos, ch in enumerate(ch_indices):
    ax_a20.plot(time_array_hd, Y_hd[:, ch],
                label=f'Ch {ch+1}', color=hd_colors[ch_idx_pos], linewidth=0.8)

ax_a20.set_xlabel('Time (ms)', fontsize=6)
ax_a20.set_ylabel('HD signal (a.u.)', fontsize=6)
ax_a20.tick_params(labelsize=5)
ax_a20.legend(fontsize=5, frameon=False, loc='best')

# [2,1]: Frequency content of first and last channel
for ch_idx_pos, ch in enumerate(ch_indices):
    f, Pxx = welch(Y_hd[:, ch], fs=fs, nperseg=min(512, len(Y_hd)))
    freq_lim = np.searchsorted(f, freq_max)
    ax_a21.semilogy(f[:freq_lim], Pxx[:freq_lim],
                    label=f'Ch {ch+1}', color=hd_colors[ch_idx_pos], linewidth=0.8)

ax_a21.set_xlabel('Frequency (Hz)', fontsize=6)
ax_a21.set_ylabel('Power', fontsize=6)
ax_a21.tick_params(labelsize=5)
ax_a21.legend(fontsize=5, frameon=False, loc='best')

# Set same y-scale for all power plots (column 1: frequency content)
all_power_axes = [ax_a01, ax_a11, ax_a21]
y_mins = []
y_maxs = []
for ax in all_power_axes:
    ylim = ax.get_ylim()
    y_mins.append(ylim[0])
    y_maxs.append(ylim[1])

common_ylim = [min(y_mins), max(y_maxs)]
for ax in all_power_axes:
    ax.set_ylim(common_ylim)

# [2,2]: Participation ratio bar plot for 5 HD channels
part_ratio_hd = compute_participation_ratio(Y_hd)

ax_a22.bar([0], [part_ratio_hd], color='grey', alpha=0.9, width=0.5)
ax_a22.set_ylabel('Participation ratio', fontsize=6)
ax_a22.set_xticks([0])
ax_a22.set_xticklabels([f'{k_hd} HD'], fontsize=5)
ax_a22.tick_params(labelsize=5)
ax_a22.set_ylim([0, max(part_ratio_hd * 1.2, 1)])

# Add panel label to subplot_a
ax_a00.text(-0.3, 1.15, 'a', transform=ax_a00.transAxes,
            fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL B (ax_02): FREQUENCY CONTENT COMPARISON WITH CV
# =============================================================================

print("Plotting panel b: Frequency content comparison...")

# Prepare data for thin vertical bars
band_names = list(frequency_bands.keys())
n_bands = len(band_names)
n_signals = len(all_relative_powers[band_names[0]])

# Create x-positions for bars
bar_width = 0.8 / n_signals  # Very thin bars
x_positions = {}

for band_idx, band_name in enumerate(band_names):
    # Center the group of bars at band_idx
    x_center = band_idx
    x_start = x_center - 0.4
    x_positions[band_name] = np.linspace(x_start, x_start + 0.8, n_signals)

# Color gradient
colors_viridis = plt.cm.viridis(np.linspace(0, 1, n_signals))

# Plot thin bars for each signal
for signal_idx in range(n_signals):
    for band_idx, band_name in enumerate(band_names):
        relative_power = all_relative_powers[band_name][signal_idx]
        x_pos = x_positions[band_name][signal_idx]

        ax_02.bar(x_pos, relative_power, width=bar_width,
                 color=colors_viridis[signal_idx], alpha=0.6,
                 edgecolor='none')

# Add CV values on top of each band
y_max = 0
for band_name in band_names:
    y_max = max(y_max, max(all_relative_powers[band_name]))

for band_idx, band_name in enumerate(band_names):
    cv_value = cv_per_band[band_name]
    ax_02.text(band_idx, y_max * 1.05, f'CV={cv_value:.1f}%',
              ha='center', va='bottom', fontsize=5)

ax_02.set_xticks(range(n_bands))
ax_02.set_xticklabels(band_names, fontsize=6)
ax_02.set_xlabel('Frequency band', fontsize=7)
ax_02.set_ylabel('Relative power (%)', fontsize=7)
ax_02.tick_params(labelsize=6)
ax_02.set_ylim([0, y_max * 1.15])

# Panel label
ax_02.text(-0.25, 1.05, 'b', transform=ax_02.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL C (ax_12): R² VS RMSE SCATTER
# =============================================================================

print("Plotting panel c: R² vs RMSE scatter...")

ax_12.scatter(r2_folds, rmse_folds, s=3, c='#2c3e50',
             alpha=0.3, edgecolors='none', rasterized=True)

# Add Spearman correlation text only
corr_text = f'ρ = {spearman_corr:.3f}'
ax_12.text(0.05, 0.95, corr_text, transform=ax_12.transAxes,
          fontsize=6, va='top', ha='left',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

ax_12.set_xlabel('R² (CV folds)', fontsize=7)
ax_12.set_ylabel('RMSE (CV folds)', fontsize=7)
ax_12.tick_params(labelsize=6)

# Panel label
ax_12.text(-0.25, 1.05, 'c', transform=ax_12.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# SAVE FIGURE
# =============================================================================

output_svg = 'supplementary_figure_2.svg'
output_pdf = 'supplementary_figure_2.pdf'

plt.savefig(output_svg, format='svg', dpi=450, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=450, bbox_inches='tight')

print()
print(f"Supplementary figure 2 saved as '{output_svg}' and '{output_pdf}'")
print()
print("="*80)
print("COMPLETE!")
print("="*80)
print("\nFigure layout:")
print("  Panel a (merged 2x2 → 3x3 grid):")
print("    Row 0: RNN activity (time traces, frequency, PR histogram)")
print("    Row 1: PCA analysis (PC1&PC20 traces, frequency, PR bar)")
print("    Row 2: HD signal (Ch1&Ch20 traces, frequency, PR bar)")
print("  Panel b: Frequency content comparison with CV values")
print("  Panel c: R² vs RMSE scatter plot")
print("\nAll specifications comply with Nature Neuroscience guidelines")
print("="*80)
