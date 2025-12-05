# plot_main_figure_3.py
"""
Main Figure 3: Network classification of high-dimensional inputs
Follows Nature Neuroscience specifications

Updated for v7.0.0:
- Supports overlapping/partitioned data types
- Reads BOTH encoding and classification data files for scatter plot
- Uses per-session values for scatter (not means)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
from scipy.stats import spearmanr, pearsonr

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
# CONFIGURATION
# =============================================================================

# Which data type to plot (can be "overlapping", "partitioned", or "both")
PLOT_DATA_TYPE = "overlapping"  # Options: "overlapping", "partitioned", "both"

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("PLOTTING MAIN FIGURE 3")
print("="*80)

# Load classification data
classification_file = os.path.join(project_root, 'data_curation', 'network_classification_data.pkl')
with open(classification_file, 'rb') as f:
    classification_data = pickle.load(f)

# Load encoding data (for scatter plot)
encoding_file = os.path.join(project_root, 'data_curation', 'network_encoding_data.pkl')
with open(encoding_file, 'rb') as f:
    encoding_data = pickle.load(f)

# Check available data types (intersection of both files)
class_types = set(classification_data.get('available_data_types', ['overlapping']))
enc_types = set(encoding_data.get('available_data_types', ['overlapping']))
available_types = list(class_types & enc_types)

print(f"Classification data types: {list(class_types)}")
print(f"Encoding data types: {list(enc_types)}")
print(f"Available for both: {available_types}")

# Determine which types to plot
if PLOT_DATA_TYPE == "both":
    types_to_plot = [t for t in ["overlapping", "partitioned"] if t in available_types]
else:
    types_to_plot = [PLOT_DATA_TYPE] if PLOT_DATA_TYPE in available_types else available_types[:1]

print(f"Plotting data types: {types_to_plot}")

# Get results (use results_by_type if available)
if 'results_by_type' in classification_data:
    class_results_by_type = classification_data['results_by_type']
else:
    class_results_by_type = {available_types[0]: {'accuracy_vs_d': classification_data['accuracy_vs_d']}}

if 'results_by_type' in encoding_data:
    enc_results_by_type = encoding_data['results_by_type']
else:
    enc_results_by_type = {available_types[0]: {'r2_vs_d': encoding_data['r2_vs_d']}}

print("Data loaded successfully!")
print()

# =============================================================================
# PREPARE SCATTER PLOT DATA (PER-SESSION VALUES)
# =============================================================================

print("Preparing scatter plot data (per-session values)...")

scatter_data = {}

for data_type in types_to_plot:
    accuracy_vs_d = class_results_by_type[data_type]['accuracy_vs_d']
    r2_vs_d = enc_results_by_type[data_type]['r2_vs_d']

    encoding_r2_values = []
    classification_accuracy_values = []
    k_values_scatter = []
    d_values_scatter = []
    session_ids = []

    k_values = accuracy_vs_d['k_values']

    for k_val in k_values:
        for d in range(1, k_val + 1):
            # Get per-session values
            acc_per_session = accuracy_vs_d.get('per_session', {}).get(k_val, {}).get(d, {})
            r2_per_session = r2_vs_d.get('per_session', {}).get(k_val, {}).get(d, {})

            # Find common sessions
            common_sessions = set(acc_per_session.keys()) & set(r2_per_session.keys())

            for session in common_sessions:
                acc = acc_per_session[session]
                r2 = r2_per_session[session]

                if not np.isnan(acc) and not np.isnan(r2):
                    encoding_r2_values.append(r2)
                    classification_accuracy_values.append(acc)
                    k_values_scatter.append(k_val)
                    d_values_scatter.append(d)
                    session_ids.append(session)

    # Compute correlations
    if len(encoding_r2_values) > 2:
        spearman_corr, spearman_p = spearmanr(encoding_r2_values, classification_accuracy_values)
        pearson_corr, pearson_p = pearsonr(encoding_r2_values, classification_accuracy_values)
    else:
        spearman_corr, spearman_p = np.nan, np.nan
        pearson_corr, pearson_p = np.nan, np.nan

    scatter_data[data_type] = {
        'encoding_r2': np.array(encoding_r2_values),
        'classification_accuracy': np.array(classification_accuracy_values),
        'k_values': np.array(k_values_scatter),
        'd_values': np.array(d_values_scatter),
        'session_ids': np.array(session_ids),
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p
    }

    print(f"  {data_type}: {len(encoding_r2_values)} scatter points, ρ = {spearman_corr:.3f}")

print()

# =============================================================================
# CREATE FIGURE
# =============================================================================

row_height_unit = 1
hspace = 0.4
height_ratios = [1.5, 2]
new_height = row_height_unit * (sum(height_ratios) + hspace * (len(height_ratios) - 1))

fig = plt.figure(figsize=(7.2, new_height))
main_gs = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=height_ratios,
                            hspace=hspace)

# ROW 1: Merged for schematic
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[0], wspace=0.5)

# ROW 2: Three panels
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[1], wspace=0.5)

# ROW 1: All merged, axis off
ax_a = fig.add_subplot(row1_gs[:])

# ROW 2: Three separate panels
ax_b = fig.add_subplot(row2_gs[0])
ax_c = fig.add_subplot(row2_gs[1])
ax_d = fig.add_subplot(row2_gs[2])

# =============================================================================
# PANEL A: EMPTY (FOR MANUAL SCHEMATIC)
# =============================================================================

ax_a.axis('off')
ax_a.text(-0.06, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL B: CLASSIFICATION ACCURACY VS INTRINSIC DIMENSIONALITY
# =============================================================================

# Line styles for different data types
linestyles = {'overlapping': '-', 'partitioned': '--'}
type_labels = {'overlapping': 'Overlap', 'partitioned': 'Partition'}

for data_type in types_to_plot:
    accuracy_vs_d = class_results_by_type[data_type]['accuracy_vs_d']
    k_values = accuracy_vs_d['k_values']
    colors_plasma = plt.cm.plasma(np.linspace(0.1, 0.9, len(k_values)))
    ls = linestyles.get(data_type, '-')

    for idx, k_val in enumerate(k_values):
        d_values = accuracy_vs_d['d_values'][k_val]
        means = [accuracy_vs_d['mean'][k_val][d] for d in d_values]
        stds = [accuracy_vs_d['std'][k_val][d] for d in d_values]

        means = np.array(means)
        stds = np.array(stds)

        label = f'k={k_val}' if len(types_to_plot) == 1 else f'k={k_val} ({type_labels[data_type]})'
        ax_b.plot(d_values, means, 'o' + ls, color=colors_plasma[idx],
                  linewidth=1.5, markersize=4, label=label)
        # Uncomment to add error bands:
        # ax_b.fill_between(d_values, means - stds, means + stds,
        #                   color=colors_plasma[idx], alpha=0.3)

ax_b.set_xlabel('Input intrinsic dim. (d)', fontsize=7)
ax_b.set_ylabel('Classification accuracy', fontsize=7)
ax_b.tick_params(labelsize=6)
ax_b.legend(fontsize=5, frameon=False, loc='best', ncol=1)

# Panel label
ax_b.text(-0.25, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL C: ENCODING VS CLASSIFICATION SCATTER (PER-SESSION)
# =============================================================================

# Use first data type for scatter (or combine if "both")
first_type = types_to_plot[0]
k_unique = class_results_by_type[first_type]['accuracy_vs_d']['k_values']
colors_plasma = plt.cm.plasma(np.linspace(0.1, 0.9, len(k_unique)))
k_to_color = {k: colors_plasma[i] for i, k in enumerate(k_unique)}

# Markers for different data types
markers = {'overlapping': 'o', 'partitioned': 's'}

for data_type in types_to_plot:
    sd = scatter_data[data_type]
    scatter_colors = [k_to_color[k] for k in sd['k_values']]
    marker = markers.get(data_type, 'o')
    alpha = 0.7 if len(types_to_plot) == 1 else 0.5

    ax_c.scatter(sd['encoding_r2'], sd['classification_accuracy'],
                 s=15, c=scatter_colors, alpha=alpha, edgecolors='white',
                 linewidth=0.3, marker=marker,
                 label=type_labels.get(data_type, data_type) if len(types_to_plot) > 1 else None)

# Add Spearman correlation text (for first type)
spearman_corr = scatter_data[first_type]['spearman_corr']
corr_text = f'ρ = {spearman_corr:.3f}'
if len(types_to_plot) > 1:
    corr_text = f'{first_type}: ρ = {spearman_corr:.3f}'
ax_c.text(0.05, 0.95, corr_text, transform=ax_c.transAxes,
          fontsize=6, va='top', ha='left',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

# Legend for k values
for i, k in enumerate(k_unique):
    ax_c.scatter([], [], s=10, c=[colors_plasma[i]], label=f'k = {k}')

# Add data type markers to legend if plotting both
if len(types_to_plot) > 1:
    for dt in types_to_plot:
        ax_c.scatter([], [], s=10, c='gray', marker=markers[dt], label=type_labels[dt])

ax_c.legend(fontsize=5, frameon=False, loc='lower right', ncol=2 if len(types_to_plot) > 1 else 1)

ax_c.set_xlabel('Encoding accuracy (R²)', fontsize=7)
ax_c.set_ylabel('Classification accuracy', fontsize=7)
ax_c.tick_params(labelsize=6)

# Panel label
ax_c.text(-0.25, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL D: EMPTY (NOT DESIGNED YET)
# =============================================================================

ax_d.text(-0.25, 1.05, 'd', transform=ax_d.transAxes,
          fontsize=7, fontweight='bold', va='top')

ax_d.set_ylabel('Classification accuracy', fontsize=7)
ax_d.set_xlabel('g_std', fontsize=7)
ax_d.tick_params(labelsize=6)

# =============================================================================
# SAVE FIGURE
# =============================================================================

# Add data type to filename
suffix = f'_{PLOT_DATA_TYPE}' if PLOT_DATA_TYPE != "both" else '_all'
output_svg = f'main_figure_3{suffix}.svg'
output_pdf = f'main_figure_3{suffix}.pdf'

plt.savefig(output_svg, format='svg', dpi=450, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=450, bbox_inches='tight')

print(f"Main figure 3 saved as '{output_svg}' and '{output_pdf}'")
print()
print("="*80)
print("COMPLETE!")
print("="*80)
print(f"\nData types plotted: {types_to_plot}")
print(f"\nScatter plot statistics:")
for dt in types_to_plot:
    sd = scatter_data[dt]
    print(f"  {dt}:")
    print(f"    - Points: {len(sd['encoding_r2'])} (per-session values)")
    print(f"    - Spearman ρ = {sd['spearman_corr']:.3f} (p = {sd['spearman_p']:.2e})")
    print(f"    - Pearson r = {sd['pearson_corr']:.3f} (p = {sd['pearson_p']:.2e})")
print("\nFigure layout:")
print("  Row 1: Panel a (empty for manual schematic)")
print("  Row 2: Panel b (accuracy vs d), Panel c (encoding vs classification), Panel d (empty)")
print("\nAll specifications comply with Nature Neuroscience guidelines")
print("="*80)
