# plot_main_figure_2.py
"""
Main Figure 2: Network encoding of high-dimensional inputs
Follows Nature Neuroscience specifications

Updated for v7.0.0: Supports overlapping/partitioned data types
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

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
# If "both", will plot overlapping as solid lines and partitioned as dashed
PLOT_DATA_TYPE = "overlapping"  # Options: "overlapping", "partitioned", "both"

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("PLOTTING MAIN FIGURE 2")
print("="*80)

data_file = os.path.join(project_root, 'data_curation', 'network_encoding_data.pkl')
with open(data_file, 'rb') as f:
    data = pickle.load(f)

# Check available data types
available_types = data.get('available_data_types', ['overlapping'])
print(f"Available data types: {available_types}")

# Determine which types to plot
if PLOT_DATA_TYPE == "both":
    types_to_plot = [t for t in ["overlapping", "partitioned"] if t in available_types]
else:
    types_to_plot = [PLOT_DATA_TYPE] if PLOT_DATA_TYPE in available_types else available_types[:1]

print(f"Plotting data types: {types_to_plot}")

# Extract example patterns (same for all types)
pattern_d1 = data['pattern_d1']
pattern_d2 = data['pattern_d2']
k = data['pattern_k']
dt = data['pattern_dt']

# Get results (use results_by_type if available, else fall back to top-level)
if 'results_by_type' in data:
    results_by_type = data['results_by_type']
else:
    # Backward compatibility
    results_by_type = {
        available_types[0]: {
            'r2_vs_d': data['r2_vs_d'],
            'pr_vs_d': data['pr_vs_d'],
            'empirical_dims': data['empirical_dims'],
            'theoretical_dims': data['theoretical_dims']
        }
    }

# Use first type for histogram data
first_type = types_to_plot[0]
empirical_dims = results_by_type[first_type]['empirical_dims']
theoretical_dims = results_by_type[first_type]['theoretical_dims']

print("Data loaded successfully!")
print(f"pattern_d1 shape: {pattern_d1.shape}")
print(f"pattern_d2 shape: {pattern_d2.shape}")
print(f"k = {k}")
print()

# Safety check: k should match the pattern dimensions
k = int(k)
if pattern_d1.shape[1] != k:
    print(f"\n⚠ WARNING: pattern_d1 has {pattern_d1.shape[1]} channels but k={k}")
    k = pattern_d1.shape[1]

# =============================================================================
# CREATE FIGURE
# =============================================================================

row_height_unit = 1.
hspace = 0.4
height_ratios = [1, 1.5, 2]
new_height = row_height_unit * (sum(height_ratios) + hspace * (len(height_ratios) - 1))

fig = plt.figure(figsize=(7.2, new_height))
main_gs = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=height_ratios,
                            hspace=hspace)

# ROW 1: Custom width ratios
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[0],
                                            width_ratios=[1.2, 1.2, 0.6],
                                            wspace=0.5)

# ROW 2 and ROW 3: Normal equal widths
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[1], wspace=0.5)
row3_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[2], wspace=0.5)

# ROW 1: Panel a (merged 0:2) and panel b
ax_a_merged = row1_gs[0:2]
ax_b = fig.add_subplot(row1_gs[2])

# Split panel a into two columns (d=1 and d=4)
ax_a_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_a_merged, wspace=0.32)
ax_a1 = fig.add_subplot(ax_a_gs[0])
ax_a2 = fig.add_subplot(ax_a_gs[1])

# Manual position adjustment
bottom_shift = -0.04
left_shift = -0.005
for ax in [ax_a1, ax_a2]:
    pos = ax.get_position()
    ax.set_position([pos.x0+left_shift, pos.y0 + bottom_shift,
                     pos.width+left_shift, pos.height-bottom_shift])

# ROW 2: Merged panel (axis off for manual schematic)
ax_c = fig.add_subplot(row2_gs[:])

# ROW 3: Three panels
ax_d = fig.add_subplot(row3_gs[0])
ax_e = fig.add_subplot(row3_gs[1])
ax_f = fig.add_subplot(row3_gs[2])

# =============================================================================
# PANEL A1: HD INPUT d=1
# =============================================================================

# Colors for channels
channel_colors = plt.cm.viridis(np.linspace(0., 0.8, k))
# Swap colors for better visualization
if k >= 4:
    a = channel_colors[0].copy()
    b = channel_colors[1].copy()
    c = channel_colors[2].copy()
    d_color = channel_colors[3].copy()
    channel_colors[1] = d_color
    channel_colors[2] = b
    channel_colors[3] = c

# Time parameters
stimulus_duration = 300.0
samples_100ms = int(100 / dt)

# Plot d=1 time series (offset for visibility)
for i in range(k):
    offset = 3*(i+1) if i < k-1 else 0
    ax_a1.plot(offset + pattern_d1[:, i], linewidth=0.8,
               color=channel_colors[i], alpha=0.9)

ax_a1.set_yticks([3*i for i in range(1, k+1)])
ax_a1.set_yticklabels([str(i) for i in range(1, k+1)], fontsize=6)
ax_a1.set_ylabel("HD input feature", fontsize=7)

ax_a1.spines['top'].set_visible(False)
ax_a1.spines['right'].set_visible(False)
ax_a1.spines['bottom'].set_visible(False)
ax_a1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Scale bars
x_start = 0
x_end = samples_100ms
hbar_y = 0.5
ax_a1.hlines(y=hbar_y, xmin=x_start, xmax=x_end, color='black', linewidth=1)
ax_a1.text((x_start + x_end) / 2, hbar_y - 0.3, '100 ms',
           ha='center', va='top', fontsize=6)

vline_x = stimulus_duration / dt * 1.05
vline_y_start = 0
vline_y_end = 4
ax_a1.vlines(x=vline_x, ymin=vline_y_start, ymax=vline_y_end,
             color='black', linewidth=1)
ax_a1.text(vline_x*1.05, 2, '4 Hz', ha='center', va='center',
           fontsize=6, rotation=90)

# Label
ax_a1.text(0.02, 1.0, f'd = 1, k = {k}', transform=ax_a1.transAxes,
           fontsize=7, ha='left', va='top')

# Panel label
ax_a1.text(-0.17, 1.07, 'a', transform=ax_a1.transAxes,
           fontsize=7, fontweight='bold', va='top')

ax_a1.set_ylim([0, 14])

# =============================================================================
# PANEL A2: HD INPUT d=k
# =============================================================================

# Plot d=k time series
for i in range(k):
    offset = 3*(i+1) if i < k-1 else 0
    ax_a2.plot(offset + pattern_d2[:, i], linewidth=0.8,
               color=channel_colors[i], alpha=0.9)

ax_a2.set_yticks([3*i for i in range(1, k+1)])
ax_a2.set_yticklabels([str(i) for i in range(1, k+1)], fontsize=6)
ax_a2.set_ylabel("HD input feature", fontsize=7)

ax_a2.spines['top'].set_visible(False)
ax_a2.spines['right'].set_visible(False)
ax_a2.spines['bottom'].set_visible(False)
ax_a2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

ax_a2.hlines(y=hbar_y, xmin=x_start, xmax=x_end, color='black', linewidth=1)
ax_a2.text((x_start + x_end) / 2, hbar_y - 0.3, '100 ms',
           ha='center', va='top', fontsize=6)

ax_a2.vlines(x=vline_x, ymin=vline_y_start, ymax=vline_y_end,
             color='black', linewidth=1)
ax_a2.text(vline_x*1.05, 2, '4 Hz', ha='center', va='center',
           fontsize=6, rotation=90)

ax_a2.text(0.02, 1.0, f'd = {k}, k = {k}', transform=ax_a2.transAxes,
           fontsize=7, ha='left', va='top')

ax_a2.set_ylim([0, 14])

# =============================================================================
# PANEL B: HISTOGRAM OF EMPIRICAL/THEORETICAL DIMENSIONALITY RATIO
# =============================================================================

# Calculate the ratio
dim_ratio = empirical_dims / theoretical_dims

# Create histogram
ax_b.hist(dim_ratio, bins=20, color='#404040', alpha=0.7, edgecolor='black', linewidth=0.5)

# Add vertical line at ratio=1 (perfect match)
ax_b.axvline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Perfect match')

ax_b.set_xlabel('Empirical / Theoretical dim.', fontsize=7)
ax_b.set_ylabel('Count', fontsize=7)
ax_b.tick_params(labelsize=6)
ax_b.legend(fontsize=5, frameon=False, loc='best')

# Add mean and std as text
mean_ratio = np.mean(dim_ratio)
std_ratio = np.std(dim_ratio)
ax_b.text(0.95, 0.95, f'μ = {mean_ratio:.2f}\nσ = {std_ratio:.2f}',
         transform=ax_b.transAxes, fontsize=5, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

# Panel label
ax_b.text(-0.5, 1.1, 'b', transform=ax_b.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL C: EMPTY (FOR MANUAL SCHEMATIC)
# =============================================================================

ax_c.axis('off')
ax_c.text(-0.06, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL D: ENCODING ACCURACY (R²) VS INTRINSIC DIMENSIONALITY
# =============================================================================

# Line styles for different data types
linestyles = {'overlapping': '-', 'partitioned': '--'}
type_labels = {'overlapping': 'Overlap', 'partitioned': 'Partition'}

for data_type in types_to_plot:
    r2_vs_d = results_by_type[data_type]['r2_vs_d']
    k_values = r2_vs_d['k_values']
    colors_plasma = plt.cm.plasma(np.linspace(0.1, 0.9, len(k_values)))
    ls = linestyles.get(data_type, '-')

    for idx, k_val in enumerate(k_values):
        d_values = r2_vs_d['d_values'][k_val]
        means = [r2_vs_d['mean'][k_val][d] for d in d_values]
        stds = [r2_vs_d['std'][k_val][d] for d in d_values]

        means = np.array(means)
        stds = np.array(stds)

        label = f'k={k_val}' if len(types_to_plot) == 1 else f'k={k_val} ({type_labels[data_type]})'
        ax_d.plot(d_values, means, 'o' + ls, color=colors_plasma[idx],
                  linewidth=1.5, markersize=4, label=label)
        ax_d.fill_between(d_values, means - stds, means + stds,
                          color=colors_plasma[idx], alpha=0.2)

ax_d.set_xlabel('Input intrinsic dim. (d)', fontsize=7)
ax_d.set_ylabel('Encoding accuracy (R²)', fontsize=7)
ax_d.tick_params(labelsize=6)
ax_d.legend(fontsize=5, frameon=False, loc='best', ncol=1)

# Panel label
ax_d.text(-0.25, 1.05, 'd', transform=ax_d.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL E: NETWORK DIMENSIONALITY (PR) VS INTRINSIC DIMENSIONALITY
# =============================================================================

for data_type in types_to_plot:
    pr_vs_d = results_by_type[data_type]['pr_vs_d']
    k_values = pr_vs_d['k_values']
    colors_plasma = plt.cm.plasma(np.linspace(0.1, 0.9, len(k_values)))
    ls = linestyles.get(data_type, '-')

    for idx, k_val in enumerate(k_values):
        d_values = pr_vs_d['d_values'][k_val]
        means = [pr_vs_d['mean'][k_val][d] for d in d_values]
        stds = [pr_vs_d['std'][k_val][d] for d in d_values]

        means = np.array(means)
        stds = np.array(stds)

        label = f'k={k_val}' if len(types_to_plot) == 1 else f'k={k_val} ({type_labels[data_type]})'
        ax_e.plot(d_values, means, 'o' + ls, color=colors_plasma[idx],
                  linewidth=1.5, markersize=4, label=label)
        ax_e.fill_between(d_values, means - stds, means + stds,
                          color=colors_plasma[idx], alpha=0.2)

ax_e.set_xlabel('Input intrinsic dim. (d)', fontsize=7)
ax_e.set_ylabel('Network activity dim. (PR)', fontsize=7)
ax_e.tick_params(labelsize=6)
ax_e.legend(fontsize=5, frameon=False, loc='best', ncol=1)

# Panel label
ax_e.text(-0.25, 1.05, 'e', transform=ax_e.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL F: EMPTY (NOT DESIGNED YET)
# =============================================================================

ax_f.axis('off')
ax_f.text(-0.25, 1.05, 'f', transform=ax_f.transAxes,
          fontsize=7, fontweight='bold', va='top')

# =============================================================================
# SAVE FIGURE
# =============================================================================

# Add data type to filename if plotting specific type
suffix = f'_{PLOT_DATA_TYPE}' if PLOT_DATA_TYPE != "both" else '_all'
output_svg = f'main_figure_2{suffix}.svg'
output_pdf = f'main_figure_2{suffix}.pdf'

plt.savefig(output_svg, format='svg', dpi=450, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=450, bbox_inches='tight')

print(f"Main figure 2 saved as '{output_svg}' and '{output_pdf}'")
print()
print("="*80)
print("COMPLETE!")
print("="*80)
print(f"\nData types plotted: {types_to_plot}")
print("\nFigure layout:")
print("  Row 1: Panel a (HD inputs d=1 and d=k), Panel b (empirical vs theoretical)")
print("  Row 2: Panel c (empty for manual schematic)")
print("  Row 3: Panel d (R² vs d), Panel e (Network PR vs d), Panel f (empty)")
print("\nAll specifications comply with Nature Neuroscience guidelines")
print("="*80)
