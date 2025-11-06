# plots/plot_supplementary_figure.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

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

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("PLOTTING SUPPLEMENTARY FIGURE")
print("="*80)

data_file = os.path.join(project_root, 'data_curation', 'network_dynamics_data.pkl')
with open(data_file, 'rb') as f:
    data = pickle.load(f)

# Extract data
io_data = data['io_mapping']
g_std_values_io = io_data['g_std_values']
input_rates_io = io_data['input_rates']
output_rates_dict = io_data['output_rates_dict']

pr_values = data['scatter_pr_dim']['pr_values']
dim_values = data['scatter_pr_dim']['dim_values']
lz_values = data['scatter_lz_kistler']['lz_values']
kistler_values = data['scatter_lz_kistler']['kistler_values']

# Perturbation data
pert_data = data['perturbation_data']
n_neurons_pert = pert_data['n_neurons']
perturbation_time = pert_data['perturbation_time']
perturbation_neuron = pert_data['perturbation_neuron']
plot_start_time = pert_data['plot_start_time']
plot_end_time = pert_data['plot_end_time']
bin_size_pert = pert_data['bin_size']

spike_diff_g06 = pert_data['spike_diff_g06']
spike_diff_g2 = pert_data['spike_diff_g2']
symbol_seq_g06 = pert_data['symbol_seq_g06']
symbol_seq_g2 = pert_data['symbol_seq_g2']
common_spikes_g06 = pert_data['common_spikes_g06']
ctrl_only_spikes_g06 = pert_data['ctrl_only_spikes_g06']
pert_only_spikes_g06 = pert_data['pert_only_spikes_g06']

print("Data loaded successfully!")
print()

# =============================================================================
# CREATE FIGURE - FOLLOWING MAIN FIGURE STRUCTURE
# =============================================================================

row_height_unit = 1.
hspace = 0.4
wspace = 0.5
height_ratios = [2, 2]
new_height = row_height_unit * (sum(height_ratios) + hspace * (len(height_ratios) - 1))

fig = plt.figure(figsize=(7.2, new_height))

main_gs = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=height_ratios,
                            hspace=hspace,
                            wspace=wspace)

# Top row
ax_00 = fig.add_subplot(main_gs[0, 0])
ax_01 = fig.add_subplot(main_gs[0, 1])
ax_02 = fig.add_subplot(main_gs[0, 2])

# Bottom row - All panels have space for pattern subplot, but Panel C doesn't plot anything
# Panel C - split for raster + empty pattern space
gs_10 = main_gs[1, 0].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
ax_10_top = fig.add_subplot(gs_10[0])
ax_10_bottom = fig.add_subplot(gs_10[1])

# Panel D - split for raster + pattern
gs_11 = main_gs[1, 1].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
ax_11_top = fig.add_subplot(gs_11[0])
ax_11_bottom = fig.add_subplot(gs_11[1])

# Panel E - split for raster + pattern
gs_12 = main_gs[1, 2].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
ax_12_top = fig.add_subplot(gs_12[0])
ax_12_bottom = fig.add_subplot(gs_12[1])

# =============================================================================
# PANEL A: INPUT-OUTPUT MAPPING (g=0 and g=1) - NO R², NO DASHED LINES
# =============================================================================

io_colors = ['orange', '#000000']  # Gray for g=0, BLACK for g=1
io_labels = ['gˢᵗᵈ=0.0', 'gˢᵗᵈ=1.0']

for idx, g_std in enumerate(g_std_values_io):
    output_rates = output_rates_dict[g_std]

    ax_00.plot(input_rates_io, output_rates, 'o-', color=io_colors[idx],
               markersize=1.5,linewidth=0.5, label=io_labels[idx])

ax_00.set_xlabel('Input rate (Hz)', fontsize=7)
ax_00.set_ylabel('Network mean firing rate (Hz)', fontsize=7)
ax_00.tick_params(labelsize=6)
ax_00.legend(fontsize=6, frameon=False, loc='lower right', borderpad=0)
ax_00.text(-0.27, 1.05, 'a', transform=ax_00.transAxes,
           fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL B: PR VS DIMENSIONALITY SCATTER - BLACK DOTS
# =============================================================================

ax_01.scatter(pr_values, dim_values, s=2, c='#000000', alpha=0.5,
             edgecolors='none', rasterized=True)

ax_01.set_xlabel('Participation ratio dimensionality', fontsize=7)
ax_01.set_ylabel('PCA dimensionality', fontsize=7)
ax_01.tick_params(labelsize=6)
ax_01.text(-0.27, 1.05, 'b', transform=ax_01.transAxes,
           fontsize=7, fontweight='bold', va='top')

if len(pr_values) > 0:
    spearman_corr = data['scatter_pr_dim']['spearman_rho']
    ax_01.text(0.05, 0.95, f'ρ = {spearman_corr:.3f}', transform=ax_01.transAxes,
               fontsize=6, verticalalignment='top')


# =============================================================================
# PANEL F: LZ VS KISTLER SCATTER - BLACK DOTS (NO LOG)
# =============================================================================



# Your existing plot
ax_02.scatter(lz_values, kistler_values, s=2, c='#000000', alpha=0.5,
             edgecolors='none', rasterized=True)
ax_02.set_xlabel('Lempel-Ziv complexity', fontsize=7)
ax_02.set_ylabel('Kistler distance', fontsize=7)
ax_02.tick_params(labelsize=6)
ax_02.text(-0.27, 1.05, 'f', transform=ax_02.transAxes,
           fontsize=7, fontweight='bold', va='top')
if len(lz_values) > 0:
    spearman_corr = data['scatter_lz_kistler']['spearman_rho']
    ax_02.text(0.4, 0.85, f'ρ = {spearman_corr:.3f}', transform=ax_02.transAxes,
               fontsize=6, verticalalignment='top')

# Create inset axes in the upper left corner
axins = ax_02.inset_axes([0.3, 0.2, 0.5, 0.4])

# Plot the same data in the inset
axins.scatter(lz_values, kistler_values, s=2, c='#000000', alpha=0.5,
                edgecolors='none', rasterized=True)

# Set the zoom limits
x_min, x_max = ax_02.get_xlim()
x_min, x_max = 1450, 2750
axins.set_xlim(x_min, x_max)
axins.set_ylim(0.997, 0.9995)

# Adjust tick labels
axins.tick_params(labelsize=5)

# Draw a rectangle on the main plot showing the zoomed region

rect = Rectangle((x_min, 0.98), x_max - x_min, 0.04,
                    linewidth=0.3, edgecolor='red', facecolor='none')
ax_02.add_patch(rect)

# Manually draw only 2 connector lines (left and right corners)
# Left connector: from bottom-left of rect to bottom-left of inset
con1 = ConnectionPatch((x_min, 0.98+0.04), (0, 1),
                        coordsA="data", coordsB="axes fraction",
                        axesA=ax_02, axesB=axins,
                        color="r", linewidth=0.3)
ax_02.add_artist(con1)

# Right connector: from bottom-right of rect to bottom-right of inset
con2 = ConnectionPatch((x_max, 0.98+0.04), (1, 1),
                        coordsA="data", coordsB="axes fraction",
                        axesA=ax_02, axesB=axins,
                        color="r", linewidth=0.3)
ax_02.add_artist(con2)


# =============================================================================
# PANEL C: COLORED RASTER (g=0.6) - WITH EMPTY PATTERN SPACE
# =============================================================================

if common_spikes_g06:
    times_common = [t for t, n in common_spikes_g06]
    neurons_common = [n for t, n in common_spikes_g06]
    ax_10_top.scatter(times_common, neurons_common, s=1.5, c='green', alpha=0.6,
                     label='Common', rasterized=True, zorder=4)

if ctrl_only_spikes_g06:
    times_ctrl = [t for t, n in ctrl_only_spikes_g06]
    neurons_ctrl = [n for t, n in ctrl_only_spikes_g06]
    ax_10_top.scatter(times_ctrl, neurons_ctrl, s=1.5, c='blue', alpha=0.7,
                     label='Control-only', rasterized=True, zorder=2)

if pert_only_spikes_g06:
    times_pert = [t for t, n in pert_only_spikes_g06]
    neurons_pert = [n for t, n in pert_only_spikes_g06]
    ax_10_top.scatter(times_pert, neurons_pert, s=1.5, c='red', alpha=0.7,
                     label='Perturbed-only', rasterized=True, zorder=3)

ax_10_top.axvline(perturbation_time, color='gray', linestyle='--', linewidth=1, zorder=1)
ax_10_top.axhline(perturbation_neuron, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)

ax_10_top.set_ylabel('Neuron ID', fontsize=7)
ax_10_top.set_xlim(plot_start_time, plot_end_time)
ax_10_top.set_ylim(0, n_neurons_pert)
ax_10_top.tick_params(labelsize=6)
ax_10_top.text(-0.28, 1.05, 'c', transform=ax_10_top.transAxes,
               fontsize=7, fontweight='bold', va='top')
xticks_original = np.array([495, 500, 505, 510, 515])
xticks_labels = xticks_original - perturbation_time
ax_10_top.set_xticks(xticks_original)
ax_10_top.set_xticklabels(xticks_labels.astype(int))
ax_10_top.set_xlabel('Time relative to perturbation (ms)', fontsize=7)

# Add white background rectangle for legend
legend = ax_10_top.legend(loc='upper right', fontsize=5, frameon=True, facecolor='white', borderpad=0.2,handletextpad=0.,framealpha=1.0)
legend.set_zorder(10)


ax_10_bottom.axis('off')
# # Empty pattern space - no plotting, but axis exists
# ax_10_bottom.set_xlabel('Time relative to perturbation (ms)', fontsize=7)
# ax_10_bottom.set_xlim(plot_start_time, plot_end_time)
# # Convert x-axis: 500ms becomes 0
# xticks_original = np.array([495, 500, 505, 510, 515])
# xticks_labels = xticks_original - perturbation_time
# ax_10_bottom.set_xticks(xticks_original)
# ax_10_bottom.set_xticklabels(xticks_labels.astype(int))
# ax_10_bottom.tick_params(labelsize=6)
# ax_10_bottom.spines['top'].set_visible(False)
# ax_10_bottom.spines['right'].set_visible(False)
# ax_10_bottom.spines['left'].set_visible(False)
# ax_10_bottom.spines['bottom'].set_visible(False)
# ax_10_bottom.set_yticks([])

# =============================================================================
# PANEL D: SPIKE DIFFERENCES (g=0.6) WITH PATTERN
# =============================================================================

start_bin = int(plot_start_time / bin_size_pert)
end_bin = int(plot_end_time / bin_size_pert)

spike_diff_zoom = spike_diff_g06[:, start_bin:end_bin]
diff_neurons, diff_bins = np.where(spike_diff_zoom == 1)
diff_times = diff_bins * bin_size_pert + plot_start_time

ax_11_top.scatter(diff_times, diff_neurons, s=1, c='k', alpha=0.7, rasterized=True)
ax_11_top.axvline(perturbation_time, color='gray', linestyle='--', linewidth=1)
ax_11_top.axhline(perturbation_neuron, color='gray', linestyle=':', linewidth=1, alpha=0.7)

ax_11_top.set_ylabel('Neuron ID', fontsize=7)
ax_11_top.set_xlim(plot_start_time, plot_end_time)
ax_11_top.set_ylim(0, n_neurons_pert)
ax_11_top.tick_params(labelsize=6, labelbottom=False)
ax_11_top.text(-0.28, 1.05, 'd', transform=ax_11_top.transAxes,
               fontsize=7, fontweight='bold', va='top')

# Add gstd text
ax_11_top.text(0.02, 0.98, 'gˢᵗᵈ=0.6', transform=ax_11_top.transAxes,
               fontsize=6, verticalalignment='top', horizontalalignment='left')

# Spatial pattern for g=0.6
symbol_seq_zoom = symbol_seq_g06[start_bin:end_bin]
time_bins_zoom = np.arange(len(symbol_seq_zoom)) * bin_size_pert + plot_start_time

ax_11_bottom.plot(time_bins_zoom, symbol_seq_zoom, linewidth=0.5, color='k')
ax_11_bottom.axvline(perturbation_time, color='gray', linestyle='--', linewidth=1)
ax_11_bottom.set_xlabel('Time relative to perturbation (ms)', fontsize=7)
ax_11_bottom.set_ylabel('Pattern ID', fontsize=6)
ax_11_bottom.set_xlim(plot_start_time, plot_end_time)
# Convert x-axis: 500ms becomes 0
ax_11_bottom.set_xticks(xticks_original)
ax_11_bottom.set_xticklabels(xticks_labels.astype(int))
ax_11_bottom.tick_params(labelsize=6)

# =============================================================================
# PANEL E: SPIKE DIFFERENCES (g=2.0) WITH PATTERN
# =============================================================================

spike_diff_zoom_g2 = spike_diff_g2[:, start_bin:end_bin]
diff_neurons_g2, diff_bins_g2 = np.where(spike_diff_zoom_g2 == 1)
diff_times_g2 = diff_bins_g2 * bin_size_pert + plot_start_time

ax_12_top.scatter(diff_times_g2, diff_neurons_g2, s=1, c='k', alpha=0.7, rasterized=True)
ax_12_top.axvline(perturbation_time, color='gray', linestyle='--', linewidth=1)
ax_12_top.axhline(perturbation_neuron, color='gray', linestyle=':', linewidth=1, alpha=0.7)

ax_12_top.set_ylabel('Neuron ID', fontsize=7)
ax_12_top.set_xlim(plot_start_time, plot_end_time)
ax_12_top.set_ylim(0, n_neurons_pert)
ax_12_top.tick_params(labelsize=6, labelbottom=False)
ax_12_top.text(-0.28, 1.05, 'e', transform=ax_12_top.transAxes,
           fontsize=7, fontweight='bold', va='top')

# Add gstd text
ax_12_top.text(0.02, 0.98, 'gˢᵗᵈ=2.0', transform=ax_12_top.transAxes,
               fontsize=6, verticalalignment='top', horizontalalignment='left')

# Spatial patterns - both g=0.6 and g=2.0
symbol_seq_zoom_g2 = symbol_seq_g2[start_bin:end_bin]

ax_12_bottom.plot(time_bins_zoom, symbol_seq_zoom_g2, linewidth=0.5, color='k', zorder=3)
ax_12_bottom.plot(time_bins_zoom, symbol_seq_zoom, linewidth=0.5, color='k',
                 alpha=0.4, linestyle='--', label='gˢᵗᵈ=0.6', zorder=2)
ax_12_bottom.axvline(perturbation_time, color='gray', linestyle='--', linewidth=1)

ax_12_bottom.set_xlabel('Time relative to perturbation (ms)', fontsize=7)
ax_12_bottom.set_ylabel('Pattern ID', fontsize=6)
ax_12_bottom.set_xlim(plot_start_time, plot_end_time)
# Convert x-axis: 500ms becomes 0
ax_12_bottom.set_xticks(xticks_original)
ax_12_bottom.set_xticklabels(xticks_labels.astype(int))
ax_12_bottom.tick_params(labelsize=6)
ax_12_bottom.legend(loc='upper left', bbox_to_anchor=(0.3, 0.99), borderpad=0, fontsize=5, frameon=False)

# Sync y-limits for spatial patterns
max_pattern_id = max(symbol_seq_zoom.max(), symbol_seq_zoom_g2.max())
min_pattern_id = min(symbol_seq_zoom.min(), symbol_seq_zoom_g2.min())

for ax_bottom in [ax_11_bottom, ax_12_bottom]:
    ax_bottom.set_ylim(min_pattern_id - 10, max_pattern_id + 10)

# =============================================================================
# SAVE FIGURE
# =============================================================================

ax_00.yaxis.set_label_coords(-0.20, 0.5)
ax_10_top.yaxis.set_label_coords(-0.20, 0.5)

ax_01.yaxis.set_label_coords(-0.20, 0.5)
ax_11_top.yaxis.set_label_coords(-0.20, 0.5)
ax_11_bottom.yaxis.set_label_coords(-0.20, 0.5)

ax_02.yaxis.set_label_coords(-0.2, 0.5)
ax_12_top.yaxis.set_label_coords(-0.2, 0.5)
ax_12_bottom.yaxis.set_label_coords(-0.2, 0.5)

output_svg = os.path.join(script_dir, 'supplementary_figure_1.svg')
output_pdf = os.path.join(script_dir, 'supplementary_figure_1.pdf')

plt.savefig(output_svg, format='svg', dpi=450, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=450, bbox_inches='tight')

print(f"Supplementary figure saved as '{output_svg}' and '{output_pdf}'")
print()
print("="*80)
print("COMPLETE!")
print("="*80)
