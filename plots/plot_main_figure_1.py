# plots/plot_main_figure.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

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
print("PLOTTING MAIN FIGURE")
print("="*80)

data_file = os.path.join(project_root, 'data_curation', 'network_dynamics_data.pkl')
with open(data_file, 'rb') as f:
    data = pickle.load(f)

# Extract data
spike_times_main = data['spike_times_rate30']
firing_rate_means_main = data['firing_rate_means_main']
firing_rate_stds_main = data['firing_rate_stds_main']
pr_stats = data['pr_stats']
lz_stats = data['lz_stats']
g_values = data['g_values']
static_input_rates = data['rate_values']
n_neurons = data['n_neurons']
duration = data['duration']
transient_time = data['transient_time']

print("Data loaded successfully!")
print()

# =============================================================================
# CREATE FIGURE
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

ax_00 = fig.add_subplot(main_gs[0, 0])

# Panel B positioning
off_y = 0.004
input_bbox = main_gs[0, 1:3].get_position(fig)
input_height = input_bbox.height * 0.15
input_y = input_bbox.y1 - input_height

ax_01_input = fig.add_axes([input_bbox.x0 - 0.025, input_y - off_y,
                            input_bbox.width + 0.025, input_height - off_y])

raster_height = input_bbox.height * 0.6
raster_y = input_y - raster_height - 0.02

ax_01_top = fig.add_axes([input_bbox.x0, raster_y - off_y,
                          input_bbox.width, raster_height - off_y])

pop_height = input_bbox.height * 0.15
pop_y = raster_y - pop_height - 0.02

ax_01_bottom = fig.add_axes([input_bbox.x0, pop_y - off_y,
                             input_bbox.width, pop_height - off_y])

ax_10 = fig.add_subplot(main_gs[1, 0])
ax_11 = fig.add_subplot(main_gs[1, 1])
ax_12 = fig.add_subplot(main_gs[1, 2])

# =============================================================================
# PANEL A: Placeholder
# =============================================================================

ax_00.set_xlim(0, 1)
ax_00.set_ylim(0, 1)
ax_00.axis('off')
ax_00.text(-0.25, 1.05, 'a', transform=ax_00.transAxes,
           fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL B: INPUT PLOT
# =============================================================================

colors = ['#99CCFF', '#66B3FF', '#3399FF', '#0080FF', '#0059B3', '#003D80']
static_input_rates_aux = 0.3 + np.array([1, 2, 3, 4, 5, 6])

transient_time_1 = -40
duration_1 = 800
t_input = np.arange(transient_time_1, duration_1)

for i, val in enumerate(static_input_rates_aux):
    y = np.where(t_input < 0, 0, val)
    ax_01_input.plot(t_input, y, color=colors[i], linewidth=0.7,
                    label=f'{static_input_rates[i]} Hz')

for spine in ax_01_input.spines.values():
    spine.set_visible(False)
ax_01_input.set_xticks([])
ax_01_input.set_yticks([])
ax_01_input.set_xlim(transient_time_1, duration_1)
ax_01_input.legend(fontsize=5, frameon=False, loc='upper right',
                   bbox_to_anchor=(0.9, 0.3), ncol=6, borderpad=0, columnspacing=1)
ax_01_input.set_ylabel("Static\ninput", fontsize=7, labelpad=3.)

# =============================================================================
# PANEL B: RASTER PLOT
# =============================================================================

times = [s[0] for s in spike_times_main]
neuron_ids = [s[1] for s in spike_times_main]

times = np.array(times)
neuron_ids = np.array(neuron_ids)

ax_01_top.vlines(
    times,
    neuron_ids - 0.9,   # tick height
    neuron_ids + 0.9,
    color='black',
    linewidth=0.3,
    alpha=0.9
)
ax_01_top.axvline(transient_time, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax_01_top.set_ylabel('Neuron ID', fontsize=7, verticalalignment='top', labelpad=10)
ax_01_top.set_xlim(0, duration)
ax_01_top.set_ylim(0, n_neurons)
ax_01_top.tick_params(labelsize=6, labelbottom=False)
ax_01_top.text(-0.13, 1.49, 'b', transform=ax_01_top.transAxes,
               fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL B: POPULATION RATE
# =============================================================================

bin_size_pop = 10.0
bins = np.arange(0, duration + bin_size_pop, bin_size_pop)
spike_counts, _ = np.histogram(times, bins=bins)
firing_rate = spike_counts / (bin_size_pop / 1000.0) / n_neurons
bin_centers = bins[:-1] + bin_size_pop/2

ax_01_bottom.plot(bin_centers, firing_rate, 'k-', linewidth=1)
ax_01_bottom.axvline(transient_time, color='red', linestyle='--', linewidth=1, alpha=0.7,
                     label='Steady state')
ax_01_bottom.set_xlabel('Time (ms)', fontsize=7)
ax_01_bottom.set_ylabel('Pop. rate\n(Hz)', fontsize=7, verticalalignment='top', labelpad=17)
ax_01_bottom.set_xlim(0, duration)
ax_01_bottom.tick_params(labelsize=6)
ax_01_bottom.legend(fontsize=5, frameon=False, loc='upper right', borderpad=0)

# =============================================================================
# PANEL C: FIRING RATES VS G_STD
# =============================================================================

for idx, static_rate in enumerate(static_input_rates):
    means = np.array(firing_rate_means_main[static_rate])
    stds = np.array(firing_rate_stds_main[static_rate])

    ax_10.plot(g_values, means, 'o-', color=colors[idx], linewidth=1.5,
               markersize=4, label=f'{static_rate} Hz')
    ax_10.fill_between(g_values, means - stds, means + stds,
                       color=colors[idx], alpha=0.3)

ax_10.set_xticks([0.6, 1, 1.4, 1.8])

ax_10.set_xlabel('gˢᵗᵈ', fontsize=7)
ax_10.set_ylabel('Neuronal firing rate (Hz)', fontsize=7)
ax_10.tick_params(labelsize=6)
ax_10.set_ylim([14, 51])
ax_10.legend(fontsize=5, frameon=False, loc='best', ncol=3, bbox_to_anchor=(1., 1.),
             columnspacing=1.3, borderpad=0)
ax_10.text(-0.25, 1.05, 'c', transform=ax_10.transAxes,
           fontsize=7, fontweight='bold', va='top')



# =============================================================================
# PANEL D: PARTICIPATION RATIO
# =============================================================================

for idx, static_rate in enumerate(static_input_rates):
    pr_means = []
    pr_stds = []

    for g_val in g_values:
        pr_means.append(pr_stats[static_rate][g_val]['mean'])
        pr_stds.append(pr_stats[static_rate][g_val]['std'])

    pr_means = np.array(pr_means)
    pr_stds = np.array(pr_stds)

    ax_11.plot(g_values, pr_means, 'o-', color=colors[idx], linewidth=1.5,
               markersize=4, label=f'{static_rate} Hz')
    ax_11.fill_between(g_values, pr_means - pr_stds, pr_means + pr_stds,
                       color=colors[idx], alpha=0.3)

ax_11.set_xticks([0.6, 1, 1.4, 1.8])

ax_11.set_xlabel('gˢᵗᵈ', fontsize=7)
ax_11.set_ylabel('Participation ratio', fontsize=7)
ax_11.tick_params(labelsize=6)
ax_11.set_ylim([10,121])
ax_11.legend(fontsize=5, frameon=False, loc='best', borderpad=0)
ax_11.text(-0.27, 1.05, 'd', transform=ax_11.transAxes,
           fontsize=7, fontweight='bold', va='top')

# =============================================================================
# PANEL E: LEMPEL-ZIV COMPLEXITY
# =============================================================================

for idx, static_rate in enumerate(static_input_rates):
    lz_means = []
    lz_stds = []
    for g_val in g_values:
        lz_means.append(lz_stats[static_rate][g_val]['mean'])
        lz_stds.append(lz_stats[static_rate][g_val]['std'])
    lz_means = np.array(lz_means)
    lz_stds = np.array(lz_stds)
    ax_12.plot(g_values, lz_means, 'o-', color=colors[idx], linewidth=1.5,
               markersize=4, label=f'{static_rate} Hz')
    ax_12.fill_between(g_values, lz_means - lz_stds, lz_means + lz_stds,
                       color=colors[idx], alpha=0.3)

# Scientific notation for y-axis
ax_12.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax_12.set_xticks([0.6, 1, 1.4, 1.8])

ax_12.set_xlabel('gˢᵗᵈ', fontsize=7)
ax_12.set_ylabel('Lempel-Ziv complexity', fontsize=7)
ax_12.tick_params(labelsize=6)
ax_12.legend(fontsize=5, frameon=False, loc='best', borderpad=0)
ax_12.text(-0.25, 1.05, 'e', transform=ax_12.transAxes,
           fontsize=7, fontweight='bold', va='top')
ax_12.yaxis.get_offset_text().set_fontsize(5)

# =============================================================================
# SAVE FIGURE
# =============================================================================

# fig.align_ylabels([ax_01_top, ax_01_bottom])

ax_01_input.yaxis.set_label_coords(-0.015, 0.4)
ax_01_top.yaxis.set_label_coords(-0.12, 0.5)
ax_01_bottom.yaxis.set_label_coords(-0.12, 0.5)

# ax_11.yaxis.set_label_coords(-0.23, 0.5)


output_svg = os.path.join(script_dir, 'main_figure_1.svg')
output_pdf = os.path.join(script_dir, 'main_figure_1.pdf')

plt.savefig(output_svg, format='svg', dpi=450, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=450, bbox_inches='tight')

print(f"Main figure saved as '{output_svg}' and '{output_pdf}'")
print()
print("="*80)
print("COMPLETE!")
print("="*80)
