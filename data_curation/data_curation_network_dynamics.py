# data_curation/data_curation_network_dynamics.py
"""
Data curation script for network dynamics experiments.

UPDATED:
- Uses simulate() method (renamed from simulate_network_dynamics)
- Uses cached transient states where available
- For g_std=0: simulates from scratch (no cached states)
- For perturbation analysis: simulates from scratch to show pre-perturbation period
"""
import numpy as np
import pickle
import os
import sys
from scipy.stats import spearmanr

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change working directory to script directory so files save here
os.chdir(script_dir)

# Add project root to path (one level up from script directory)
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.spiking_network import SpikingRNN

# =============================================================================
# CONFIGURATION
# =============================================================================

# Session range to use for stability sweep data
START_SESSION = 0
END_SESSION = 20  # Exclusive (goes from START_SESSION to END_SESSION-1)

# Parameters for stability sweep (Panels D, E, Supp B, F)
G_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
RATE_VALUES = [30, 31, 32, 33, 34, 35]
DATA_DIR = os.path.join(project_root, 'results/data/stability/')

# Parameters for input-output mapping (Supplementary Panel A)
G_STD_VALUES_IO = [0.0, 1.0]  # Two curves
INPUT_RATES_IO = np.linspace(30, 35, 100)

# Parameters for perturbation analysis (Supp Panels C, D, E)
# UPDATED: Perturbation at 1000ms (not 500ms) to show full transient
N_NEURONS_PERT = 1000
DT_PERT = 0.1
DURATION_PRE_PERT = 1000.0  # CHANGED: Full transient period
DURATION_POST_PERT = 15.0   # CHANGED: Only need short post-perturbation window
DURATION_TOTAL_PERT = DURATION_PRE_PERT + DURATION_POST_PERT  # 1015ms
PERTURBATION_TIME = 1000.0  # CHANGED: Perturbation at 1000ms
G_STD_BASE_PERT = 0.6
STATIC_RATE_PERT = 30.0
V_TH_STD_PERT = 0
TRIAL_ID_PERT = 1
SESSION_ID_PERT = 0
PERTURBATION_NEURON_IDX = 0

# Network simulation parameters
N_NEURONS = 1000
DT = 0.1
TRANSIENT_TIME = 1000.0  # Standard transient time
ANALYSIS_DURATION = 300.0  # Duration for firing rate analysis
V_TH_STD = 0
TRIAL_ID = 1

# Cached states directory
TRANSIENT_CACHE_DIR = os.path.join(project_root, 'results/cached_states')

NETWORK_PARAMS = {
    'v_th_distribution': 'normal',
    'static_input_strength': 10.0,
    'dynamic_input_strength': 1.0,
    'readout_weight_scale': 1.0
}

# Output file (will save in script directory)
OUTPUT_FILE = 'network_dynamics_data.pkl'

print("="*80)
print("DATA CURATION FOR NETWORK DYNAMIC EXPERIMENTS")
print("="*80)
print(f"Working directory: {os.getcwd()}")
print(f"Will save data to: {os.path.abspath(OUTPUT_FILE)}")
print(f"Sessions to use: {START_SESSION} to {END_SESSION-1}")
print(f"G values: {G_VALUES}")
print(f"Rate values: {RATE_VALUES}")
print(f"Input-output mapping: g_std={G_STD_VALUES_IO}, {len(INPUT_RATES_IO)} input rates")
print(f"Cached states directory: {TRANSIENT_CACHE_DIR}")
print()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_spike_matrix(spike_list, n_neurons, duration, step_size):
    """Create spike matrix from spike list"""
    n_steps = int(np.ceil(duration / step_size))
    spike_matrix = np.zeros((n_steps, n_neurons))

    for spike_time, neuron_id in spike_list:
        time_bin = int(round(spike_time / step_size))
        if 0 <= time_bin < n_steps and 0 <= neuron_id < n_neurons:
            spike_matrix[time_bin, neuron_id] += 1

    return spike_matrix


def get_mean_firing_rate(spike_matrix, transient_time, step_size):
    """Get mean firing rate across all neurons, excluding transient"""
    transient_bins = int(transient_time / step_size)
    spike_matrix_analysis = spike_matrix[transient_bins:, :]
    mean_rate = np.mean(spike_matrix_analysis) / (step_size / 1000.0)
    return mean_rate


def get_firing_rates_from_matrix(spike_matrix, transient_time, step_size):
    """Get per-neuron firing rates, excluding transient"""
    transient_bins = int(transient_time / step_size)
    spike_matrix_analysis = spike_matrix[transient_bins:, :]
    firing_rates = np.mean(spike_matrix_analysis, axis=0) / (step_size / 1000.0)
    return firing_rates


def get_firing_rate_from_spikes(spike_list, n_neurons, duration_ms):
    """Compute mean firing rate directly from spike list"""
    duration_s = duration_ms / 1000.0
    return len(spike_list) / (n_neurons * duration_s)


def spikes_to_binary(spikes, num_neurons, duration, bin_size):
    """Convert spike times to binary matrix."""
    num_bins = int(duration / bin_size)
    binary_matrix = np.zeros((num_neurons, num_bins), dtype=int)

    for spike_time, neuron_id in spikes:
        time_bin = int(round(spike_time / bin_size))
        if 0 <= time_bin < num_bins:
            binary_matrix[neuron_id, time_bin] = 1

    return binary_matrix


def compute_spatial_patterns(spike_diff_matrix, bin_size):
    """
    Compute spatial pattern IDs over time from spike difference matrix.

    Returns:
        symbol_seq: Array of pattern IDs over time
        pattern_dict: Dictionary mapping patterns to IDs
    """
    spatial_patterns = [tuple(spike_diff_matrix[:, t])
                       for t in range(spike_diff_matrix.shape[1])]

    pattern_dict = {}
    symbol_seq = []
    next_id = 0

    for pat in spatial_patterns:
        if pat not in pattern_dict:
            pattern_dict[pat] = next_id
            next_id += 1
        symbol_seq.append(pattern_dict[pat])

    return np.array(symbol_seq), pattern_dict


def categorize_spikes(spikes_ctrl, spikes_pert, time_start, time_end, bin_size):
    """
    Categorize spikes into common, control-only, and perturbed-only.

    Returns:
        common_spikes: [(time, neuron), ...] in both networks
        ctrl_only: [(time, neuron), ...] only in control
        pert_only: [(time, neuron), ...] only in perturbed
    """
    ctrl_set = set([(round(t/bin_size)*bin_size, n) for t, n in spikes_ctrl
                     if time_start <= t <= time_end])
    pert_set = set([(round(t/bin_size)*bin_size, n) for t, n in spikes_pert
                     if time_start <= t <= time_end])

    common_set = ctrl_set & pert_set
    ctrl_only_set = ctrl_set - pert_set
    pert_only_set = pert_set - ctrl_set

    return list(common_set), list(ctrl_only_set), list(pert_only_set)


def load_cached_transient_state(session_id, g_std, v_th_std, static_rate, trial_id,
                                 cache_dir=TRANSIENT_CACHE_DIR):
    """
    Load pre-cached transient state from file.

    Returns:
        state dict if found, None otherwise
    """
    filename = os.path.join(cache_dir,
        f"session_{session_id}_g_{g_std:.3f}_vth_{v_th_std:.3f}_"
        f"rate_{static_rate:.1f}_trial_states.pkl")

    if not os.path.exists(filename):
        return None

    try:
        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['trial_states'][trial_id]
    except (KeyError, IndexError, Exception) as e:
        print(f"  Warning: Could not load cached state: {e}")
        return None


def simulate_with_cached_state(session_id, g_std, v_th_std, static_rate, trial_id,
                                analysis_duration, n_neurons=N_NEURONS, dt=DT):
    """
    Simulate network using cached transient state if available.

    For g_std=0: No cached states exist, simulate from scratch
    For g_std≠0: Load cached state at 1000ms, simulate analysis_duration more

    Returns:
        spike_list: List of (time, neuron_id) tuples for analysis period
        used_cache: Boolean indicating if cached state was used
    """
    # Try to load cached state
    cached_state = None
    if abs(g_std) > 1e-6:  # g_std != 0
        cached_state = load_cached_transient_state(
            session_id, g_std, v_th_std, static_rate, trial_id)

    network = SpikingRNN(n_neurons=n_neurons, dt=dt, synaptic_mode="filter",
                        static_input_mode="common_tonic")
    network.initialize_network(session_id, v_th_std, g_std, **NETWORK_PARAMS)

    if cached_state is not None:
        # Use cached state: restore and simulate analysis period only
        network.restore_state(cached_state)
        transient_end_time = cached_state['current_time']

        spikes = network.simulate(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=analysis_duration,
            static_input_rate=static_rate,
            continue_from_state=True
        )

        # Adjust spike times to be relative to analysis period start
        spikes_adjusted = [(t - transient_end_time, n) for t, n in spikes
                          if t >= transient_end_time]
        return spikes_adjusted, True
    else:
        # No cached state: simulate from scratch with transient
        total_duration = TRANSIENT_TIME + analysis_duration

        spikes = network.simulate(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=total_duration,
            static_input_rate=static_rate,
            continue_from_state=False
        )

        # Extract only spikes after transient, adjust times
        spikes_adjusted = [(t - TRANSIENT_TIME, n) for t, n in spikes
                          if t >= TRANSIENT_TIME]
        return spikes_adjusted, False


# =============================================================================
# PART 1: INPUT-OUTPUT MAPPING (SUPPLEMENTARY PANEL A)
# =============================================================================

print("="*80)
print("PART 1: INPUT-OUTPUT MAPPING FOR SUPPLEMENTARY PANEL A")
print("="*80)

input_rates_io = INPUT_RATES_IO
output_rates_io_dict = {}  # {g_std: [output_rates]}
spike_times_rate30 = None  # Will store spike times for rate=30Hz from g=1.0

bin_size = 2.0  # ms

for g_std_io in G_STD_VALUES_IO:
    print(f"\nRunning for g_std = {g_std_io}")
    print("-" * 80)

    # Check if cached states are available for this g_std
    if abs(g_std_io) > 1e-6:
        test_cache = load_cached_transient_state(0, g_std_io, V_TH_STD, 30.0, TRIAL_ID)
        if test_cache is not None:
            print(f"  Using cached transient states (1000ms pre-computed)")
        else:
            print(f"  No cached states found, will simulate full transient")
    else:
        print(f"  g_std=0: Will simulate full transient from scratch")

    output_rates_io = []

    for idx, input_rate in enumerate(input_rates_io):
        session_id = int(10000 + int(g_std_io * 1000) + idx)

        # Use cached states if available
        spikes, used_cache = simulate_with_cached_state(
            session_id=session_id,
            g_std=g_std_io,
            v_th_std=V_TH_STD,
            static_rate=input_rate,
            trial_id=TRIAL_ID,
            analysis_duration=ANALYSIS_DURATION
        )

        # Compute firing rate from analysis period spikes
        mean_rate = get_firing_rate_from_spikes(spikes, N_NEURONS, ANALYSIS_DURATION)
        output_rates_io.append(mean_rate)

        # Save spike times for rate=30Hz from g=1.0 for Main Panel B raster plot
        # Need FULL simulation for raster (including transient)
        if abs(input_rate - 30.0) < 0.1 and abs(g_std_io - 1.0) < 0.01:
            # For raster plot, we need spikes from t=0 (including transient)
            # So simulate from scratch for this specific case
            print(f"  *** Generating full spike times for rate=30Hz, g=1.0 (for raster plot)")
            network_raster = SpikingRNN(n_neurons=N_NEURONS, dt=DT, synaptic_mode="filter",
                                       static_input_mode="common_tonic")
            network_raster.initialize_network(session_id, V_TH_STD, g_std_io, **NETWORK_PARAMS)

            # Simulate full duration including transient for raster visualization
            raster_duration = TRANSIENT_TIME + ANALYSIS_DURATION  # 1300ms total
            spike_times_rate30 = network_raster.simulate(
                session_id=session_id,
                v_th_std=V_TH_STD,
                g_std=g_std_io,
                trial_id=TRIAL_ID,
                duration=raster_duration,
                static_input_rate=input_rate,
                continue_from_state=False
            )
            del network_raster

        if (idx + 1) % 20 == 0:
            cache_str = "(cached)" if used_cache else "(fresh)"
            print(f"  [{idx+1}/{len(input_rates_io)}] Input={input_rate:.2f} Hz → Output={mean_rate:.2f} Hz {cache_str}")

    output_rates_io_dict[g_std_io] = np.array(output_rates_io)

    print(f"  Output range: {np.min(output_rates_io):.2f} - {np.max(output_rates_io):.2f} Hz")

print(f"\nInput-output mapping complete!")
print()

# =============================================================================
# PART 2: FIRING RATES VS G_STD (MAIN FIGURE PANEL C)
# =============================================================================

print("="*80)
print("PART 2: FIRING RATES VS G_STD FOR MAIN PANEL C")
print("="*80)

firing_rate_means_main = {rate: [] for rate in RATE_VALUES}
firing_rate_stds_main = {rate: [] for rate in RATE_VALUES}

sim_count = 0
total_sims = len(G_VALUES) * len(RATE_VALUES)

for g_val in G_VALUES:
    for static_rate in RATE_VALUES:
        sim_count += 1
        session_id = 30000 + sim_count

        print(f"  [{sim_count}/{total_sims}] g={g_val:.2f}, rate={static_rate} Hz...", end=" ")

        # Use cached states if available
        spikes, used_cache = simulate_with_cached_state(
            session_id=session_id,
            g_std=g_val,
            v_th_std=V_TH_STD,
            static_rate=static_rate,
            trial_id=TRIAL_ID,
            analysis_duration=ANALYSIS_DURATION
        )

        # Create spike matrix for per-neuron analysis
        spike_matrix = create_spike_matrix(spikes, N_NEURONS, ANALYSIS_DURATION, bin_size)
        # No transient removal needed - spikes are already from analysis period only
        firing_rates = np.mean(spike_matrix, axis=0) / (bin_size / 1000.0)

        mean_rate = np.mean(firing_rates)
        std_rate = np.std(firing_rates)

        firing_rate_means_main[static_rate].append(mean_rate)
        firing_rate_stds_main[static_rate].append(std_rate)

        cache_str = "(cached)" if used_cache else "(fresh)"
        print(f"Mean={mean_rate:.2f} Hz {cache_str}")

print(f"\nFiring rate analysis complete!")
print()

# =============================================================================
# PART 3: STABILITY SWEEP DATA (MAIN PANELS D, E & SUPP PANELS B, F)
# =============================================================================

print("="*80)
print("PART 3: STABILITY SWEEP DATA COLLECTION")
print("="*80)

# Storage for collected data
participation_ratio_data = {}
lz_complexity_data = {}
dimensionality_data = {}
kistler_data = {}

# For scatter plots
all_pr_values = []
all_dim_values = []
all_lz_values = []
all_kistler_values = []

missing_files = []

# Initialize nested dictionaries
for rate in RATE_VALUES:
    participation_ratio_data[rate] = {g: [] for g in G_VALUES}
    lz_complexity_data[rate] = {g: [] for g in G_VALUES}
    dimensionality_data[rate] = {g: [] for g in G_VALUES}
    kistler_data[rate] = {g: [] for g in G_VALUES}

n_sessions = END_SESSION - START_SESSION
total_files = n_sessions * len(G_VALUES) * len(RATE_VALUES)
processed = 0
found = 0

print(f"Looking for {total_files} files...")
print()

for session in range(START_SESSION, END_SESSION):
    for g in G_VALUES:
        for rate in RATE_VALUES:
            filename = f'stability_session_{session}_vth_0.000_g_{g:.3f}_rate_{rate}.pkl'
            filepath = os.path.join(DATA_DIR, filename)

            processed += 1

            if not os.path.exists(filepath):
                missing_files.append((session, g, rate))
                print(f"[{processed}/{total_files}] MISSING FILE: session={session}, g={g}, rate={rate}")
                continue

            try:
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)

                result = results[0]

                # Extract data
                pr_mean = result['participation_ratio_2ms_mean']
                lz_mean = result['lz_spatial_patterns_mean']
                dim_mean = result['dimensionality_2ms_mean']
                kistler_mean = result['kistler_delta_2.0ms_mean']

                # Check if any data is None or NaN
                if pr_mean is None or np.isnan(pr_mean):
                    print(f"[{processed}/{total_files}] INVALID DATA (PR is None/NaN): session={session}, g={g}, rate={rate}")
                    missing_files.append((session, g, rate))
                    continue

                if lz_mean is None or np.isnan(lz_mean):
                    print(f"[{processed}/{total_files}] INVALID DATA (LZ is None/NaN): session={session}, g={g}, rate={rate}")
                    missing_files.append((session, g, rate))
                    continue

                if dim_mean is None or np.isnan(dim_mean):
                    print(f"[{processed}/{total_files}] INVALID DATA (Dim is None/NaN): session={session}, g={g}, rate={rate}")
                    missing_files.append((session, g, rate))
                    continue

                if kistler_mean is None or np.isnan(kistler_mean):
                    print(f"[{processed}/{total_files}] INVALID DATA (Kistler is None/NaN): session={session}, g={g}, rate={rate}")
                    missing_files.append((session, g, rate))
                    continue

                # Store in dictionaries (only if all data is valid)
                participation_ratio_data[rate][g].append(pr_mean)
                lz_complexity_data[rate][g].append(lz_mean)
                dimensionality_data[rate][g].append(dim_mean)
                kistler_data[rate][g].append(kistler_mean)

                # Store for scatter plots
                all_pr_values.append(pr_mean)
                all_dim_values.append(dim_mean)
                all_lz_values.append(lz_mean)
                all_kistler_values.append(kistler_mean)

                found += 1

            except Exception as e:
                print(f"[{processed}/{total_files}] ERROR loading: session={session}, g={g}, rate={rate}")
                print(f"    Error: {e}")
                missing_files.append((session, g, rate))

print()
print("="*80)
print(f"Data collection complete: {found}/{total_files} files with valid data")
print(f"Missing or invalid: {len(missing_files)}")
print("="*80)
print()

if missing_files:
    print("MISSING/INVALID FILES SUMMARY:")
    print("-" * 80)
    for session, g, rate in missing_files:
        print(f"  Session {session:2d}, g={g:.1f}, rate={rate}")
    print()

# =============================================================================
# COMPUTE STATISTICS FOR MAIN PANELS D & E
# =============================================================================

print("Computing statistics across sessions...")
print()

pr_stats = {}
lz_stats = {}

for rate in RATE_VALUES:
    pr_stats[rate] = {}
    lz_stats[rate] = {}

    for g in G_VALUES:
        # Participation ratio stats
        pr_values = participation_ratio_data[rate][g]
        if len(pr_values) > 0:
            pr_stats[rate][g] = {
                'mean': np.mean(pr_values),
                'std': np.std(pr_values),
                'n_samples': len(pr_values),
                'values': pr_values
            }
        else:
            pr_stats[rate][g] = {
                'mean': np.nan,
                'std': np.nan,
                'n_samples': 0,
                'values': []
            }
            print(f"WARNING: No valid data for PR at rate={rate}, g={g}")

        # LZ complexity stats
        lz_values = lz_complexity_data[rate][g]
        if len(lz_values) > 0:
            lz_stats[rate][g] = {
                'mean': np.mean(lz_values),
                'std': np.std(lz_values),
                'n_samples': len(lz_values),
                'values': lz_values
            }
        else:
            lz_stats[rate][g] = {
                'mean': np.nan,
                'std': np.nan,
                'n_samples': 0,
                'values': []
            }
            print(f"WARNING: No valid data for LZ at rate={rate}, g={g}")

# =============================================================================
# COMPUTE CORRELATIONS (BOTH PEARSON AND SPEARMAN)
# =============================================================================

print("Computing correlations...")
print()

# PR vs Dimensionality
if len(all_pr_values) > 0:
    pearson_pr_dim = np.corrcoef(all_pr_values, all_dim_values)[0, 1]
    spearman_pr_dim, _ = spearmanr(all_pr_values, all_dim_values)
    print(f"PR vs Dimensionality:")
    print(f"  Pearson r = {pearson_pr_dim:.3f}")
    print(f"  Spearman ρ = {spearman_pr_dim:.3f}")
else:
    pearson_pr_dim = np.nan
    spearman_pr_dim = np.nan

# LZ vs Kistler
if len(all_lz_values) > 0:
    pearson_lz_kistler = np.corrcoef(all_lz_values, all_kistler_values)[0, 1]
    spearman_lz_kistler, _ = spearmanr(all_lz_values, all_kistler_values)
    print(f"LZ vs Kistler:")
    print(f"  Pearson r = {pearson_lz_kistler:.3f}")
    print(f"  Spearman ρ = {spearman_lz_kistler:.3f}")
else:
    pearson_lz_kistler = np.nan
    spearman_lz_kistler = np.nan

print()

# =============================================================================
# PART 4: PERTURBATION ANALYSIS (SUPPLEMENTARY PANELS C, D, E)
# =============================================================================

print("="*80)
print("PART 4: PERTURBATION ANALYSIS FOR SUPPLEMENTARY PANELS C, D, E")
print("="*80)

from src.rng_utils import get_rng

# Get perturbation neuron
rng = get_rng(SESSION_ID_PERT, V_TH_STD_PERT, G_STD_BASE_PERT, 0, 'perturbation_targets')
perturbation_neurons = rng.choice(N_NEURONS_PERT, size=min(100, N_NEURONS_PERT), replace=False)
perturbation_neuron = int(perturbation_neurons[PERTURBATION_NEURON_IDX])

print(f"Base g_std: {G_STD_BASE_PERT}")
print(f"Perturbation neuron: {perturbation_neuron}")
print(f"Total duration: {DURATION_TOTAL_PERT} ms")
print(f"Perturbation time: {PERTURBATION_TIME} ms")
print()

network_params_pert = {
    'v_th_distribution': 'normal',
    'static_input_strength': 10.0,
    'readout_weight_scale': 1.0
}

# =============================================================================
# Step 1: Simulate from scratch 0→1000ms (NOT using cached states)
# This is needed to show spikes ~5ms BEFORE perturbation
# =============================================================================
print(f"Step 1: Transient 0→{int(DURATION_PRE_PERT)}ms (simulating from scratch)...")
network = SpikingRNN(N_NEURONS_PERT, dt=DT_PERT, synaptic_mode="filter",
                     static_input_mode="common_tonic")
network.initialize_network(SESSION_ID_PERT, V_TH_STD_PERT, G_STD_BASE_PERT, **network_params_pert)

spikes_transient = network.simulate(
    session_id=SESSION_ID_PERT,
    v_th_std=V_TH_STD_PERT,
    g_std=G_STD_BASE_PERT,
    trial_id=TRIAL_ID_PERT,
    duration=DURATION_PRE_PERT,
    static_input_rate=STATIC_RATE_PERT,
    continue_from_state=False
)

state_at_perturbation = network.save_state()
del network

print(f"  Spikes during transient: {len(spikes_transient)}")
print(f"  State saved at t={state_at_perturbation['current_time']:.1f}ms")
print()

# =============================================================================
# Step 2: Control g=0.6 (from perturbation time onward)
# =============================================================================
print(f"Step 2: Control (g=0.6) {int(PERTURBATION_TIME)}→{int(PERTURBATION_TIME + DURATION_POST_PERT)}ms...")
network_ctrl_g06 = SpikingRNN(N_NEURONS_PERT, dt=DT_PERT, synaptic_mode="filter",
                               static_input_mode="common_tonic")
network_ctrl_g06.initialize_network(SESSION_ID_PERT, V_TH_STD_PERT, G_STD_BASE_PERT, **network_params_pert)
network_ctrl_g06.restore_state(state_at_perturbation)

spikes_ctrl_g06_post = network_ctrl_g06.simulate(
    session_id=SESSION_ID_PERT,
    v_th_std=V_TH_STD_PERT,
    g_std=G_STD_BASE_PERT,
    trial_id=TRIAL_ID_PERT,
    duration=DURATION_POST_PERT,
    static_input_rate=STATIC_RATE_PERT,
    continue_from_state=True
)

spikes_control_g06 = spikes_transient + [(t, n) for t, n in spikes_ctrl_g06_post]
del network_ctrl_g06

print(f"  Total spikes: {len(spikes_control_g06)}")
print()

# =============================================================================
# Step 3: Perturbed g=0.6
# =============================================================================
print(f"Step 3: Perturbed (g=0.6) {int(PERTURBATION_TIME)}→{int(PERTURBATION_TIME + DURATION_POST_PERT)}ms...")
network_pert_g06 = SpikingRNN(N_NEURONS_PERT, dt=DT_PERT, synaptic_mode="filter",
                               static_input_mode="common_tonic")
network_pert_g06.initialize_network(SESSION_ID_PERT, V_TH_STD_PERT, G_STD_BASE_PERT, **network_params_pert)
network_pert_g06.restore_state(state_at_perturbation)

spikes_pert_g06_post = network_pert_g06.simulate(
    session_id=SESSION_ID_PERT,
    v_th_std=V_TH_STD_PERT,
    g_std=G_STD_BASE_PERT,
    trial_id=TRIAL_ID_PERT,
    duration=DURATION_POST_PERT,
    static_input_rate=STATIC_RATE_PERT,
    perturbation_time=state_at_perturbation['current_time'],
    perturbation_neuron=perturbation_neuron,
    continue_from_state=True
)

spikes_perturbed_g06 = spikes_transient + [(t, n) for t, n in spikes_pert_g06_post]
del network_pert_g06

print(f"  Total spikes: {len(spikes_perturbed_g06)}")
print()

# =============================================================================
# Step 4: Control g=2.0
# =============================================================================
print(f"Step 4: Control (g=2.0) {int(PERTURBATION_TIME)}→{int(PERTURBATION_TIME + DURATION_POST_PERT)}ms...")
g_scale_factor = 2.0 / G_STD_BASE_PERT

network_ctrl_g2 = SpikingRNN(N_NEURONS_PERT, dt=DT_PERT, synaptic_mode="filter",
                              static_input_mode="common_tonic")
network_ctrl_g2.initialize_network(SESSION_ID_PERT, V_TH_STD_PERT, G_STD_BASE_PERT, **network_params_pert)
network_ctrl_g2.recurrent_synapses.weight_matrix.data *= g_scale_factor
network_ctrl_g2.restore_state(state_at_perturbation)

spikes_ctrl_g2_post = network_ctrl_g2.simulate(
    session_id=SESSION_ID_PERT,
    v_th_std=V_TH_STD_PERT,
    g_std=G_STD_BASE_PERT,
    trial_id=TRIAL_ID_PERT,
    duration=DURATION_POST_PERT,
    static_input_rate=STATIC_RATE_PERT,
    continue_from_state=True
)

spikes_control_g2 = spikes_transient + [(t, n) for t, n in spikes_ctrl_g2_post]
del network_ctrl_g2

print(f"  Total spikes: {len(spikes_control_g2)}")
print()

# =============================================================================
# Step 5: Perturbed g=2.0
# =============================================================================
print(f"Step 5: Perturbed (g=2.0) {int(PERTURBATION_TIME)}→{int(PERTURBATION_TIME + DURATION_POST_PERT)}ms...")
network_pert_g2 = SpikingRNN(N_NEURONS_PERT, dt=DT_PERT, synaptic_mode="filter",
                              static_input_mode="common_tonic")
network_pert_g2.initialize_network(SESSION_ID_PERT, V_TH_STD_PERT, G_STD_BASE_PERT, **network_params_pert)
network_pert_g2.recurrent_synapses.weight_matrix.data *= g_scale_factor
network_pert_g2.restore_state(state_at_perturbation)

spikes_pert_g2_post = network_pert_g2.simulate(
    session_id=SESSION_ID_PERT,
    v_th_std=V_TH_STD_PERT,
    g_std=G_STD_BASE_PERT,
    trial_id=TRIAL_ID_PERT,
    duration=DURATION_POST_PERT,
    static_input_rate=STATIC_RATE_PERT,
    perturbation_time=state_at_perturbation['current_time'],
    perturbation_neuron=perturbation_neuron,
    continue_from_state=True
)

spikes_perturbed_g2 = spikes_transient + [(t, n) for t, n in spikes_pert_g2_post]
del network_pert_g2

print(f"  Total spikes: {len(spikes_perturbed_g2)}")
print()

# =============================================================================
# Compute spike difference matrices
# =============================================================================
print("Computing spike difference matrices...")
bin_size_pert = DT_PERT

matrix_control_g06 = spikes_to_binary(spikes_control_g06, N_NEURONS_PERT, DURATION_TOTAL_PERT, bin_size_pert)
matrix_perturbed_g06 = spikes_to_binary(spikes_perturbed_g06, N_NEURONS_PERT, DURATION_TOTAL_PERT, bin_size_pert)
matrix_control_g2 = spikes_to_binary(spikes_control_g2, N_NEURONS_PERT, DURATION_TOTAL_PERT, bin_size_pert)
matrix_perturbed_g2 = spikes_to_binary(spikes_perturbed_g2, N_NEURONS_PERT, DURATION_TOTAL_PERT, bin_size_pert)

spike_diff_g06 = (matrix_control_g06 != matrix_perturbed_g06).astype(int)
spike_diff_g2 = (matrix_control_g2 != matrix_perturbed_g2).astype(int)

print(f"Spike differences g=0.6: {spike_diff_g06.sum()}")
print(f"Spike differences g=2.0: {spike_diff_g2.sum()}")
print()

# Compute spatial patterns
print("Computing spatial patterns...")
symbol_seq_g06, pattern_dict_g06 = compute_spatial_patterns(spike_diff_g06, bin_size_pert)
symbol_seq_g2, pattern_dict_g2 = compute_spatial_patterns(spike_diff_g2, bin_size_pert)

print(f"Spatial patterns g=0.6: {len(pattern_dict_g06)} unique patterns")
print(f"Spatial patterns g=2.0: {len(pattern_dict_g2)} unique patterns")
print()

# Categorize spikes for colored raster
# UPDATED: Plot window around 1000ms perturbation time
plot_start_time = PERTURBATION_TIME - 5.0  # 995ms
plot_end_time = PERTURBATION_TIME + 15.0   # 1015ms

common_spikes_g06, ctrl_only_spikes_g06, pert_only_spikes_g06 = categorize_spikes(
    spikes_control_g06, spikes_perturbed_g06, plot_start_time, plot_end_time, bin_size_pert
)

print(f"Spike categorization (g=0.6):")
print(f"  Common: {len(common_spikes_g06)}")
print(f"  Control-only: {len(ctrl_only_spikes_g06)}")
print(f"  Perturbed-only: {len(pert_only_spikes_g06)}")
print()

print("Perturbation analysis complete!")
print()

# =============================================================================
# SAVE ALL RESULTS
# =============================================================================

output_data = {
    # Configuration
    'start_session': START_SESSION,
    'end_session': END_SESSION,
    'n_sessions_used': END_SESSION - START_SESSION,
    'g_values': G_VALUES,
    'rate_values': RATE_VALUES,
    'n_neurons': N_NEURONS,
    'duration': TRANSIENT_TIME + ANALYSIS_DURATION,  # Total duration for raster (original key name)
    'transient_time': TRANSIENT_TIME,
    'analysis_duration': ANALYSIS_DURATION,

    # Main Figure Panel B (raster plot)
    'spike_times_rate30': spike_times_rate30,

    # Main Figure Panel C (firing rates vs g_std)
    'firing_rate_means_main': firing_rate_means_main,
    'firing_rate_stds_main': firing_rate_stds_main,

    # Main Figure Panels D & E
    'pr_stats': pr_stats,
    'lz_stats': lz_stats,

    # Supplementary Panel A (input-output mapping for g=0 and g=1)
    'io_mapping': {
        'g_std_values': G_STD_VALUES_IO,
        'input_rates': input_rates_io,
        'output_rates_dict': output_rates_io_dict  # {g_std: [output_rates]}
    },

    # Supplementary Figure Panel B (PR vs Dimensionality scatter)
    'scatter_pr_dim': {
        'pr_values': all_pr_values,
        'dim_values': all_dim_values,
        'pearson_r': pearson_pr_dim,
        'spearman_rho': spearman_pr_dim
    },

    # Supplementary Figure Panel F (LZ vs Kistler scatter)
    'scatter_lz_kistler': {
        'lz_values': all_lz_values,
        'kistler_values': all_kistler_values,
        'pearson_r': pearson_lz_kistler,
        'spearman_rho': spearman_lz_kistler
    },

    # Supplementary Figure Panels C, D, E (perturbation analysis)
    'perturbation_data': {
        'n_neurons': N_NEURONS_PERT,
        'duration_total': DURATION_TOTAL_PERT,
        'perturbation_time': PERTURBATION_TIME,
        'perturbation_neuron': perturbation_neuron,
        'bin_size': bin_size_pert,
        'plot_start_time': plot_start_time,
        'plot_end_time': plot_end_time,
        # Spike times
        'spikes_control_g06': spikes_control_g06,
        'spikes_perturbed_g06': spikes_perturbed_g06,
        'spikes_control_g2': spikes_control_g2,
        'spikes_perturbed_g2': spikes_perturbed_g2,
        # Spike difference matrices
        'spike_diff_g06': spike_diff_g06,
        'spike_diff_g2': spike_diff_g2,
        # Spatial patterns
        'symbol_seq_g06': symbol_seq_g06,
        'symbol_seq_g2': symbol_seq_g2,
        # Categorized spikes for raster
        'common_spikes_g06': common_spikes_g06,
        'ctrl_only_spikes_g06': ctrl_only_spikes_g06,
        'pert_only_spikes_g06': pert_only_spikes_g06,
    },

    # Metadata
    'missing_files': missing_files,
    'n_files_found': found,
    'n_files_expected': total_files
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(output_data, f)

print()
print("="*80)
print(f"DATA CURATION COMPLETE! Results saved to: {os.path.abspath(OUTPUT_FILE)}")
print("="*80)
print()
print("Summary:")
print(f"  Sessions used: {START_SESSION} to {END_SESSION-1}")
print(f"  G values: {len(G_VALUES)}")
print(f"  Rate values: {len(RATE_VALUES)}")
print(f"  Files with valid data: {found}/{total_files} ({100*found/total_files:.1f}%)" if total_files > 0 else "  Files with valid data: N/A")
print(f"  Input-output curves: {len(G_STD_VALUES_IO)} (g_std = {G_STD_VALUES_IO})")
print(f"  Input-output points per curve: {len(input_rates_io)}")
print(f"  Scatter plot points (PR-Dim): {len(all_pr_values)}")
if len(all_pr_values) > 0:
    print(f"    Pearson r = {pearson_pr_dim:.3f}, Spearman ρ = {spearman_pr_dim:.3f}")
print(f"  Scatter plot points (LZ-Kistler): {len(all_lz_values)}")
if len(all_lz_values) > 0:
    print(f"    Pearson r = {pearson_lz_kistler:.3f}, Spearman ρ = {spearman_lz_kistler:.3f}")
print(f"  Perturbation simulations: Complete")
print(f"    Perturbation time: {PERTURBATION_TIME} ms")
print(f"    Plot window: {plot_start_time} - {plot_end_time} ms")
print()
