# experiments/chaos_experiment.py - Fixed with single session runs and clean structure
"""
Chaos analysis experiment with random network structure per parameter combination and single session execution.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Tuple, Any

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.spike_analysis import analyze_perturbation_response_enhanced
except ImportError:
    try:
        from ..src.spiking_network import SpikingRNN
        from ..src.rng_utils import get_rng
        from ..analysis.spike_analysis import analyze_perturbation_response_enhanced
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        src_dir = os.path.join(project_root, 'src')
        analysis_dir = os.path.join(project_root, 'analysis')
        sys.path.insert(0, src_dir)
        sys.path.insert(0, analysis_dir)
        from spiking_network import SpikingRNN
        from rng_utils import get_rng
        from spike_analysis import analyze_perturbation_response_enhanced

class ChaosExperiment:
    """Chaos experiment with random structure and single session execution."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1, synaptic_mode: str = "dynamic"):
        self.n_neurons = n_neurons
        self.dt = dt
        self.synaptic_mode = synaptic_mode

        # Timing parameters
        self.pre_perturbation_time = 50.0
        self.post_perturbation_time = 300.0
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time
        self.perturbation_time = self.pre_perturbation_time
        self.n_perturbation_trials = 100  # 100 trials per combination

    def get_perturbation_neurons(self, session_id: int, v_th_std: float, g_std: float) -> np.ndarray:
        """Get perturbation neurons for this parameter combination."""
        # Generate perturbation neurons directly using parameter-dependent RNG
        rng = get_rng(session_id, v_th_std, g_std, 0, 'perturbation_targets')
        sample_size = min(100, self.n_neurons)
        return rng.choice(self.n_neurons, size=sample_size, replace=False)

    def run_single_perturbation(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                              v_th_distribution: str, perturbation_neuron_idx: int,
                              static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single perturbation with random structure per parameter combination."""

        # Create identical networks with random structure (depends on session + parameters)
        network_control = SpikingRNN(self.n_neurons, dt=self.dt, synaptic_mode=self.synaptic_mode)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt, synaptic_mode=self.synaptic_mode)

        # Network parameters with direct heterogeneity
        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 1.0,
            'dynamic_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        # Initialize both networks with identical random structure (same session + parameters)
        for network in [network_control, network_perturbed]:
            network.initialize_network(session_id, v_th_std, g_std, **network_params)

        # Get perturbation neuron from parameter-dependent list
        perturbation_neurons = self.get_perturbation_neurons(session_id, v_th_std, g_std)
        available_neurons = len(perturbation_neurons)
        safe_idx = perturbation_neuron_idx % available_neurons
        perturbation_neuron = int(perturbation_neurons[safe_idx])

        # Run control simulation (no perturbation)
        spikes_control = network_control.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate
        )

        # Run perturbed simulation (auxiliary spike at perturbation_time)
        spikes_perturbed = network_perturbed.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate,
            perturbation_time=self.perturbation_time,
            perturbation_neuron=perturbation_neuron
        )

        # Use enhanced analysis function
        analysis_results = analyze_perturbation_response_enhanced(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=self.perturbation_time,
            simulation_end=self.total_duration,
            perturbed_neuron=perturbation_neuron
        )

        # Add perturbation info
        analysis_results['perturbation_neuron'] = perturbation_neuron
        analysis_results['perturbation_neuron_idx'] = perturbation_neuron_idx

        return analysis_results

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run parameter combination for single session."""
        start_time = time.time()

        # Store all trial results for statistics
        trial_results = []

        # Use different perturbation neuron indices for 100 trials
        perturbation_neuron_indices = list(range(self.n_perturbation_trials))

        for trial_idx in range(self.n_perturbation_trials):
            trial_id = trial_idx + 1

            trial_result = self.run_single_perturbation(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                v_th_distribution=v_th_distribution,
                perturbation_neuron_idx=perturbation_neuron_indices[trial_idx],
                static_input_rate=static_input_rate
            )

            trial_results.append(trial_result)

        # Compile statistics
        lz_complexities = [r['lz_complexity'] for r in trial_results]
        hamming_slopes = [r['hamming_slope'] for r in trial_results]
        total_spike_diffs = [r['total_spike_differences'] for r in trial_results]
        intrinsic_dims = [r['intrinsic_dimensionality'] for r in trial_results]
        effective_dims = [r['effective_dimensionality'] for r in trial_results]
        participation_ratios = [r['participation_ratio'] for r in trial_results]
        total_variances = [r['total_variance'] for r in trial_results]
        gamma_coincidences = [r['gamma_coincidence'] for r in trial_results]

        results = {
            # Parameter information
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'synaptic_mode': self.synaptic_mode,

            # Chaos measures - arrays and statistics
            'lz_complexities': np.array(lz_complexities),
            'hamming_slopes': np.array(hamming_slopes),
            'lz_mean': np.mean(lz_complexities),
            'lz_std': np.std(lz_complexities),
            'hamming_mean': np.mean(hamming_slopes),
            'hamming_std': np.std(hamming_slopes),

            # Enhanced measures
            'total_spike_differences': np.array(total_spike_diffs),
            'spike_diff_mean': np.mean(total_spike_diffs),
            'spike_diff_std': np.std(total_spike_diffs),

            'intrinsic_dimensionalities': np.array(intrinsic_dims),
            'effective_dimensionalities': np.array(effective_dims),
            'participation_ratios': np.array(participation_ratios),
            'total_variances': np.array(total_variances),
            'intrinsic_dim_mean': np.mean(intrinsic_dims),
            'intrinsic_dim_std': np.std(intrinsic_dims),
            'effective_dim_mean': np.mean(effective_dims),
            'effective_dim_std': np.std(effective_dims),
            'participation_ratio_mean': np.mean(participation_ratios),
            'participation_ratio_std': np.std(participation_ratios),
            'total_variance_mean': np.mean(total_variances),
            'total_variance_std': np.std(total_variances),

            'gamma_coincidences': np.array(gamma_coincidences),
            'gamma_coincidence_mean': np.mean(gamma_coincidences),
            'gamma_coincidence_std': np.std(gamma_coincidences),

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'perturbation_neurons': [r['perturbation_neuron'] for r in trial_results]
        }

        return results

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, v_th_distributions: List[str],
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full experiment for single session."""
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

        results = []

        total_combinations = len(v_th_stds) * len(g_stds) * len(v_th_distributions) * len(static_input_rates)
        print(f"Starting single-session experiment: {total_combinations} parameter combinations")
        print(f"  Session ID: {session_id}")
        print(f"  v_th_stds: {len(v_th_stds)} (range: {np.min(v_th_stds):.3f}-{np.max(v_th_stds):.3f})")
        print(f"  g_stds: {len(g_stds)} (range: {np.min(g_stds):.3f}-{np.max(g_stds):.3f})")
        print(f"  v_th_distributions: {v_th_distributions}")
        print(f"  Synaptic mode: {self.synaptic_mode}")
        print(f"  Trials per combination: {self.n_perturbation_trials}")

        combo_idx = 0
        for input_rate in static_input_rates:
            for v_th_dist in v_th_distributions:
                for i, v_th_std in enumerate(v_th_stds):
                    for j, g_std in enumerate(g_stds):
                        combo_idx += 1

                        print(f"[{combo_idx}/{total_combinations}] Processing: "
                              f"v_th_std={v_th_std:.3f}, g_std={g_std:.3f}, "
                              f"dist={v_th_dist}, rate={input_rate:.0f}Hz")

                        result = self.run_parameter_combination(
                            session_id=session_id,
                            v_th_std=v_th_std,
                            g_std=g_std,
                            v_th_distribution=v_th_dist,
                            static_input_rate=input_rate
                        )

                        result['combination_index'] = combo_idx
                        results.append(result)

                        print(f"  LZ: {result['lz_mean']:.2f}±{result['lz_std']:.2f}")
                        print(f"  Hamming: {result['hamming_mean']:.4f}±{result['hamming_std']:.4f}")
                        print(f"  Time: {result['computation_time']:.1f}s")

        print(f"Single-session experiment completed: {len(results)} combinations processed")
        return results


def create_parameter_grid(n_v_th_points: int = 10, n_g_points: int = 10,
                         v_th_std_range: Tuple[float, float] = (0.0, 4.0),
                         g_std_range: Tuple[float, float] = (0.0, 4.0),
                         input_rate_range: Tuple[float, float] = (50.0, 500.0),
                         n_input_rates: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create parameter grids with direct heterogeneity values."""
    v_th_stds = np.linspace(v_th_std_range[0], v_th_std_range[1], n_v_th_points)
    g_stds = np.linspace(g_std_range[0], g_std_range[1], n_g_points)
    static_input_rates = np.linspace(input_rate_range[0], input_rate_range[1], n_input_rates)

    return v_th_stds, g_stds, static_input_rates


def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    """Save experimental results."""
    if not os.path.isabs(filename):
        if use_data_subdir:
            results_dir = os.path.join(os.getcwd(), "results", "data")
        else:
            results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        full_path = os.path.join(results_dir, filename)
    else:
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        full_path = filename

    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

    file_size = os.path.getsize(full_path) / (1024 * 1024)
    print(f"Single-session results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load experimental results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Single-session results loaded: {len(results)} combinations from {filename}")
    return results

def average_across_sessions(results_files: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple single-session result files and average them.

    Args:
        results_files: List of pickle file paths from different sessions

    Returns:
        List of averaged results across sessions
    """
    print(f"Averaging results across {len(results_files)} sessions...")

    # Load all session results
    all_session_results = []
    for file_path in results_files:
        session_results = load_results(file_path)
        all_session_results.append(session_results)
        print(f"  Loaded session with {len(session_results)} combinations")

    # Check that all sessions have same parameter combinations
    n_combinations = len(all_session_results[0])
    for i, session_results in enumerate(all_session_results[1:], 1):
        if len(session_results) != n_combinations:
            raise ValueError(f"Session {i+1} has {len(session_results)} combinations, expected {n_combinations}")

    # Average across sessions for each parameter combination
    averaged_results = []

    for combo_idx in range(n_combinations):
        # Get this combination from all sessions
        combo_results = [session_results[combo_idx] for session_results in all_session_results]

        # Extract parameter info from first session (should be identical across sessions)
        first_result = combo_results[0]

        # Extract all metrics from all sessions for this combination
        all_lz = np.concatenate([r['lz_complexities'] for r in combo_results])
        all_hamming = np.concatenate([r['hamming_slopes'] for r in combo_results])
        all_spike_diffs = np.concatenate([r['total_spike_differences'] for r in combo_results])
        all_intrinsic_dims = np.concatenate([r['intrinsic_dimensionalities'] for r in combo_results])
        all_effective_dims = np.concatenate([r['effective_dimensionalities'] for r in combo_results])
        all_participation_ratios = np.concatenate([r['participation_ratios'] for r in combo_results])
        all_total_variances = np.concatenate([r['total_variances'] for r in combo_results])
        all_gamma_coincidences = np.concatenate([r['gamma_coincidences'] for r in combo_results])

        # Create averaged result
        averaged_result = {
            # Parameter information (from first session)
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'synaptic_mode': first_result['synaptic_mode'],
            'combination_index': first_result['combination_index'],

            # Session-averaged chaos measures
            'lz_mean': np.mean(all_lz),
            'lz_std': np.std(all_lz),
            'hamming_mean': np.mean(all_hamming),
            'hamming_std': np.std(all_hamming),

            'spike_diff_mean': np.mean(all_spike_diffs),
            'spike_diff_std': np.std(all_spike_diffs),

            'intrinsic_dim_mean': np.mean(all_intrinsic_dims),
            'intrinsic_dim_std': np.std(all_intrinsic_dims),
            'effective_dim_mean': np.mean(all_effective_dims),
            'effective_dim_std': np.std(all_effective_dims),
            'participation_ratio_mean': np.mean(all_participation_ratios),
            'participation_ratio_std': np.std(all_participation_ratios),
            'total_variance_mean': np.mean(all_total_variances),
            'total_variance_std': np.std(all_total_variances),

            'gamma_coincidence_mean': np.mean(all_gamma_coincidences),
            'gamma_coincidence_std': np.std(all_gamma_coincidences),

            # Metadata
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': len(all_lz),
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results]
        }

        averaged_results.append(averaged_result)

        if (combo_idx + 1) % 10 == 0:
            print(f"  Averaged {combo_idx + 1}/{n_combinations} combinations")

    print(f"Session averaging completed: {len(averaged_results)} combinations averaged")
    return averaged_results
