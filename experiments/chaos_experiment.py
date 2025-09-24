# experiments/chaos_experiment.py - Complete implementation with enhanced measures
"""
Chaos analysis experiment with random network structure per parameter combination and comprehensive analysis.
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
    """Enhanced chaos experiment with comprehensive analysis measures."""

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
        rng = get_rng(session_id, v_th_std, g_std, 0, 'perturbation_targets')
        sample_size = min(100, self.n_neurons)
        return rng.choice(self.n_neurons, size=sample_size, replace=False)

    def run_single_perturbation(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                              v_th_distribution: str, perturbation_neuron_idx: int,
                              static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single perturbation with random structure per parameter combination."""

        # Create identical networks with random structure
        network_control = SpikingRNN(self.n_neurons, dt=self.dt, synaptic_mode=self.synaptic_mode)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt, synaptic_mode=self.synaptic_mode)

        # Network parameters
        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 1.0,
            'dynamic_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        # Initialize both networks with identical structure
        for network in [network_control, network_perturbed]:
            network.initialize_network(session_id, v_th_std, g_std, **network_params)

        # Get perturbation neuron
        perturbation_neurons = self.get_perturbation_neurons(session_id, v_th_std, g_std)
        available_neurons = len(perturbation_neurons)
        safe_idx = perturbation_neuron_idx % available_neurons
        perturbation_neuron = int(perturbation_neurons[safe_idx])

        # Run control simulation
        spikes_control = network_control.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate
        )

        # Run perturbed simulation
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

        # Enhanced analysis with all new measures
        analysis_results = analyze_perturbation_response_enhanced(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=self.perturbation_time,
            simulation_end=self.total_duration,
            perturbed_neuron=perturbation_neuron
        )

        # Add perturbation metadata
        analysis_results['perturbation_neuron'] = perturbation_neuron
        analysis_results['perturbation_neuron_idx'] = perturbation_neuron_idx

        return analysis_results

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run parameter combination with comprehensive statistics."""
        start_time = time.time()

        # Store all trial results
        trial_results = []
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

        # Extract arrays for all measures
        arrays = self._extract_trial_arrays(trial_results)

        # Compute all statistics
        stats = self._compute_all_statistics(arrays)

        # Additional computed statistics
        additional_stats = self._compute_additional_statistics(trial_results)

        # Compile complete results
        results = {
            # Parameter information
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'synaptic_mode': self.synaptic_mode,

            # Raw arrays (for session averaging)
            **arrays,

            # Summary statistics
            **stats,

            # Additional analysis
            **additional_stats,

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'perturbation_neurons': [r['perturbation_neuron'] for r in trial_results]
        }

        return results

    def _extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from all trial results."""
        arrays = {}

        # Basic chaos measures
        arrays['lz_matrix_flattened_values'] = np.array([r['lz_matrix_flattened'] for r in trial_results])
        arrays['hamming_slopes'] = np.array([r['hamming_slope'] for r in trial_results])
        arrays['total_spike_differences'] = np.array([r['total_spike_differences'] for r in trial_results])

        # Enhanced LZ and PCI measures
        arrays['lz_spatial_patterns_values'] = np.array([r['lz_spatial_patterns'] for r in trial_results])
        arrays['pci_raw_values'] = np.array([r['pci_raw'] for r in trial_results])
        arrays['pci_normalized_values'] = np.array([r['pci_normalized'] for r in trial_results])
        arrays['pci_with_threshold_values'] = np.array([r['pci_with_threshold'] for r in trial_results])

        # Coincidence measures
        arrays['kistler_delta_2ms_values'] = np.array([r['kistler_delta_2ms'] for r in trial_results])
        arrays['kistler_delta_5ms_values'] = np.array([r['kistler_delta_5ms'] for r in trial_results])
        arrays['gamma_window_5ms_values'] = np.array([r['gamma_window_5ms'] for r in trial_results])
        arrays['gamma_window_10ms_values'] = np.array([r['gamma_window_10ms'] for r in trial_results])

        # Dimensionality measures (multiple bin sizes)
        bin_sizes = ['bin_2ms', 'bin_5ms', 'bin_20ms']
        metrics = ['intrinsic_dimensionality', 'effective_dimensionality', 'participation_ratio', 'total_variance']

        for bin_size in bin_sizes:
            for metric in metrics:
                key = f'{metric}_{bin_size}_values'
                values = []
                for result in trial_results:
                    if 'dimensionality_metrics' in result and bin_size in result['dimensionality_metrics']:
                        values.append(result['dimensionality_metrics'][bin_size][metric])
                    else:
                        values.append(0.0)  # Default value
                arrays[key] = np.array(values)

        # Firing rate statistics
        firing_metrics = [
            'control_mean_firing_rate', 'control_std_firing_rate', 'control_percent_silent', 'control_percent_active',
            'perturbed_mean_firing_rate', 'perturbed_std_firing_rate', 'perturbed_percent_silent', 'perturbed_percent_active'
        ]

        for metric in firing_metrics:
            control_key = metric.replace('control_', '')
            perturbed_key = metric.replace('perturbed_', '')

            values = []
            for result in trial_results:
                if 'control_firing_stats' in result and control_key in result['control_firing_stats']:
                    values.append(result['control_firing_stats'][control_key])
                elif 'perturbed_firing_stats' in result and perturbed_key in result['perturbed_firing_stats']:
                    values.append(result['perturbed_firing_stats'][perturbed_key])
                else:
                    values.append(0.0)
            arrays[f'{metric}_values'] = np.array(values)

        return arrays

    def _compute_all_statistics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute mean and std for all arrays."""
        stats = {}

        for key, array in arrays.items():
            if key.endswith('_values'):
                base_name = key[:-7]  # Remove '_values' suffix
                stats[f'{base_name}_mean'] = float(np.mean(array))
                stats[f'{base_name}_std'] = float(np.std(array))

        return stats

    def _compute_additional_statistics(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Compute additional statistics like stability and spatial patterns."""
        additional = {}

        # Pattern stability statistics
        stable_periods = [r.get('stable_period') for r in trial_results]
        stable_count = sum(1 for sp in stable_periods if sp is not None)
        additional['stable_pattern_fraction'] = stable_count / len(stable_periods)
        additional['stable_pattern_count'] = stable_count

        if stable_count > 0:
            stable_ones = [sp for sp in stable_periods if sp is not None]
            periods = [sp['period'] for sp in stable_ones]
            repeats = [sp['repeats'] for sp in stable_ones]
            onset_times = [sp['onset_time'] for sp in stable_ones]

            additional.update({
                'stable_period_mean': float(np.mean(periods)),
                'stable_period_std': float(np.std(periods)),
                'stable_repeats_mean': float(np.mean(repeats)),
                'stable_repeats_std': float(np.std(repeats)),
                'stable_onset_time_mean': float(np.mean(onset_times)),
                'stable_onset_time_std': float(np.std(onset_times))
            })
        else:
            additional.update({
                'stable_period_mean': 0.0,
                'stable_period_std': 0.0,
                'stable_repeats_mean': 0.0,
                'stable_repeats_std': 0.0,
                'stable_onset_time_mean': 0.0,
                'stable_onset_time_std': 0.0
            })

        # Spatial pattern statistics
        spatial_entropies = [r.get('spatial_entropy', 0.0) for r in trial_results]
        pattern_fractions = [r.get('pattern_fraction', 0.0) for r in trial_results]
        unique_patterns = [r.get('unique_patterns', 0) for r in trial_results]

        additional.update({
            'spatial_entropy_mean': float(np.mean(spatial_entropies)),
            'spatial_entropy_std': float(np.std(spatial_entropies)),
            'pattern_fraction_mean': float(np.mean(pattern_fractions)),
            'pattern_fraction_std': float(np.std(pattern_fractions)),
            'unique_patterns_mean': float(np.mean(unique_patterns)),
            'unique_patterns_std': float(np.std(unique_patterns))
        })

        return additional

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, v_th_distributions: List[str],
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full experiment for single session with extended analysis."""
        if static_input_rates is None:
            # Extended rate range based on synaptic mode
            if self.synaptic_mode == "dynamic":
                static_input_rates = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
            else:
                static_input_rates = np.array([50.0, 100.0, 200.0, 500.0])

        results = []
        total_combinations = len(v_th_stds) * len(g_stds) * len(v_th_distributions) * len(static_input_rates)

        print(f"Starting enhanced chaos experiment: {total_combinations} parameter combinations")
        print(f"  Session ID: {session_id}")
        print(f"  v_th_stds: {len(v_th_stds)} (range: {np.min(v_th_stds):.3f}-{np.max(v_th_stds):.3f})")
        print(f"  g_stds: {len(g_stds)} (range: {np.min(g_stds):.3f}-{np.max(g_stds):.3f})")
        print(f"  v_th_distributions: {v_th_distributions}")
        print(f"  Static rates: {static_input_rates}")
        print(f"  Synaptic mode: {self.synaptic_mode}")
        print(f"  Trials per combination: {self.n_perturbation_trials}")

        combo_idx = 0
        for input_rate in static_input_rates:
            for v_th_dist in v_th_distributions:
                for v_th_std in v_th_stds:
                    for g_std in g_stds:
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

                        # Enhanced progress reporting
                        print(f"  LZ (flattened): {result['lz_matrix_flattened_mean']:.2f}±{result['lz_matrix_flattened_std']:.2f}")
                        print(f"  LZ (spatial): {result['lz_spatial_patterns_mean']:.2f}±{result['lz_spatial_patterns_std']:.2f}")
                        print(f"  PCI (normalized): {result['pci_normalized_mean']:.3f}±{result['pci_normalized_std']:.3f}")
                        print(f"  Kistler (2ms): {result['kistler_delta_2ms_mean']:.3f}±{result['kistler_delta_2ms_std']:.3f}")
                        print(f"  Silent neurons: {result['control_percent_silent_mean']:.1f}%")
                        print(f"  Stable patterns: {result['stable_pattern_fraction']:.2f}")
                        print(f"  Time: {result['computation_time']:.1f}s")

        print(f"Enhanced experiment completed: {len(results)} combinations processed")
        return results


def create_parameter_grid(n_v_th_points: int = 10, n_g_points: int = 10,
                         v_th_std_range: Tuple[float, float] = (0.0, 4.0),
                         g_std_range: Tuple[float, float] = (0.0, 4.0),
                         input_rate_range: Tuple[float, float] = (50.0, 1000.0),
                         n_input_rates: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create parameter grids with extended input rate range."""
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
    print(f"Enhanced results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load experimental results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Enhanced results loaded: {len(results)} combinations from {filename}")
    return results

def average_across_sessions(results_files: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple single-session result files and average them across sessions.

    Args:
        results_files: List of pickle file paths from different sessions

    Returns:
        List of averaged results across sessions
    """
    print(f"Averaging enhanced results across {len(results_files)} sessions...")

    # Load all session results
    all_session_results = []
    for file_path in results_files:
        session_results = load_results(file_path)
        all_session_results.append(session_results)
        print(f"  Loaded session with {len(session_results)} combinations")

    # Verify consistency
    n_combinations = len(all_session_results[0])
    for i, session_results in enumerate(all_session_results[1:], 1):
        if len(session_results) != n_combinations:
            raise ValueError(f"Session {i+1} has {len(session_results)} combinations, expected {n_combinations}")

    # Average across sessions
    averaged_results = []

    for combo_idx in range(n_combinations):
        combo_results = [session_results[combo_idx] for session_results in all_session_results]
        first_result = combo_results[0]

        # Extract and concatenate all arrays across sessions
        concatenated_arrays = {}
        array_keys = [k for k in first_result.keys() if k.endswith('_values')]

        for key in array_keys:
            all_values = np.concatenate([r[key] for r in combo_results if key in r])
            concatenated_arrays[key] = all_values

        # Create averaged result
        averaged_result = {
            # Parameter information
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'synaptic_mode': first_result['synaptic_mode'],
            'combination_index': first_result['combination_index'],

            # Session-averaged statistics
            **{key.replace('_values', '_mean'): float(np.mean(array))
               for key, array in concatenated_arrays.items()},
            **{key.replace('_values', '_std'): float(np.std(array))
               for key, array in concatenated_arrays.items()},

            # Stability and pattern statistics
            'stable_pattern_fraction': np.mean([r['stable_pattern_fraction'] for r in combo_results]),
            'stable_pattern_count': np.sum([r['stable_pattern_count'] for r in combo_results]),
            'spatial_entropy_mean': np.mean([r['spatial_entropy_mean'] for r in combo_results]),
            'pattern_fraction_mean': np.mean([r['pattern_fraction_mean'] for r in combo_results]),
            'unique_patterns_mean': np.mean([r['unique_patterns_mean'] for r in combo_results]),

            # Metadata
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': sum(len(concatenated_arrays[list(concatenated_arrays.keys())[0]])),
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results]
        }

        averaged_results.append(averaged_result)

        if (combo_idx + 1) % 10 == 0:
            print(f"  Averaged {combo_idx + 1}/{n_combinations} combinations")

    print(f"Session averaging completed: {len(averaged_results)} combinations averaged")
    return averaged_results
