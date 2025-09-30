# experiments/stability_experiment.py - Network stability analysis with updated measures
"""
Network stability experiment with full-simulation LZ analysis and settling time.
"""

import numpy as np
import os
import sys
import time
import pickle
import random
from typing import Dict, List, Tuple, Any

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.stability_analysis import analyze_perturbation_response
except ImportError:
    try:
        from ..src.spiking_network import SpikingRNN
        from ..src.rng_utils import get_rng
        from ..analysis.stability_analysis import analyze_perturbation_response
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        src_dir = os.path.join(project_root, 'src')
        analysis_dir = os.path.join(project_root, 'analysis')
        sys.path.insert(0, src_dir)
        sys.path.insert(0, analysis_dir)
        from spiking_network import SpikingRNN
        from rng_utils import get_rng
        from stability_analysis import analyze_perturbation_response

class StabilityExperiment:
    """Network stability experiment with perturbation analysis."""

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
            'static_input_strength': 10.0,
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

        # Stability analysis with updated measures
        analysis_results = analyze_perturbation_response(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=self.perturbation_time,
            simulation_end=self.total_duration,
            perturbed_neuron=perturbation_neuron,
            dt=self.dt  # Pass dt parameter
        )

        # Add perturbation metadata
        analysis_results['perturbation_neuron'] = perturbation_neuron
        analysis_results['perturbation_neuron_idx'] = perturbation_neuron_idx

        return analysis_results

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run parameter combination with stability analysis."""
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
        """Extract arrays from trial results (updated measures)."""
        arrays = {}

        # LZ complexity
        arrays['lz_spatial_patterns_values'] = np.array([r['lz_spatial_patterns'] for r in trial_results])

        # Shannon entropies
        arrays['shannon_entropy_symbols_values'] = np.array([r['shannon_entropy_symbols'] for r in trial_results])
        arrays['shannon_entropy_spikes_values'] = np.array([r['shannon_entropy_spikes'] for r in trial_results])

        # Pattern statistics
        arrays['unique_patterns_count_values'] = np.array([r['unique_patterns_count'] for r in trial_results])
        arrays['post_pert_symbol_sum_values'] = np.array([r['post_pert_symbol_sum'] for r in trial_results])
        arrays['total_spike_differences_values'] = np.array([r['total_spike_differences'] for r in trial_results])

        # Settling time
        arrays['settling_time_ms_values'] = np.array([r['settling_time_ms'] for r in trial_results])

        # Coincidence measures (unchanged)
        arrays['kistler_delta_2ms_values'] = np.array([r['kistler_delta_2ms'] for r in trial_results])
        arrays['kistler_delta_5ms_values'] = np.array([r['kistler_delta_5ms'] for r in trial_results])
        arrays['gamma_window_2ms_values'] = np.array([r['gamma_window_2ms'] for r in trial_results])
        arrays['gamma_window_5ms_values'] = np.array([r['gamma_window_5ms'] for r in trial_results])

        return arrays

    def _compute_all_statistics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute mean and std for all arrays."""
        stats = {}

        for key, array in arrays.items():
            if key.endswith('_values'):
                base_name = key[:-7]  # Remove '_values' suffix

                # Use nanmean and nanstd to handle NaN values properly
                # This is especially important for settling_time_ms which may have NaN
                stats[f'{base_name}_mean'] = float(np.nanmean(array))
                stats[f'{base_name}_std'] = float(np.nanstd(array))

        return stats

    def _compute_additional_statistics(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Compute additional statistics for stability."""
        additional = {}

        # Settling time statistics
        settling_times = [r.get('settling_time_ms', np.nan) for r in trial_results]
        settled_count = sum(1 for st in settling_times if not np.isnan(st))

        additional['settled_fraction'] = settled_count / len(settling_times) if settling_times else 0.0
        additional['settled_count'] = settled_count

        # Only compute median here - mean and std already computed in _compute_all_statistics
        # as settling_time_ms_mean and settling_time_ms_std
        if settled_count > 0:
            valid_times = [st for st in settling_times if not np.isnan(st)]
            additional['settling_time_median'] = float(np.median(valid_times))
        else:
            additional['settling_time_median'] = np.nan

        return additional

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, v_th_distributions: List[str],
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full stability experiment for single session with randomized job distribution."""
        if static_input_rates is None:
            if self.synaptic_mode == "dynamic":
                static_input_rates = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
            else:
                static_input_rates = np.array([50.0, 100.0, 200.0, 500.0])

        # Create all parameter combinations
        all_combinations = []
        combo_idx = 0
        for input_rate in static_input_rates:
            for v_th_dist in v_th_distributions:
                for v_th_std in v_th_stds:
                    for g_std in g_stds:
                        all_combinations.append({
                            'combo_idx': combo_idx,
                            'session_id': session_id,
                            'v_th_std': v_th_std,
                            'g_std': g_std,
                            'v_th_distribution': v_th_dist,
                            'static_input_rate': input_rate
                        })
                        combo_idx += 1

        # RANDOMIZE job order for better CPU load balancing
        random.shuffle(all_combinations)

        total_combinations = len(all_combinations)

        print(f"Starting stability experiment with randomized job distribution: {total_combinations} combinations")
        print(f"  Session ID: {session_id}")
        print(f"  v_th_stds: {len(v_th_stds)} (range: {np.min(v_th_stds):.3f}-{np.max(v_th_stds):.3f})")
        print(f"  g_stds: {len(g_stds)} (range: {np.min(g_stds):.3f}-{np.max(g_stds):.3f})")
        print(f"  v_th_distributions: {v_th_distributions}")
        print(f"  Static rates: {static_input_rates}")
        print(f"  Synaptic mode: {self.synaptic_mode}")
        print(f"  Job order: RANDOMIZED for load balancing")

        results = []
        for i, combo in enumerate(all_combinations):
            print(f"[{i+1}/{total_combinations}] Processing randomized job:")
            print(f"    v_th_std={combo['v_th_std']:.3f}, g_std={combo['g_std']:.3f}")
            print(f"    dist={combo['v_th_distribution']}, rate={combo['static_input_rate']:.0f}Hz")

            result = self.run_parameter_combination(
                session_id=combo['session_id'],
                v_th_std=combo['v_th_std'],
                g_std=combo['g_std'],
                v_th_distribution=combo['v_th_distribution'],
                static_input_rate=combo['static_input_rate']
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

            # Progress reporting for updated measures
            print(f"  LZ (spatial): {result['lz_spatial_patterns_mean']:.2f}±{result['lz_spatial_patterns_std']:.2f}")
            print(f"  Shannon (symbols): {result['shannon_entropy_symbols_mean']:.3f}±{result['shannon_entropy_symbols_std']:.3f}")
            print(f"  Unique patterns: {result['unique_patterns_count_mean']:.1f}±{result['unique_patterns_count_std']:.1f}")
            print(f"  Settling time: {result['settling_time_ms_mean']:.1f}±{result['settling_time_ms_std']:.1f} ms")
            print(f"  Settled fraction: {result['settled_fraction']:.2f}")
            print(f"  Kistler (2ms): {result['kistler_delta_2ms_mean']:.3f}±{result['kistler_delta_2ms_std']:.3f}")
            print(f"  Time: {result['computation_time']:.1f}s")

        # Sort results back by original combination index for consistency
        results.sort(key=lambda x: x['original_combination_index'])

        print(f"Stability experiment completed: {len(results)} combinations processed")
        print("Jobs were processed in randomized order for optimal CPU load balancing")
        return results


def create_parameter_grid(n_v_th_points: int = 10, n_g_points: int = 10,
                         v_th_std_range: Tuple[float, float] = (0.0, 4.0),
                         g_std_range: Tuple[float, float] = (0.0, 4.0),
                         input_rate_range: Tuple[float, float] = (50.0, 1000.0),
                         n_input_rates: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create parameter grids."""
    v_th_stds = np.linspace(v_th_std_range[0], v_th_std_range[1], n_v_th_points)
    g_stds = np.linspace(g_std_range[0], g_std_range[1], n_g_points)
    static_input_rates = np.geomspace(input_rate_range[0], input_rate_range[1], n_input_rates)

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
    print(f"Stability results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load experimental results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Stability results loaded: {len(results)} combinations from {filename}")
    return results

def average_across_sessions(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average stability results across sessions."""
    print(f"Averaging stability results across {len(results_files)} sessions...")

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

        # Extract and concatenate arrays across sessions
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
            'original_combination_index': first_result.get('original_combination_index', combo_idx),

            # Session-averaged statistics
            **{key.replace('_values', '_mean'): float(np.mean(array))
               for key, array in concatenated_arrays.items()},
            **{key.replace('_values', '_std'): float(np.std(array))
               for key, array in concatenated_arrays.items()},

            # Settling statistics
            'settled_fraction': np.mean([r['settled_fraction'] for r in combo_results]),
            'settled_count': np.sum([r['settled_count'] for r in combo_results]),
            'settling_time_ms_mean': np.mean([r['settling_time_ms_mean'] for r in combo_results]),
            'settling_time_ms_std': np.std([r['settling_time_ms_mean'] for r in combo_results]),  # Std of means across sessions
            'settling_time_median': np.mean([r.get('settling_time_median', np.nan) for r in combo_results
                                            if not np.isnan(r.get('settling_time_median', np.nan))])
                                   if any(not np.isnan(r.get('settling_time_median', np.nan)) for r in combo_results)
                                   else np.nan,

            # Metadata
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': len(concatenated_arrays[list(concatenated_arrays.keys())[0]]) if concatenated_arrays else 0,
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results]
        }

        averaged_results.append(averaged_result)

        if (combo_idx + 1) % 10 == 0:
            print(f"  Averaged {combo_idx + 1}/{n_combinations} combinations")

    print(f"Session averaging completed: {len(averaged_results)} combinations averaged")
    return averaged_results
