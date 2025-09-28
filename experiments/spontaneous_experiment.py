# experiments/spontaneous_experiment.py - Spontaneous activity analysis experiment
"""
Spontaneous activity analysis: firing rates, dimensionality, silent neurons.
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
    from analysis.spontaneous_analysis import analyze_spontaneous_activity
except ImportError:
    try:
        from ..src.spiking_network import SpikingRNN
        from ..src.rng_utils import get_rng
        from ..analysis.spontaneous_analysis import analyze_spontaneous_activity
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        src_dir = os.path.join(project_root, 'src')
        analysis_dir = os.path.join(project_root, 'analysis')
        sys.path.insert(0, src_dir)
        sys.path.insert(0, analysis_dir)
        from spiking_network import SpikingRNN
        from rng_utils import get_rng
        from spontaneous_analysis import analyze_spontaneous_activity

class SpontaneousExperiment:
    """Spontaneous activity analysis experiment."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1, synaptic_mode: str = "dynamic"):
        self.n_neurons = n_neurons
        self.dt = dt
        self.synaptic_mode = synaptic_mode
        self.n_trials = 10  # Fewer trials since we're measuring spontaneous activity

    def run_single_trial(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                        v_th_distribution: str, duration: float,
                        static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single spontaneous activity trial."""

        # Create network with random structure
        network = SpikingRNN(self.n_neurons, dt=self.dt, synaptic_mode=self.synaptic_mode)

        # Network parameters
        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 10.0,  # Enhanced connectivity strength
            'dynamic_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        # Initialize network
        network.initialize_network(session_id, v_th_std, g_std, **network_params)

        # Run spontaneous activity simulation
        spikes = network.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=duration,
            static_input_rate=static_input_rate
        )

        # Analyze spontaneous activity
        analysis_results = analyze_spontaneous_activity(
            spikes=spikes,
            num_neurons=self.n_neurons,
            duration=duration
        )

        return analysis_results

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0,
                                duration: float = 5000.0) -> Dict[str, Any]:
        """Run parameter combination for spontaneous activity analysis."""
        start_time = time.time()

        # Store all trial results
        trial_results = []

        for trial_idx in range(self.n_trials):
            trial_id = trial_idx + 1

            trial_result = self.run_single_trial(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                v_th_distribution=v_th_distribution,
                duration=duration,
                static_input_rate=static_input_rate
            )

            trial_results.append(trial_result)

        # Extract arrays for all measures
        arrays = self._extract_trial_arrays(trial_results)

        # Compute all statistics
        stats = self._compute_all_statistics(arrays)

        # Compile results
        results = {
            # Parameter information
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'duration': duration,
            'synaptic_mode': self.synaptic_mode,

            # Raw arrays (for session averaging)
            **arrays,

            # Summary statistics
            **stats,

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time
        }

        return results

    def _extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results."""
        arrays = {}

        # Firing rate statistics
        firing_metrics = [
            'mean_firing_rate', 'std_firing_rate', 'min_firing_rate', 'max_firing_rate',
            'silent_neurons', 'active_neurons', 'percent_silent', 'percent_active'
        ]

        for metric in firing_metrics:
            values = [r['firing_stats'][metric] for r in trial_results]
            arrays[f'{metric}_values'] = np.array(values)

        # Dimensionality measures (multiple bin sizes)
        bin_sizes = ['bin_0.1ms', 'bin_2.0ms', 'bin_5.0ms', 'bin_20.0ms', 'bin_50.0ms', 'bin_100.0ms']
        dim_metrics = ['intrinsic_dimensionality', 'effective_dimensionality', 'participation_ratio', 'total_variance']

        for bin_size in bin_sizes:
            for metric in dim_metrics:
                values = []
                for result in trial_results:
                    if bin_size in result['dimensionality_metrics']:
                        values.append(result['dimensionality_metrics'][bin_size][metric])
                    else:
                        values.append(0.0)
                arrays[f'{metric}_{bin_size}_values'] = np.array(values)

        poisson_metrics = [
            'mean_cv_isi', 'std_cv_isi', 'mean_fano_factor', 'std_fano_factor',
            'poisson_isi_fraction', 'poisson_count_fraction'
        ]

        for metric in poisson_metrics:
            values = []
            for result in trial_results:
                if 'poisson_analysis' in result and 'population_statistics' in result['poisson_analysis']:
                    values.append(result['poisson_analysis']['population_statistics'][metric])
                else:
                    values.append(np.nan)
            arrays[f'{metric}_values'] = np.array(values)

        # Basic trial info
        arrays['total_spikes_values'] = np.array([r['total_spikes'] for r in trial_results])
        arrays['steady_state_spikes_values'] = np.array([r['steady_state_spikes'] for r in trial_results])

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

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, v_th_distributions: List[str],
                          static_input_rates: np.ndarray = None,
                          duration: float = 5000.0) -> List[Dict[str, Any]]:
        """Run full spontaneous activity experiment with randomized job distribution."""
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
                            'static_input_rate': input_rate,
                            'duration': duration
                        })
                        combo_idx += 1

        # RANDOMIZE job order for better CPU load balancing
        random.shuffle(all_combinations)

        total_combinations = len(all_combinations)

        print(f"Starting spontaneous activity experiment with randomized job distribution: {total_combinations} combinations")
        print(f"  Session ID: {session_id}")
        print(f"  Duration: {duration:.0f} ms")
        print(f"  v_th_stds: {len(v_th_stds)} (range: {np.min(v_th_stds):.3f}-{np.max(v_th_stds):.3f})")
        print(f"  g_stds: {len(g_stds)} (range: {np.min(g_stds):.3f}-{np.max(g_stds):.3f})")
        print(f"  v_th_distributions: {v_th_distributions}")
        print(f"  Static rates: {static_input_rates}")
        print(f"  Synaptic mode: {self.synaptic_mode}")
        print(f"  Bin sizes for dimensionality: 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms")
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
                static_input_rate=combo['static_input_rate'],
                duration=combo['duration']
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

            # Progress reporting
            print(f"  Mean firing rate: {result['mean_firing_rate_mean']:.2f}±{result['mean_firing_rate_std']:.2f} Hz")
            print(f"  Silent neurons: {result['percent_silent_mean']:.1f}±{result['percent_silent_std']:.1f}%")
            print(f"  Dimensionality (5ms): {result['effective_dimensionality_bin_5.0ms_mean']:.1f}±{result['effective_dimensionality_bin_5.0ms_std']:.1f}")
            print(f"  CV ISI: {result.get('mean_cv_isi_mean', 'N/A'):.2f}±{result.get('mean_cv_isi_std', 'N/A'):.2f}")
            print(f"  Poisson-like: {result.get('poisson_isi_fraction_mean', 'N/A'):.1%}")
            print(f"  Total spikes: {result['total_spikes_mean']:.0f}±{result['total_spikes_std']:.0f}")

        # Sort results back by original combination index for consistency
        results.sort(key=lambda x: x['original_combination_index'])

        print(f"Spontaneous activity experiment completed: {len(results)} combinations processed")
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
    print(f"Spontaneous activity results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load experimental results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Spontaneous activity results loaded: {len(results)} combinations from {filename}")
    return results

def average_across_sessions(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average spontaneous activity results across sessions."""
    print(f"Averaging spontaneous activity results across {len(results_files)} sessions...")

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
            'duration': first_result['duration'],
            'synaptic_mode': first_result['synaptic_mode'],
            'original_combination_index': first_result.get('original_combination_index', combo_idx),

            # Session-averaged statistics
            **{key.replace('_values', '_mean'): float(np.mean(array))
               for key, array in concatenated_arrays.items()},
            **{key.replace('_values', '_std'): float(np.std(array))
               for key, array in concatenated_arrays.items()},

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
