# experiments/spontaneous_experiment.py - Refactored with base class
"""
Spontaneous activity analysis: firing rates, dimensionality, silent neurons.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Any

# Import base class and utilities
from .base_experiment import BaseExperiment

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.spontaneous_analysis import analyze_spontaneous_activity
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    for subdir in ['src', 'analysis']:
        sys.path.insert(0, os.path.join(project_root, subdir))
    from src.spiking_network import SpikingRNN  # FIXED: was 'from spiking_network'
    from src.rng_utils import get_rng  # FIXED: was 'from rng_utils'
    from analysis.spontaneous_analysis import analyze_spontaneous_activity


class SpontaneousExperiment(BaseExperiment):
    """Spontaneous activity analysis experiment."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 synaptic_mode: str = "filter", static_input_mode: str = "independent"):
        super().__init__(n_neurons, dt)

        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.n_trials = 10

    def run_single_trial(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                        v_th_distribution: str, duration: float,
                        static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single spontaneous activity trial."""
        network = SpikingRNN(self.n_neurons, dt=self.dt,
                            synaptic_mode=self.synaptic_mode,
                            static_input_mode=self.static_input_mode)

        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 10.0,
            'readout_weight_scale': 1.0
        }

        network.initialize_network(session_id, v_th_std, g_std, **network_params)

        spikes = network.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=duration,
            static_input_rate=static_input_rate
        )

        # Analyze with UPDATED 200ms transient time
        analysis_results = analyze_spontaneous_activity(
            spikes=spikes,
            num_neurons=self.n_neurons,
            duration=duration,
            transient_time=200.0  # UPDATED
        )

        return analysis_results

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0,
                                duration: float = 5000.0) -> Dict[str, Any]:
        """Run parameter combination for spontaneous activity analysis."""
        start_time = time.time()

        trial_results = []
        for trial_idx in range(self.n_trials):
            trial_result = self.run_single_trial(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_idx + 1,
                v_th_distribution=v_th_distribution,
                duration=duration,
                static_input_rate=static_input_rate
            )
            trial_results.append(trial_result)

        # Extract arrays and compute statistics using base class
        arrays = self.extract_trial_arrays(trial_results)
        stats = self.compute_all_statistics(arrays)

        # Compile results
        results = {
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'duration': duration,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            **arrays,
            **stats,
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time
        }

        return results

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results (required by base class)."""
        arrays = {}

        # Firing rate statistics
        firing_metrics = [
            'mean_firing_rate', 'std_firing_rate', 'min_firing_rate', 'max_firing_rate',
            'silent_neurons', 'active_neurons', 'percent_silent', 'percent_active'
        ]

        for metric in firing_metrics:
            values = [r['firing_stats'][metric] for r in trial_results]
            arrays[f'{metric}_values'] = np.array(values)

        # Dimensionality measures
        bin_sizes = ['bin_0.1ms', 'bin_2.0ms', 'bin_5.0ms', 'bin_20.0ms', 'bin_50.0ms', 'bin_100.0ms']
        dim_metrics = ['intrinsic_dimensionality', 'effective_dimensionality', 'participation_ratio', 'total_variance']

        for bin_size in bin_sizes:
            for metric in dim_metrics:
                values = [r['dimensionality_metrics'][bin_size][metric] if bin_size in r['dimensionality_metrics'] else 0.0
                         for r in trial_results]
                arrays[f'{metric}_{bin_size}_values'] = np.array(values)

        # Poisson metrics
        poisson_metrics = [
            'mean_cv_isi', 'std_cv_isi', 'mean_fano_factor', 'std_fano_factor',
            'poisson_isi_fraction', 'poisson_count_fraction'
        ]

        for metric in poisson_metrics:
            values = [r['poisson_analysis']['population_statistics'][metric]
                     if 'poisson_analysis' in r and 'population_statistics' in r['poisson_analysis']
                     else np.nan
                     for r in trial_results]
            arrays[f'{metric}_values'] = np.array(values)

        # Basic trial info
        arrays['total_spikes_values'] = np.array([r['total_spikes'] for r in trial_results])
        arrays['steady_state_spikes_values'] = np.array([r['steady_state_spikes'] for r in trial_results])

        return arrays

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, v_th_distribution: str = "normal",
                          static_input_rates: np.ndarray = None,
                          duration: float = 5000.0) -> List[Dict[str, Any]]:
        """Run full spontaneous activity experiment."""
        if static_input_rates is None:
            static_input_rates = np.array([50.0, 100.0, 200.0, 500.0, 1000.0]) if self.synaptic_mode == "filter" \
                                 else np.array([50.0, 100.0, 200.0, 500.0])

        # Use base class method for parameter combinations
        all_combinations = self.create_parameter_combinations(
            session_id=session_id,
            v_th_stds=v_th_stds,
            g_stds=g_stds,
            static_input_rates=static_input_rates,
            v_th_distribution=v_th_distribution,
            duration=duration
        )

        print(f"Starting spontaneous activity experiment: {len(all_combinations)} combinations")

        results = []
        for i, combo in enumerate(all_combinations):
            print(f"[{i+1}/{len(all_combinations)}]: v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}")

            result = self.run_parameter_combination(
                session_id=combo['session_id'],
                v_th_std=combo['v_th_std'],
                g_std=combo['g_std'],
                v_th_distribution=combo['v_th_distribution'],
                static_input_rate=combo['static_input_rate'],
                duration=combo.get('duration', duration)
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

        print(f"Spontaneous activity experiment completed")
        return results


# Keep existing save/load/average functions for compatibility
def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    if not os.path.isabs(filename):
        if use_data_subdir:
            results_dir = os.path.join(os.getcwd(), "results", "data")
            full_path = os.path.join(results_dir, filename)
        else:
            full_path = os.path.join(os.getcwd(), filename)
    else:
        full_path = filename

    directory = os.path.dirname(full_path)
    os.makedirs(directory, exist_ok=True)

    with open(full_path, 'wb') as f:
        pickle.dump(results, f)


def load_results(filename: str) -> List[Dict[str, Any]]:
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Spontaneous activity results loaded: {len(results)} combinations from {filename}")
    return results


def average_across_sessions(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average spontaneous activity results across sessions."""
    print(f"Averaging spontaneous activity results across {len(results_files)} sessions...")

    all_session_results = [load_results(f) for f in results_files]
    n_combinations = len(all_session_results[0])

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

        # Create averaged result with statistics
        averaged_result = {
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'duration': first_result['duration'],
            'synaptic_mode': first_result['synaptic_mode'],
            'static_input_mode': first_result['static_input_mode'],
            'original_combination_index': first_result.get('original_combination_index', combo_idx),
            **{key.replace('_values', '_mean'): BaseExperiment.compute_safe_mean(array)
               for key, array in concatenated_arrays.items()},
            **{key.replace('_values', '_std'): BaseExperiment.compute_safe_std(array)
               for key, array in concatenated_arrays.items()},
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': len(concatenated_arrays[list(concatenated_arrays.keys())[0]]) if concatenated_arrays else 0,
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results]
        }

        averaged_results.append(averaged_result)

    print(f"Session averaging completed: {len(averaged_results)} combinations averaged")
    return averaged_results
