# experiments/stability_experiment.py - Refactored with base class
"""
Network stability experiment with full-simulation LZ analysis and settling time.
"""

import numpy as np
import os
import sys
import time
from typing import Dict, List, Any

# Import base class
from .base_experiment import BaseExperiment

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.stability_analysis import analyze_perturbation_response
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    for subdir in ['src', 'analysis']:
        sys.path.insert(0, os.path.join(project_root, subdir))
    from spiking_network import SpikingRNN
    from rng_utils import get_rng
    from stability_analysis import analyze_perturbation_response


class StabilityExperiment(BaseExperiment):
    """Network stability experiment with perturbation analysis."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 synaptic_mode: str = "filter", static_input_mode: str = "independent"):
        super().__init__(n_neurons, dt)

        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode

        # Timing parameters - perturbation_time IS the transient (200ms)
        self.pre_perturbation_time = 200.0  # UPDATED
        self.post_perturbation_time = 300.0
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time
        self.perturbation_time = self.pre_perturbation_time
        self.n_perturbation_trials = 100

    def get_perturbation_neurons(self, session_id: int, v_th_std: float, g_std: float) -> np.ndarray:
        """Get perturbation neurons for this parameter combination."""
        rng = get_rng(session_id, v_th_std, g_std, 0, 'perturbation_targets')
        sample_size = min(100, self.n_neurons)
        return rng.choice(self.n_neurons, size=sample_size, replace=False)

    def run_single_perturbation(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                              v_th_distribution: str, perturbation_neuron_idx: int,
                              static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single perturbation with random structure per parameter combination."""

        # Create identical networks
        network_control = SpikingRNN(self.n_neurons, dt=self.dt,
                                     synaptic_mode=self.synaptic_mode,
                                     static_input_mode=self.static_input_mode)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt,
                                       synaptic_mode=self.synaptic_mode,
                                       static_input_mode=self.static_input_mode)

        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 10.0,
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

        # Stability analysis
        analysis_results = analyze_perturbation_response(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=self.perturbation_time,
            simulation_end=self.total_duration,
            perturbed_neuron=perturbation_neuron,
            dt=self.dt
        )

        analysis_results['perturbation_neuron'] = perturbation_neuron
        analysis_results['perturbation_neuron_idx'] = perturbation_neuron_idx

        return analysis_results

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run parameter combination with stability analysis."""
        start_time = time.time()

        trial_results = []
        perturbation_neuron_indices = list(range(self.n_perturbation_trials))

        for trial_idx in range(self.n_perturbation_trials):
            trial_result = self.run_single_perturbation(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_idx + 1,
                v_th_distribution=v_th_distribution,
                perturbation_neuron_idx=perturbation_neuron_indices[trial_idx],
                static_input_rate=static_input_rate
            )
            trial_results.append(trial_result)

        # Extract arrays and compute statistics using base class
        arrays = self.extract_trial_arrays(trial_results)
        stats = self.compute_all_statistics(arrays)

        # Additional computed statistics
        additional_stats = self._compute_additional_statistics(trial_results)

        # Compile results
        results = {
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            **arrays,
            **stats,
            **additional_stats,
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'perturbation_neurons': [r['perturbation_neuron'] for r in trial_results]
        }

        return results

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results (required by base class)."""
        arrays = {}

        # LZ complexity measures
        arrays['lz_spatial_patterns_values'] = np.array([r['lz_spatial_patterns'] for r in trial_results])
        arrays['lz_column_wise_values'] = np.array([r['lz_column_wise'] for r in trial_results])

        # Shannon entropies
        arrays['shannon_entropy_symbols_values'] = np.array([r['shannon_entropy_symbols'] for r in trial_results])
        arrays['shannon_entropy_spikes_values'] = np.array([r['shannon_entropy_spikes'] for r in trial_results])

        # Pattern statistics
        arrays['unique_patterns_count_values'] = np.array([r['unique_patterns_count'] for r in trial_results])
        arrays['post_pert_symbol_sum_values'] = np.array([r['post_pert_symbol_sum'] for r in trial_results])
        arrays['total_spike_differences_values'] = np.array([r['total_spike_differences'] for r in trial_results])

        # Settling time
        arrays['settling_time_ms_values'] = np.array([r['settling_time_ms'] for r in trial_results])

        # Coincidence measures (includes 0.1ms)
        arrays['kistler_delta_0.1ms_values'] = np.array([r['kistler_delta_0.1ms'] for r in trial_results])
        arrays['kistler_delta_2.0ms_values'] = np.array([r['kistler_delta_2.0ms'] for r in trial_results])
        arrays['kistler_delta_5.0ms_values'] = np.array([r['kistler_delta_5.0ms'] for r in trial_results])
        arrays['gamma_window_0.1ms_values'] = np.array([r['gamma_window_0.1ms'] for r in trial_results])
        arrays['gamma_window_2.0ms_values'] = np.array([r['gamma_window_2.0ms'] for r in trial_results])
        arrays['gamma_window_5.0ms_values'] = np.array([r['gamma_window_5.0ms'] for r in trial_results])

        return arrays

    def _compute_additional_statistics(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Compute additional statistics for stability."""
        additional = {}

        # Settling time statistics
        settling_times = [r.get('settling_time_ms', np.nan) for r in trial_results]
        settled_count = sum(1 for st in settling_times if not np.isnan(st))

        additional['settled_fraction'] = settled_count / len(settling_times) if settling_times else 0.0
        additional['settled_count'] = settled_count

        if settled_count > 0:
            valid_times = [st for st in settling_times if not np.isnan(st)]
            additional['settling_time_median'] = float(np.median(valid_times))
        else:
            additional['settling_time_median'] = np.nan

        return additional

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, v_th_distribution: str = "normal",
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full stability experiment with randomized job distribution."""
        if static_input_rates is None:
            static_input_rates = np.array([50.0, 100.0, 200.0, 500.0, 1000.0]) if self.synaptic_mode == "filter" \
                                 else np.array([50.0, 100.0, 200.0, 500.0])

        # Use base class method for parameter combinations
        all_combinations = self.create_parameter_combinations(
            session_id=session_id,
            v_th_stds=v_th_stds,
            g_stds=g_stds,
            static_input_rates=static_input_rates,
            v_th_distribution=v_th_distribution
        )

        print(f"Starting stability experiment: {len(all_combinations)} combinations")

        results = []
        for i, combo in enumerate(all_combinations):
            print(f"[{i+1}/{len(all_combinations)}]: v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}")

            result = self.run_parameter_combination(
                session_id=combo['session_id'],
                v_th_std=combo['v_th_std'],
                g_std=combo['g_std'],
                v_th_distribution=combo['v_th_distribution'],
                static_input_rate=combo['static_input_rate']
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

        print(f"Stability experiment completed")
        return results
