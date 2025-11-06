# experiments/stability_experiment.py - Refactored with base class
"""
Network stability experiment with full-simulation LZ analysis and settling time.
MODIFIED: Pre-perturbation time changed from 200ms to 500ms (total duration 800ms).
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
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.stability_analysis import analyze_perturbation_response


class StabilityExperiment(BaseExperiment):
    """Network stability experiment with perturbation analysis."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 synaptic_mode: str = "filter", static_input_mode: str = "independent"):
        super().__init__(n_neurons, dt)

        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode

        # Timing parameters - MODIFIED: 500ms pre-perturbation (was 200ms)
        self.pre_perturbation_time = 500.0  # ms
        self.post_perturbation_time = 300.0  # ms
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time  # 800ms
        self.perturbation_time = self.pre_perturbation_time  # Perturbation at 500ms
        self.n_perturbation_trials = 100

    def get_perturbation_neurons(self, session_id: int, v_th_std: float, g_std: float) -> np.ndarray:
        """Get perturbation neurons for this parameter combination."""
        rng = get_rng(session_id, v_th_std, g_std, 0, 'perturbation_targets')
        sample_size = min(100, self.n_neurons)
        return rng.choice(self.n_neurons, size=sample_size, replace=False)

    def run_single_perturbation(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                              v_th_distribution: str, perturbation_neuron_idx: int,
                              static_input_rate: float = 200.0) -> Dict[str, Any]:
        """
        Run single perturbation with OPTIMIZED simulation.

        Strategy:
        1. Simulate one network 0→500ms (transient)
        2. Save state at 500ms
        3. Create control & perturbed from saved state
        4. Control: 500→800ms (no perturbation)
        5. Perturbed: 500→800ms (auxiliary spike at t=500ms)

        Saves ~40% computation time by avoiding duplicate 0→500ms simulation!
        """

        # Create single network for transient simulation
        network = SpikingRNN(self.n_neurons, dt=self.dt,
                            synaptic_mode=self.synaptic_mode,
                            static_input_mode=self.static_input_mode)

        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 10.0,
            'readout_weight_scale': 1.0
        }

        network.initialize_network(session_id, v_th_std, g_std, **network_params)

        # Get perturbation neuron
        perturbation_neurons = self.get_perturbation_neurons(session_id, v_th_std, g_std)
        available_neurons = len(perturbation_neurons)
        safe_idx = perturbation_neuron_idx % available_neurons
        perturbation_neuron = int(perturbation_neurons[safe_idx])

        # PHASE 1: Simulate transient period (0→500ms) ONCE
        spikes_transient = network.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.pre_perturbation_time,  # 500ms
            static_input_rate=static_input_rate
        )

        # Save state at 500ms
        state_at_500ms = network.save_state()

        # PHASE 2a: Control network - continue from 500ms to 800ms (no perturbation)
        network_control = SpikingRNN(self.n_neurons, dt=self.dt,
                                     synaptic_mode=self.synaptic_mode,
                                     static_input_mode=self.static_input_mode)
        network_control.initialize_network(session_id, v_th_std, g_std, **network_params)
        network_control.restore_state(state_at_500ms)

        spikes_control = network_control.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.post_perturbation_time,  # 300ms
            static_input_rate=static_input_rate,
            continue_from_state=True  # Continue from restored state at 500ms
        )

        del network  # Free transient network
        del network_control  # Free control network

        # PHASE 2b: Perturbed network - continue from 500ms with perturbation at t=0 (=500ms absolute)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt,
                                       synaptic_mode=self.synaptic_mode,
                                       static_input_mode=self.static_input_mode)
        network_perturbed.initialize_network(session_id, v_th_std, g_std, **network_params)
        network_perturbed.restore_state(state_at_500ms)

        spikes_perturbed = network_perturbed.simulate_network_dynamics(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.post_perturbation_time,  # 300ms
            static_input_rate=static_input_rate,
            perturbation_time=state_at_500ms['current_time'],  # Perturbation at restored time (500ms)
            perturbation_neuron=perturbation_neuron,
            continue_from_state=True  # Continue from restored state at 500ms
        )

        del network_perturbed  # Free perturbed network

        # Analyze perturbation response (stability)
        stability_analysis = analyze_perturbation_response(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            perturbation_time=self.perturbation_time,  # 500ms absolute
            perturbed_neuron=perturbation_neuron,
            num_neurons=self.n_neurons,
            simulation_end=self.total_duration,
            dt=self.dt
        )

        # Analyze spontaneous activity on control network's post-transient period
        try:
            from analysis.spontaneous_analysis import analyze_spontaneous_activity
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
            from analysis.spontaneous_analysis import analyze_spontaneous_activity

        spontaneous_analysis = analyze_spontaneous_activity(
            spikes=spikes_control,
            num_neurons=self.n_neurons,
            duration=self.total_duration,
            transient_time=self.pre_perturbation_time  # 500ms transient
        )

        # Flatten spontaneous analysis (it returns nested dicts)
        spontaneous_flat = {
            **spontaneous_analysis['firing_stats'],
            'dimensionality': spontaneous_analysis['dimensionality_metrics'],
            'cv_isi': spontaneous_analysis['poisson_analysis']['population_statistics']['mean_cv_isi'],
            'fano_factor': spontaneous_analysis['poisson_analysis']['population_statistics']['mean_fano_factor']
        }

        # Merge both analyses
        result = {**stability_analysis, **spontaneous_flat}

        return result

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str = "normal",
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run stability analysis for parameter combination."""
        start_time = time.time()

        trial_results = []
        for trial_idx in range(self.n_perturbation_trials):
            trial_result = self.run_single_perturbation(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_idx + 1,
                v_th_distribution=v_th_distribution,
                perturbation_neuron_idx=trial_idx,
                static_input_rate=static_input_rate
            )
            trial_results.append(trial_result)

        # Extract arrays and compute statistics using base class
        arrays = self.extract_trial_arrays(trial_results)
        stats = self.compute_all_statistics(arrays)

        results = {
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            'perturbation_time': self.perturbation_time,
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            **stats
        }

        return results

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results (stability + spontaneous)."""

        return {
            # Stability metrics
            'lz_spatial_patterns': np.array([r['lz_spatial_patterns'] for r in trial_results]),
            'lz_column_wise': np.array([r['lz_column_wise'] for r in trial_results]),
            'settling_time_ms': np.array([r['settling_time_ms'] for r in trial_results]),
            'kistler_delta_2.0ms': np.array([r['kistler_delta_2.0ms'] for r in trial_results]),
            'gamma_window_2.0ms': np.array([r['gamma_window_2.0ms'] for r in trial_results]),

            # Spontaneous metrics from control network
            'mean_firing_rate': np.array([r['mean_firing_rate'] for r in trial_results]),
            'percent_silent': np.array([r['percent_silent'] for r in trial_results]),
            'cv_isi': np.array([r['cv_isi'] for r in trial_results]),
            'fano_factor': np.array([r['fano_factor'] for r in trial_results]),

            # Dimensionality - effective dimensionality
            'dimensionality_2ms': np.array([r['dimensionality']['bin_2.0ms']['effective_dimensionality'] for r in trial_results]),
            'dimensionality_5ms': np.array([r['dimensionality']['bin_5.0ms']['effective_dimensionality'] for r in trial_results]),
            'dimensionality_20ms': np.array([r['dimensionality']['bin_20.0ms']['effective_dimensionality'] for r in trial_results]),

            # NEW: Participation ratio
            'participation_ratio_2ms': np.array([r['dimensionality']['bin_2.0ms']['participation_ratio'] for r in trial_results]),
            'participation_ratio_5ms': np.array([r['dimensionality']['bin_5.0ms']['participation_ratio'] for r in trial_results]),
            'participation_ratio_20ms': np.array([r['dimensionality']['bin_20.0ms']['participation_ratio'] for r in trial_results]),

            # NEW: Intrinsic dimensionality (cumulative variance threshold)
            'intrinsic_dim_2ms': np.array([r['dimensionality']['bin_2.0ms']['intrinsic_dimensionality'] for r in trial_results]),
            'intrinsic_dim_5ms': np.array([r['dimensionality']['bin_5.0ms']['intrinsic_dimensionality'] for r in trial_results]),
            'intrinsic_dim_20ms': np.array([r['dimensionality']['bin_20.0ms']['intrinsic_dimensionality'] for r in trial_results]),
        }

    def compute_all_statistics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute summary statistics for all metrics (override base method)."""
        stats = {}

        for key, values in arrays.items():
            # Remove NaN values
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                stats[f'{key}_mean'] = float(np.mean(valid_values))
                stats[f'{key}_std'] = float(np.std(valid_values))
                stats[f'{key}_median'] = float(np.median(valid_values))
                stats[f'{key}_min'] = float(np.min(valid_values))
                stats[f'{key}_max'] = float(np.max(valid_values))
            else:
                stats[f'{key}_mean'] = np.nan
                stats[f'{key}_std'] = np.nan
                stats[f'{key}_median'] = np.nan
                stats[f'{key}_min'] = np.nan
                stats[f'{key}_max'] = np.nan

        return stats


    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray, g_stds: np.ndarray,
                          v_th_distribution: str = "normal",
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full stability experiment."""
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

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
            print(f"[{i+1}/{len(all_combinations)}]: v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}, rate={combo['static_input_rate']:.0f}")

            result = self.run_parameter_combination(
                session_id=combo['session_id'],
                v_th_std=combo['v_th_std'],
                g_std=combo['g_std'],
                v_th_distribution=combo['v_th_distribution'],
                static_input_rate=combo['static_input_rate']
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

        print(f"Stability experiment completed: {len(results)} combinations")
        return results
