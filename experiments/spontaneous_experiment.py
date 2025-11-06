# experiments/spontaneous_experiment.py - Refactored with base class
"""
Spontaneous activity analysis: firing rates, dimensionality, silent neurons.
MODIFIED: Transient time changed from 200ms to 500ms.
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
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
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

        # Analyze with UPDATED 500ms transient time (was 200ms)
        analysis_results = analyze_spontaneous_activity(
            spikes=spikes,
            num_neurons=self.n_neurons,
            duration=duration,
            transient_time=500.0  # UPDATED
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

        results = {
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'duration': duration,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            **stats
        }

        return results

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results."""
        return {
            'mean_firing_rate': np.array([r['mean_firing_rate'] for r in trial_results]),
            'percent_silent': np.array([r['percent_silent'] for r in trial_results]),
            'cv_isi': np.array([r['cv_isi_mean'] for r in trial_results]),
            'burst_fraction': np.array([r['burst_fraction'] for r in trial_results])
        }

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray, g_stds: np.ndarray,
                          v_th_distribution: str = "normal",
                          static_input_rates: np.ndarray = None,
                          duration: float = 5000.0) -> List[Dict[str, Any]]:
        """Run full spontaneous activity experiment."""
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

        all_combinations = self.create_parameter_combinations(
            session_id=session_id,
            v_th_stds=v_th_stds,
            g_stds=g_stds,
            static_input_rates=static_input_rates,
            v_th_distribution=v_th_distribution,
            duration=duration
        )

        print(f"Starting spontaneous experiment: {len(all_combinations)} combinations")
        print(f"  Duration: {duration/1000:.1f}s")

        results = []
        for i, combo in enumerate(all_combinations):
            print(f"[{i+1}/{len(all_combinations)}]: v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}, rate={combo['static_input_rate']:.0f}")

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

        print(f"Spontaneous experiment completed: {len(results)} combinations")
        return results
