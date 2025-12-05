# experiments/stability_experiment.py
"""
Network stability experiment with cached transient states.
Tests perturbation response using pre-computed transient states.
"""

import numpy as np
import os
import sys
import pickle
from typing import Dict, List, Any

# Import base class
from .base_experiment import BaseExperiment

try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.stability_analysis import analyze_perturbation_response
    from analysis.spontaneous_analysis import compute_activity_dimensionality_multi_bin
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    for subdir in ['src', 'analysis']:
        sys.path.insert(0, os.path.join(project_root, subdir))
    from spiking_network import SpikingRNN
    from rng_utils import get_rng
    from stability_analysis import analyze_perturbation_response
    from analysis.spontaneous_analysis import compute_activity_dimensionality_multi_bin



class StabilityExperiment(BaseExperiment):
    """
    Network stability experiment with perturbation analysis.

    Can operate in two modes:
    1. use_cached_transients=True: Load pre-computed 1000ms transient states
    2. use_cached_transients=False: Simulate transient from scratch (legacy)
    """

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 synaptic_mode: str = "filter",
                 static_input_mode: str = "independent",
                 transient_cache_dir: str = "results/cached_states",
                 use_cached_transients: bool = True):
        """
        Initialize stability experiment.

        Args:
            n_neurons: Number of neurons
            dt: Time step (ms)
            synaptic_mode: Synaptic dynamics mode
            static_input_mode: Static input mode
            transient_cache_dir: Directory with cached transient states
            use_cached_transients: If True, use cached states; if False, simulate fresh
        """
        super().__init__(n_neurons, dt)

        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.transient_cache_dir = transient_cache_dir
        self.use_cached_transients = use_cached_transients

        # Timing parameters
        self.pre_perturbation_time = 1000.0  # ms (only for legacy mode)
        self.post_perturbation_time = 300.0  # ms
        self.n_perturbation_trials = 100

    def get_perturbation_neurons(self, session_id: int, v_th_std: float,
                                g_std: float) -> np.ndarray:
        """
        Get perturbation neurons for this parameter combination.

        Args:
            session_id: Session identifier
            v_th_std: Threshold heterogeneity std
            g_std: Weight heterogeneity std

        Returns:
            Array of neuron indices to perturb
        """
        rng = get_rng(session_id, v_th_std, g_std, 0, 'perturbation_targets')
        sample_size = min(100, self.n_neurons)
        return rng.choice(self.n_neurons, size=sample_size, replace=False)

    def load_cached_transient_state(self, session_id: int, g_std: float,
                                   v_th_std: float, static_rate: float,
                                   trial_id: int) -> Dict[str, Any]:
        """
        Load pre-cached transient state.

        Args:
            session_id: Session identifier
            g_std: Weight heterogeneity std
            v_th_std: Threshold heterogeneity std
            static_rate: Static input rate (Hz)
            trial_id: Trial identifier

        Returns:
            Network state dictionary
        """
        filename = os.path.join(self.transient_cache_dir,
            f"session_{session_id}_g_{g_std:.3f}_vth_{v_th_std:.3f}_"
            f"rate_{static_rate:.1f}_trial_states.pkl")

        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)

        return cache_data['trial_states'][trial_id]

    def run_single_perturbation(self, session_id: int, v_th_std: float,
                            g_std: float, trial_id: int,
                            v_th_distribution: str, perturbation_neuron_idx: int,
                            static_input_rate: float = 30.0) -> Dict[str, Any]:
        """
        Run single perturbation trial.
        Spikes are returned relative to stimulus onset (t=0).
        """
        # Get perturbation neuron
        perturbation_neurons = self.get_perturbation_neurons(session_id, v_th_std, g_std)
        available_neurons = len(perturbation_neurons)
        safe_idx = perturbation_neuron_idx % available_neurons
        perturbation_neuron = int(perturbation_neurons[safe_idx])

        if self.use_cached_transients:
            # Load cached transient state
            initial_state = self.load_cached_transient_state(
                session_id, g_std, v_th_std, static_input_rate, trial_id)
            transient_time = initial_state['current_time']

            # Control network
            network_control = SpikingRNN(
                n_neurons=self.n_neurons,
                dt=self.dt,
                synaptic_mode=self.synaptic_mode,
                static_input_mode=self.static_input_mode,
                n_hd_channels=0
            )
            network_control.initialize_network(session_id, v_th_std, g_std, v_th_distribution)
            network_control.restore_state(initial_state)

            spikes_control_raw = network_control.simulate(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                duration=self.post_perturbation_time,
                static_input_rate=static_input_rate,
                continue_from_state=True
            )

            # Subtract transient time so spikes start at t=0
            spikes_control = [(t - transient_time, nid)
                            for t, nid in spikes_control_raw
                            if t >= transient_time]

            # Perturbed network
            network_perturbed = SpikingRNN(
                n_neurons=self.n_neurons,
                dt=self.dt,
                synaptic_mode=self.synaptic_mode,
                static_input_mode=self.static_input_mode,
                n_hd_channels=0
            )
            network_perturbed.initialize_network(session_id, v_th_std, g_std, v_th_distribution)
            network_perturbed.restore_state(initial_state)

            spikes_perturbed_raw = network_perturbed.simulate(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                duration=self.post_perturbation_time,
                static_input_rate=static_input_rate,
                perturbation_time=transient_time,  # Perturbation at start
                perturbation_neuron=perturbation_neuron,
                continue_from_state=True
            )

            # Subtract transient time so spikes start at t=0
            spikes_perturbed = [(t - transient_time, nid)
                            for t, nid in spikes_perturbed_raw
                            if t >= transient_time]

        else:
            # Simulate transient from scratch
            network = SpikingRNN(
                n_neurons=self.n_neurons,
                dt=self.dt,
                synaptic_mode=self.synaptic_mode,
                static_input_mode=self.static_input_mode,
                n_hd_channels=0
            )
            network.initialize_network(session_id, v_th_std, g_std, v_th_distribution)

            network.simulate(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                duration=self.pre_perturbation_time,
                static_input_rate=static_input_rate,
                continue_from_state=False
            )

            state_at_transient = network.save_state()
            transient_time = state_at_transient['current_time']

            # Control network
            network_control = SpikingRNN(
                n_neurons=self.n_neurons,
                dt=self.dt,
                synaptic_mode=self.synaptic_mode,
                static_input_mode=self.static_input_mode,
                n_hd_channels=0
            )
            network_control.initialize_network(session_id, v_th_std, g_std, v_th_distribution)
            network_control.restore_state(state_at_transient)

            spikes_control_raw = network_control.simulate(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                duration=self.post_perturbation_time,
                static_input_rate=static_input_rate,
                continue_from_state=True
            )

            spikes_control = [(t - transient_time, nid)
                            for t, nid in spikes_control_raw
                            if t >= transient_time]

            # Perturbed network
            network_perturbed = SpikingRNN(
                n_neurons=self.n_neurons,
                dt=self.dt,
                synaptic_mode=self.synaptic_mode,
                static_input_mode=self.static_input_mode,
                n_hd_channels=0
            )
            network_perturbed.initialize_network(session_id, v_th_std, g_std, v_th_distribution)
            network_perturbed.restore_state(state_at_transient)

            spikes_perturbed_raw = network_perturbed.simulate(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                duration=self.post_perturbation_time,
                static_input_rate=static_input_rate,
                perturbation_time=transient_time,
                perturbation_neuron=perturbation_neuron,
                continue_from_state=True
            )

            spikes_perturbed = [(t - transient_time, nid)
                            for t, nid in spikes_perturbed_raw
                            if t >= transient_time]

        # Analyze - spikes now start at t=0, perturbation is at t=0
        stability_analysis = analyze_perturbation_response(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=0.0,  # Perturbation is now at t=0
            simulation_end=self.post_perturbation_time,
            perturbed_neuron=perturbation_neuron,
            dt=self.dt
        )

        # =====================================================================
        # NEW: Compute participation ratio and dimensionality from control spikes
        # =====================================================================
        dimensionality_metrics = compute_activity_dimensionality_multi_bin(
            spikes=spikes_control,
            num_neurons=self.n_neurons,
            duration=self.post_perturbation_time,
            bin_sizes=[2.0]  # Use 2ms bin size to match expected keys
        )

        # Extract metrics for 2ms bin size
        if 'bin_2.0ms' in dimensionality_metrics:
            dim_2ms = dimensionality_metrics['bin_2.0ms']
            stability_analysis['participation_ratio_2ms'] = dim_2ms.get('participation_ratio', np.nan)
            stability_analysis['dimensionality_2ms'] = dim_2ms.get('effective_dimensionality', np.nan)
            stability_analysis['intrinsic_dimensionality_2ms'] = dim_2ms.get('intrinsic_dimensionality', np.nan)
            stability_analysis['total_variance_2ms'] = dim_2ms.get('total_variance', np.nan)
        else:
            stability_analysis['participation_ratio_2ms'] = np.nan
            stability_analysis['dimensionality_2ms'] = np.nan
            stability_analysis['intrinsic_dimensionality_2ms'] = np.nan
            stability_analysis['total_variance_2ms'] = np.nan

        return stability_analysis

    def run_parameter_combination(self, session_id: int, v_th_std: float,
                                 g_std: float, v_th_distribution: str = "normal",
                                 static_input_rate: float = 30.0) -> Dict[str, Any]:
        """
        Run stability experiment for single parameter combination.

        Args:
            session_id: Session identifier
            v_th_std: Threshold heterogeneity std
            g_std: Weight heterogeneity std
            v_th_distribution: Threshold distribution type
            static_input_rate: Static input rate (Hz)

        Returns:
            Aggregated results across all perturbation trials
        """
        results_all_trials = []

        for trial_id in range(self.n_perturbation_trials):
            result = self.run_single_perturbation(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                v_th_distribution=v_th_distribution,
                perturbation_neuron_idx=trial_id,
                static_input_rate=static_input_rate
            )
            results_all_trials.append(result)

            if (trial_id + 1) % 10 == 0:
                print(f"  Completed {trial_id + 1}/{self.n_perturbation_trials} trials")

        # Aggregate results
        aggregated = self._aggregate_trial_results(results_all_trials)
        aggregated['session_id'] = session_id
        aggregated['v_th_std'] = v_th_std
        aggregated['g_std'] = g_std
        aggregated['static_input_rate'] = static_input_rate
        aggregated['v_th_distribution'] = v_th_distribution
        aggregated['n_trials'] = self.n_perturbation_trials

        return aggregated

    def _aggregate_trial_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate stability metrics across trials.

        Args:
            results: List of per-trial result dictionaries

        Returns:
            Dictionary with mean, std, and all values for each metric
        """
        # Collect arrays of each metric
        arrays = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float, np.number)):
                arrays[key] = np.array([r[key] for r in results])

        # Compute statistics
        stats = {}
        for key, values in arrays.items():
            valid = values[~np.isnan(values)]
            # Always create _mean and _std keys (NaN if no valid values)
            stats[f'{key}_mean'] = float(np.mean(valid)) if len(valid) > 0 else float('nan')
            stats[f'{key}_std'] = float(np.std(valid)) if len(valid) > 0 else float('nan')
            stats[f'{key}_values'] = values.tolist()

        return stats

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results (required by base class)."""
        return {}

