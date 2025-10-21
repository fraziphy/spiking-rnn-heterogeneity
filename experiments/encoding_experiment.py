
# experiments/encoding_experiment.py - Updated to use inputs subdirectory
"""
Encoding capacity experiment: study how networks encode high-dimensional inputs.
Updated to use inputs/ subdirectory for HD signal caching.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Any, Optional

# Import base class and utilities
from .base_experiment import BaseExperiment

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.hd_input import HDInputGenerator
    from src.rng_utils import get_rng
    from analysis.encoding_analysis import decode_hd_input
    from analysis.statistics_utils import get_extreme_combinations, is_extreme_combo
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    for subdir in ['src', 'analysis']:
        sys.path.insert(0, os.path.join(project_root, subdir))
    from src.spiking_network import SpikingRNN  # FIXED
    from src.hd_input import HDInputGenerator  # FIXED
    from src.rng_utils import get_rng  # FIXED
    from analysis.encoding_analysis import decode_hd_input
    from analysis.statistics_utils import get_extreme_combinations, is_extreme_combo


class EncodingExperiment(BaseExperiment):
    """Encoding capacity experiment with HD inputs."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 synaptic_mode: str = "filter",
                 static_input_mode: str = "independent",
                 hd_input_mode: str = "independent",
                 embed_dim: int = 10,
                 signal_cache_dir: str = "hd_signals"):
        """
        Initialize encoding experiment.

        Args:
            n_neurons: Number of RNN neurons
            dt: Time step (ms)
            synaptic_mode: "pulse" or "filter"
            static_input_mode: Static background mode
            hd_input_mode: HD input mode
            embed_dim: HD embedding dimensionality
            signal_cache_dir: Directory for HD signal caching
        """
        super().__init__(n_neurons, dt)

        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.hd_input_mode = hd_input_mode
        self.embed_dim = embed_dim

        # Timing parameters
        self.transient_time = 200.0  # ms
        self.encoding_time = 300.0  # ms
        self.total_duration = self.transient_time + self.encoding_time

        # Number of trials
        self.n_trials = 100

        # HD input generator with caching in inputs/ subdirectory
        self.hd_generator = HDInputGenerator(
            embed_dim=embed_dim,
            dt=dt,
            signal_cache_dir=os.path.join(signal_cache_dir, 'inputs')  # UPDATED
        )

    def run_single_trial(self, session_id: int, v_th_std: float, g_std: float,
                        trial_id: int, hd_dim: int,
                        v_th_distribution: str = "normal",
                        static_input_rate: float = 200.0,
                        hd_noise_std: float = 0.5,
                        hd_rate_scale: float = 1.0,
                        return_thresholds: bool = False) -> Dict[str, Any]:
        """
        Run single encoding trial.

        Args:
            return_thresholds: If True, return spike thresholds (for low-dim experiments)
        """
        # Create network
        network = SpikingRNN(
            self.n_neurons,
            dt=self.dt,
            synaptic_mode=self.synaptic_mode,
            static_input_mode=self.static_input_mode,
            hd_input_mode=self.hd_input_mode,
            n_hd_channels=self.embed_dim
        )

        # Initialize network
        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 10.0,
            'hd_connection_prob': 0.3,
            'hd_input_strength': 50.0,
            'readout_weight_scale': 1.0
        }

        network.initialize_network(
            session_id, v_th_std, g_std,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim,
            **network_params
        )

        # Generate HD input for this trial
        hd_input_patterns = self.hd_generator.generate_trial_input(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            hd_dim=hd_dim,
            noise_std=hd_noise_std,
            rate_scale=hd_rate_scale
        )

        # Run encoding simulation
        spike_times, readout_history = network.simulate_encoding_task(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.total_duration,
            hd_input_patterns=hd_input_patterns,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim,
            static_input_rate=static_input_rate,
            transient_time=self.transient_time
        )

        # Extract encoding period spikes (after transient)
        encoding_spikes = [(t - self.transient_time, nid)
                          for t, nid in spike_times
                          if t >= self.transient_time]

        result = {
            'spike_times': encoding_spikes,
            'n_spikes': len(encoding_spikes),
            'trial_id': trial_id
        }

        # Optionally return spike thresholds (for low-dim experiments)
        if return_thresholds:
            result['spike_thresholds'] = network.neurons.spike_thresholds.copy()

        return result

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                 hd_dim: int,
                                 v_th_distribution: str = "normal",
                                 static_input_rate: float = 200.0,
                                 hd_noise_std: float = 0.5,
                                 hd_rate_scale: float = 1.0,
                                 extreme_combos: List = None) -> Dict[str, Any]:
        """
        Run full parameter combination with multiple trials.

        Args:
            extreme_combos: List of extreme (v_th_std, g_std) combinations for neuron data saving
        """
        start_time = time.time()

        # Initialize HD generator base input (cached)
        self.hd_generator.initialize_base_input(
            session_id=session_id,
            hd_dim=hd_dim,
            rate_rnn_params={'n_neurons': 1000, 'T': 500.0, 'g': 2.0}
        )

        # Determine if we should save neuron-level data
        save_neuron_data = False
        if hd_dim == 1 and self.embed_dim == 1 and extreme_combos is not None:
            save_neuron_data = is_extreme_combo(v_th_std, g_std, extreme_combos)

        # Run all trials
        trial_results = []
        spike_thresholds = None

        for trial_idx in range(self.n_trials):
            trial_id = trial_idx + 1

            # Only get thresholds on first trial if saving neuron data
            return_thresholds = (save_neuron_data and trial_idx == 0)

            trial_result = self.run_single_trial(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                hd_dim=hd_dim,
                v_th_distribution=v_th_distribution,
                static_input_rate=static_input_rate,
                hd_noise_std=hd_noise_std,
                hd_rate_scale=hd_rate_scale,
                return_thresholds=return_thresholds
            )

            # Extract and store thresholds from first trial
            if return_thresholds and 'spike_thresholds' in trial_result:
                spike_thresholds = trial_result.pop('spike_thresholds')

            trial_results.append(trial_result)

        # Extract basic statistics
        n_spikes_array = np.array([r['n_spikes'] for r in trial_results])

        # Perform decoding analysis
        print(f"    Running decoding analysis...")

        decoding_results = decode_hd_input(
            trial_results=trial_results,
            hd_input_ground_truth=self.hd_generator.Y_base,
            n_neurons=self.n_neurons,
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim,
            encoding_duration=self.encoding_time,
            dt=self.dt,
            tau=10.0,
            lambda_reg=1e-3,
            n_splits=self.n_trials  # LOOCV
        )

        # Prepare decoding results based on dimensionality
        if save_neuron_data:
            # Low-dim: keep neuron-level data
            decoding_data = {
                'test_rmse_mean': decoding_results['test_rmse_mean'],
                'test_r2_mean': decoding_results['test_r2_mean'],
                'test_correlation_mean': decoding_results['test_correlation_mean'],
                'decoder_weights': decoding_results['decoder_weights'],
                'spike_jitter_per_fold': decoding_results['spike_jitter_per_fold'],
                'spike_thresholds': spike_thresholds
            }
        else:
            # High-dim or non-extreme: only summary statistics
            # Extract dimensionality metrics
            weight_pr_per_fold = [svd['participation_ratio'] for svd in decoding_results['weight_svd_analysis']]
            weight_ed_per_fold = [svd['effective_dim_95'] for svd in decoding_results['weight_svd_analysis']]
            decoded_pr_per_fold = [pca['participation_ratio'] for pca in decoding_results['decoded_pca_analysis']]
            decoded_ed_per_fold = [pca['effective_dim_95'] for pca in decoding_results['decoded_pca_analysis']]

            decoding_data = {
                'test_rmse_mean': decoding_results['test_rmse_mean'],
                'test_r2_mean': decoding_results['test_r2_mean'],
                'test_correlation_mean': decoding_results['test_correlation_mean'],
                'weight_participation_ratio_mean': float(np.mean(weight_pr_per_fold)),
                'weight_effective_dim_mean': float(np.mean(weight_ed_per_fold)),
                'decoded_participation_ratio_mean': float(np.mean(decoded_pr_per_fold)),
                'decoded_effective_dim_mean': float(np.mean(decoded_ed_per_fold))
            }

        # Compile results
        results = {
            # Parameter information
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'hd_dim': hd_dim,
            'embed_dim': self.embed_dim,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'hd_noise_std': hd_noise_std,
            'hd_rate_scale': hd_rate_scale,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            'hd_input_mode': self.hd_input_mode,

            # Basic statistics
            'n_spikes_mean': float(np.mean(n_spikes_array)),
            'n_spikes_std': float(np.std(n_spikes_array)),

            # HD input statistics
            'hd_base_stats': self.hd_generator.get_base_statistics(),

            # Decoding analysis (smart storage)
            'decoding': decoding_data,

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'transient_time': self.transient_time,
            'encoding_time': self.encoding_time,
            'saved_neuron_data': save_neuron_data
        }

        return results

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results (required by base class)."""
        # Encoding experiment doesn't use trial-level arrays
        return {}

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, hd_dims: np.ndarray,
                          v_th_distribution: str = "normal",
                          static_input_rates: np.ndarray = None,
                          hd_noise_std: float = 0.5,
                          hd_rate_scale: float = 1.0) -> List[Dict[str, Any]]:
        """Run full encoding experiment with randomized job distribution."""
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

        # Get extreme combinations for smart storage
        extreme_combos = get_extreme_combinations(v_th_stds, g_stds)

        # Create parameter combinations using base class method
        all_combinations = self.create_parameter_combinations(
            session_id=session_id,
            v_th_stds=v_th_stds,
            g_stds=g_stds,
            static_input_rates=static_input_rates,
            v_th_distribution=v_th_distribution,
            hd_dims=hd_dims,  # Encoding-specific
            hd_noise_std=hd_noise_std,
            hd_rate_scale=hd_rate_scale
        )

        total_combinations = len(all_combinations)

        print(f"Starting encoding experiment: {total_combinations} combinations")
        print(f"  Session ID: {session_id}")
        print(f"  Extreme combos for neuron data: {len(extreme_combos)}")

        results = []
        for i, combo in enumerate(all_combinations):
            print(f"[{i+1}/{total_combinations}]:")
            print(f"    v_th={combo['v_th_std']:.3f}, g={combo['g_std']:.3f}, hd={combo['hd_dim']}")

            result = self.run_parameter_combination(
                session_id=combo['session_id'],
                v_th_std=combo['v_th_std'],
                g_std=combo['g_std'],
                hd_dim=combo['hd_dim'],
                v_th_distribution=combo['v_th_distribution'],
                static_input_rate=combo['static_input_rate'],
                hd_noise_std=combo.get('hd_noise_std', 0.5),
                hd_rate_scale=combo.get('hd_rate_scale', 1.0),
                extreme_combos=extreme_combos
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

        print(f"Encoding experiment completed: {len(results)} combinations processed")
        return results
