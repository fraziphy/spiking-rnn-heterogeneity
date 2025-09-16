# encoding_experiment.py
"""
Encoding capacity experiment for studying information processing in spiking RNNs.
Uses 20-channel dynamic Poisson input to test network encoding capabilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import time
from spiking_network import SpikingRNN
from rng_utils import get_rng

class EncodingExperiment:
    """Experiment class for studying encoding capacity."""

    def __init__(self, n_neurons: int = 1000, n_input_channels: int = 20, dt: float = 0.1):
        """
        Initialize encoding experiment.

        Args:
            n_neurons: Number of neurons in RNN
            n_input_channels: Number of input channels
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.n_input_channels = n_input_channels
        self.dt = dt

    def generate_input_patterns(self, session_id: int, trial_id: int,
                              duration: float, n_patterns: int = 10,
                              base_rate: float = 20.0) -> np.ndarray:
        """
        Generate dynamic input patterns for encoding study.

        Args:
            session_id: Session ID for RNG
            trial_id: Trial ID for RNG
            duration: Pattern duration (ms)
            n_patterns: Number of different patterns
            base_rate: Base firing rate (Hz)

        Returns:
            Array of input patterns (time_steps x n_channels)
        """
        rng = get_rng(session_id, 0, trial_id, 'input_patterns')

        n_time_steps = int(duration / self.dt)

        # Create random patterns with different rate modulations
        patterns = np.zeros((n_time_steps, self.n_input_channels))

        # Each pattern is a different combination of channel activations
        for step in range(n_time_steps):
            # Randomly modulate each channel
            modulation = rng.uniform(0.1, 2.0, self.n_input_channels)
            patterns[step, :] = base_rate * modulation

        return patterns

    def run_encoding_trial(self, session_id: int, block_id: int, trial_id: int,
                          v_th_std: float, g_std: float,
                          input_patterns: np.ndarray) -> Dict[str, Any]:
        """
        Run single encoding trial.

        Args:
            session_id, block_id, trial_id: RNG parameters
            v_th_std: Spike threshold heterogeneity
            g_std: Synaptic weight heterogeneity
            input_patterns: Input rate patterns

        Returns:
            Dictionary with encoding results
        """
        # Create network
        network = SpikingRNN(self.n_neurons, self.n_input_channels, dt=self.dt)

        # Initialize network
        network_params = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'static_input_strength': 0.5,  # Background activity
            'dynamic_input_strength': 2.0,  # Task-related input
            'dynamic_connection_prob': 0.3,  # 30% connectivity as specified
            'readout_weight_scale': 1.0
        }

        network.initialize_network(session_id, block_id, **network_params)

        # Run simulation
        duration = len(input_patterns) * self.dt
        spike_times, readout_history = network.simulate_encoding_task(
            session_id, block_id, trial_id,
            duration=duration,
            input_patterns=input_patterns,
            static_input_rate=5.0  # Low background rate
        )

        # Analyze encoding performance
        # TODO: Add specific encoding metrics here
        # - Mutual information between input and spikes
        # - Population vector decoding accuracy
        # - Temporal coding fidelity

        return {
            'spike_times': spike_times,
            'readout_history': readout_history,
            'n_spikes': len(spike_times),
            'network_info': network.get_network_info(),
            'input_patterns': input_patterns
        }

    def run_parameter_combination(self, session_id: int, block_id: int,
                                v_th_std: float, g_std: float,
                                n_trials: int = 10) -> Dict[str, Any]:
        """
        Run encoding experiment for single parameter combination.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            v_th_std: Spike threshold heterogeneity
            g_std: Synaptic weight heterogeneity
            n_trials: Number of trials to run

        Returns:
            Dictionary with results
        """
        trial_results = []

        for trial_id in range(1, n_trials + 1):
            # Generate input patterns for this trial
            input_patterns = self.generate_input_patterns(
                session_id, trial_id, duration=1000.0  # 1 second trial
            )

            # Run encoding trial
            result = self.run_encoding_trial(
                session_id, block_id, trial_id, v_th_std, g_std, input_patterns
            )

            trial_results.append(result)

        # Compute statistics across trials
        n_spikes_all = [r['n_spikes'] for r in trial_results]

        results = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'trial_results': trial_results,
            'n_spikes_mean': np.mean(n_spikes_all),
            'n_spikes_std': np.std(n_spikes_all),
            'n_trials': len(trial_results)
        }

        return results

def save_encoding_results(results: List[Dict[str, Any]], filename: str):
    """
    Save encoding results to file.

    Args:
        results: List of result dictionaries
        filename: Output filename
    """
    import pickle

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Encoding results saved to {filename}")

if __name__ == "__main__":
    # Example usage
    experiment = EncodingExperiment(n_neurons=1000, n_input_channels=20)

    # Test single parameter combination
    session_id = 1
    block_id = 0

    result = experiment.run_parameter_combination(
        session_id, block_id, v_th_std=0.1, g_std=0.1, n_trials=5
    )

    print(f"Encoding experiment completed:")
    print(f"  Mean spikes per trial: {result['n_spikes_mean']:.1f} ± {result['n_spikes_std']:.1f}")

    # Save results
    save_encoding_results([result], f"encoding_test_session_{session_id}.pkl")
