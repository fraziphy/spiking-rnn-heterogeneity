# task_performance_experiment.py
"""
Task performance experiment for studying computational capabilities in spiking RNNs.
Uses 20-channel input and 10-neuron readout layer to evaluate task performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import time
from spiking_network import SpikingRNN
from rng_utils import get_rng

class TaskPerformanceExperiment:
    """Experiment class for studying task performance."""

    def __init__(self, n_neurons: int = 1000, n_input_channels: int = 20,
                 n_readout_neurons: int = 10, dt: float = 0.1):
        """
        Initialize task performance experiment.

        Args:
            n_neurons: Number of neurons in RNN
            n_input_channels: Number of input channels
            n_readout_neurons: Number of readout neurons
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.n_input_channels = n_input_channels
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt

    def generate_task_data(self, session_id: int, trial_id: int,
                          duration: float, task_type: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate input patterns and target outputs for a specific task.

        Args:
            session_id: Session ID for RNG
            trial_id: Trial ID for RNG
            duration: Trial duration (ms)
            task_type: Type of task ('classification', 'regression', 'memory')

        Returns:
            Tuple of (input_patterns, target_outputs)
        """
        rng = get_rng(session_id, 0, trial_id, 'task_generation')

        n_time_steps = int(duration / self.dt)

        input_patterns = np.zeros((n_time_steps, self.n_input_channels))
        target_outputs = np.zeros((n_time_steps, self.n_readout_neurons))

        if task_type == 'classification':
            # Simple classification task: sum of active channels determines class
            for step in range(n_time_steps):
                # Random input pattern
                active_channels = rng.choice(
                    self.n_input_channels,
                    size=rng.integers(1, 6),  # 1-5 active channels
                    replace=False
                )

                input_patterns[step, active_channels] = rng.uniform(10, 30)  # Hz

                # Target: number of active channels determines output neuron
                n_active = len(active_channels)
                target_class = min(n_active - 1, self.n_readout_neurons - 1)
                target_outputs[step, target_class] = 1.0

        elif task_type == 'regression':
            # Regression task: predict sum of input magnitudes
            for step in range(n_time_steps):
                input_rates = rng.uniform(0, 20, self.n_input_channels)
                input_patterns[step, :] = input_rates

                # Target: normalized sum across readout neurons
                total_input = np.sum(input_rates)
                normalized_target = total_input / (20 * self.n_input_channels)  # Normalize
                target_outputs[step, :] = normalized_target

        elif task_type == 'memory':
            # Working memory task: remember first input pattern
            # Set initial pattern
            initial_pattern = rng.uniform(5, 25, self.n_input_channels)
            input_patterns[0, :] = initial_pattern

            # Noise for middle period
            for step in range(1, n_time_steps - 1):
                input_patterns[step, :] = rng.uniform(0, 5)  # Low noise

            # Target: maintain memory of initial pattern in readout
            memory_signal = np.mean(initial_pattern)
            normalized_memory = memory_signal / 25.0  # Normalize to [0,1]

            # Distribute memory signal across readout neurons
            for neuron_idx in range(self.n_readout_neurons):
                target_outputs[:, neuron_idx] = normalized_memory * (neuron_idx + 1) / self.n_readout_neurons

        return input_patterns, target_outputs

    def run_task_trial(self, session_id: int, block_id: int, trial_id: int,
                      v_th_std: float, g_std: float,
                      input_patterns: np.ndarray,
                      target_outputs: np.ndarray) -> Dict[str, Any]:
        """
        Run single task performance trial.

        Args:
            session_id, block_id, trial_id: RNG parameters
            v_th_std: Spike threshold heterogeneity
            g_std: Synaptic weight heterogeneity
            input_patterns: Input rate patterns
            target_outputs: Target readout activities

        Returns:
            Dictionary with task performance results
        """
        # Create network
        network = SpikingRNN(self.n_neurons, self.n_input_channels,
                           self.n_readout_neurons, dt=self.dt)

        # Initialize network
        network_params = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'static_input_strength': 0.5,
            'dynamic_input_strength': 2.0,
            'dynamic_connection_prob': 0.3,
            'readout_weight_scale': 1.0
        }

        network.initialize_network(session_id, block_id, **network_params)

        # Run simulation
        duration = len(input_patterns) * self.dt
        performance = network.simulate_task_performance(
            session_id, block_id, trial_id,
            duration=duration,
            input_patterns=input_patterns,
            target_outputs=target_outputs,
            static_input_rate=5.0
        )

        return performance

    def run_parameter_combination(self, session_id: int, block_id: int,
                                v_th_std: float, g_std: float,
                                task_type: str = 'classification',
                                n_trials: int = 10) -> Dict[str, Any]:
        """
        Run task performance experiment for single parameter combination.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            v_th_std: Spike threshold heterogeneity
            g_std: Synaptic weight heterogeneity
            task_type: Type of task to run
            n_trials: Number of trials to run

        Returns:
            Dictionary with results
        """
        trial_results = []

        for trial_id in range(1, n_trials + 1):
            # Generate task data
            input_patterns, target_outputs = self.generate_task_data(
                session_id, trial_id, duration=500.0, task_type=task_type  # 500ms trial
            )

            # Run task trial
            result = self.run_task_trial(
                session_id, block_id, trial_id, v_th_std, g_std,
                input_patterns, target_outputs
            )

            trial_results.append(result)

        # Compute statistics across trials
        mse_values = [r['mse'] for r in trial_results]
        correlation_values = [r['mean_correlation'] for r in trial_results]
        n_spikes_values = [r['n_spikes'] for r in trial_results]

        results = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'task_type': task_type,
            'trial_results': trial_results,
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'correlation_mean': np.mean(correlation_values),
            'correlation_std': np.std(correlation_values),
            'n_spikes_mean': np.mean(n_spikes_values),
            'n_spikes_std': np.std(n_spikes_values),
            'n_trials': len(trial_results)
        }

        return results

    def run_full_task_experiment(self, session_id: int,
                               v_th_std_values: np.ndarray,
                               g_std_values: np.ndarray,
                               task_type: str = 'classification') -> List[Dict[str, Any]]:
        """
        Run complete task performance experiment across parameter space.

        Args:
            session_id: Session ID for RNG
            v_th_std_values: Array of spike threshold heterogeneity values
            g_std_values: Array of synaptic weight heterogeneity values
            task_type: Type of task to run

        Returns:
            List of result dictionaries
        """
        results = []

        block_id = 0
        for v_th_std in v_th_std_values:
            for g_std in g_std_values:
                print(f"Running task '{task_type}': v_th_std={v_th_std:.3f}, g_std={g_std:.3f}")

                start_time = time.time()
                result = self.run_parameter_combination(
                    session_id, block_id, v_th_std, g_std, task_type
                )
                end_time = time.time()

                result['computation_time'] = end_time - start_time
                result['block_id'] = block_id

                results.append(result)

                print(f"  MSE: {result['mse_mean']:.4f} ± {result['mse_std']:.4f}")
                print(f"  Correlation: {result['correlation_mean']:.3f} ± {result['correlation_std']:.3f}")
                print(f"  Time: {result['computation_time']:.1f}s")

                block_id += 1

        return results

def save_task_results(results: List[Dict[str, Any]], filename: str):
    """
    Save task performance results to file.

    Args:
        results: List of result dictionaries
        filename: Output filename
    """
    import pickle

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Task performance results saved to {filename}")

if __name__ == "__main__":
    # Example usage
    experiment = TaskPerformanceExperiment(
        n_neurons=1000, n_input_channels=20, n_readout_neurons=10
    )

    # Test single parameter combination
    session_id = 1
    block_id = 0

    result = experiment.run_parameter_combination(
        session_id, block_id, v_th_std=0.1, g_std=0.1,
        task_type='classification', n_trials=5
    )

    print(f"Task performance experiment completed:")
    print(f"  MSE: {result['mse_mean']:.4f} ± {result['mse_std']:.4f}")
    print(f"  Correlation: {result['correlation_mean']:.3f} ± {result['correlation_std']:.3f}")

    # Save results
    save_task_results([result], f"task_test_session_{session_id}.pkl")
