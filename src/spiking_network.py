# spiking_network.py
"""
Main spiking RNN network class combining all components.
Updated to handle different types of Poisson inputs and readout layer.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from lif_neuron import LIFNeuron
from synaptic_model import ExponentialSynapses, StaticPoissonInput, DynamicPoissonInput, ReadoutLayer
from rng_utils import get_rng

class SpikingRNN:
    """Complete spiking RNN with LIF neurons, exponential synapses, and flexible input types."""

    def __init__(self, n_neurons: int = 1000, n_input_channels: int = 20,
                 n_readout_neurons: int = 10, dt: float = 0.1):
        """
        Initialize spiking RNN.

        Args:
            n_neurons: Number of RNN neurons
            n_input_channels: Number of dynamic input channels
            n_readout_neurons: Number of readout neurons
            dt: Time step in ms
        """
        self.n_neurons = n_neurons
        self.n_input_channels = n_input_channels
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt

        # Initialize components
        self.neurons = LIFNeuron(n_neurons, dt)
        self.synapses = ExponentialSynapses(n_neurons, dt)
        self.static_input = StaticPoissonInput(n_neurons, dt)
        self.dynamic_input = DynamicPoissonInput(n_neurons, n_input_channels, dt)
        self.readout = ReadoutLayer(n_neurons, n_readout_neurons, dt)

        # Simulation state
        self.current_time = 0.0
        self.spike_times = []
        self.readout_history = []

    def initialize_network(self, session_id: int, block_id: int,
                          v_th_std: float = 0.1, g_std: float = 0.1,
                          **kwargs):
        """
        Initialize network parameters.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            v_th_std: Standard deviation of spike thresholds
            g_std: Standard deviation of synaptic weights
            **kwargs: Additional parameters
        """
        # Get parameters with defaults
        v_th_mean = kwargs.get('v_th_mean', -55.0)
        g_mean = kwargs.get('g_mean', 0.0)
        connection_prob = kwargs.get('connection_prob', 0.1)
        static_input_strength = kwargs.get('static_input_strength', 1.0)
        dynamic_input_strength = kwargs.get('dynamic_input_strength', 1.0)
        dynamic_connection_prob = kwargs.get('dynamic_connection_prob', 0.3)
        readout_weight_scale = kwargs.get('readout_weight_scale', 1.0)

        # Initialize neuron parameters
        self.neurons.initialize_parameters(
            session_id, block_id, v_th_mean, v_th_std
        )

        # Initialize synaptic weights
        self.synapses.initialize_weights(
            session_id, block_id, g_mean, g_std, connection_prob
        )

        # Initialize static Poisson input
        self.static_input.initialize_parameters(static_input_strength)

        # Initialize dynamic Poisson input
        self.dynamic_input.initialize_connectivity(
            session_id, block_id, dynamic_connection_prob, dynamic_input_strength
        )

        # Initialize readout layer
        self.readout.initialize_weights(
            session_id, block_id, readout_weight_scale
        )

    def reset_simulation(self, session_id: int, block_id: int, trial_id: int):
        """
        Reset simulation state for new trial.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            trial_id: Trial ID for RNG
        """
        self.current_time = 0.0
        self.spike_times = []
        self.readout_history = []

        # Initialize neuron states
        self.neurons.initialize_state(session_id, block_id, trial_id)

    def step(self, session_id: int, block_id: int, trial_id: int,
             static_input_rate: float = 0.0,
             dynamic_input_rates: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
        """
        Execute one simulation time step.

        Args:
            session_id, block_id, trial_id: RNG parameters
            static_input_rate: Static Poisson input rate (Hz)
            dynamic_input_rates: Array of dynamic input rates (Hz) for each channel

        Returns:
            Tuple of (spike_indices, readout_activity)
        """
        # Initialize total input current
        total_input = np.zeros(self.n_neurons)

        # 1. Static Poisson input (background activity)
        if static_input_rate > 0:
            static_current = self.static_input.update(
                session_id, block_id, trial_id, static_input_rate
            )
            total_input += static_current

        # 2. Dynamic Poisson input (task-related activity)
        if dynamic_input_rates is not None and len(dynamic_input_rates) > 0:
            dynamic_current = self.dynamic_input.update(
                session_id, block_id, trial_id, dynamic_input_rates
            )
            total_input += dynamic_current

        # 3. Recurrent synaptic input
        if len(self.spike_times) > 0:
            # Get spikes from current time step
            current_spikes = [neuron_id for t, neuron_id in self.spike_times
                            if abs(t - self.current_time) < self.dt/2]
        else:
            current_spikes = []

        synaptic_current = self.synapses.update(current_spikes)
        total_input += synaptic_current

        # 4. Update neurons
        membrane_potentials, spike_indices = self.neurons.update(
            self.current_time, total_input
        )

        # 5. Update readout layer
        readout_activity = self.readout.update(spike_indices)

        # Record spikes and readout
        for neuron_id in spike_indices:
            self.spike_times.append((self.current_time, neuron_id))

        self.readout_history.append((self.current_time, readout_activity.copy()))

        self.current_time += self.dt

        return spike_indices, readout_activity

    def inject_perturbation(self, neuron_id: int):
        """
        Inject auxiliary spike for perturbation analysis.

        Args:
            neuron_id: ID of neuron to perturb
        """
        self.neurons.inject_auxiliary_spike(neuron_id, self.current_time)
        # Record the perturbation spike
        self.spike_times.append((self.current_time, neuron_id))

    def simulate_network_dynamics(self, session_id: int, block_id: int, trial_id: int,
                                 duration: float, static_input_rate: float = 0.0,
                                 perturbation_time: float = None,
                                 perturbation_neuron: int = None) -> List[Tuple[float, int]]:
        """
        Run simulation for network dynamics study (no dynamic inputs).

        Args:
            session_id, block_id, trial_id: RNG parameters
            duration: Simulation duration (ms)
            static_input_rate: Background Poisson rate (Hz)
            perturbation_time: Time to inject perturbation (ms)
            perturbation_neuron: Neuron to perturb (if None, random)

        Returns:
            List of (spike_time, neuron_id) tuples
        """
        # Reset simulation
        self.reset_simulation(session_id, block_id, trial_id)

        # Choose random neuron for perturbation if not specified
        if perturbation_time is not None and perturbation_neuron is None:
            rng = get_rng(session_id, block_id, trial_id, 'perturbation')
            perturbation_neuron = rng.integers(0, self.n_neurons)

        # Main simulation loop
        n_steps = int(duration / self.dt)

        for step in range(n_steps):
            # Check for perturbation
            if (perturbation_time is not None and
                abs(self.current_time - perturbation_time) < self.dt/2):
                self.inject_perturbation(perturbation_neuron)

            # Execute time step (only static input for network dynamics)
            self.step(session_id, block_id, trial_id,
                     static_input_rate=static_input_rate,
                     dynamic_input_rates=None)

        return self.spike_times.copy()

    def simulate_encoding_task(self, session_id: int, block_id: int, trial_id: int,
                              duration: float, input_patterns: np.ndarray,
                              static_input_rate: float = 0.0) -> Tuple[List[Tuple[float, int]], List[Tuple[float, np.ndarray]]]:
        """
        Run simulation for encoding capacity study with dynamic inputs.

        Args:
            session_id, block_id, trial_id: RNG parameters
            duration: Simulation duration (ms)
            input_patterns: Array of input rates over time (time_steps x n_channels)
            static_input_rate: Background Poisson rate (Hz)

        Returns:
            Tuple of (spike_times, readout_history)
        """
        # Reset simulation
        self.reset_simulation(session_id, block_id, trial_id)

        # Main simulation loop
        n_steps = int(duration / self.dt)

        for step in range(n_steps):
            # Get dynamic input rates for this time step
            if step < len(input_patterns):
                dynamic_rates = input_patterns[step] # Dynamic input is arranged as time * channels
            else:
                dynamic_rates = np.zeros(self.n_input_channels)

            # Execute time step
            self.step(session_id, block_id, trial_id,
                     static_input_rate=static_input_rate,
                     dynamic_input_rates=dynamic_rates)

        return self.spike_times.copy(), self.readout_history.copy()

    def simulate_task_performance(self, session_id: int, block_id: int, trial_id: int,
                                 duration: float, input_patterns: np.ndarray,
                                 target_outputs: np.ndarray,
                                 static_input_rate: float = 0.0) -> Dict[str, Any]:
        """
        Run simulation for task performance evaluation.

        Args:
            session_id, block_id, trial_id: RNG parameters
            duration: Simulation duration (ms)
            input_patterns: Array of input rates over time (time_steps x n_channels)
            target_outputs: Target readout activity (time_steps x n_readout)
            static_input_rate: Background Poisson rate (Hz)

        Returns:
            Dictionary with performance metrics
        """
        # Run encoding simulation
        spike_times, readout_history = self.simulate_encoding_task(
            session_id, block_id, trial_id, duration, input_patterns, static_input_rate
        )

        # Extract readout activities
        readout_times = [t for t, _ in readout_history]
        readout_activities = np.array([activity for _, activity in readout_history])

        # Compute performance metrics
        if len(readout_activities) > 0 and len(target_outputs) > 0:
            # Align lengths
            min_length = min(len(readout_activities), len(target_outputs))
            readout_aligned = readout_activities[:min_length]
            target_aligned = target_outputs[:min_length]

            # Mean squared error
            mse = np.mean((readout_aligned - target_aligned) ** 2)

            # Correlation (if multiple time points)
            if min_length > 1:
                correlations = []
                for i in range(self.n_readout_neurons):
                    if np.std(readout_aligned[:, i]) > 0 and np.std(target_aligned[:, i]) > 0:
                        corr = np.corrcoef(readout_aligned[:, i], target_aligned[:, i])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                mean_correlation = np.mean(correlations) if correlations else 0.0
            else:
                mean_correlation = 0.0

            performance = {
                'mse': mse,
                'mean_correlation': mean_correlation,
                'n_spikes': len(spike_times),
                'readout_activities': readout_aligned,
                'target_outputs': target_aligned,
                'spike_times': spike_times
            }
        else:
            performance = {
                'mse': float('inf'),
                'mean_correlation': 0.0,
                'n_spikes': len(spike_times),
                'readout_activities': readout_activities,
                'target_outputs': target_outputs,
                'spike_times': spike_times
            }

        return performance

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information for analysis."""
        info = {
            'n_neurons': self.n_neurons,
            'n_input_channels': self.n_input_channels,
            'n_readout_neurons': self.n_readout_neurons,
            'dt': self.dt,
            'spike_thresholds': self.neurons.spike_thresholds.copy(),
            'weight_matrix_nnz': self.synapses.weight_matrix.nnz,
            'readout_weights_shape': self.readout.readout_weights.shape
        }

        # Add input connectivity info
        dynamic_info = self.dynamic_input.get_connectivity_info()
        info.update({f'dynamic_input_{k}': v for k, v in dynamic_info.items()})

        return info
