# src/spiking_network.py - Clean version with HD input support, no unused DynamicPoissonInput
"""
Spiking RNN network with HD dynamic input support for encoding experiments.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from lif_neuron import LIFNeuron
from synaptic_model import Synapse, StaticPoissonInput, HDDynamicInput, ReadoutLayer
from rng_utils import get_rng

class SpikingRNN:
    """Spiking RNN with HD input support for encoding experiments."""

    def __init__(self, n_neurons: int = 1000, n_readout_neurons: int = 10, dt: float = 0.1,
                 synaptic_mode: str = "filter", static_input_mode: str = "independent",
                 hd_input_mode: str = "independent", n_hd_channels: int = 10):
        """Initialize spiking RNN with HD input support."""
        self.n_neurons = n_neurons
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt
        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.hd_input_mode = hd_input_mode
        self.n_hd_channels = n_hd_channels

        # Initialize components
        self.neurons = LIFNeuron(n_neurons, dt)
        self.recurrent_synapses = Synapse(n_neurons, dt, synaptic_mode)
        self.static_input_synapses = Synapse(n_neurons, dt, synaptic_mode)
        self.hd_input_synapses = Synapse(n_neurons, dt, synaptic_mode)

        self.static_input = StaticPoissonInput(n_neurons, dt, static_input_mode)
        self.hd_input = HDDynamicInput(n_neurons, n_hd_channels, dt, hd_input_mode)
        self.readout = ReadoutLayer(n_neurons, n_readout_neurons, dt)

        # Simulation state
        self.current_time = 0.0
        self.spike_times = []
        self.readout_history = []

    def initialize_network(self, session_id: int, v_th_std: float, g_std: float,
                          v_th_distribution: str = "normal",
                          hd_dim: int = 0, embed_dim: int = 0,
                          **kwargs):
        """
        Initialize network with optional HD input support.

        Args:
            session_id: Session ID for reproducibility
            v_th_std: Direct standard deviation for spike thresholds
            g_std: Direct standard deviation for synaptic weights
            v_th_distribution: "normal" or "uniform" threshold distribution
            hd_dim: HD intrinsic dimensionality (0 means no HD input)
            embed_dim: HD embedding dimensionality (0 means no HD input)
            **kwargs: Additional parameters
        """
        # Get parameters with defaults
        readout_weight_scale = kwargs.get('readout_weight_scale', 1.0)
        hd_connection_prob = kwargs.get('hd_connection_prob', 0.3)
        hd_input_strength = kwargs.get('hd_input_strength', 1.0)

        # Initialize neuron parameters
        self.neurons.initialize_parameters(
            session_id=session_id,
            v_th_std=v_th_std,
            trial_id=0,
            v_th_mean=-55.0,
            v_th_distribution=v_th_distribution
        )

        # Initialize synaptic weights
        self.recurrent_synapses.initialize_weights(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            g_mean=0.0,
            connection_prob=0.1
        )

        # Initialize static Poisson input
        self.static_input.initialize_parameters(kwargs.get('static_input_strength', 10.0))

        # Initialize HD input connectivity (if HD experiment)
        if hd_dim > 0 and embed_dim > 0:
            self.hd_input.initialize_connectivity(
                session_id=session_id,
                hd_dim=hd_dim,
                embed_dim=embed_dim,
                connection_prob=hd_connection_prob,
                input_strength=hd_input_strength
            )

        # Initialize readout layer
        self.readout.initialize_weights(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            weight_scale=readout_weight_scale
        )

    def reset_simulation(self, session_id: int, v_th_std: float, g_std: float, trial_id: int):
        """Reset simulation state for new trial."""
        self.current_time = 0.0
        self.spike_times = []
        self.readout_history = []

        # Initialize neuron states (trial-dependent)
        self.neurons.initialize_state(session_id, v_th_std, g_std, trial_id)

    def step(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
            static_input_rate: float = 0.0,
            hd_input_rates: Optional[np.ndarray] = None,
            hd_dim: int = 0, embed_dim: int = 0,
            time_step: int = 0) -> Tuple[List[int], np.ndarray]:
        """Execute one simulation time step with optional HD input."""
        # Initialize total input current
        total_input = np.zeros(self.n_neurons)

        # 1. Static Poisson input
        if static_input_rate > 0:
            static_events = self.static_input.generate_events(
                session_id, v_th_std, g_std, trial_id, static_input_rate, time_step
            )
            static_current = self.static_input_synapses.apply_to_input(static_events)
            total_input += static_current

        # 2. HD input (for encoding experiments)
        if hd_input_rates is not None and len(hd_input_rates) > 0:
            hd_events = self.hd_input.generate_events(
                session_id, v_th_std, g_std, trial_id, hd_dim, embed_dim,
                hd_input_rates, time_step
            )
            hd_current = self.hd_input_synapses.apply_to_input(hd_events)
            total_input += hd_current

        # 3. Recurrent synaptic input
        spiked_last_step = np.abs(self.neurons.last_spike_time - (self.current_time - self.dt)) < self.dt/2
        current_spikes = np.where(spiked_last_step)[0].tolist()

        synaptic_current = self.recurrent_synapses.update(current_spikes)
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
        """Inject auxiliary spike for perturbation analysis."""
        self.neurons.inject_auxiliary_spike(neuron_id, self.current_time)
        self.spike_times.append((self.current_time, neuron_id))

    def simulate_network_dynamics(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                                 duration: float, static_input_rate: float = 0.0,
                                 perturbation_time: float = None,
                                 perturbation_neuron: int = None) -> List[Tuple[float, int]]:
        """Run simulation for network stability study."""
        self.reset_simulation(session_id, v_th_std, g_std, trial_id)

        n_steps = int(duration / self.dt)

        for step in range(n_steps):
            # Check for perturbation
            if (perturbation_time is not None and
                abs(self.current_time - perturbation_time) < self.dt/2):
                if perturbation_neuron is not None:
                    self.inject_perturbation(perturbation_neuron)

            # Execute time step (only static input for network stability)
            self.step(session_id, v_th_std, g_std, trial_id,
                     static_input_rate=static_input_rate,
                     time_step=step)

        return self.spike_times.copy()

    def simulate_encoding_task(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                              duration: float, hd_input_patterns: np.ndarray,
                              hd_dim: int, embed_dim: int,
                              static_input_rate: float = 0.0,
                              transient_time: float = 50.0) -> Tuple[List[Tuple[float, int]], List[Tuple[float, np.ndarray]]]:
        """
        Run simulation for encoding capacity study with HD inputs.

        Args:
            session_id: Session ID
            v_th_std: Threshold std
            g_std: Weight std
            trial_id: Trial ID
            duration: Total simulation duration (ms)
            hd_input_patterns: HD input rates, shape (n_timesteps, n_hd_channels)
            hd_dim: HD intrinsic dimensionality
            embed_dim: HD embedding dimensionality
            static_input_rate: Background static input rate
            transient_time: Transient period before HD input (ms)

        Returns:
            spike_times: List of (time, neuron_id) tuples
            readout_history: List of (time, readout_activity) tuples
        """
        self.reset_simulation(session_id, v_th_std, g_std, trial_id)

        n_steps = int(duration / self.dt)
        transient_steps = int(transient_time / self.dt)

        for step in range(n_steps):
            # Determine if we're in transient or encoding period
            if step < transient_steps:
                # Transient period: only static background, no HD input
                self.step(session_id, v_th_std, g_std, trial_id,
                         static_input_rate=static_input_rate,
                         time_step=step)
            else:
                # Encoding period: static background + HD input
                encoding_step = step - transient_steps

                if encoding_step < len(hd_input_patterns):
                    hd_rates = hd_input_patterns[encoding_step]
                else:
                    hd_rates = np.zeros(self.n_hd_channels)

                self.step(session_id, v_th_std, g_std, trial_id,
                         static_input_rate=static_input_rate,
                         hd_input_rates=hd_rates,
                         hd_dim=hd_dim,
                         embed_dim=embed_dim,
                         time_step=step)

        return self.spike_times.copy(), self.readout_history.copy()

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information including HD input details."""
        info = {
            'n_neurons': self.n_neurons,
            'spike_thresholds': self.neurons.spike_thresholds if self.neurons.spike_thresholds is not None else [],
            'n_readout_neurons': self.n_readout_neurons,
            'n_hd_channels': self.n_hd_channels,
            'dt': self.dt,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            'hd_input_mode': self.hd_input_mode,
            'weight_matrix_nnz': self.recurrent_synapses.weight_matrix.nnz if self.recurrent_synapses.weight_matrix is not None else 0,
            'readout_weights_shape': self.readout.readout_weights.shape if self.readout.readout_weights is not None else None
        }

        # Add threshold statistics
        threshold_stats = self.neurons.get_threshold_statistics()
        info.update({f'threshold_{k}': v for k, v in threshold_stats.items()})

        # Add weight statistics
        weight_stats = self.recurrent_synapses.get_weight_statistics()
        info.update({f'weight_{k}': v for k, v in weight_stats.items()})

        # Add HD input connectivity info
        hd_info = self.hd_input.get_connectivity_info()
        info.update({f'hd_input_{k}': v for k, v in hd_info.items()})

        return info
