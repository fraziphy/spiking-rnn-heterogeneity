# src/spiking_network.py
"""
Spiking RNN network with LIF neurons and synaptic dynamics.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .lif_neuron import LIFNeuron
from .synaptic_model import Synapse, StaticPoissonInput, HDDynamicInput
from .rng_utils import get_rng

class SpikingRNN:
    """Spiking RNN with conditional readout synapse computation."""

    def __init__(self, n_neurons: int = 1000, n_readout_neurons: int = 10, dt: float = 0.1,
                 synaptic_mode: str = "filter", static_input_mode: str = "independent",
                 hd_input_mode: str = "independent", n_hd_channels: int = 10,
                 hd_connection_mode: str = "overlapping",
                 use_readout_synapses: bool = False):
        """
        Initialize spiking RNN with conditional readout.

        Args:
            hd_connection_mode: "overlapping" (30% random) or "partitioned" (equal division)
            use_readout_synapses: If True, initialize and update readout synapses
        """
        self.n_neurons = n_neurons
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt
        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.hd_input_mode = hd_input_mode
        self.n_hd_channels = n_hd_channels
        self.hd_connection_mode = hd_connection_mode
        self.use_readout_synapses = use_readout_synapses

        # Initialize neurons
        self.neurons = LIFNeuron(n_neurons, dt)

        # Initialize synapses
        self.recurrent_synapses = Synapse(n_neurons, dt, synaptic_mode)
        self.static_input_synapses = Synapse(n_neurons, dt, synaptic_mode)
        self.hd_input_synapses = Synapse(n_neurons, dt, synaptic_mode)

        # Readout synapses (only if needed)
        if use_readout_synapses:
            self.readout_synapses = Synapse(n_readout_neurons, dt, synaptic_mode)
            self.readout_activity = np.zeros(n_readout_neurons)
        else:
            self.readout_synapses = None
            self.readout_activity = None

        # Initialize input generators
        self.static_input = StaticPoissonInput(n_neurons, dt, static_input_mode)
        self.hd_input = HDDynamicInput(n_neurons, n_hd_channels, dt, hd_input_mode, hd_connection_mode)

        # Simulation state
        self.current_time = 0.0
        self.spike_times = []

    def initialize_network(self, session_id: int, v_th_std: float, g_std: float,
                          v_th_distribution: str = "normal",
                          hd_dim: int = 0, embed_dim: int = 0,
                          **kwargs):
        """Initialize network parameters."""
        hd_connection_prob = kwargs.get('hd_connection_prob', 0.3)
        hd_input_strength = kwargs.get('hd_input_strength', 1.0)

        # 1. Initialize neuron parameters
        self.neurons.initialize_parameters(
            session_id=session_id,
            v_th_std=v_th_std,
            trial_id=0,
            v_th_mean=-55.0,
            v_th_distribution=v_th_distribution
        )

        # 2. Initialize recurrent synaptic weights
        self.recurrent_synapses.initialize_weights(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            g_mean=0.0,
            connection_prob=0.1
        )

        # 3. Initialize static input
        self.static_input.initialize_parameters(kwargs.get('static_input_strength', 10.0))

        # 4. Initialize HD input connectivity
        if hd_dim > 0 and embed_dim > 0:
            self.hd_input.initialize_connectivity(
                session_id=session_id,
                hd_dim=hd_dim,
                embed_dim=embed_dim,
                connection_prob=hd_connection_prob,
                input_strength=hd_input_strength
            )

        # 5. Initialize readout synaptic weights (ONLY if needed)
        if self.use_readout_synapses:
            self.readout_synapses.initialize_weights(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=0.0,
                g_mean=1.0 / np.sqrt(self.n_neurons),
                connection_prob=1.0,
                n_source_neurons=self.n_neurons
            )

    def reset_simulation(self, session_id: int, v_th_std: float, g_std: float, trial_id: int):
        """Reset simulation state for new trial."""
        self.current_time = 0.0
        self.spike_times = []

        if self.use_readout_synapses:
            self.readout_activity = np.zeros(self.n_readout_neurons)

        # Initialize neuron states (trial-dependent)
        self.neurons.initialize_state(session_id, v_th_std, g_std, trial_id)

    def step(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
            static_input_rate: float = 0.0,
            hd_input_rates: Optional[np.ndarray] = None,
            hd_dim: int = 0, embed_dim: int = 0,
            time_step: int = 0) -> Tuple[List[int], Optional[np.ndarray]]:
        """Execute one simulation time step."""
        total_input = np.zeros(self.n_neurons)

        # 1. Static Poisson input → synapses → RNN
        if static_input_rate > 0:
            static_events = self.static_input.generate_events(
                session_id, v_th_std, g_std, trial_id, static_input_rate, time_step
            )
            static_current = self.static_input_synapses.apply_to_input(static_events)
            total_input += static_current

        # 2. HD input → synapses → RNN
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

        # 4. Update RNN neurons
        membrane_potentials, spike_indices = self.neurons.update(
            self.current_time, total_input
        )

        # 5. RNN spikes → readout synapses (ONLY if needed)
        if self.use_readout_synapses:
            self.readout_activity = self.readout_synapses.update(spike_indices)
        else:
            self.readout_activity = None

        # Record spikes
        for neuron_id in spike_indices:
            self.spike_times.append((self.current_time, neuron_id))

        self.current_time += self.dt

        return spike_indices, self.readout_activity

    def inject_perturbation(self, neuron_id: int):
        """Inject auxiliary spike for perturbation analysis."""
        self.neurons.inject_auxiliary_spike(neuron_id, self.current_time)
        self.spike_times.append((self.current_time, neuron_id))

    def simulate(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                duration: float, static_input_rate: float = 0.0,
                hd_input_patterns: Optional[np.ndarray] = None,
                hd_dim: int = 0, embed_dim: int = 0,
                perturbation_time: float = None,
                perturbation_neuron: int = None,
                continue_from_state: bool = False) -> List[Tuple[float, int]]:
        """
        Run simulation - THE ONLY SIMULATE METHOD.

        Args:
            continue_from_state: If True, continue from current state (for cached transients)
        """
        if not continue_from_state:
            self.reset_simulation(session_id, v_th_std, g_std, trial_id)
            start_step = 0
        else:
            start_step = int(self.current_time / self.dt)

        n_steps = int(duration / self.dt)

        for step in range(n_steps):
            actual_time_step = start_step + step

            # Check for perturbation
            if (perturbation_time is not None and
                abs(self.current_time - perturbation_time) < self.dt/2):
                if perturbation_neuron is not None:
                    self.inject_perturbation(perturbation_neuron)

            # Determine HD input
            hd_rates = None
            if hd_input_patterns is not None:
                if step < len(hd_input_patterns):
                    hd_rates = hd_input_patterns[step]
                else:
                    hd_rates = np.zeros(self.n_hd_channels) if self.n_hd_channels > 0 else None

            # Execute time step
            self.step(session_id, v_th_std, g_std, trial_id,
                     static_input_rate=static_input_rate,
                     hd_input_rates=hd_rates,
                     hd_dim=hd_dim,
                     embed_dim=embed_dim,
                     time_step=actual_time_step)

        return self.spike_times.copy()

    def save_state(self):
        """Save current network state (for caching transient states)."""
        return {
            'current_time': self.current_time,
            'spike_times': [],  # Empty - we start fresh from transient state
            'neuron_v_membrane': self.neurons.v_membrane.copy(),
            'neuron_last_spike': self.neurons.last_spike_time.copy(),
            'neuron_refractory': self.neurons.refractory_timer.copy(),
            'recurrent_synaptic_current': self.recurrent_synapses.synaptic_current.copy(),
            'static_synaptic_current': self.static_input_synapses.synaptic_current.copy()
        }

    def restore_state(self, state):
        """Restore previously saved network state."""
        self.current_time = state['current_time']
        self.spike_times = []  # Always start fresh with empty spike history
        self.neurons.v_membrane = state['neuron_v_membrane'].copy()
        self.neurons.last_spike_time = state['neuron_last_spike'].copy()
        self.neurons.refractory_timer = state['neuron_refractory'].copy()
        self.recurrent_synapses.synaptic_current = state['recurrent_synaptic_current'].copy()
        self.static_input_synapses.synaptic_current = state['static_synaptic_current'].copy()

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        info = {
            'n_neurons': self.n_neurons,
            'n_readout_neurons': self.n_readout_neurons,
            'use_readout_synapses': self.use_readout_synapses,
            'n_hd_channels': self.n_hd_channels,
            'dt': self.dt,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            'hd_input_mode': self.hd_input_mode,
            'hd_connection_mode': self.hd_connection_mode
        }

        threshold_stats = self.neurons.get_threshold_statistics()
        info.update({f'threshold_{k}': v for k, v in threshold_stats.items()})

        weight_stats = self.recurrent_synapses.get_weight_statistics()
        info.update({f'recurrent_weight_{k}': v for k, v in weight_stats.items()})

        if self.use_readout_synapses:
            readout_stats = self.readout_synapses.get_weight_statistics()
            info.update({f'readout_synapse_{k}': v for k, v in readout_stats.items()})

        hd_info = self.hd_input.get_connectivity_info()
        info.update({f'hd_input_{k}': v for k, v in hd_info.items()})

        return info
