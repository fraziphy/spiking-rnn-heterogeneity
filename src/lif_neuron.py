# lif_neuron.py
"""
Leaky Integrate-and-Fire (LIF) neuron model with heterogeneous spike thresholds.
"""

import numpy as np
from typing import Tuple, List
from rng_utils import get_rng

class LIFNeuron:
    """LIF neuron with biological parameters."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """
        Initialize LIF neuron population.

        Args:
            n_neurons: Number of neurons
            dt: Time step in ms
        """
        self.n_neurons = n_neurons
        self.dt = dt

        # Biological parameters
        self.tau_m = 20.0  # Membrane time constant (ms)
        self.v_rest = -70.0  # Resting potential (mV)
        self.v_reset = -80.0  # Reset potential (mV)
        self.tau_ref = 2.0  # Refractory period (ms)

        # State variables
        self.v_membrane = None
        self.spike_thresholds = None
        self.refractory_timer = None
        self.last_spike_time = None

    def initialize_parameters(self, session_id: int, block_id: int,
                            v_th_mean: float = -55.0, v_th_std: float = 0.1):
        """
        Initialize spike thresholds with heterogeneity.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            v_th_mean: Mean spike threshold (mV)
            v_th_std: Standard deviation of spike thresholds
        """
        rng = get_rng(session_id, block_id, 0, 'spike_threshold')

        # Generate heterogeneous spike thresholds
        self.spike_thresholds = rng.normal(
            v_th_mean, v_th_std, self.n_neurons
        )

    def initialize_state(self, session_id: int, block_id: int, trial_id: int):
        """
        Initialize neuron states.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            trial_id: Trial ID for RNG
        """
        rng = get_rng(session_id, block_id, trial_id, 'initial_state')

        # Initialize membrane potentials near resting potential
        self.v_membrane = rng.uniform(
            self.v_reset, self.v_rest, self.n_neurons
        )

        # Initialize refractory timers
        self.refractory_timer = np.zeros(self.n_neurons)
        self.last_spike_time = -np.inf * np.ones(self.n_neurons)

    def update(self, t: float, synaptic_input: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Update neuron states for one time step.

        Args:
            t: Current time (ms)
            synaptic_input: Synaptic input current for each neuron

        Returns:
            Tuple of (membrane_potentials, spike_indices)
        """
        # Update refractory timer
        self.refractory_timer = np.maximum(
            0, self.refractory_timer - self.dt
        )

        # Only update neurons not in refractory period
        active_mask = self.refractory_timer <= 0

        # Membrane dynamics (exponential decay + input)
        dv_dt = (self.v_rest - self.v_membrane) / self.tau_m + synaptic_input
        self.v_membrane[active_mask] += dv_dt[active_mask] * self.dt

        # Check for spikes
        spike_mask = (self.v_membrane >= self.spike_thresholds) & active_mask
        spike_indices = np.where(spike_mask)[0].tolist()

        # Reset spiking neurons
        if len(spike_indices) > 0:
            self.v_membrane[spike_indices] = self.v_reset
            self.refractory_timer[spike_indices] = self.tau_ref
            self.last_spike_time[spike_indices] = t

        return self.v_membrane.copy(), spike_indices

    def inject_auxiliary_spike(self, neuron_id: int, t: float):
        """
        Inject auxiliary spike for perturbation analysis.

        Args:
            neuron_id: ID of neuron to spike
            t: Current time
        """
        self.v_membrane[neuron_id] = self.v_reset
        self.refractory_timer[neuron_id] = self.tau_ref
        self.last_spike_time[neuron_id] = t
