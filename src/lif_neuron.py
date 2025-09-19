# src/lif_neuron.py - Updated with fixed structure and multiplier scaling
"""
Leaky Integrate-and-Fire neuron model with fixed base distributions and scaling.
"""

import numpy as np
from typing import Tuple, List, Dict
from rng_utils import get_rng, generate_base_distributions

class LIFNeuron:
    """LIF neuron with fixed base structure and heterogeneity scaling."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """Initialize LIF neuron population."""
        self.n_neurons = n_neurons
        self.dt = dt

        # Biological parameters
        self.tau_m = 20.0  # Membrane time constant (ms)
        self.v_rest = -70.0  # Resting potential (mV)
        self.v_reset = -80.0  # Reset potential (mV)
        self.tau_ref = 2.0  # Refractory period (ms)

        # Base distributions and actual parameters
        self.base_distributions = None
        self.spike_thresholds = None
        self.v_th_multiplier = None

        # State variables
        self.v_membrane = None
        self.refractory_timer = None
        self.last_spike_time = None

    def initialize_base_distributions(self, session_id: int):
        """Initialize base distributions that remain fixed across parameter combinations."""
        if self.base_distributions is None:
            self.base_distributions = generate_base_distributions(
                session_id=session_id,
                n_neurons=self.n_neurons
            )

    def initialize_parameters(self, session_id: int, block_id: int,
                            v_th_mean: float = -55.0,
                            v_th_multiplier: float = 1.0):
        """
        Initialize spike thresholds using base distribution and multiplier.

        Args:
            session_id: Session ID for base distributions
            block_id: Block ID (not used for structure)
            v_th_mean: Target mean (should be -55.0, verified)
            v_th_multiplier: Multiplier for base heterogeneity (1-100)
        """
        # Ensure base distributions exist
        self.initialize_base_distributions(session_id)

        # Verify mean is exactly -55.0 (critical for proper scaling)
        if abs(v_th_mean - (-55.0)) > 1e-10:
            raise ValueError(f"v_th_mean must be exactly -55.0, got {v_th_mean}")

        # Store multiplier for reference
        self.v_th_multiplier = v_th_multiplier

        # Scale base thresholds: v_th = -55.0 + (base_v_th - (-55.0)) * multiplier
        # This preserves the mean at -55.0 while scaling the heterogeneity
        base_v_th = self.base_distributions['base_v_th']

        # Apply scaling: mean stays -55, heterogeneity scales by multiplier
        self.spike_thresholds = v_th_mean + (base_v_th - v_th_mean) * v_th_multiplier

        # Verification (critical check)
        actual_mean = np.mean(self.spike_thresholds)
        if abs(actual_mean - v_th_mean) > 1e-10:
            raise RuntimeError(f"Mean preservation failed: expected {v_th_mean}, got {actual_mean}")

    def initialize_state(self, session_id: int, block_id: int, trial_id: int):
        """Initialize neuron states (varies with trial_id)."""
        rng = get_rng(session_id, block_id, trial_id, 'initial_state')

        # Initialize membrane potentials near resting potential
        self.v_membrane = rng.uniform(
            self.v_reset, self.v_rest, self.n_neurons
        )

        # Initialize refractory timers
        self.refractory_timer = np.zeros(self.n_neurons)
        self.last_spike_time = -np.inf * np.ones(self.n_neurons)

    def update(self, t: float, synaptic_input: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Update neuron states for one time step."""
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
        """Inject auxiliary spike for perturbation analysis."""
        self.v_membrane[neuron_id] = self.v_reset
        self.refractory_timer[neuron_id] = self.tau_ref
        self.last_spike_time[neuron_id] = t

    def get_threshold_statistics(self) -> Dict[str, float]:
        """Get statistics about current threshold configuration."""
        if self.spike_thresholds is None:
            return {'error': 'Thresholds not initialized'}

        return {
            'mean': float(np.mean(self.spike_thresholds)),
            'std': float(np.std(self.spike_thresholds)),
            'min': float(np.min(self.spike_thresholds)),
            'max': float(np.max(self.spike_thresholds)),
            'multiplier': float(self.v_th_multiplier) if self.v_th_multiplier else 0.0,
            'base_std': float(np.std(self.base_distributions['base_v_th'])) if self.base_distributions else 0.0
        }
