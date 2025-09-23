# src/lif_neuron.py - Fixed with mean centering and clean structure
"""
Leaky Integrate-and-Fire neuron model with direct heterogeneity and exact mean preservation.
"""

import numpy as np
from typing import Tuple, List, Dict
from rng_utils import get_rng

class LIFNeuron:
    """LIF neuron with direct heterogeneity and exact mean preservation."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """Initialize LIF neuron population."""
        self.n_neurons = n_neurons
        self.dt = dt

        # Biological parameters
        self.tau_m = 20.0  # Membrane time constant (ms)
        self.v_rest = -70.0  # Resting potential (mV)
        self.v_reset = -80.0  # Reset potential (mV)
        self.tau_ref = 2.0  # Refractory period (ms)

        # Parameters and state variables
        self.spike_thresholds = None
        self.v_th_std = None
        self.v_th_distribution = None

        # State variables
        self.v_membrane = None
        self.refractory_timer = None
        self.last_spike_time = None

    def initialize_parameters(self, session_id: int, v_th_std: float, trial_id: int,
                            v_th_mean: float = -55.0,
                            v_th_distribution: str = "normal"):
        """
        Initialize spike thresholds with direct heterogeneity and exact mean preservation.

        Args:
            session_id: Session ID for reproducibility
            v_th_std: Direct standard deviation for thresholds
            trial_id: Trial ID (not used for structure)
            v_th_mean: Target mean threshold (default: -55.0)
            v_th_distribution: "normal" or "uniform"
        """
        # Store parameters
        self.v_th_std = v_th_std
        self.v_th_distribution = v_th_distribution

        # Get RNG (structure depends on session + std, not trial)
        rng = get_rng(session_id, v_th_std, 0.0, 0, 'spike_thresholds')

        # Generate thresholds based on distribution
        if v_th_std == 0.0:
            # Homogeneous case
            self.spike_thresholds = np.full(self.n_neurons, v_th_mean)
        elif v_th_distribution == "normal":
            # Normal distribution
            self.spike_thresholds = rng.normal(v_th_mean, v_th_std, self.n_neurons)
            # Center to exact mean with iterative correction for maximum precision
            for _ in range(2):  # Two iterations should be enough
                current_mean = np.mean(self.spike_thresholds, dtype=np.float64)
                self.spike_thresholds = self.spike_thresholds - (current_mean - v_th_mean)

        elif v_th_distribution == "uniform":
            # Uniform distribution with same std as normal
            half_width = v_th_std * np.sqrt(12) / 2
            a = v_th_mean - half_width
            b = v_th_mean + half_width
            self.spike_thresholds = rng.uniform(a, b, self.n_neurons)
            # Center to exact mean with iterative correction
            for _ in range(2):
                current_mean = np.mean(self.spike_thresholds, dtype=np.float64)
                self.spike_thresholds = self.spike_thresholds - (current_mean - v_th_mean)
        else:
            raise ValueError(f"Unknown distribution: {v_th_distribution}. Use 'normal' or 'uniform'")

        # Ensure thresholds are above reset potential
        eps = 1e-9
        self.spike_thresholds = np.clip(self.spike_thresholds, self.v_reset + eps, None)

        # Verify mean preservation (for debugging)
        actual_mean = np.mean(self.spike_thresholds)
        if abs(actual_mean - v_th_mean) > 1e-10:
            print(f"Warning: Threshold mean not preserved: {actual_mean:.12f} vs {v_th_mean}")

    def initialize_state(self, session_id: int, v_th_std: float, g_std: float, trial_id: int):
        """Initialize neuron states (varies with trial_id)."""
        rng = get_rng(session_id, v_th_std, g_std, trial_id, 'initial_state')

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
            'distribution': self.v_th_distribution,
            'target_std': float(self.v_th_std) if self.v_th_std else 0.0,
            'target_mean': -55.0
        }
