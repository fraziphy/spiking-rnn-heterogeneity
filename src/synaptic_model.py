# src/synaptic_model.py - Fixed with mean centering and clean structure
"""
Synaptic model with immediate vs dynamic modes, mean centering, and normalized synaptic impact.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import sparse
from rng_utils import get_rng

class ExponentialSynapses:
    """Synapses with immediate vs dynamic modes, mean centering, and impact normalization."""

    def __init__(self, n_neurons: int, dt: float = 0.1, synaptic_mode: str = "dynamic"):
        """Initialize synaptic model."""
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0  # Synaptic time constant (ms)

        # Synaptic mode: "immediate" or "dynamic"
        self.synaptic_mode = synaptic_mode

        # Parameters and state
        self.weight_matrix = None
        self.g_std = None
        self.synaptic_current = None

    def initialize_weights(self, session_id: int, v_th_std: float, g_std: float,
                          g_mean: float = 0.0, connection_prob: float = 0.1):
        """Initialize synaptic weights with direct heterogeneity and exact mean preservation."""

        # Store parameters
        self.g_std = g_std

        # Get RNGs for structure (depends on session + parameters)
        weight_rng = get_rng(session_id, v_th_std, g_std, 0, 'synaptic_weights')
        conn_rng = get_rng(session_id, v_th_std, g_std, 0, 'connectivity')

        # Generate connectivity pattern
        connectivity = conn_rng.random((self.n_neurons, self.n_neurons)) < connection_prob
        np.fill_diagonal(connectivity, False)

        # Generate weights for connected synapses
        n_connections = np.sum(connectivity)
        if n_connections > 0:
            if g_std == 0.0:
                # Homogeneous weights
                weights = np.full(n_connections, g_mean)
            else:
                # Heterogeneous weights with exact mean centering
                weights = weight_rng.normal(g_mean, g_std, n_connections)
                weights = weights - np.mean(weights) + g_mean  # Force exact mean

            # Apply impact normalization for fair comparison
            if self.synaptic_mode == "immediate":
                # Scale up immediate synapses to match total dynamic impact
                normalization_factor = self.tau_syn / self.dt
                weights = weights * normalization_factor

            # Create sparse matrix
            rows, cols = np.where(connectivity)
            self.weight_matrix = sparse.csr_matrix(
                (weights, (rows, cols)),
                shape=(self.n_neurons, self.n_neurons)
            )

            # Verify mean preservation (for debugging)
            if n_connections > 1:
                if self.synaptic_mode == "immediate":
                    effective_weights = weights / normalization_factor
                    actual_mean = np.mean(effective_weights)
                else:
                    actual_mean = np.mean(weights)

                if abs(actual_mean - g_mean) > 1e-8:
                    print(f"Warning: Weight mean not preserved: {actual_mean:.10f} vs {g_mean}")
        else:
            # No connections
            self.weight_matrix = sparse.csr_matrix((self.n_neurons, self.n_neurons))

        # Initialize synaptic current
        self.synaptic_current = np.zeros(self.n_neurons)

    def update(self, spike_indices: List[int]) -> np.ndarray:
        """Update synaptic currents with mode-dependent dynamics."""

        if self.synaptic_mode == "dynamic":
            # Dynamic synapses with exponential decay
            self.synaptic_current *= np.exp(-self.dt / self.tau_syn)

            # Add contribution from new spikes
            if len(spike_indices) > 0:
                spike_contribution = self.weight_matrix[:, spike_indices].sum(axis=1).A1
                self.synaptic_current += spike_contribution

        elif self.synaptic_mode == "immediate":
            # Immediate synapses (like Set 2)
            self.synaptic_current.fill(0.0)  # Reset to zero each timestep

            # Add immediate contribution from current spikes
            if len(spike_indices) > 0:
                spike_contribution = self.weight_matrix[:, spike_indices].sum(axis=1).A1
                self.synaptic_current = spike_contribution
        else:
            raise ValueError(f"Unknown synaptic mode: {self.synaptic_mode}")

        return self.synaptic_current.copy()

    def get_weight_statistics(self) -> Dict[str, float]:
        """Get statistics about current weight configuration."""
        if self.weight_matrix is None:
            return {'error': 'Weights not initialized'}

        # Get non-zero weights
        weight_data = self.weight_matrix.data

        if len(weight_data) == 0:
            return {'error': 'No connections'}

        # Compute effective weights (accounting for normalization)
        if self.synaptic_mode == "immediate":
            effective_weights = weight_data / (self.tau_syn / self.dt)
        else:
            effective_weights = weight_data

        return {
            'mean': float(np.mean(weight_data)),
            'std': float(np.std(weight_data)),
            'effective_mean': float(np.mean(effective_weights)),
            'effective_std': float(np.std(effective_weights)),
            'min': float(np.min(weight_data)),
            'max': float(np.max(weight_data)),
            'n_connections': len(weight_data),
            'connection_density': len(weight_data) / (self.n_neurons ** 2),
            'target_std': float(self.g_std) if self.g_std else 0.0,
            'target_mean': 0.0,
            'synaptic_mode': self.synaptic_mode,
            'normalization_factor': float(self.tau_syn / self.dt) if self.synaptic_mode == "immediate" else 1.0
        }

class StaticPoissonInput:
    """Static Poisson process input."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0
        self.input_strength = None
        self.input_current = None

    def initialize_parameters(self, input_strength: float = 1.0):
        self.input_strength = input_strength
        self.input_current = np.zeros(self.n_neurons)

    def update(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
               rate: float = 10.0) -> np.ndarray:
        """Update with trial-dependent Poisson process."""
        # Exponential decay
        self.input_current *= np.exp(-self.dt / self.tau_syn)

        # Generate independent Poisson spikes for each neuron
        if rate > 0:
            rng = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson')

            spike_prob = rate * (self.dt / 1000.0)
            spike_mask = rng.random(self.n_neurons) < spike_prob
            self.input_current[spike_mask] += self.input_strength

        return self.input_current.copy()

class DynamicPoissonInput:
    """Dynamic Poisson input with parameter-dependent connectivity."""

    def __init__(self, n_neurons: int, n_channels: int = 20, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.n_channels = n_channels
        self.dt = dt
        self.tau_syn = 5.0

        # Structure that varies with parameters
        self.connectivity_matrix = None
        self.input_strength = None
        self.input_current = None

    def initialize_connectivity(self, session_id: int, v_th_std: float, g_std: float,
                              connection_prob: float = 0.3,
                              input_strength: float = 1.0):
        """Initialize input connectivity based on parameters."""
        # Get parameter-dependent connectivity
        rng = get_rng(session_id, v_th_std, g_std, 0, 'dynamic_input_connectivity')

        # Generate connectivity matrix
        self.connectivity_matrix = rng.random((self.n_neurons, self.n_channels)) < connection_prob

        # Set input strength
        self.input_strength = input_strength
        self.input_current = np.zeros(self.n_neurons)

    def update(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
               rates: np.ndarray) -> np.ndarray:
        """Update with trial-dependent spike generation."""
        # Exponential decay
        self.input_current *= np.exp(-self.dt / self.tau_syn)

        # Generate spikes for each channel (trial-dependent)
        if len(rates) > 0:
            rng = get_rng(session_id, v_th_std, g_std, trial_id, 'dynamic_poisson_spikes')

            spike_probs = rates * (self.dt / 1000.0)
            channel_spikes = rng.random(self.n_channels) < spike_probs

            if np.any(channel_spikes):
                spiking_channels = np.where(channel_spikes)[0]
                input_contribution = np.sum(
                    self.connectivity_matrix[:, spiking_channels], axis=1
                ) * self.input_strength
                self.input_current += input_contribution

        return self.input_current.copy()

    def get_connectivity_info(self) -> Dict[str, Any]:
        """Get connectivity information."""
        if self.connectivity_matrix is not None:
            n_connections_per_channel = np.sum(self.connectivity_matrix, axis=0)
            return {
                'n_channels': self.n_channels,
                'connections_per_channel_mean': float(np.mean(n_connections_per_channel)),
                'connections_per_channel_std': float(np.std(n_connections_per_channel)),
                'total_connections': int(np.sum(self.connectivity_matrix)),
                'connection_density': float(np.mean(self.connectivity_matrix))
            }
        else:
            return {}

class ReadoutLayer:
    """Readout layer with parameter-dependent weights."""

    def __init__(self, n_rnn_neurons: int, n_readout_neurons: int = 10, dt: float = 0.1):
        self.n_rnn_neurons = n_rnn_neurons
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt
        self.tau_readout = 20.0
        self.readout_weights = None
        self.readout_activity = None

    def initialize_weights(self, session_id: int, v_th_std: float, g_std: float,
                          weight_scale: float = 1.0):
        """Initialize readout weights based on parameters."""
        rng = get_rng(session_id, v_th_std, g_std, 0, 'readout_weights')

        self.readout_weights = rng.normal(
            0.0, weight_scale / np.sqrt(self.n_rnn_neurons),
            (self.n_readout_neurons, self.n_rnn_neurons)
        )
        self.readout_activity = np.zeros(self.n_readout_neurons)

    def update(self, rnn_spike_indices: List[int]) -> np.ndarray:
        """Update readout activity."""
        self.readout_activity *= np.exp(-self.dt / self.tau_readout)

        if len(rnn_spike_indices) > 0:
            spike_contribution = np.sum(
                self.readout_weights[:, rnn_spike_indices], axis=1
            )
            self.readout_activity += spike_contribution

        return self.readout_activity.copy()

    def get_output(self) -> np.ndarray:
        """Get current readout output."""
        return self.readout_activity.copy()
