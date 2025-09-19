# src/synaptic_model.py - Updated with fixed structure and multiplier scaling
"""
Synaptic model with fixed base structure and heterogeneity scaling.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import sparse
from rng_utils import get_rng, generate_base_distributions

class ExponentialSynapses:
    """Exponential decay synapses with fixed base structure and scaling."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """Initialize synaptic model."""
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0  # Synaptic time constant (ms)

        # Base distributions and current parameters
        self.base_distributions = None
        self.weight_matrix = None
        self.g_multiplier = None
        self.synaptic_current = None

    def initialize_base_distributions(self, session_id: int):
        """Initialize base distributions that remain fixed across parameter combinations."""
        if self.base_distributions is None:
            self.base_distributions = generate_base_distributions(
                session_id=session_id,
                n_neurons=self.n_neurons
            )

    def initialize_weights(self, session_id: int, block_id: int,
                        g_mean: float = 0.0, g_multiplier: float = 1.0,
                        connection_prob: float = 0.1):
        """Initialize synaptic weights using base distributions and multiplier."""

        # Ensure base distributions exist
        self.initialize_base_distributions(session_id)

        # Verify mean is exactly 0.0
        if abs(g_mean - 0.0) > 1e-10:
            raise ValueError(f"g_mean must be exactly 0.0, got {g_mean}")

        # Store multiplier for reference
        self.g_multiplier = g_multiplier

        # Get connectivity from base distributions
        connectivity = self.base_distributions['connectivity']

        # Get base weights only for connected positions
        base_g = self.base_distributions['base_g']
        connected_indices = np.where(connectivity)
        base_connected_weights = base_g[connected_indices]

        # Apply multiplier to the base connected weights (scaling step)
        scaled_connected_weights = base_connected_weights * g_multiplier

        # THEN center them to ensure exact mean = g_mean (0.0)
        if len(scaled_connected_weights) > 0:
            scaled_connected_weights = scaled_connected_weights - np.mean(scaled_connected_weights) + g_mean

        # Create sparse matrix directly from connected weights
        rows, cols = connected_indices
        self.weight_matrix = sparse.csr_matrix(
            (scaled_connected_weights, (rows, cols)),
            shape=(self.n_neurons, self.n_neurons)
        )

        # Verify mean preservation
        if len(scaled_connected_weights) > 0:
            actual_mean = np.mean(scaled_connected_weights)
            if abs(actual_mean - g_mean) > 1e-10:
                print(f"Warning: Connected weights mean {actual_mean:.10f}, expected {g_mean}")

        # Initialize synaptic current
        self.synaptic_current = np.zeros(self.n_neurons)

    def update(self, spike_indices: List[int]) -> np.ndarray:
        """Update synaptic currents."""
        # Exponential decay
        self.synaptic_current *= np.exp(-self.dt / self.tau_syn)

        # Add contribution from new spikes
        if len(spike_indices) > 0:
            # Sum contributions from all spiking neurons
            spike_contribution = self.weight_matrix[:, spike_indices].sum(axis=1).A1
            self.synaptic_current += spike_contribution

        return self.synaptic_current.copy()

    def get_weight_statistics(self) -> Dict[str, float]:
        """Get statistics about current weight configuration."""
        if self.weight_matrix is None:
            return {'error': 'Weights not initialized'}

        # Get non-zero weights
        weight_data = self.weight_matrix.data

        if len(weight_data) == 0:
            return {'error': 'No connections'}

        return {
            'mean': float(np.mean(weight_data)),
            'std': float(np.std(weight_data)),
            'min': float(np.min(weight_data)),
            'max': float(np.max(weight_data)),
            'n_connections': len(weight_data),
            'connection_density': len(weight_data) / (self.n_neurons ** 2),
            'multiplier': float(self.g_multiplier) if self.g_multiplier else 0.0,
            'base_std': float(np.std(self.base_distributions['base_g'])) if self.base_distributions else 0.0
        }

class StaticPoissonInput:
    """Static Poisson process input - unchanged from original."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0
        self.input_strength = None
        self.input_current = None

    def initialize_parameters(self, input_strength: float = 1.0):
        self.input_strength = input_strength
        self.input_current = np.zeros(self.n_neurons)

    def update(self, session_id: int, block_id: int, trial_id: int,
               rate: float = 10.0) -> np.ndarray:
        """Update with trial-dependent Poisson process."""
        # Exponential decay
        self.input_current *= np.exp(-self.dt / self.tau_syn)

        # Generate independent Poisson spikes for each neuron
        if rate > 0:
            rng = get_rng(session_id, block_id, trial_id, 'static_poisson')

            spike_prob = rate * (self.dt / 1000.0)
            spike_mask = rng.random(self.n_neurons) < spike_prob
            self.input_current[spike_mask] += self.input_strength

        return self.input_current.copy()

class DynamicPoissonInput:
    """Dynamic Poisson input with fixed connectivity structure."""

    def __init__(self, n_neurons: int, n_channels: int = 20, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.n_channels = n_channels
        self.dt = dt
        self.tau_syn = 5.0

        # Fixed structure
        self.base_distributions = None
        self.connectivity_matrix = None
        self.input_strength = None
        self.input_current = None

    def initialize_base_distributions(self, session_id: int):
        """Initialize base distributions for fixed connectivity."""
        if self.base_distributions is None:
            self.base_distributions = generate_base_distributions(
                session_id=session_id,
                n_neurons=self.n_neurons,
                n_input_channels=self.n_channels
            )

    def initialize_connectivity(self, session_id: int, block_id: int,
                              connection_prob: float = 0.3,
                              input_strength: float = 1.0):
        """Initialize input connectivity using fixed base structure."""
        # Ensure base distributions exist
        self.initialize_base_distributions(session_id)

        # Use fixed connectivity from base distributions
        self.connectivity_matrix = self.base_distributions['dynamic_connectivity']

        # Verify connection probability matches approximately
        actual_conn_prob = np.mean(self.connectivity_matrix)
        if abs(actual_conn_prob - connection_prob) > 0.05:
            print(f"Warning: Expected dynamic conn_prob {connection_prob}, got {actual_conn_prob}")

        # Set input strength
        self.input_strength = input_strength
        self.input_current = np.zeros(self.n_neurons)

    def update(self, session_id: int, block_id: int, trial_id: int,
               rates: np.ndarray) -> np.ndarray:
        """Update with trial-dependent spike generation."""
        # Exponential decay
        self.input_current *= np.exp(-self.dt / self.tau_syn)

        # Generate spikes for each channel (trial-dependent)
        if len(rates) > 0:
            rng = get_rng(session_id, block_id, trial_id, 'dynamic_poisson_spikes')

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
    """Readout layer - unchanged from original."""

    def __init__(self, n_rnn_neurons: int, n_readout_neurons: int = 10, dt: float = 0.1):
        self.n_rnn_neurons = n_rnn_neurons
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt
        self.tau_readout = 20.0
        self.readout_weights = None
        self.readout_activity = None

    def initialize_weights(self, session_id: int, block_id: int,
                          weight_scale: float = 1.0):
        """Initialize readout weights (could be made session-only if desired)."""
        rng = get_rng(session_id, block_id, 0, 'readout_weights')

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
