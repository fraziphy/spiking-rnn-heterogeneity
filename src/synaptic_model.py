# src/synaptic_model.py - Fixed: Removed ReadoutLayer (synapses handle filtering)
"""
Synaptic model with pulse vs filter modes.
Input classes generate spike events or tonic values WITHOUT filtering.
Synapse class applies the filtering based on mode.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import sparse
from .rng_utils import get_rng

class Synapse:
    """Synapses with pulse vs filter modes. Can be used for recurrent, input, or readout connections."""

    def __init__(self, n_neurons: int, dt: float = 0.1, synaptic_mode: str = "filter"):
        """Initialize synaptic model."""
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0  # Synaptic time constant (ms)

        # Synaptic mode: "pulse" or "filter"
        if synaptic_mode not in ["pulse", "filter"]:
            raise ValueError(f"synaptic_mode must be 'pulse' or 'filter', got '{synaptic_mode}'")
        self.synaptic_mode = synaptic_mode

        # Parameters and state
        self.weight_matrix = None
        self.g_std = None
        self.synaptic_current = None

    def initialize_weights(self, session_id: int, v_th_std: float, g_std: float,
                           g_mean: float = 0.0, connection_prob: float = 0.1,
                           n_source_neurons: int = None):
        """Initialize synaptic weights with direct heterogeneity and exact mean preservation."""

        # Store parameters
        self.g_std = g_std

        # Get RNGs for structure (depends on session + parameters)
        weight_rng = get_rng(session_id, v_th_std, g_std, 0, 'synaptic_weights')
        conn_rng = get_rng(session_id, v_th_std, g_std, 0, 'connectivity')

        # Determine source and target neuron counts
        n_source = n_source_neurons if n_source_neurons is not None else self.n_neurons
        n_target = self.n_neurons

        # Generate connectivity pattern
        connectivity = conn_rng.random((n_target, n_source)) < connection_prob
        if n_source == n_target:
            np.fill_diagonal(connectivity, False)  # Only for square matrices

        # Generate weights for connected synapses
        n_connections = np.sum(connectivity)
        if n_connections > 0:
            if g_std == 0.0:
                # Homogeneous weights
                weights = np.full(n_connections, g_mean)
            else:
                # Heterogeneous weights with exact mean centering
                scale = 1 / (self.n_neurons * connection_prob)
                weights = weight_rng.normal(g_mean, scale * g_std, n_connections)
                weights = weights - np.mean(weights) + g_mean  # Force exact mean

            # Apply impact normalization for fair comparison
            if self.synaptic_mode == "pulse":
                # Scale up pulse synapses to match total filter impact
                normalization_factor = self.tau_syn / self.dt
                weights = weights * normalization_factor

            # Create sparse matrix
            rows, cols = np.where(connectivity)
            self.weight_matrix = sparse.csr_matrix(
                (weights, (rows, cols)),
                shape=(n_target, n_source)
            )
        else:
            # No connections
            self.weight_matrix = sparse.csr_matrix((self.n_neurons, self.n_neurons))

        # Initialize synaptic current
        self.synaptic_current = np.zeros(self.n_neurons)

    def update(self, spike_indices: List[int]) -> np.ndarray:
        """Update synaptic currents with mode-dependent dynamics."""

        if self.synaptic_mode == "filter":
            # Filter synapses with exponential decay
            self.synaptic_current *= np.exp(-self.dt / self.tau_syn)

            # Add contribution from new spikes
            if len(spike_indices) > 0:
                spike_contribution = self.weight_matrix[:, spike_indices].sum(axis=1).A1
                self.synaptic_current += spike_contribution

        elif self.synaptic_mode == "pulse":
            # Pulse synapses
            self.synaptic_current.fill(0.0)  # Reset to zero each timestep

            # Add immediate contribution from current spikes
            if len(spike_indices) > 0:
                spike_contribution = self.weight_matrix[:, spike_indices].sum(axis=1).A1
                self.synaptic_current = spike_contribution

        return self.synaptic_current.copy()

    def apply_to_input(self, input_events: np.ndarray) -> np.ndarray:
        """
        Apply synaptic filtering to input events.

        Args:
            input_events: Array of input values (spike events or tonic input)

        Returns:
            Filtered synaptic current
        """
        # Initialize if needed (for input synapses that don't call initialize_weights)
        if self.synaptic_current is None:
            self.synaptic_current = np.zeros(self.n_neurons)

        if self.synaptic_mode == "filter":
            # Filter: exponential decay + add new input
            self.synaptic_current *= np.exp(-self.dt / self.tau_syn)
            self.synaptic_current += input_events

        elif self.synaptic_mode == "pulse":
            # Pulse: replace with new input (no accumulation)
            # For pulse mode, input needs same normalization as recurrent
            normalization_factor = self.tau_syn / self.dt
            self.synaptic_current = input_events * normalization_factor

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
        if self.synaptic_mode == "pulse":
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
            'normalization_factor': float(self.tau_syn / self.dt) if self.synaptic_mode == "pulse" else 1.0
        }


class StaticPoissonInput:
    """
    Static Poisson process input with three modes.
    Generates spike events or tonic values WITHOUT synaptic filtering.
    """

    def __init__(self, n_neurons: int, dt: float = 0.1, static_input_mode: str = "independent"):
        self.n_neurons = n_neurons
        self.dt = dt
        self.input_strength = None

        # Three modes: independent, common_stochastic, common_tonic
        if static_input_mode not in ['independent', 'common_stochastic', 'common_tonic']:
            raise ValueError(f"static_input_mode must be 'independent', 'common_stochastic', or 'common_tonic', got '{static_input_mode}'")
        self.static_input_mode = static_input_mode

    def initialize_parameters(self, input_strength: float = 1.0):
        """Initialize input strength."""
        self.input_strength = input_strength

    def generate_events(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                       rate: float = 10.0, time_step: int = 0) -> np.ndarray:
        """
        Generate input events (spikes or tonic values) WITHOUT synaptic filtering.

        Returns:
            Array of input events (to be filtered by synapse)
        """
        events = np.zeros(self.n_neurons)

        if rate > 0:
            spike_prob = rate * (self.dt / 1000.0)

            if self.static_input_mode == 'independent':
                # Independent stochastic: Each neuron gets independent Poisson spikes
                rng = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson', time_step, rate)
                spike_mask = rng.random(self.n_neurons) < spike_prob
                events[spike_mask] = self.input_strength

            elif self.static_input_mode == 'common_stochastic':
                # Common stochastic: All neurons receive identical Poisson spikes
                rng = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson', time_step, rate)
                single_spike = rng.random() < spike_prob
                if single_spike:
                    events[:] = self.input_strength  # All neurons get same spike

            elif self.static_input_mode == 'common_tonic':
                # Common tonic: Deterministic fractional input (expected value, zero variance)
                events[:] = self.input_strength * spike_prob  # All neurons get constant fractional input

        return events


class HDDynamicInput:
    """
    High-dimensional dynamic input for encoding experiments.
    Each of k channels connects to random 30% of RNN neurons.
    Supports three modes: independent, common_stochastic, common_tonic.
    """

    def __init__(self, n_neurons: int, n_channels: int = 10, dt: float = 0.1,
                 hd_input_mode: str = "independent"):
        """
        Initialize HD dynamic input.

        Args:
            n_neurons: Number of RNN neurons
            n_channels: Number of HD input channels (embedding dimensionality k)
            dt: Time step (ms)
            hd_input_mode: "independent", "common_stochastic", or "common_tonic"
        """
        self.n_neurons = n_neurons
        self.n_channels = n_channels
        self.dt = dt

        if hd_input_mode not in ['independent', 'common_stochastic', 'common_tonic']:
            raise ValueError(f"hd_input_mode must be 'independent', 'common_stochastic', or 'common_tonic', got '{hd_input_mode}'")
        self.hd_input_mode = hd_input_mode

        # Connectivity matrix: which neurons receive which channels
        self.connectivity_matrix = None  # shape (n_neurons, n_channels), boolean
        self.input_strength = None

    def initialize_connectivity(self, session_id: int, hd_dim: int, embed_dim: int,
                               connection_prob: float = 0.3,
                               input_strength: float = 1.0):
        """
        Initialize connectivity: each channel connects to random 30% of neurons.
        Fixed per session/hd_dim/embed_dim combination.

        Args:
            session_id: Session ID
            hd_dim: Intrinsic dimensionality (affects seed)
            embed_dim: Embedding dimensionality (affects seed)
            connection_prob: Connection probability per channel (default 0.3)
            input_strength: Input strength multiplier
        """
        # Get RNG for connectivity (fixed per session/hd_dim/embed_dim)
        rng = get_rng(session_id, 0.0, 0.0, 0, 'hd_input_connectivity',
                     hd_dim=hd_dim, embed_dim=embed_dim)

        # Generate connectivity matrix: each column is a channel
        self.connectivity_matrix = rng.random((self.n_neurons, self.n_channels)) < connection_prob

        self.input_strength = input_strength

    def generate_events(self, session_id: int, v_th_std: float, g_std: float,
                       trial_id: int, hd_dim: int, embed_dim: int,
                       rates: np.ndarray, time_step: int = 0) -> np.ndarray:
        """
        Generate input events from HD rates for current timestep.

        Args:
            session_id: Session ID
            v_th_std: Threshold std
            g_std: Weight std
            trial_id: Trial ID
            hd_dim: Intrinsic dimensionality
            embed_dim: Embedding dimensionality
            rates: HD input rates for this timestep, shape (n_channels,)
            time_step: Current time step index

        Returns:
            events: Input events for each neuron, shape (n_neurons,)
        """
        events = np.zeros(self.n_neurons)

        if self.connectivity_matrix is None:
            raise ValueError("Must call initialize_connectivity() first")

        # Convert rates to spike probabilities
        spike_probs = rates * (self.dt / 1000.0)  # rates in Hz, dt in ms

        for channel_idx in range(self.n_channels):
            if spike_probs[channel_idx] <= 0:
                continue

            # Find neurons receiving this channel
            receiving_neurons = np.where(self.connectivity_matrix[:, channel_idx])[0]

            if len(receiving_neurons) == 0:
                continue

            if self.hd_input_mode == 'independent':
                # Each neuron gets independent Poisson sample
                rng_offset = get_rng(session_id, v_th_std, g_std, trial_id,
                                    f'hd_poisson_ch_{channel_idx}', time_step=time_step,
                                    hd_dim=hd_dim, embed_dim=embed_dim)

                spike_mask = rng_offset.random(len(receiving_neurons)) < spike_probs[channel_idx]
                events[receiving_neurons[spike_mask]] += self.input_strength

            elif self.hd_input_mode == 'common_stochastic':
                # All neurons receiving this channel get identical Poisson sample
                # But different across channels
                rng = get_rng(session_id, v_th_std, g_std, trial_id,
                             f'hd_poisson_common_ch_{channel_idx}', time_step=time_step,
                             hd_dim=hd_dim, embed_dim=embed_dim)

                single_spike = rng.random() < spike_probs[channel_idx]
                if single_spike:
                    events[receiving_neurons] += self.input_strength

            elif self.hd_input_mode == 'common_tonic':
                # All neurons get deterministic expected value (no Poisson)
                events[receiving_neurons] += self.input_strength * spike_probs[channel_idx]

        return events

    def get_connectivity_info(self) -> dict:
        """Get connectivity statistics."""
        if self.connectivity_matrix is None:
            return {'error': 'Connectivity not initialized'}

        # Neurons per channel
        neurons_per_channel = np.sum(self.connectivity_matrix, axis=0)

        # Channels per neuron
        channels_per_neuron = np.sum(self.connectivity_matrix, axis=1)

        # Overlap between channels (pairwise)
        n_overlaps = []
        for i in range(self.n_channels):
            for j in range(i+1, self.n_channels):
                overlap = np.sum(self.connectivity_matrix[:, i] & self.connectivity_matrix[:, j])
                n_overlaps.append(overlap)

        return {
            'n_channels': self.n_channels,
            'n_neurons': self.n_neurons,
            'hd_input_mode': self.hd_input_mode,
            'neurons_per_channel_mean': float(np.mean(neurons_per_channel)),
            'neurons_per_channel_std': float(np.std(neurons_per_channel)),
            'channels_per_neuron_mean': float(np.mean(channels_per_neuron)),
            'channels_per_neuron_std': float(np.std(channels_per_neuron)),
            'pairwise_overlap_mean': float(np.mean(n_overlaps)) if n_overlaps else 0.0,
            'pairwise_overlap_std': float(np.std(n_overlaps)) if n_overlaps else 0.0,
            'total_connections': int(np.sum(self.connectivity_matrix)),
            'connection_density': float(np.mean(self.connectivity_matrix))
        }
