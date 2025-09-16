# synaptic_model.py
"""
Synaptic model with exponential decay and different types of Poisson inputs.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import sparse
from rng_utils import get_rng

class ExponentialSynapses:
    """Exponential decay synapses with heterogeneous weights."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """
        Initialize synaptic model.

        Args:
            n_neurons: Number of neurons
            dt: Time step in ms
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0  # Synaptic time constant (ms)

        # Synaptic variables
        self.weight_matrix = None
        self.synaptic_current = None

    def initialize_weights(self, session_id: int, block_id: int,
                          g_mean: float = 0.0, g_std: float = 0.1,
                          connection_prob: float = 0.1):
        """
        Initialize synaptic weight matrix with heterogeneity.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            g_mean: Mean synaptic strength
            g_std: Standard deviation of synaptic strengths
            connection_prob: Connection probability
        """
        rng_weights = get_rng(session_id, block_id, 0, 'synaptic_weights')
        rng_conn = get_rng(session_id, block_id, 0, 'connectivity')

        # Create connectivity matrix
        conn_matrix = rng_conn.random((self.n_neurons, self.n_neurons)) < connection_prob

        # No self-connections
        np.fill_diagonal(conn_matrix, False)

        # Generate weights for existing connections
        n_connections = np.sum(conn_matrix)
        weights = rng_weights.normal(g_mean, g_std, n_connections)

        # Create sparse weight matrix
        self.weight_matrix = sparse.csr_matrix(
            (weights, np.where(conn_matrix)),
            shape=(self.n_neurons, self.n_neurons)
        )

        # Initialize synaptic current
        self.synaptic_current = np.zeros(self.n_neurons)

    def update(self, spike_indices: List[int]) -> np.ndarray:
        """
        Update synaptic currents.

        Args:
            spike_indices: List of neurons that spiked this timestep

        Returns:
            Synaptic input current for each neuron
        """
        # Exponential decay
        self.synaptic_current *= np.exp(-self.dt / self.tau_syn)

        # Add contribution from new spikes
        if len(spike_indices) > 0:
            # Sum contributions from all spiking neurons
            spike_contribution = self.weight_matrix[:, spike_indices].sum(axis=1).A1
            self.synaptic_current += spike_contribution

        return self.synaptic_current.copy()

class StaticPoissonInput:
    """
    Static Poisson process input - all neurons receive independent Poisson spikes
    with identical synaptic strength but independent random processes.
    """

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """
        Initialize static Poisson input.

        Args:
            n_neurons: Number of RNN neurons
            dt: Time step in ms
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_syn = 5.0  # Input synaptic time constant

        # Input parameters
        self.input_strength = None
        self.input_current = None

    def initialize_parameters(self, input_strength: float = 1.0):
        """
        Initialize static Poisson input parameters.

        Args:
            input_strength: Identical synaptic strength for all neurons
        """
        self.input_strength = input_strength
        self.input_current = np.zeros(self.n_neurons)

    def update(self, session_id: int, block_id: int, trial_id: int,
               rate: float = 10.0) -> np.ndarray:
        """
        Update input currents from static Poisson process.

        Args:
            session_id, block_id, trial_id: RNG parameters
            rate: Firing rate in Hz (same for all neurons)

        Returns:
            Input current for each neuron
        """
        # Exponential decay
        self.input_current *= np.exp(-self.dt / self.tau_syn)

        # Generate independent Poisson spikes for each neuron
        if rate > 0:
            rng = get_rng(session_id, block_id, trial_id, 'static_poisson')

            # Probability of spike in dt for each neuron
            spike_prob = rate * (self.dt / 1000.0)  # Convert dt to seconds

            # Generate spikes independently for each neuron
            spike_mask = rng.random(self.n_neurons) < spike_prob

            # Add input for neurons that received spikes (identical strength)
            self.input_current[spike_mask] += self.input_strength

        return self.input_current.copy()

class DynamicPoissonInput:
    """
    Dynamic Poisson process input with multiple channels for encoding studies.
    Each channel connects to 30% of neurons with identical synaptic strength.
    """

    def __init__(self, n_neurons: int, n_channels: int = 20, dt: float = 0.1):
        """
        Initialize dynamic Poisson input.

        Args:
            n_neurons: Number of RNN neurons
            n_channels: Number of input channels
            dt: Time step in ms
        """
        self.n_neurons = n_neurons
        self.n_channels = n_channels
        self.dt = dt
        self.tau_syn = 5.0  # Input synaptic time constant

        # Input connectivity and weights
        self.connectivity_matrix = None
        self.input_strength = None
        self.input_current = None

    def initialize_connectivity(self, session_id: int, block_id: int,
                              connection_prob: float = 0.3,
                              input_strength: float = 1.0):
        """
        Initialize input connectivity.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            connection_prob: Probability of connection from channel to neuron (30%)
            input_strength: Identical strength for all connections
        """
        rng = get_rng(session_id, block_id, 0, 'dynamic_poisson_connectivity')

        # Create connectivity matrix (neurons x channels)
        # Each channel connects to connection_prob fraction of neurons
        self.connectivity_matrix = rng.random((self.n_neurons, self.n_channels)) < connection_prob

        # All connections have identical weight
        self.input_strength = input_strength

        # Initialize input current
        self.input_current = np.zeros(self.n_neurons)

    def update(self, session_id: int, block_id: int, trial_id: int,
               rates: np.ndarray) -> np.ndarray:
        """
        Update input currents from dynamic Poisson channels.

        Args:
            session_id, block_id, trial_id: RNG parameters
            rates: Array of firing rates for each channel (Hz)

        Returns:
            Input current for each neuron
        """
        # Exponential decay
        self.input_current *= np.exp(-self.dt / self.tau_syn)

        # Generate spikes for each channel
        if len(rates) > 0:
            rng = get_rng(session_id, block_id, trial_id, 'dynamic_poisson_spikes')

            # Probability of spike in dt for each channel
            spike_probs = rates * (self.dt / 1000.0)  # Convert dt to seconds

            # Generate spikes for each channel
            channel_spikes = rng.random(self.n_channels) < spike_probs

            if np.any(channel_spikes):
                # Add input from spiking channels to connected neurons
                spiking_channels = np.where(channel_spikes)[0]

                # Sum contributions from all spiking channels (identical strength)
                input_contribution = np.sum(
                    self.connectivity_matrix[:, spiking_channels], axis=1
                ) * self.input_strength

                self.input_current += input_contribution

        return self.input_current.copy()

    def get_connectivity_info(self) -> dict:
        """
        Get connectivity information for analysis.

        Returns:
            Dictionary with connectivity statistics
        """
        if self.connectivity_matrix is not None:
            # Calculate overlap between channels
            n_connections_per_channel = np.sum(self.connectivity_matrix, axis=0)

            # Pairwise overlaps
            overlaps = []
            for i in range(self.n_channels):
                for j in range(i+1, self.n_channels):
                    overlap = np.sum(
                        self.connectivity_matrix[:, i] & self.connectivity_matrix[:, j]
                    )
                    total_i = n_connections_per_channel[i]
                    total_j = n_connections_per_channel[j]
                    if total_i > 0 and total_j > 0:
                        overlap_fraction = overlap / min(total_i, total_j)
                        overlaps.append(overlap_fraction)

            return {
                'n_channels': self.n_channels,
                'connections_per_channel_mean': np.mean(n_connections_per_channel),
                'connections_per_channel_std': np.std(n_connections_per_channel),
                'average_overlap_fraction': np.mean(overlaps) if overlaps else 0.0,
                'total_connections': np.sum(self.connectivity_matrix)
            }
        else:
            return {}

class ReadoutLayer:
    """
    Readout layer for task performance evaluation.
    Linear readout from RNN activity to task outputs.
    """

    def __init__(self, n_rnn_neurons: int, n_readout_neurons: int = 10, dt: float = 0.1):
        """
        Initialize readout layer.

        Args:
            n_rnn_neurons: Number of RNN neurons
            n_readout_neurons: Number of readout neurons
            dt: Time step in ms
        """
        self.n_rnn_neurons = n_rnn_neurons
        self.n_readout_neurons = n_readout_neurons
        self.dt = dt
        self.tau_readout = 20.0  # Readout time constant (ms)

        # Readout parameters
        self.readout_weights = None
        self.readout_activity = None

    def initialize_weights(self, session_id: int, block_id: int,
                          weight_scale: float = 1.0):
        """
        Initialize readout weights.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            weight_scale: Scale of readout weights
        """
        rng = get_rng(session_id, block_id, 0, 'readout_weights')

        # Random readout weights
        self.readout_weights = rng.normal(
            0.0, weight_scale / np.sqrt(self.n_rnn_neurons),
            (self.n_readout_neurons, self.n_rnn_neurons)
        )

        # Initialize readout activity
        self.readout_activity = np.zeros(self.n_readout_neurons)

    def update(self, rnn_spike_indices: List[int]) -> np.ndarray:
        """
        Update readout activity based on RNN spikes.

        Args:
            rnn_spike_indices: List of RNN neurons that spiked

        Returns:
            Current readout activity
        """
        # Exponential decay
        self.readout_activity *= np.exp(-self.dt / self.tau_readout)

        # Add contribution from RNN spikes
        if len(rnn_spike_indices) > 0:
            spike_contribution = np.sum(
                self.readout_weights[:, rnn_spike_indices], axis=1
            )
            self.readout_activity += spike_contribution

        return self.readout_activity.copy()

    def get_output(self) -> np.ndarray:
        """
        Get current readout output (for classification, etc.).

        Returns:
            Current readout activity
        """
        return self.readout_activity.copy()
