# analysis/common_utils.py
"""
Common utility functions shared across all analysis modules.
Extracted to eliminate code duplication.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def compute_empirical_dimensionality(data: np.ndarray) -> float:
    """
    Compute empirical dimensionality using participation ratio.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
              e.g., (n_timesteps, n_channels) for temporal signals
    
    Returns:
        Participation ratio: (sum(eigenvalues))^2 / sum(eigenvalues^2)
        - Returns 1.0 for k=1 (single feature)
        - Returns 0.0 for empty data or invalid input
    
    Example:
        >>> Y_base = np.random.randn(3000, 5)  # 3000 timesteps, 5 channels
        >>> dim = compute_empirical_dimensionality(Y_base)
        >>> 
        >>> Y_single = np.random.randn(3000, 1)  # Single channel
        >>> dim = compute_empirical_dimensionality(Y_single)  # Returns 1.0
    """
    # Handle edge cases
    if data.size == 0:
        return 0.0
    
    n_features = data.shape[1] if data.ndim > 1 else 1
    
    # Special case: k=1 (single feature)
    # Participation ratio is always 1.0 (all variance in one dimension)
    if n_features == 1:
        return 1.0
    
    try:
        # Compute covariance matrix (features in columns)
        eig = np.linalg.eigvalsh(np.cov(data, rowvar=False))
        
        # Remove negative eigenvalues (numerical errors)
        eig = eig[eig > 1e-10]
        
        if len(eig) == 0:
            return 0.0
        
        # Participation ratio
        part_ratio = (np.sum(eig) ** 2) / np.sum(eig ** 2)
        
        return float(part_ratio)
    
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def spikes_to_binary(spikes: List[Tuple[float, int]], num_neurons: int,
                    duration: float, bin_size: float) -> np.ndarray:
    """
    Convert spike times into a binary matrix.

    Used by both spontaneous_analysis and stability_analysis.

    Args:
        spikes: List of (spike_time, neuron_id) tuples
        num_neurons: Number of neurons
        duration: Duration in ms
        bin_size: Bin size in ms

    Returns:
        Binary matrix of shape (num_neurons, num_bins)
    """
    num_bins = int(duration / bin_size)
    binary_matrix = np.zeros((num_neurons, num_bins), dtype=int)

    for spike_time, neuron_id in spikes:
        time_bin = int(round(spike_time / bin_size))
        if 0 <= time_bin < num_bins:
            binary_matrix[neuron_id, time_bin] = 1

    return binary_matrix


def spikes_to_matrix(spike_list: List[Tuple[float, int]], n_steps: int,
                    n_neurons: int, step_size: float) -> np.ndarray:
    """
    Convert spike data into a spike matrix for encoding analysis.

    Args:
        spike_list: List of spikes [(time, neuron_id), ...]
        n_steps: Number of time steps
        n_neurons: Number of neurons
        step_size: Time step size in ms

    Returns:
        Spike matrix of shape (n_steps, n_neurons)
    """
    spike_matrix = np.zeros((n_steps, n_neurons))
    for spike_time, neuron_id in spike_list:
        time_bin = int(round(spike_time / step_size))
        if 0 <= time_bin < n_steps and 0 <= neuron_id < n_neurons:
            spike_matrix[time_bin, neuron_id] += 1
    return spike_matrix


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio from eigenvalues.

    Formula: PR = (sum(λ))² / sum(λ²)

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Participation ratio
    """
    if len(eigenvalues) == 0 or np.sum(eigenvalues) == 0:
        return 0.0

    return float((np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2))


def compute_effective_dimensionality(eigenvalues: np.ndarray,
                                    variance_threshold: float = 0.95) -> int:
    """
    Compute effective dimensionality (number of components explaining variance threshold).

    Args:
        eigenvalues: Array of eigenvalues (must be sorted descending)
        variance_threshold: Variance threshold (default 0.95)

    Returns:
        Effective dimensionality
    """
    if len(eigenvalues) == 0:
        return 0

    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        return 0

    cumulative_var = np.cumsum(eigenvalues) / total_variance
    effective_dim = np.searchsorted(cumulative_var, variance_threshold) + 1
    effective_dim = min(effective_dim, len(eigenvalues))

    return int(effective_dim)


def compute_dimensionality_from_covariance(data: np.ndarray,
                                          variance_threshold: float = 0.95) -> Dict[str, float]:
    """
    Compute dimensionality metrics from data covariance.

    Generalized version used across multiple analysis modules.

    Args:
        data: Data matrix (features × samples or samples × features)
        variance_threshold: Variance threshold for effective dimensionality

    Returns:
        Dictionary with dimensionality metrics
    """
    # Handle edge cases
    if data.size == 0:
        return {
            'intrinsic_dimensionality': 0.0,
            'effective_dimensionality': 0.0,
            'participation_ratio': 0.0,
            'total_variance': 0.0
        }

    # Remove zero-variance features
    feature_variance = np.var(data, axis=1) if data.shape[0] < data.shape[1] else np.var(data, axis=0)
    active_features = feature_variance > 1e-10

    if np.sum(active_features) < 2:
        return {
            'intrinsic_dimensionality': float(np.sum(active_features)),
            'effective_dimensionality': float(np.sum(active_features)),
            'participation_ratio': 1.0 if np.sum(active_features) > 0 else 0.0,
            'total_variance': 0.0
        }

    # Center data
    if data.shape[0] < data.shape[1]:  # features × samples
        active_data = data[active_features, :]
        centered = active_data - np.mean(active_data, axis=1, keepdims=True)
    else:  # samples × features
        active_data = data[:, active_features]
        centered = active_data - np.mean(active_data, axis=0, keepdims=True)

    # Compute covariance and eigenvalues
    if centered.shape[1] > 1:
        cov_matrix = np.cov(centered)
        if cov_matrix.ndim == 0:
            eigenvalues = np.array([cov_matrix])
        else:
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])
    else:
        eigenvalues = np.array([1.0])

    if len(eigenvalues) == 0:
        return {
            'intrinsic_dimensionality': 0.0,
            'effective_dimensionality': 0.0,
            'participation_ratio': 0.0,
            'total_variance': 0.0
        }

    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]

    return {
        'intrinsic_dimensionality': float(len(eigenvalues)),
        'effective_dimensionality': float(compute_effective_dimensionality(eigenvalues, variance_threshold)),
        'participation_ratio': compute_participation_ratio(eigenvalues),
        'total_variance': float(np.sum(eigenvalues))
    }



def compute_dimensionality_svd(data, variance_threshold=0.95):
    """
    Compute dimensionality metrics using SVD (faster than covariance).

    Args:
        data: (n_samples, n_features) array - e.g., (n_timebins, n_neurons)
        variance_threshold: float - threshold for cumulative variance (default 0.95)

    Returns:
        dict with dimensionality metrics
    """
    # Center data
    data_centered = data - data.mean(axis=0)

    # Handle empty or single-sample data
    if data_centered.shape[0] <= 1:
        return {
            'participation_ratio': 0.0,
            'effective_dimensionality': 0.0,
            'intrinsic_dimensionality': 0.0,
            'total_variance': 0.0,
            'n_components': 0
        }

    # SVD (faster than covariance!)
    try:
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            'participation_ratio': 0.0,
            'effective_dimensionality': 0.0,
            'intrinsic_dimensionality': 0.0,
            'total_variance': 0.0,
            'n_components': 0
        }

    # Eigenvalues of C = X.T @ X (no normalization needed!)
    eigenvalues = S ** 2

    # Remove near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return {
            'participation_ratio': 0.0,
            'effective_dimensionality': 0.0,
            'intrinsic_dimensionality': 0.0,
            'total_variance': 0.0,
            'n_components': 0
        }

    # Total variance
    total_variance = np.sum(eigenvalues)

    # Participation ratio: (sum λ)² / (sum λ²)
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    # Effective dimensionality (normalized eigenvalues)
    normalized_eigenvalues = eigenvalues / total_variance
    effective_dimensionality = np.exp(-np.sum(
        normalized_eigenvalues * np.log(normalized_eigenvalues + 1e-12)
    ))

    # Intrinsic dimensionality (cumulative variance threshold)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    cumulative_variance = np.cumsum(sorted_eigenvalues) / total_variance
    intrinsic_dimensionality = np.searchsorted(cumulative_variance, variance_threshold) + 1

    return {
        'participation_ratio': float(participation_ratio),
        'effective_dimensionality': float(effective_dimensionality),
        'intrinsic_dimensionality': float(intrinsic_dimensionality),
        'total_variance': float(total_variance),
        'n_components': len(eigenvalues)
    }



def apply_exponential_filter(spike_matrix: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """
    Apply exponential filter with time constant tau.

    Converts spike counts to filtered synaptic currents using exponential decay.
    Used for preprocessing spike data before decoding/readout.

    Args:
        spike_matrix: (T, N) spike count matrix (timesteps × neurons)
        tau: Time constant in ms (decay time of synaptic filter)
        dt: Time step in ms

    Returns:
        Filtered traces (T, N) - same shape as input

    Example:
        >>> spikes = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> filtered = apply_exponential_filter(spikes, tau=10.0, dt=0.1)
    """
    T, N = spike_matrix.shape
    filtered = np.zeros_like(spike_matrix, dtype=float)

    # Exponential decay factor
    decay = np.exp(-dt / tau)

    # Apply causal exponential filter
    for t in range(T):
        if t == 0:
            filtered[t] = spike_matrix[t] * (1 - decay)
        else:
            filtered[t] = filtered[t-1] * decay + spike_matrix[t] * (1 - decay)

    return filtered
