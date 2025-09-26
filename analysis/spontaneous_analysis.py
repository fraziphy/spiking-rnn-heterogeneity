# analysis/spontaneous_analysis.py - Spontaneous activity analysis
"""
Spontaneous activity analysis: firing rates, dimensionality, silent neurons.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict

def spikes_to_binary(spikes: List[Tuple[float, int]], num_neurons: int,
                    duration: float, bin_size: float) -> np.ndarray:
    """Convert spike times into a binary matrix."""
    num_bins = int(duration / bin_size)
    binary_matrix = np.zeros((num_neurons, num_bins), dtype=int)

    for spike_time, neuron_id in spikes:
        time_bin = int(round(spike_time / bin_size))
        if 0 <= time_bin < num_bins:
            binary_matrix[neuron_id, time_bin] = 1

    return binary_matrix

def compute_activity_dimensionality(binary_matrix: np.ndarray,
                                  variance_threshold: float = 0.95) -> Dict[str, float]:
    """
    Compute dimensionality of network activity using PCA.
    """
    if binary_matrix.size == 0:
        return {
            'intrinsic_dimensionality': 0.0,
            'effective_dimensionality': 0.0,
            'participation_ratio': 0.0,
            'total_variance': 0.0
        }

    # Remove neurons with no activity
    active_neurons = np.sum(binary_matrix, axis=1) > 0
    if np.sum(active_neurons) < 2:
        return {
            'intrinsic_dimensionality': float(np.sum(active_neurons)),
            'effective_dimensionality': float(np.sum(active_neurons)),
            'participation_ratio': 1.0 if np.sum(active_neurons) > 0 else 0.0,
            'total_variance': 0.0
        }

    active_matrix = binary_matrix[active_neurons, :]

    # Center the data
    centered_matrix = active_matrix - np.mean(active_matrix, axis=1, keepdims=True)

    if np.allclose(centered_matrix, 0):
        return {
            'intrinsic_dimensionality': 1.0,
            'effective_dimensionality': 1.0,
            'participation_ratio': 1.0,
            'total_variance': 0.0
        }

    n_active = active_matrix.shape[0]
    n_time = active_matrix.shape[1]

    if n_time > 1:
        cov_matrix = np.cov(centered_matrix)

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

    eigenvalues = np.sort(eigenvalues)[::-1]
    total_variance = np.sum(eigenvalues)

    cumulative_var = np.cumsum(eigenvalues) / total_variance
    effective_dim = np.searchsorted(cumulative_var, variance_threshold) + 1
    effective_dim = min(effective_dim, len(eigenvalues))

    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    return {
        'intrinsic_dimensionality': float(len(eigenvalues)),
        'effective_dimensionality': float(effective_dim),
        'participation_ratio': float(participation_ratio),
        'total_variance': float(total_variance)
    }

def compute_activity_dimensionality_multi_bin(spikes: List[Tuple[float, int]],
                                             num_neurons: int,
                                             duration: float,
                                             bin_sizes: List[float] = [0.1, 2.0, 5.0, 20.0, 50.0, 100.0],
                                             variance_threshold: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Compute dimensionality with multiple bin sizes for spontaneous activity.
    """
    results = {}

    for bin_size in bin_sizes:
        binary_matrix = spikes_to_binary(spikes, num_neurons, duration, bin_size)
        dim_result = compute_activity_dimensionality(binary_matrix, variance_threshold)
        results[f'bin_{bin_size}ms'] = dim_result

    return results

def analyze_firing_rates_and_silence(spikes: List[Tuple[float, int]],
                                   num_neurons: int, duration: float) -> Dict[str, float]:
    """
    Analyze firing rates and silent neurons.
    """
    spike_counts = defaultdict(int)
    for _, neuron_id in spikes:
        spike_counts[neuron_id] += 1

    # Convert to firing rates (Hz)
    firing_rates = []
    duration_seconds = duration / 1000.0  # Convert ms to seconds

    for neuron_id in range(num_neurons):
        rate = spike_counts[neuron_id] / duration_seconds
        firing_rates.append(rate)

    firing_rates = np.array(firing_rates)

    silent_neurons = np.sum(firing_rates == 0)
    active_neurons = num_neurons - silent_neurons

    return {
        'mean_firing_rate': float(np.mean(firing_rates)),
        'std_firing_rate': float(np.std(firing_rates)),
        'min_firing_rate': float(np.min(firing_rates)),
        'max_firing_rate': float(np.max(firing_rates)),
        'silent_neurons': int(silent_neurons),
        'active_neurons': int(active_neurons),
        'percent_silent': float(silent_neurons / num_neurons * 100),
        'percent_active': float(active_neurons / num_neurons * 100)
    }

def analyze_spontaneous_activity(spikes: List[Tuple[float, int]],
                               num_neurons: int,
                               duration: float) -> Dict[str, Any]:
    """
    Complete spontaneous activity analysis.

    Args:
        spikes: List of (spike_time, neuron_id) tuples
        num_neurons: Total number of neurons
        duration: Simulation duration in milliseconds

    Returns:
        Dictionary with firing rate statistics and dimensionality metrics
    """
    # Firing rate analysis
    firing_stats = analyze_firing_rates_and_silence(spikes, num_neurons, duration)

    # Dimensionality analysis with extended bin sizes
    dimensionality_metrics = compute_activity_dimensionality_multi_bin(
        spikes, num_neurons, duration,
        bin_sizes=[0.1, 2.0, 5.0, 20.0, 50.0, 100.0]
    )

    return {
        'firing_stats': firing_stats,
        'dimensionality_metrics': dimensionality_metrics,
        'duration_ms': duration,
        'total_spikes': len(spikes)
    }
