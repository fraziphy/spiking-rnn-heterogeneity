# analysis/spike_analysis.py - Enhanced with new analyses
"""
Spike analysis functions for chaos quantification with additional metrics.
Includes dimensionality analysis, matrix differences, and gamma coincidence.
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

    Args:
        binary_matrix: Binary spike matrix (neurons x time_bins)
        variance_threshold: Fraction of variance to capture for effective dimensionality

    Returns:
        Dictionary with dimensionality metrics
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

    # Center the data (subtract mean across time for each neuron)
    centered_matrix = active_matrix - np.mean(active_matrix, axis=1, keepdims=True)

    # Handle case where all activities are identical
    if np.allclose(centered_matrix, 0):
        return {
            'intrinsic_dimensionality': 1.0,
            'effective_dimensionality': 1.0,
            'participation_ratio': 1.0,
            'total_variance': 0.0
        }

    # Compute covariance matrix (neurons x neurons)
    n_active = active_matrix.shape[0]
    n_time = active_matrix.shape[1]

    if n_time > 1:
        cov_matrix = np.cov(centered_matrix)

        # Ensure covariance matrix is well-conditioned
        if cov_matrix.ndim == 0:  # Single neuron case
            eigenvalues = np.array([cov_matrix])
        else:
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])  # Remove numerical zeros
    else:
        eigenvalues = np.array([1.0])  # Single time bin case

    if len(eigenvalues) == 0:
        return {
            'intrinsic_dimensionality': 0.0,
            'effective_dimensionality': 0.0,
            'participation_ratio': 0.0,
            'total_variance': 0.0
        }

    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    total_variance = np.sum(eigenvalues)

    # Effective dimensionality: number of components needed for variance_threshold of variance
    cumulative_var = np.cumsum(eigenvalues) / total_variance
    effective_dim = np.searchsorted(cumulative_var, variance_threshold) + 1
    effective_dim = min(effective_dim, len(eigenvalues))

    # Participation ratio: (sum of eigenvalues)^2 / sum of eigenvalues^2
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    return {
        'intrinsic_dimensionality': float(len(eigenvalues)),
        'effective_dimensionality': float(effective_dim),
        'participation_ratio': float(participation_ratio),
        'total_variance': float(total_variance)
    }

def gamma_coincidence(spike_train1: List[float], spike_train2: List[float],
                     window_ms: float = 5.0) -> float:
    """
    Compute normalized gamma coincidence factor between two spike trains.

    Args:
        spike_train1: List of spike times (ms)
        spike_train2: List of spike times (ms)
        window_ms: Coincidence window in ms (default 5 ms)

    Returns:
        Normalized coincidence factor between 0 and 1
    """
    if not spike_train1 or not spike_train2:
        return 0.0

    spike_train1 = sorted(spike_train1)
    spike_train2 = sorted(spike_train2)

    count_coincidences = 0
    i, j = 0, 0

    while i < len(spike_train1) and j < len(spike_train2):
        dt = spike_train1[i] - spike_train2[j]
        if abs(dt) <= window_ms:
            count_coincidences += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    # Normalization: average of the two spike counts
    norm = (len(spike_train1) + len(spike_train2)) / 2.0
    if norm == 0:
        return 0.0

    return count_coincidences / norm

def average_gamma_coincidence(spikes1: List[Tuple[float, int]],
                            spikes2: List[Tuple[float, int]],
                            num_neurons: int,
                            window_ms: float = 5.0) -> float:
    """
    Compute average normalized gamma coincidence between two spike train sets.

    Args:
        spikes1: List of (spike_time, neuron_id) tuples for first condition
        spikes2: List of (spike_time, neuron_id) tuples for second condition
        num_neurons: Total number of neurons
        window_ms: Coincidence time window (ms)

    Returns:
        Average normalized gamma coincidence across all neurons
    """
    # Organize spikes by neuron
    spikes_net1 = defaultdict(list)
    spikes_net2 = defaultdict(list)

    for spike_time, neuron_id in spikes1:
        spikes_net1[neuron_id].append(spike_time)
    for spike_time, neuron_id in spikes2:
        spikes_net2[neuron_id].append(spike_time)

    # Compute gamma coincidence for each neuron
    coincidences = []
    for neuron_id in range(num_neurons):
        gamma_c = gamma_coincidence(
            spikes_net1[neuron_id],
            spikes_net2[neuron_id],
            window_ms=window_ms
        )
        coincidences.append(gamma_c)

    return np.mean(coincidences)

def sort_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """Sort binary matrix for Lempel-Ziv complexity calculation."""
    row_sums = np.sum(binary_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    return binary_matrix[sorted_indices, :]

def lempel_ziv_complexity(matrix_sorted: np.ndarray) -> int:
    """Compute the Lempel-Ziv complexity of a binary sequence."""
    binary_sequence = matrix_sorted.flatten(order='F')

    if not isinstance(binary_sequence, str):
        binary_sequence = ''.join(str(int(b)) for b in binary_sequence)

    n = len(binary_sequence)
    substrings = []
    i = 0

    while i < n:
        l = 1
        while i + l <= n and binary_sequence[i:i+l] in substrings:
            l += 1

        if i + l <= n:
            substrings.append(binary_sequence[i:i+l])
        else:
            substrings.append(binary_sequence[i:])

        i += l

    return len(substrings)

def compute_spike_difference_matrix_enhanced(spikes_control: List[Tuple[float, int]],
                                           spikes_perturbed: List[Tuple[float, int]],
                                           num_neurons: int, perturbation_time: float,
                                           simulation_end: float, perturbed_neuron: int,
                                           bin_size: float = 0.1) -> Tuple[np.ndarray, int]:
    """
    Compute spike difference matrix with total difference count.

    Returns:
        Tuple of (difference_matrix, total_differences)
    """
    duration_post = simulation_end - perturbation_time

    # Get spikes after perturbation
    spikes_control_post = [(t - perturbation_time, n)
                          for t, n in spikes_control if t >= perturbation_time]
    spikes_pert_post = [(t - perturbation_time, n)
                       for t, n in spikes_perturbed if t >= perturbation_time]

    # Convert to binary matrices
    matrix_control = spikes_to_binary(spikes_control_post, num_neurons,
                                    duration_post, bin_size)
    matrix_pert = spikes_to_binary(spikes_pert_post, num_neurons,
                                 duration_post, bin_size)

    # Compute difference matrix
    spike_difference = (matrix_control != matrix_pert).astype(int)

    # Add perturbation column at the beginning
    new_col = np.zeros(num_neurons, dtype=int)
    new_col[perturbed_neuron] = 1
    spike_difference = np.insert(spike_difference, 0, new_col, axis=1)

    # Compute total differences
    total_differences = int(np.sum(spike_difference))

    return spike_difference, total_differences

def compute_hamming_distance_from_matrix(difference_matrix: np.ndarray,
                                       bin_size: float = 1.0,
                                       time_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Hamming distance from precomputed difference matrix."""
    hamming_distances = np.sum(difference_matrix, axis=0)
    duration = difference_matrix.shape[1] * bin_size
    time_bins = np.arange(bin_size/2, duration, bin_size) + time_offset

    min_length = min(len(time_bins), len(hamming_distances))
    time_bins = time_bins[:min_length]
    hamming_distances = hamming_distances[:min_length]

    return time_bins, hamming_distances

def compute_chaos_slope_robust(time_bins: np.ndarray, hamming_distances: np.ndarray,
                              window_start: float = 50.0,
                              min_slope_window: int = 10) -> float:
    """Compute robust slope of Hamming distance increase."""
    valid_indices = time_bins >= window_start

    if np.sum(valid_indices) < min_slope_window:
        return 0.0

    t_fit = time_bins[valid_indices]
    h_fit = hamming_distances[valid_indices]

    if len(t_fit) < min_slope_window:
        return 0.0

    best_slope = 0.0

    for window_size in range(min_slope_window, min(len(t_fit) + 1, min_slope_window * 3)):
        for start_idx in range(len(t_fit) - window_size + 1):
            end_idx = start_idx + window_size

            t_window = t_fit[start_idx:end_idx]
            h_window = h_fit[start_idx:end_idx]

            if len(t_window) >= 2:
                slope, _ = np.polyfit(t_window, h_window, 1)

                if slope > best_slope:
                    correlation = np.corrcoef(t_window, h_window)[0, 1]
                    if not np.isnan(correlation) and correlation > 0.3:
                        best_slope = slope

        return best_slope

    if len(t_fit) >= 2:
        slope, _ = np.polyfit(t_fit, h_fit, 1)
        return max(0.0, slope)
    else:
        return 0.0

def analyze_perturbation_response_enhanced(spikes_control: List[Tuple[float, int]],
                                         spikes_perturbed: List[Tuple[float, int]],
                                         num_neurons: int, perturbation_time: float,
                                         simulation_end: float,
                                         perturbed_neuron: int) -> Dict[str, Any]:
    """
    Enhanced perturbation analysis with all three new metrics.

    Returns:
        Dictionary with all chaos and network metrics
    """
    # 1. Compute enhanced spike difference matrix for LZ complexity
    spike_difference_fine, total_spike_differences = compute_spike_difference_matrix_enhanced(
        spikes_control, spikes_perturbed, num_neurons,
        perturbation_time, simulation_end, perturbed_neuron,
        bin_size=0.1
    )

    # 2. LZ complexity analysis
    spike_difference_sorted = sort_matrix(spike_difference_fine)
    lz_complexity = lempel_ziv_complexity(spike_difference_sorted)

    # 3. Hamming distance slope analysis
    spike_difference_coarse, _ = compute_spike_difference_matrix_enhanced(
        spikes_control, spikes_perturbed, num_neurons,
        perturbation_time, simulation_end, perturbed_neuron,
        bin_size=1.0
    )

    time_bins, hamming_distances = compute_hamming_distance_from_matrix(
        spike_difference_coarse, bin_size=1.0, time_offset=0.0
    )

    hamming_slope = compute_chaos_slope_robust(
        time_bins, hamming_distances, window_start=50.0
    )

    # 4. Network activity dimensionality analysis
    # Analyze control condition activity
    duration_post = simulation_end - perturbation_time
    spikes_control_post = [(t - perturbation_time, n)
                          for t, n in spikes_control if t >= perturbation_time]

    control_binary = spikes_to_binary(spikes_control_post, num_neurons,
                                    duration_post, bin_size=1.0)

    dimensionality_metrics = compute_activity_dimensionality(control_binary)

    # 5. Gamma coincidence analysis
    # Filter spikes to post-perturbation period
    spikes_control_post_full = [(t, n) for t, n in spikes_control if t >= perturbation_time]
    spikes_perturbed_post_full = [(t, n) for t, n in spikes_perturbed if t >= perturbation_time]

    gamma_coincidence_avg = average_gamma_coincidence(
        spikes_control_post_full, spikes_perturbed_post_full,
        num_neurons, window_ms=5.0
    )

    # Compile all results
    results = {
        # Original chaos measures
        'lz_complexity': lz_complexity,
        'hamming_slope': hamming_slope,

        # New measure 1: Matrix difference magnitude
        'total_spike_differences': total_spike_differences,

        # New measure 2: Network activity dimensionality
        'intrinsic_dimensionality': dimensionality_metrics['intrinsic_dimensionality'],
        'effective_dimensionality': dimensionality_metrics['effective_dimensionality'],
        'participation_ratio': dimensionality_metrics['participation_ratio'],
        'total_variance': dimensionality_metrics['total_variance'],

        # New measure 3: Gamma coincidence
        'gamma_coincidence': gamma_coincidence_avg
    }

    return results
