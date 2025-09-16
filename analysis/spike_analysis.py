# spike_analysis.py
"""
Spike analysis functions for chaos quantification.
Optimized to avoid redundant computations.
"""

import numpy as np
from typing import List, Tuple

def spikes_to_binary(spikes: List[Tuple[float, int]], num_neurons: int,
                    duration: float, bin_size: float) -> np.ndarray:
    """
    Convert spike times into a binary matrix.

    Args:
        spikes: List of tuples (spike_time, neuron_id)
        num_neurons: Total number of neurons
        duration: Total simulation time (ms)
        bin_size: Size of each time bin (ms)

    Returns:
        binary_matrix: 2D numpy array (neurons x time bins) with binary values
    """
    num_bins = int(duration / bin_size)
    binary_matrix = np.zeros((num_neurons, num_bins), dtype=int)

    for spike_time, neuron_id in spikes:
        # Find the corresponding time bin for each spike
        time_bin = int(round(spike_time / bin_size))
        if 0 <= time_bin < num_bins:  # Ensure spike is within simulation duration
            binary_matrix[neuron_id, time_bin] = 1

    return binary_matrix

def sort_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """
    Sort binary matrix for Lempel-Ziv complexity calculation.

    Args:
        binary_matrix: Binary spike matrix

    Returns:
        Sorted binary matrix
    """
    # Sort rows by total activity (optional - can use other sorting schemes)
    row_sums = np.sum(binary_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]  # Descending order
    return binary_matrix[sorted_indices, :]

def lempel_ziv_complexity(matrix_sorted: np.ndarray) -> int:
    """
    Compute the Lempel-Ziv complexity of a binary sequence.

    Args:
        matrix_sorted: Sorted binary matrix

    Returns:
        complexity: LZ complexity value
    """
    # Flatten matrix column-wise (time-major)
    binary_sequence = matrix_sorted.flatten(order='F')

    # Convert to string
    if not isinstance(binary_sequence, str):
        binary_sequence = ''.join(str(int(b)) for b in binary_sequence)

    n = len(binary_sequence)
    substrings = []
    i = 0

    while i < n:
        l = 1
        # Find the shortest substring not seen before
        while i + l <= n and binary_sequence[i:i+l] in substrings:
            l += 1

        if i + l <= n:
            substrings.append(binary_sequence[i:i+l])
        else:
            # Handle edge case at the end
            substrings.append(binary_sequence[i:])

        i += l

    complexity = len(substrings)
    return complexity

def compute_spike_difference_matrix(spikes_control: List[Tuple[float, int]],
                                  spikes_perturbed: List[Tuple[float, int]],
                                  num_neurons: int, perturbation_time: float,
                                  simulation_end: float, perturbed_neuron: int,
                                  bin_size: float = 0.1) -> np.ndarray:
    """
    Compute spike difference matrix (shared computation for both chaos measures).

    Args:
        spikes_control: Control condition spikes
        spikes_perturbed: Perturbed condition spikes
        num_neurons: Number of neurons
        perturbation_time: Time of perturbation (ms)
        simulation_end: End of simulation (ms)
        perturbed_neuron: ID of perturbed neuron
        bin_size: Time bin size (ms)

    Returns:
        Binary matrix of spike differences with perturbation column added
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

    return spike_difference

def compute_hamming_distance_from_matrix(difference_matrix: np.ndarray,
                                       bin_size: float = 1.0,
                                       time_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hamming distance from precomputed difference matrix.

    Args:
        difference_matrix: Binary matrix of differences (neurons x time_bins)
        bin_size: Time bin size (ms)
        time_offset: Time offset for bin centers (ms)

    Returns:
        Tuple of (time_bins, hamming_distances)
    """
    # Compute Hamming distance for each time bin (sum across neurons)
    hamming_distances = np.sum(difference_matrix, axis=0)

    # Time bins (centers)
    duration = difference_matrix.shape[1] * bin_size
    time_bins = np.arange(bin_size/2, duration, bin_size) + time_offset

    # Ensure arrays have same length
    min_length = min(len(time_bins), len(hamming_distances))
    time_bins = time_bins[:min_length]
    hamming_distances = hamming_distances[:min_length]

    return time_bins, hamming_distances

def compute_chaos_slope_robust(time_bins: np.ndarray, hamming_distances: np.ndarray,
                              window_start: float = 50.0,
                              min_slope_window: int = 10) -> float:
    """
    Compute robust slope of Hamming distance increase (chaos measure).
    Measures how fast the perturbation grows over time.

    Args:
        time_bins: Time bin centers
        hamming_distances: Hamming distances over time
        window_start: Start time for slope calculation (ms after perturbation)
        min_slope_window: Minimum number of points for slope calculation

    Returns:
        Robust slope value (higher = more chaotic)
    """
    # Find indices for slope calculation
    valid_indices = time_bins >= window_start

    if np.sum(valid_indices) < min_slope_window:
        return 0.0

    t_fit = time_bins[valid_indices]
    h_fit = hamming_distances[valid_indices]

    if len(t_fit) < min_slope_window:
        return 0.0

    # Strategy: Find the period of fastest growth (steepest sustained slope)
    # This captures "how fast the perturbation grows" regardless of noise or saturation

    # Use a sliding window approach to find the best growth period
    if len(t_fit) >= min_slope_window:
        # Try different window sizes to find sustained growth
        best_slope = 0.0

        # Start with minimum window and gradually increase
        for window_size in range(min_slope_window, min(len(t_fit) + 1, min_slope_window * 3)):
            for start_idx in range(len(t_fit) - window_size + 1):
                end_idx = start_idx + window_size

                t_window = t_fit[start_idx:end_idx]
                h_window = h_fit[start_idx:end_idx]

                # Only consider windows with monotonic or mostly increasing trend
                if len(t_window) >= 2:
                    slope, _ = np.polyfit(t_window, h_window, 1)

                    # Only consider positive slopes (growth)
                    if slope > best_slope:
                        # Check if this represents a good growth period
                        # by ensuring the trend is reasonably consistent
                        correlation = np.corrcoef(t_window, h_window)[0, 1]
                        if not np.isnan(correlation) and correlation > 0.3:  # Reasonable correlation
                            best_slope = slope

        return best_slope

    # Fallback: simple linear fit if we don't have enough data for windowing
    if len(t_fit) >= 2:
        slope, _ = np.polyfit(t_fit, h_fit, 1)
        return max(0.0, slope)  # Only return positive slopes
    else:
        return 0.0

def analyze_perturbation_response(spikes_control: List[Tuple[float, int]],
                                spikes_perturbed: List[Tuple[float, int]],
                                num_neurons: int, perturbation_time: float,
                                simulation_end: float,
                                perturbed_neuron: int) -> Tuple[float, float]:
    """
    Analyze response to perturbation using both LZ complexity and Hamming slope.
    Uses shared computation to avoid redundant matrix operations.

    Args:
        spikes_control: Control condition spikes
        spikes_perturbed: Perturbed condition spikes
        num_neurons: Number of neurons
        perturbation_time: Time of perturbation (ms)
        simulation_end: End of simulation (ms)
        perturbed_neuron: ID of perturbed neuron

    Returns:
        Tuple of (lz_complexity, hamming_slope)
    """
    # 1. Compute shared spike difference matrix for LZ complexity (fine resolution)
    spike_difference_fine = compute_spike_difference_matrix(
        spikes_control, spikes_perturbed, num_neurons,
        perturbation_time, simulation_end, perturbed_neuron,
        bin_size=0.1  # Fine resolution for LZ complexity
    )

    # 2. LZ complexity analysis
    spike_difference_sorted = sort_matrix(spike_difference_fine)
    lz_complexity = lempel_ziv_complexity(spike_difference_sorted)

    # 3. Hamming distance slope analysis (coarser resolution)
    spike_difference_coarse = compute_spike_difference_matrix(
        spikes_control, spikes_perturbed, num_neurons,
        perturbation_time, simulation_end, perturbed_neuron,
        bin_size=1.0  # Coarser resolution for slope analysis
    )

    # Compute Hamming distances from precomputed difference matrix
    time_bins, hamming_distances = compute_hamming_distance_from_matrix(
        spike_difference_coarse, bin_size=1.0, time_offset=0.0
    )

    # Compute robust slope that measures growth rate
    hamming_slope = compute_chaos_slope_robust(
        time_bins, hamming_distances, window_start=50.0
    )

    return lz_complexity, hamming_slope
