# analysis/stability_analysis.py - Network stability and perturbation analysis
"""
Network stability analysis: perturbation response, coincidence measures, LZ complexity.
Optimized with unified coincidence calculation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
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

def lempel_ziv_complexity(temporal_sequence: np.ndarray) -> int:
    """Compute the Lempel-Ziv complexity of a one dimensional temporal sequence."""
    binary_sequence = ''.join(str(int(b)) for b in temporal_sequence)
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
    unique_substrings = list(dict.fromkeys(substrings))
    return len(unique_substrings)

def compute_spatial_pattern_complexity(binary_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute LZ complexity of spatial patterns (columns).
    """
    if binary_matrix.size == 0:
        return {
            'lz_spatial_patterns': 0,
            'spatial_entropy': 0.0,
            'pattern_fraction': 0.0,
            'unique_patterns': 0,
            'total_patterns': 0
        }

    # Extract spatial patterns (columns)
    spatial_patterns = []
    for t in range(binary_matrix.shape[1]):
        pattern = tuple(binary_matrix[:, t])
        spatial_patterns.append(pattern)

    if len(spatial_patterns) == 0:
        return {
            'lz_spatial_patterns': 0,
            'spatial_entropy': 0.0,
            'pattern_fraction': 0.0,
            'unique_patterns': 0,
            'total_patterns': 0
        }

    # Create mapping from patterns to symbols
    unique_patterns_dict = {}
    symbol_sequence = []
    next_symbol = 0

    for pattern in spatial_patterns:
        if pattern not in unique_patterns_dict:
            unique_patterns_dict[pattern] = next_symbol
            next_symbol += 1
        symbol_sequence.append(unique_patterns_dict[pattern])

    # Compute LZ complexity of symbol sequence
    lz_spatial = lempel_ziv_complexity(np.array(symbol_sequence))

    # Calculate pattern statistics
    total_samples = binary_matrix.size
    if total_samples == 0:
        return {
            'lz_spatial_patterns': lz_spatial,
            'spatial_entropy': 0.0,
            'pattern_fraction': 0.0,
            'unique_patterns': len(unique_patterns_dict),
            'total_patterns': len(spatial_patterns)
        }

    # Calculate p1 (fraction of 1s in the matrix) - keep for pattern_fraction
    p1 = np.sum(binary_matrix) / total_samples
    pattern_fraction = p1

    # Compute entropy over spatial pattern distribution
    pattern_counts = {}
    for pattern in spatial_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Calculate spatial pattern entropy
    total_patterns = len(spatial_patterns)
    entropy = 0.0
    if total_patterns > 0:
        for count in pattern_counts.values():
            p = count / total_patterns
            if p > 0:
                entropy -= p * np.log2(p)

    return {
        'lz_spatial_patterns': lz_spatial,
        'spatial_entropy': entropy,
        'pattern_fraction': pattern_fraction,
        'unique_patterns': len(unique_patterns_dict),
        'total_patterns': len(spatial_patterns)
    }

def find_stable_period(sequence, min_repeats=3):
    """Find if sequence ends with a repeating pattern."""
    sequence = [int(x) for x in sequence]
    n = len(sequence)

    for period in range(1, n//3 + 1):
        repeats = 0
        for start_pos in range(n - period, -1, -period):
            if start_pos + period > n:
                continue

            pattern = sequence[-period:]
            segment = sequence[start_pos:start_pos + period]

            if segment == pattern:
                repeats += 1
            else:
                break

        if repeats >= min_repeats:
            onset_time = n - (repeats * period)
            return {
                'period': period,
                'pattern': sequence[-period:],
                'repeats': repeats,
                'onset_time': onset_time,
                'stable_length': repeats * period
            }

    return None

def unified_coincidence_factor(spike_train1: List[float], spike_train2: List[float],
                              delta: float = 2.0, duration: float = None) -> Tuple[float, float]:
    """
    Unified calculation for both Kistler and Gamma coincidence factors.

    Returns:
        Tuple of (kistler_factor, gamma_factor)
    """
    if not spike_train1 or not spike_train2:
        return 0.0, 0.0

    N_data = len(spike_train1)
    N_SRM = len(spike_train2)

    if duration is None:
        duration = max(max(spike_train1), max(spike_train2))

    # Count coincidences within precision delta - SINGLE LOOP
    spike_train1_sorted = sorted(spike_train1)
    spike_train2_sorted = sorted(spike_train2)

    N_coinc = 0
    i, j = 0, 0

    while i < len(spike_train1_sorted) and j < len(spike_train2_sorted):
        dt = spike_train1_sorted[i] - spike_train2_sorted[j]
        if abs(dt) <= delta:
            N_coinc += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    # Expected coincidences
    if duration > 0:
        rate_SRM = N_SRM / duration  # Rate in spikes/ms
        expected_coinc = rate_SRM * delta * N_data
    else:
        expected_coinc = 0


    N = 1 - rate_SRM * delta # Normalization factor for Kistler coincidence

    # Gamma coincidence and Kistler coincidence
    if (N_data * N_SRM) == 0:
        gamma = float('nan')
        kistler = float('nan')
    else:
        gamma = (N_coinc - expected_coinc) / (0.5 * (N_data + N_SRM))
        if N <= 0:
            kistler = float('nan')
        else:
            kistler = gamma / N

    return kistler, gamma

def average_coincidence_multi_window(spikes1: List[Tuple[float, int]],
                                   spikes2: List[Tuple[float, int]],
                                   num_neurons: int,
                                   delta_values: List[float] = [2.0, 5.0],
                                   duration: float = None) -> Dict[str, float]:
    """
    Compute average coincidence using unified calculation for efficiency.
    """
    # Organize spikes by neuron
    spikes_net1 = defaultdict(list)
    spikes_net2 = defaultdict(list)

    for spike_time, neuron_id in spikes1:
        spikes_net1[neuron_id].append(spike_time)
    for spike_time, neuron_id in spikes2:
        spikes_net2[neuron_id].append(spike_time)

    results = {}

    for delta in delta_values:
        kistler_values = []
        gamma_values = []

        for neuron_id in range(num_neurons):
            kistler_c, gamma_c = unified_coincidence_factor(
                spikes_net1[neuron_id],
                spikes_net2[neuron_id],
                delta=delta,
                duration=duration
            )

            if not np.isnan(kistler_c):
                kistler_values.append(kistler_c)
            if not np.isnan(gamma_c):
                gamma_values.append(gamma_c)

        if kistler_values:
            results[f'kistler_delta_{delta:.0f}ms'] = np.mean(kistler_values)
        else:
            results[f'kistler_delta_{delta:.0f}ms'] = float('nan')

        if gamma_values:
            results[f'gamma_window_{delta:.0f}ms'] = np.mean(gamma_values)
        else:
            results[f'gamma_window_{delta:.0f}ms'] = float('nan')

    return results

def sort_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """Sort binary matrix for Lempel-Ziv complexity calculation."""
    row_sums = np.sum(binary_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    return binary_matrix[sorted_indices, :]

def compute_spike_difference_matrix(spikes_control: List[Tuple[float, int]],
                                  spikes_perturbed: List[Tuple[float, int]],
                                  num_neurons: int, perturbation_time: float,
                                  simulation_end: float, perturbed_neuron: int,
                                  bin_size: float = 2.0) -> Tuple[np.ndarray, int]:
    """Compute spike difference matrix with total difference count."""
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
                if np.std(t_window) == 0 or np.std(h_window) == 0:
                    slope, _ = np.polyfit(t_window, h_window, 1)
                    if slope > best_slope:
                        best_slope = slope
                else:
                    slope, _ = np.polyfit(t_window, h_window, 1)
                    if slope > best_slope:
                        correlation = np.corrcoef(t_window, h_window)[0, 1]
                        if not np.isnan(correlation) and correlation > 0.3:
                            best_slope = slope

    if len(t_fit) >= 2:
        slope, _ = np.polyfit(t_fit, h_fit, 1)
        return max(0.0, slope, best_slope)
    else:
        return max(0.0, best_slope)

def analyze_perturbation_response(spikes_control: List[Tuple[float, int]],
                                spikes_perturbed: List[Tuple[float, int]],
                                num_neurons: int, perturbation_time: float,
                                simulation_end: float,
                                perturbed_neuron: int) -> Dict[str, Any]:
    """
    Enhanced perturbation analysis with optimized coincidence calculation.
    """
    # 1. Compute spike difference matrix for LZ complexity (2ms bins)
    spike_difference_fine, total_spike_differences = compute_spike_difference_matrix(
        spikes_control, spikes_perturbed, num_neurons,
        perturbation_time, simulation_end, perturbed_neuron,
        bin_size=2.0
    )

    # 2. Spatial pattern complexity (no PCI measures)
    spatial_results = compute_spatial_pattern_complexity(spike_difference_fine)

    # 3. Pattern stability analysis
    spatial_patterns = []
    for t in range(spike_difference_fine.shape[1]):
        pattern = tuple(spike_difference_fine[:, t])
        spatial_patterns.append(pattern)

    unique_patterns_dict = {}
    symbol_sequence = []
    next_symbol = 0

    for pattern in spatial_patterns:
        if pattern not in unique_patterns_dict:
            unique_patterns_dict[pattern] = next_symbol
            next_symbol += 1
        symbol_sequence.append(unique_patterns_dict[pattern])

    stable_period_info = find_stable_period(symbol_sequence, min_repeats=3)

    # 4. Hamming distance slope analysis (using 1ms bins)
    spike_difference_coarse, _ = compute_spike_difference_matrix(
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

    # 5. Unified coincidence analysis (optimized)
    duration_post = simulation_end - perturbation_time
    spikes_control_post = [(t, n) for t, n in spikes_control if t >= perturbation_time]
    spikes_perturbed_post = [(t, n) for t, n in spikes_perturbed if t >= perturbation_time]

    coincidence_results = average_coincidence_multi_window(
        spikes_control_post, spikes_perturbed_post,
        num_neurons, delta_values=[2.0, 5.0], duration=duration_post
    )

    # Compile results (removed PCI measures)
    results = {
        # Spatial LZ complexity
        'lz_spatial_patterns': spatial_results['lz_spatial_patterns'],
        'hamming_slope': hamming_slope,

        # Spatial pattern statistics
        'spatial_entropy': spatial_results['spatial_entropy'],
        'pattern_fraction': spatial_results['pattern_fraction'],
        'unique_patterns': spatial_results['unique_patterns'],
        'total_patterns': spatial_results['total_patterns'],

        # Pattern stability
        'stable_period': stable_period_info,

        # Matrix difference magnitude
        'total_spike_differences': total_spike_differences,

        # Unified coincidence measures
        **coincidence_results,
    }

    return results
