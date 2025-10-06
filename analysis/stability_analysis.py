# analysis/stability_analysis.py - Refactored with common utilities
"""
Network stability analysis: full-simulation difference patterns, LZ complexity,
settling time, and coincidence measures.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from scipy.stats import entropy

# Import from common utilities
from .common_utils import spikes_to_binary


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


def compute_shannon_entropy(sequence: np.ndarray) -> float:
    """Compute Shannon entropy of a discrete sequence."""
    if len(sequence) == 0:
        return 0.0

    unique, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / len(sequence)
    return float(entropy(probabilities, base=2))


def find_settling_time(symbol_sequence: np.ndarray, perturbation_bin: int,
                      bin_size: float, min_zero_duration_ms: float = 50.0) -> float:
    """
    Find time when network settles back to baseline (zero state).

    Args:
        symbol_sequence: Full symbol sequence from time 0
        perturbation_bin: Bin index where perturbation occurred
        bin_size: Bin size in milliseconds
        min_zero_duration_ms: Minimum duration of zeros to consider "settled"

    Returns:
        Settling time in milliseconds (from perturbation), or NaN if never settles
    """
    post_pert_seq = symbol_sequence[perturbation_bin:]

    if len(post_pert_seq) == 0:
        return np.nan

    min_zero_bins = int(min_zero_duration_ms / bin_size)

    # Search from the end backwards
    for end_idx in range(len(post_pert_seq), min_zero_bins - 1, -1):
        start_idx = end_idx - min_zero_bins
        window = post_pert_seq[start_idx:end_idx]

        if np.all(window == 0):
            settling_time_ms = start_idx * bin_size
            return float(settling_time_ms)

    return np.nan


def unified_coincidence_factor(spike_train1: List[float], spike_train2: List[float],
                              delta: float = 2.0, duration: float = None) -> Tuple[float, float]:
    """
    Unified calculation for both Kistler and Gamma coincidence factors.

    Returns:
        Tuple of (kistler_factor, gamma_factor)
    """
    if not spike_train1 or not spike_train2:
        return float('nan'), float('nan')

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
        rate_SRM = N_SRM / duration
        expected_coinc = rate_SRM * delta * N_data
    else:
        expected_coinc = 0

    N = 1 - rate_SRM * delta

    # Gamma coincidence and Kistler coincidence
    gamma = (N_coinc - expected_coinc) / (0.5 * (N_data + N_SRM))
    if N <= 0:
        kistler = float('nan')
    else:
        kistler = gamma / N

    return kistler, gamma


def average_coincidence_multi_window(spikes1: List[Tuple[float, int]],
                                   spikes2: List[Tuple[float, int]],
                                   num_neurons: int,
                                   delta_values: List[float] = [0.1, 2.0, 5.0],
                                   duration: float = None) -> Dict[str, float]:
    """
    Compute average coincidence using unified calculation for efficiency.
    Now includes 0.1ms precision window.
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
            results[f'kistler_delta_{delta:.1f}ms'] = np.mean(kistler_values)
        else:
            results[f'kistler_delta_{delta:.1f}ms'] = float('nan')

        if gamma_values:
            results[f'gamma_window_{delta:.1f}ms'] = np.mean(gamma_values)
        else:
            results[f'gamma_window_{delta:.1f}ms'] = float('nan')

    return results


def analyze_perturbation_response(spikes_control: List[Tuple[float, int]],
                                spikes_perturbed: List[Tuple[float, int]],
                                num_neurons: int, perturbation_time: float,
                                simulation_end: float,
                                perturbed_neuron: int,
                                dt: float = 0.1) -> Dict[str, Any]:
    """
    Enhanced perturbation analysis with full-simulation difference patterns and lz_column_wise.

    Args:
        spikes_control: Control spike times [(time, neuron_id), ...]
        spikes_perturbed: Perturbed spike times [(time, neuron_id), ...]
        num_neurons: Number of neurons
        perturbation_time: Time of perturbation (ms) - UPDATED TO 200ms
        simulation_end: End time of simulation (ms)
        perturbed_neuron: ID of perturbed neuron
        dt: Time step size (ms), default 0.1 ms

    Returns:
        Dictionary with stability measures including lz_column_wise
    """

    bin_size = dt
    total_duration = simulation_end

    # 1. Convert full spike trains to binary matrices (using common utility)
    matrix_control = spikes_to_binary(spikes_control, num_neurons,
                                     total_duration, bin_size)
    matrix_perturbed = spikes_to_binary(spikes_perturbed, num_neurons,
                                       total_duration, bin_size)

    # 2. Compute spike difference matrix (full simulation)
    spike_diff_full = (matrix_control != matrix_perturbed).astype(int)

    # 3. Get perturbation bin index
    pert_bin = int(perturbation_time / bin_size)

    # 4. Extract spatial patterns and create symbol sequence
    spatial_patterns = [tuple(spike_diff_full[:, t])
                       for t in range(spike_diff_full.shape[1])]

    pattern_dict = {}
    symbol_seq = []
    next_id = 0

    for pat in spatial_patterns:
        if pat not in pattern_dict:
            pattern_dict[pat] = next_id
            next_id += 1
        symbol_seq.append(pattern_dict[pat])

    symbol_seq = np.array(symbol_seq)

    # 5. LZ complexity of post-perturbation symbol sequence
    lz_spatial = lempel_ziv_complexity(symbol_seq[pert_bin:])

    # 6. NEW: LZ column-wise (activity-sorted, column-major flattening)
    matrix_post = spike_diff_full[:, pert_bin:]
    activity = matrix_post.sum(axis=1)
    sorted_indices = np.argsort(activity)
    matrix_sorted = matrix_post[sorted_indices, :]
    lz_column_wise = lempel_ziv_complexity(matrix_sorted.flatten(order='F'))

    # 7. Shannon entropies (BOTH post-perturbation only)
    shannon_entropy_symbols = compute_shannon_entropy(symbol_seq[pert_bin:])

    spike_diff_post = spike_diff_full[:, pert_bin:]
    shannon_entropy_spikes = compute_shannon_entropy(spike_diff_post.flatten())

    # 8. Pattern diversity and activity
    unique_patterns_count = len(pattern_dict)
    post_pert_symbol_sum = int(np.sum(symbol_seq[pert_bin:]))
    total_spike_differences = int(spike_diff_full.sum())

    # 9. Settling time (time to return to baseline)
    settling_time_ms = find_settling_time(symbol_seq, pert_bin, bin_size,
                                         min_zero_duration_ms=50.0)

    # 10. Unified coincidence analysis with 0.1ms, 2ms, 5ms windows (post-perturbation only)
    duration_post = simulation_end - perturbation_time
    spikes_control_post = [(t, n) for t, n in spikes_control if t >= perturbation_time]
    spikes_perturbed_post = [(t, n) for t, n in spikes_perturbed if t >= perturbation_time]

    coincidence_results = average_coincidence_multi_window(
        spikes_control_post, spikes_perturbed_post,
        num_neurons, delta_values=[0.1, 2.0, 5.0], duration=duration_post
    )

    # Compile results
    results = {
        # LZ complexity measures
        'lz_spatial_patterns': lz_spatial,
        'lz_column_wise': lz_column_wise,  # NEW

        # Shannon entropies
        'shannon_entropy_symbols': shannon_entropy_symbols,
        'shannon_entropy_spikes': shannon_entropy_spikes,

        # Pattern statistics
        'unique_patterns_count': unique_patterns_count,
        'post_pert_symbol_sum': post_pert_symbol_sum,
        'total_spike_differences': total_spike_differences,

        # Settling dynamics
        'settling_time_ms': settling_time_ms,

        # Coincidence measures (now includes 0.1ms)
        **coincidence_results,

        # Metadata
        'perturbation_bin': pert_bin,
        'bin_size': bin_size,
        'perturbation_time': perturbation_time,
        'simulation_duration': simulation_end
    }

    return results
