# analysis/spike_analysis.py - Enhanced with Kistler coincidence, PCI, and new analyses
"""
Spike analysis functions for chaos quantification with additional metrics.
Includes dimensionality analysis, matrix differences, Kistler coincidence factor, and PCI.
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

def lempel_ziv_matrix_flattened(matrix_sorted: np.ndarray) -> int:
    """Compute the Lempel-Ziv complexity of a binary matrix by flattening."""
    matrix_flatten = matrix_sorted.flatten(order='F')
    return lempel_ziv_complexity(matrix_flatten)

def compute_spatial_pattern_complexity(binary_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute complexity measures based on spatial patterns (columns).
    Returns LZ complexity of spatial patterns and PCI measures.
    """
    if binary_matrix.size == 0:
        return {
            'lz_spatial_patterns': 0,
            'pci_raw': 0.0,
            'pci_normalized': 0.0,
            'pci_with_threshold': 0.0,
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
            'pci_raw': 0.0,
            'pci_normalized': 0.0,
            'pci_with_threshold': 0.0,
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

    # Compute spatial entropy for PCI normalization
    total_samples = binary_matrix.size
    if total_samples == 0:
        return {
            'lz_spatial_patterns': lz_spatial,
            'pci_raw': 0.0,
            'pci_normalized': 0.0,
            'pci_with_threshold': 0.0,
            'spatial_entropy': 0.0,
            'pattern_fraction': 0.0,
            'unique_patterns': len(unique_patterns_dict),
            'total_patterns': len(spatial_patterns)
        }

    # Calculate p1 (fraction of 1s in the matrix)
    p1 = np.sum(binary_matrix) / total_samples
    pattern_fraction = p1

    # Compute entropy H(L) = -p1*log2(p1) - (1-p1)*log2(1-p1)
    if p1 == 0 or p1 == 1:
        entropy = 0.0
    else:
        entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)

    # PCI normalization factor: L * H(L) / log2(L)
    L = len(spatial_patterns)
    if L <= 1 or entropy == 0:
        normalization_factor = 1.0
    else:
        normalization_factor = L * entropy / np.log2(L)

    # Raw PCI (without normalization)
    pci_raw = float(lz_spatial)

    # Normalized PCI
    pci_normalized = pci_raw / normalization_factor if normalization_factor > 0 else 0.0

    # PCI with activation threshold (p1 > 0.01 as per Casali paper)
    pci_with_threshold = pci_normalized if pattern_fraction > 0.01 else 0.0

    return {
        'lz_spatial_patterns': lz_spatial,
        'pci_raw': pci_raw,
        'pci_normalized': pci_normalized,
        'pci_with_threshold': pci_with_threshold,
        'spatial_entropy': entropy,
        'pattern_fraction': pattern_fraction,
        'unique_patterns': len(unique_patterns_dict),
        'total_patterns': len(spatial_patterns)
    }

def find_stable_period(sequence, min_repeats=3):
    """
    Find if sequence ends with a repeating pattern.

    Args:
        sequence: Input sequence
        min_repeats: Minimum number of repetitions to consider stable

    Returns:
        dict with period info or None
    """
    sequence = [int(x) for x in sequence]
    n = len(sequence)

    # Try different period lengths, starting from shortest
    for period in range(1, n//3 + 1):  # Need at least 3 repeats
        # Check how far back we can find this period
        repeats = 0
        for start_pos in range(n - period, -1, -period):
            if start_pos + period > n:
                continue

            # Check if this segment matches the pattern
            pattern = sequence[-period:]  # Pattern from end
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

    return None  # No stable period found

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

def compute_activity_dimensionality_multi_bin(binary_matrix: np.ndarray,
                                             bin_sizes: List[float] = [2.0, 5.0, 20.0],
                                             variance_threshold: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Compute dimensionality of network activity using PCA with multiple bin sizes.

    Args:
        binary_matrix: Binary spike matrix (neurons x time_bins)
        bin_sizes: List of bin sizes to test in ms
        variance_threshold: Fraction of variance to capture for effective dimensionality

    Returns:
        Dictionary with dimensionality metrics for each bin size
    """
    results = {}

    for bin_size in bin_sizes:
        # Rebin the matrix if needed
        if bin_size != 2.0:  # Assuming input is at 2ms resolution
            rebin_factor = int(bin_size / 2.0)
            if rebin_factor > 1 and binary_matrix.shape[1] >= rebin_factor:
                # Rebin by summing adjacent bins and clipping to 0/1
                n_new_bins = binary_matrix.shape[1] // rebin_factor
                rebinned_matrix = np.zeros((binary_matrix.shape[0], n_new_bins), dtype=int)

                for i in range(n_new_bins):
                    start_idx = i * rebin_factor
                    end_idx = min((i + 1) * rebin_factor, binary_matrix.shape[1])
                    rebinned_matrix[:, i] = np.clip(
                        np.sum(binary_matrix[:, start_idx:end_idx], axis=1), 0, 1
                    )
                matrix_to_use = rebinned_matrix
            else:
                matrix_to_use = binary_matrix
        else:
            matrix_to_use = binary_matrix

        # Compute dimensionality for this bin size
        dim_result = compute_activity_dimensionality(matrix_to_use, variance_threshold)
        results[f'bin_{bin_size:.0f}ms'] = dim_result

    return results

def kistler_coincidence_factor(spike_train1: List[float], spike_train2: List[float],
                              delta: float = 2.0, duration: float = None) -> float:
    """
    Compute Kistler coincidence factor Γ between two spike trains.

    Based on Kistler et al. (1997) formula:
    Γ = (N_coinc - <N_coinc>) / (1/2 * (N_data + N_SRM) * N)

    Args:
        spike_train1: Reference spike train (data)
        spike_train2: Predicted spike train (SRM)
        delta: Precision window in ms
        duration: Total duration for rate calculation (if None, inferred)

    Returns:
        Coincidence factor between 0 and 1
    """
    if not spike_train1 or not spike_train2:
        return 0.0

    N_data = len(spike_train1)
    N_SRM = len(spike_train2)

    if duration is None:
        duration = max(max(spike_train1), max(spike_train2))

    # Count coincidences within precision delta
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

    # Expected coincidences for homogeneous Poisson process
    if duration > 0:
        rate_SRM = N_SRM / duration  # Rate in Hz (assuming duration in ms, convert accordingly)
        expected_coinc = 2 * rate_SRM * delta * N_data
    else:
        expected_coinc = 0

    # Normalization factor N = 1 - 2*rate*delta
    N = 1 - 2 * (rate_SRM * delta / 1000.0) if duration > 0 else 1  # Convert delta to seconds

    # Compute Γ
    if N > 0 and (N_data + N_SRM) > 0:
        gamma = (N_coinc - expected_coinc) / (0.5 * (N_data + N_SRM) * N)
    else:
        gamma = 0.0

    return max(0.0, min(1.0, gamma))  # Clamp between 0 and 1

def gamma_coincidence(spike_train1: List[float], spike_train2: List[float],
                     window_ms: float = 5.0) -> float:
    """
    Compute normalized gamma coincidence factor between two spike trains (original method).

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

def average_kistler_coincidence(spikes1: List[Tuple[float, int]],
                               spikes2: List[Tuple[float, int]],
                               num_neurons: int,
                               delta_values: List[float] = [2.0, 5.0],
                               duration: float = None) -> Dict[str, float]:
    """
    Compute average Kistler coincidence between two spike train sets.

    Args:
        spikes1: List of (spike_time, neuron_id) tuples for first condition
        spikes2: List of (spike_time, neuron_id) tuples for second condition
        num_neurons: Total number of neurons
        delta_values: List of precision windows to test
        duration: Total simulation duration

    Returns:
        Dictionary with average coincidence factors for each delta
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
        # Compute Kistler coincidence for each neuron
        coincidences = []
        for neuron_id in range(num_neurons):
            kistler_c = kistler_coincidence_factor(
                spikes_net1[neuron_id],
                spikes_net2[neuron_id],
                delta=delta,
                duration=duration
            )
            coincidences.append(kistler_c)

        results[f'kistler_delta_{delta:.0f}ms'] = np.mean(coincidences)

    return results

def average_gamma_coincidence_multi_window(spikes1: List[Tuple[float, int]],
                                          spikes2: List[Tuple[float, int]],
                                          num_neurons: int,
                                          window_values: List[float] = [5.0, 10.0]) -> Dict[str, float]:
    """
    Compute average normalized gamma coincidence between two spike train sets with multiple windows.

    Args:
        spikes1: List of (spike_time, neuron_id) tuples for first condition
        spikes2: List of (spike_time, neuron_id) tuples for second condition
        num_neurons: Total number of neurons
        window_values: List of coincidence time windows (ms)

    Returns:
        Dictionary with average gamma coincidence for each window
    """
    # Organize spikes by neuron
    spikes_net1 = defaultdict(list)
    spikes_net2 = defaultdict(list)

    for spike_time, neuron_id in spikes1:
        spikes_net1[neuron_id].append(spike_time)
    for spike_time, neuron_id in spikes2:
        spikes_net2[neuron_id].append(spike_time)

    results = {}

    for window_ms in window_values:
        # Compute gamma coincidence for each neuron
        coincidences = []
        for neuron_id in range(num_neurons):
            gamma_c = gamma_coincidence(
                spikes_net1[neuron_id],
                spikes_net2[neuron_id],
                window_ms=window_ms
            )
            coincidences.append(gamma_c)

        results[f'gamma_window_{window_ms:.0f}ms'] = np.mean(coincidences)

    return results

# Backward compatibility - keep old function names
def average_gamma_coincidence(spikes1: List[Tuple[float, int]],
                            spikes2: List[Tuple[float, int]],
                            num_neurons: int,
                            window_ms: float = 5.0) -> float:
    """
    Compute average normalized gamma coincidence between two spike train sets (original function).

    This is kept for backward compatibility. Use average_gamma_coincidence_multi_window for enhanced analysis.
    """
    results = average_gamma_coincidence_multi_window(spikes1, spikes2, num_neurons, [window_ms])
    return results[f'gamma_window_{window_ms:.0f}ms']

def analyze_firing_rates_and_silence(spikes: List[Tuple[float, int]],
                                   num_neurons: int, duration: float) -> Dict[str, float]:
    """
    Analyze firing rates and silent neurons.

    Args:
        spikes: List of (spike_time, neuron_id) tuples
        num_neurons: Total number of neurons
        duration: Simulation duration in ms

    Returns:
        Dictionary with firing rate statistics
    """
    # Count spikes per neuron
    spike_counts = defaultdict(int)
    for _, neuron_id in spikes:
        spike_counts[neuron_id] += 1

    # Convert to firing rates (Hz)
    firing_rates = []
    for neuron_id in range(num_neurons):
        rate = (spike_counts[neuron_id] / duration) * 1000.0  # Convert ms to s
        firing_rates.append(rate)

    firing_rates = np.array(firing_rates)

    # Count silent neurons
    silent_neurons = np.sum(firing_rates == 0)
    active_neurons = num_neurons - silent_neurons

    # Statistics
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

def sort_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """Sort binary matrix for Lempel-Ziv complexity calculation."""
    row_sums = np.sum(binary_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    return binary_matrix[sorted_indices, :]

def compute_spike_difference_matrix_enhanced(spikes_control: List[Tuple[float, int]],
                                           spikes_perturbed: List[Tuple[float, int]],
                                           num_neurons: int, perturbation_time: float,
                                           simulation_end: float, perturbed_neuron: int,
                                           bin_size: float = 2.0) -> Tuple[np.ndarray, int]:
    """
    Compute spike difference matrix with total difference count using 2ms bins.

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
    Enhanced perturbation analysis with all complexity measures.

    Returns:
        Dictionary with all chaos and network metrics
    """
    # 1. Compute enhanced spike difference matrix for LZ complexity (2ms bins)
    spike_difference_fine, total_spike_differences = compute_spike_difference_matrix_enhanced(
        spikes_control, spikes_perturbed, num_neurons,
        perturbation_time, simulation_end, perturbed_neuron,
        bin_size=2.0
    )

    # 2. LZ complexity analysis - flattened matrix
    spike_difference_sorted = sort_matrix(spike_difference_fine)
    lz_flattened = lempel_ziv_matrix_flattened(spike_difference_sorted)

    # 3. Spatial pattern complexity and PCI measures
    spatial_results = compute_spatial_pattern_complexity(spike_difference_fine)

    # 4. Pattern stability analysis
    # Convert difference matrix to symbol sequence for stability analysis
    spatial_patterns = []
    for t in range(spike_difference_fine.shape[1]):
        pattern = tuple(spike_difference_fine[:, t])
        spatial_patterns.append(pattern)

    # Create symbol sequence
    unique_patterns_dict = {}
    symbol_sequence = []
    next_symbol = 0

    for pattern in spatial_patterns:
        if pattern not in unique_patterns_dict:
            unique_patterns_dict[pattern] = next_symbol
            next_symbol += 1
        symbol_sequence.append(unique_patterns_dict[pattern])

    # Find stable periods
    stable_period_info = find_stable_period(symbol_sequence, min_repeats=3)

    # 5. Hamming distance slope analysis (using 1ms bins for slope)
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

    # 6. Network activity dimensionality analysis with multiple bin sizes
    duration_post = simulation_end - perturbation_time
    spikes_control_post_full = [(t, n) for t, n in spikes_control if t >= perturbation_time]

    control_binary = spikes_to_binary(spikes_control_post_full, num_neurons,
                                    duration_post, bin_size=2.0)

    dimensionality_metrics = compute_activity_dimensionality_multi_bin(
        control_binary, bin_sizes=[2.0, 5.0, 20.0]
    )

    # 7. Kistler coincidence analysis with multiple deltas
    spikes_perturbed_post_full = [(t, n) for t, n in spikes_perturbed if t >= perturbation_time]

    kistler_results = average_kistler_coincidence(
        spikes_control_post_full, spikes_perturbed_post_full,
        num_neurons, delta_values=[2.0, 5.0], duration=duration_post
    )

    # 8. Gamma coincidence analysis with multiple windows
    gamma_results = average_gamma_coincidence_multi_window(
        spikes_control_post_full, spikes_perturbed_post_full,
        num_neurons, window_values=[5.0, 10.0]
    )

    # 9. Firing rate analysis for both conditions
    control_firing_stats = analyze_firing_rates_and_silence(
        spikes_control_post_full, num_neurons, duration_post
    )

    perturbed_firing_stats = analyze_firing_rates_and_silence(
        spikes_perturbed_post_full, num_neurons, duration_post
    )

    # Compile all results
    results = {
        # Original chaos measures
        'lz_matrix_flattened': lz_flattened,
        'hamming_slope': hamming_slope,

        # New LZ and PCI measures
        'lz_spatial_patterns': spatial_results['lz_spatial_patterns'],
        'pci_raw': spatial_results['pci_raw'],
        'pci_normalized': spatial_results['pci_normalized'],
        'pci_with_threshold': spatial_results['pci_with_threshold'],

        # Spatial pattern statistics
        'spatial_entropy': spatial_results['spatial_entropy'],
        'pattern_fraction': spatial_results['pattern_fraction'],
        'unique_patterns': spatial_results['unique_patterns'],
        'total_patterns': spatial_results['total_patterns'],

        # Pattern stability analysis
        'stable_period': stable_period_info,

        # Matrix difference magnitude
        'total_spike_differences': total_spike_differences,

        # Network activity dimensionality (multiple bin sizes)
        'dimensionality_metrics': dimensionality_metrics,

        # Coincidence measures
        **kistler_results,
        **gamma_results,

        # Firing rate statistics
        'control_firing_stats': control_firing_stats,
        'perturbed_firing_stats': perturbed_firing_stats,
    }

    return results
