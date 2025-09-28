# analysis/spontaneous_analysis.py - Spontaneous activity analysis with Poisson tests
"""
Spontaneous activity analysis: firing rates, dimensionality, silent neurons, and Poisson process tests.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from scipy import stats

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

def extract_neuron_spike_trains(spikes: List[Tuple[float, int]],
                               num_neurons: int) -> Dict[int, List[float]]:
    """Extract individual spike trains for each neuron."""
    spike_trains = defaultdict(list)

    for spike_time, neuron_id in spikes:
        spike_trains[neuron_id].append(spike_time)

    return dict(spike_trains)

def compute_isi_statistics(spike_train: List[float]) -> Dict[str, float]:
    """Compute inter-spike interval statistics for a single neuron."""
    if len(spike_train) < 2:
        return {
            'mean_isi': np.nan,
            'std_isi': np.nan,
            'cv_isi': np.nan,
            'n_intervals': 0
        }

    # Compute inter-spike intervals
    isis = np.diff(sorted(spike_train))

    return {
        'mean_isi': float(np.mean(isis)),
        'std_isi': float(np.std(isis)),
        'cv_isi': float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else np.nan,
        'n_intervals': len(isis)
    }

def test_exponential_isi_distribution(spike_train: List[float],
                                    alpha: float = 0.05) -> Dict[str, Any]:
    """Test if inter-spike intervals follow exponential distribution (Poisson property)."""
    if len(spike_train) < 3:
        return {
            'ks_statistic': np.nan,
            'ks_pvalue': np.nan,
            'is_exponential': False
        }

    isis = np.diff(sorted(spike_train))

    # Kolmogorov-Smirnov test against exponential distribution
    # Fit exponential: rate = 1/mean_isi
    rate = 1.0 / np.mean(isis)
    ks_stat, ks_pvalue = stats.kstest(isis, lambda x: stats.expon.cdf(x, scale=1/rate))

    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'is_exponential': ks_pvalue > alpha
    }

def test_count_poisson_distribution(spike_train: List[float],
                                  duration: float,
                                  bin_size: float = 10.0,
                                  alpha: float = 0.05) -> Dict[str, Any]:
    """Test if spike counts in time bins follow Poisson distribution."""
    if len(spike_train) == 0:
        return {
            'chi2_statistic': np.nan,
            'chi2_pvalue': np.nan,
            'is_poisson': False,
            'mean_count': 0.0,
            'var_count': 0.0,
            'fano_factor': np.nan,
            'n_bins': 0
        }

    # Create time bins
    n_bins = int(duration / bin_size)
    bins = np.linspace(0, duration, n_bins + 1)

    # Count spikes in each bin
    counts, _ = np.histogram(spike_train, bins=bins)

    # Compute statistics
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    fano_factor = var_count / mean_count if mean_count > 0 else np.nan

    # Chi-square goodness of fit test for Poisson
    if mean_count > 0 and len(counts) > 10:  # Increased minimum requirement
        try:
            # Use simpler approach - group into fewer bins
            max_count = min(int(np.max(counts)) + 1, 10)  # Limit bins

            # Observed frequencies
            observed = np.array([np.sum(counts == k) for k in range(max_count)])
            observed = np.append(observed, np.sum(counts >= max_count))

            # Expected frequencies
            expected = np.array([stats.poisson.pmf(k, mean_count) * len(counts)
                               for k in range(max_count)])
            expected = np.append(expected, stats.poisson.sf(max_count - 1, mean_count) * len(counts))

            # Remove bins with very low expected counts
            valid_mask = expected >= 5.0
            if np.sum(valid_mask) >= 2:  # Need at least 2 bins
                observed_valid = observed[valid_mask]
                expected_valid = expected[valid_mask]

                # Normalize to ensure exact sum match (fix floating point issues)
                expected_valid = expected_valid * np.sum(observed_valid) / np.sum(expected_valid)

                chi2_stat, chi2_pvalue = stats.chisquare(observed_valid, expected_valid)
            else:
                chi2_stat, chi2_pvalue = np.nan, np.nan

        except Exception as e:
            # If chi-square test fails, fall back to NaN
            chi2_stat, chi2_pvalue = np.nan, np.nan
    else:
        chi2_stat, chi2_pvalue = np.nan, np.nan

    return {
        'chi2_statistic': float(chi2_stat) if not np.isnan(chi2_stat) else np.nan,
        'chi2_pvalue': float(chi2_pvalue) if not np.isnan(chi2_pvalue) else np.nan,
        'is_poisson': chi2_pvalue > alpha if not np.isnan(chi2_pvalue) else False,
        'mean_count': float(mean_count),
        'var_count': float(var_count),
        'fano_factor': float(fano_factor) if not np.isnan(fano_factor) else np.nan,
        'n_bins': n_bins
    }

def analyze_population_poisson_properties(spikes: List[Tuple[float, int]],
                                        num_neurons: int,
                                        duration: float,
                                        bin_size: float = 10.0,
                                        min_spikes: int = 10) -> Dict[str, Any]:
    """Comprehensive Poisson analysis for the entire population."""

    # Extract spike trains
    spike_trains = extract_neuron_spike_trains(spikes, num_neurons)

    # Analyze each neuron
    neuron_results = {}
    isi_results = []
    count_results = []

    active_neurons = 0
    poisson_isi_neurons = 0
    poisson_count_neurons = 0

    for neuron_id in range(num_neurons):
        if neuron_id in spike_trains and len(spike_trains[neuron_id]) >= min_spikes:
            train = spike_trains[neuron_id]
            active_neurons += 1

            # ISI analysis
            isi_stats = compute_isi_statistics(train)
            isi_test = test_exponential_isi_distribution(train)

            # Count analysis
            count_test = test_count_poisson_distribution(train, duration, bin_size)

            neuron_results[neuron_id] = {
                'isi_statistics': isi_stats,
                'isi_exponential_test': isi_test,
                'count_poisson_test': count_test
            }

            # Collect for population statistics
            if not np.isnan(isi_stats['cv_isi']):
                isi_results.append(isi_stats['cv_isi'])

            if not np.isnan(count_test['fano_factor']):
                count_results.append(count_test['fano_factor'])

            # Count Poisson-like neurons
            if isi_test['is_exponential']:
                poisson_isi_neurons += 1
            if count_test['is_poisson']:
                poisson_count_neurons += 1

    # Population statistics
    population_stats = {
        'total_neurons': num_neurons,
        'active_neurons': active_neurons,
        'neurons_with_sufficient_spikes': len(neuron_results),
        'poisson_isi_fraction': poisson_isi_neurons / len(neuron_results) if len(neuron_results) > 0 else 0.0,
        'poisson_count_fraction': poisson_count_neurons / len(neuron_results) if len(neuron_results) > 0 else 0.0,
        'mean_cv_isi': float(np.mean(isi_results)) if isi_results else np.nan,
        'std_cv_isi': float(np.std(isi_results)) if isi_results else np.nan,
        'mean_fano_factor': float(np.mean(count_results)) if count_results else np.nan,
        'std_fano_factor': float(np.std(count_results)) if count_results else np.nan,
        'expected_cv_poisson': 1.0,  # CV should be 1 for Poisson
        'expected_fano_poisson': 1.0,  # Fano factor should be 1 for Poisson
    }

    return {
        'population_statistics': population_stats,
        'individual_neurons': neuron_results,
        'analysis_parameters': {
            'duration': duration,
            'bin_size': bin_size,
            'min_spikes_threshold': min_spikes
        }
    }

def analyze_spontaneous_activity(spikes: List[Tuple[float, int]],
                               num_neurons: int,
                               duration: float,
                               transient_time: float = 25.0) -> Dict[str, Any]:
    """
    Complete spontaneous activity analysis with transient removal and Poisson tests.

    Args:
        spikes: List of (spike_time, neuron_id) tuples
        num_neurons: Total number of neurons
        duration: Simulation duration in milliseconds
        transient_time: Time to discard at beginning (milliseconds)

    Returns:
        Dictionary with firing rate statistics, dimensionality metrics, and Poisson analysis
    """

    # Remove transient period
    steady_spikes = [(t - transient_time, n) for t, n in spikes if t >= transient_time]
    steady_duration = duration - transient_time

    # Firing rate analysis
    firing_stats = analyze_firing_rates_and_silence(steady_spikes, num_neurons, steady_duration)

    # Dimensionality analysis with extended bin sizes
    dimensionality_metrics = compute_activity_dimensionality_multi_bin(
        steady_spikes, num_neurons, steady_duration,
        bin_sizes=[0.1, 2.0, 5.0, 20.0, 50.0, 100.0]
    )

    # Poisson process analysis
    poisson_analysis = analyze_population_poisson_properties(
        steady_spikes, num_neurons, steady_duration
    )

    return {
        'firing_stats': firing_stats,
        'dimensionality_metrics': dimensionality_metrics,
        'poisson_analysis': poisson_analysis,
        'transient_time': transient_time,
        'steady_state_duration': steady_duration,
        'duration_ms': duration,
        'total_spikes': len(spikes),
        'steady_state_spikes': len(steady_spikes)
    }
