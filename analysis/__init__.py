# analysis/__init__.py
"""
Analysis modules for spiking RNN experiments.
"""

# Common utilities (used by multiple analysis modules)
from .common_utils import (
    spikes_to_binary,
    spikes_to_matrix,
    compute_participation_ratio,
    compute_effective_dimensionality,
    compute_dimensionality_from_covariance
)

# Statistics utilities
from .statistics_utils import (
    get_extreme_combinations,
    is_extreme_combo,
    compute_hierarchical_stats
)

# Spontaneous activity analysis
from .spontaneous_analysis import (
    analyze_firing_rates_and_silence,
    compute_activity_dimensionality_multi_bin,
    extract_neuron_spike_trains,
    compute_isi_statistics,
    test_exponential_isi_distribution,
    test_count_poisson_distribution,
    analyze_population_poisson_properties,
    analyze_spontaneous_activity
)

# Stability analysis
from .stability_analysis import (
    lempel_ziv_complexity,
    compute_shannon_entropy,
    find_settling_time,
    unified_coincidence_factor,
    average_coincidence_multi_window,
    analyze_perturbation_response
)

# Encoding analysis
from .encoding_analysis import decode_hd_input

__all__ = [
    # Common utilities
    'spikes_to_binary',
    'spikes_to_matrix',
    'compute_participation_ratio',
    'compute_effective_dimensionality',
    'compute_dimensionality_from_covariance',

    # Statistics utilities
    'get_extreme_combinations',
    'is_extreme_combo',
    'compute_hierarchical_stats',

    # Spontaneous analysis
    'analyze_firing_rates_and_silence',
    'compute_activity_dimensionality_multi_bin',
    'extract_neuron_spike_trains',
    'compute_isi_statistics',
    'test_exponential_isi_distribution',
    'test_count_poisson_distribution',
    'analyze_population_poisson_properties',
    'analyze_spontaneous_activity',

    # Stability analysis
    'lempel_ziv_complexity',
    'compute_shannon_entropy',
    'find_settling_time',
    'unified_coincidence_factor',
    'average_coincidence_multi_window',
    'analyze_perturbation_response',

    # Encoding analysis
    'decode_hd_input'
]
