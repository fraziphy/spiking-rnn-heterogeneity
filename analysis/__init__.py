# analysis/__init__.py
"""
Analysis modules for spiking RNN experiments.

This package contains split analysis functionality:
- Spontaneous activity: firing rates, dimensionality, Poisson tests
- Stability: LZ complexity (spatial + column-wise), Shannon entropy, settling time, coincidence measures
- Encoding: decoding analysis, encoding capacity metrics
"""

# Spontaneous activity analysis
from .spontaneous_analysis import (
    spikes_to_binary,
    compute_activity_dimensionality,
    compute_activity_dimensionality_multi_bin,
    analyze_firing_rates_and_silence,
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
from .encoding_analysis import (
    decode_hd_input,
    analyze_encoding_capacity,
    compare_across_hd_dims
)

__all__ = [
    # Spontaneous analysis
    'spikes_to_binary',
    'compute_activity_dimensionality',
    'compute_activity_dimensionality_multi_bin',
    'analyze_firing_rates_and_silence',
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
    'decode_hd_input',
    'analyze_encoding_capacity',
    'compare_across_hd_dims'
]
