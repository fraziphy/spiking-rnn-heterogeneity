# analysis/__init__.py
"""
Analysis modules for spiking RNN experiments.

This package contains split analysis functionality:
- Spontaneous activity: firing rates, dimensionality, Poisson tests
- Stability: LZ complexity, Shannon entropy, settling time, coincidence measures
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

# Stability analysis (updated with new measures)
from .stability_analysis import (
    spikes_to_binary,
    lempel_ziv_complexity,
    compute_shannon_entropy,
    find_settling_time,
    unified_coincidence_factor,
    average_coincidence_multi_window,
    analyze_perturbation_response
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

    # Stability analysis (updated)
    'lempel_ziv_complexity',
    'compute_shannon_entropy',
    'find_settling_time',
    'unified_coincidence_factor',
    'average_coincidence_multi_window',
    'analyze_perturbation_response'
]
