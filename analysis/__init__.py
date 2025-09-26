# analysis/__init__.py
"""
Analysis and measurement tools for spiking RNN studies.

This package contains tools split into two main analysis types:

1. Spontaneous Activity Analysis (spontaneous_analysis.py):
   - Firing rate statistics (mean, std, silent neurons %)
   - Extended dimensionality analysis with 6 bin sizes (0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms)
   - Participation ratio and variance analysis

2. Network Dynamics Analysis (stability_analysis.py):
   - LZ spatial pattern complexity
   - Unified coincidence measures (Kistler + Gamma, optimized)
   - Hamming distance slope analysis
   - Pattern stability detection
"""

from .spontaneous_analysis import (
    spikes_to_binary as spontaneous_spikes_to_binary,
    compute_activity_dimensionality,
    compute_activity_dimensionality_multi_bin,
    analyze_firing_rates_and_silence,
    analyze_spontaneous_activity
)

from .stability_analysis import (
    spikes_to_binary as stability_spikes_to_binary,
    lempel_ziv_complexity,
    compute_spatial_pattern_complexity,
    find_stable_period,
    unified_coincidence_factor,
    average_coincidence_multi_window,
    sort_matrix,
    compute_spike_difference_matrix,
    compute_hamming_distance_from_matrix,
    compute_chaos_slope_robust,
    analyze_perturbation_response
)

__all__ = [
    # Spontaneous activity analysis
    'spontaneous_spikes_to_binary',
    'compute_activity_dimensionality',
    'compute_activity_dimensionality_multi_bin',
    'analyze_firing_rates_and_silence',
    'analyze_spontaneous_activity',

    # Network stability analysis
    'stability_spikes_to_binary',
    'lempel_ziv_complexity',
    'compute_spatial_pattern_complexity',
    'find_stable_period',
    'unified_coincidence_factor',
    'average_coincidence_multi_window',
    'sort_matrix',
    'compute_spike_difference_matrix',
    'compute_hamming_distance_from_matrix',
    'compute_chaos_slope_robust',
    'analyze_perturbation_response'
]
