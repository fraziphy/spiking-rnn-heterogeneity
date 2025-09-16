# analysis/__init__.py
"""
Analysis and measurement tools for spiking RNN studies.

This package contains tools for:
- Chaos quantification (Lempel-Ziv complexity, Hamming distance slopes)
- Spike train analysis and visualization
- Statistical analysis of parameter spaces
- Data visualization and plotting utilities
"""

from .spike_analysis import (
    spikes_to_binary,
    sort_matrix,
    lempel_ziv_complexity,
    compute_spike_difference_matrix,
    compute_chaos_slope_robust,
    analyze_perturbation_response
)

__all__ = [
    'spikes_to_binary',
    'sort_matrix',
    'lempel_ziv_complexity',
    'compute_spike_difference_matrix',
    'compute_chaos_slope_robust',
    'analyze_perturbation_response'
]

# Note: Other analysis modules will be imported as they are created
# from .visualize_chaos import plot_chaos_heatmap, plot_parameter_sweep
# from .statistical_analysis import chaos_correlation_analysis
# from .parameter_exploration import find_optimal_parameters
