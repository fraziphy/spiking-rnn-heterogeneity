# experiments/__init__.py
"""
Experiment coordination and execution modules for random structure studies.

This package contains high-level experiment classes that coordinate:
- Network construction with random structure per parameter combination
- Parameter space exploration with direct heterogeneity values
- Data collection with single session execution and session averaging
- Synaptic mode comparison (immediate vs dynamic)

Split into three main experiment types:
- SpontaneousExperiment: Firing rates, dimensionality analysis, Poisson tests
- StabilityExperiment: Perturbation response, LZ complexity, Shannon entropy, settling time
- EncodingExperiment: HD input encoding capacity studies
"""

from .spontaneous_experiment import (
    SpontaneousExperiment,
    create_parameter_grid as spontaneous_create_parameter_grid,
    save_results as spontaneous_save_results,
    load_results as spontaneous_load_results,
    average_across_sessions as spontaneous_average_across_sessions
)

from .stability_experiment import (
    StabilityExperiment,
    create_parameter_grid as stability_create_parameter_grid,
    save_results as stability_save_results,
    load_results as stability_load_results,
    average_across_sessions as stability_average_across_sessions
)

from .encoding_experiment import (
    EncodingExperiment,
    create_parameter_grid as encoding_create_parameter_grid,
    save_results as encoding_save_results,
    load_results as encoding_load_results,
    average_across_sessions as encoding_average_across_sessions
)

__all__ = [
    # Spontaneous activity
    'SpontaneousExperiment',
    'spontaneous_create_parameter_grid',
    'spontaneous_save_results',
    'spontaneous_load_results',
    'spontaneous_average_across_sessions',

    # Network stability
    'StabilityExperiment',
    'stability_create_parameter_grid',
    'stability_save_results',
    'stability_load_results',
    'stability_average_across_sessions',

    # Encoding capacity
    'EncodingExperiment',
    'encoding_create_parameter_grid',
    'encoding_save_results',
    'encoding_load_results',
    'encoding_average_across_sessions'
]
