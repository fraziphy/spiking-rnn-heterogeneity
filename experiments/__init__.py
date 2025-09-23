# experiments/__init__.py
"""
Experiment coordination and execution modules for random structure studies.

This package contains high-level experiment classes that coordinate:
- Network construction with random structure per parameter combination
- Parameter space exploration with direct heterogeneity values
- Data collection with single session execution and session averaging
- Synaptic mode comparison (immediate vs dynamic)
"""

from .chaos_experiment import (
    ChaosExperiment,
    create_parameter_grid,
    save_results,
    load_results,
    average_across_sessions
)

__all__ = [
    'ChaosExperiment',
    'create_parameter_grid',
    'save_results',
    'load_results',
    'average_across_sessions'
]

# Future experiments will be added here:
# from .encoding_experiment import EncodingExperiment
# from .task_performance_experiment import TaskPerformanceExperiment
