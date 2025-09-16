# experiments/__init__.py
"""
Experiment coordination and execution modules.

This package contains high-level experiment classes that coordinate:
- Network construction and initialization
- Parameter space exploration
- Data collection and storage
- Result summarization and export
"""

from .chaos_experiment import ChaosExperiment, create_parameter_grid, save_results, load_results

__all__ = [
    'ChaosExperiment',
    'create_parameter_grid',
    'save_results',
    'load_results'
]

# Future experiments will be added here:
# from .encoding_experiment import EncodingExperiment
# from .task_performance_experiment import TaskPerformanceExperiment
