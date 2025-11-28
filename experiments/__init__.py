# experiments/__init__.py
"""
Experiment coordination and execution modules.
"""
from .base_experiment import BaseExperiment
from .stability_experiment import StabilityExperiment
from .task_performance_experiment import TaskPerformanceExperiment
from .experiment_utils import (
    save_results,
    load_results,
    average_across_sessions_stability,
    average_across_sessions_encoding,
    apply_exponential_filter,
    train_task_readout,
    predict_task_readout,
    evaluate_categorical_task,
    evaluate_temporal_task
)

__all__ = [
    'BaseExperiment',
    'StabilityExperiment',
    'TaskPerformanceExperiment',
    'save_results',
    'load_results',
    'average_across_sessions_stability',
    'average_across_sessions_encoding',
    'apply_exponential_filter',
    'train_task_readout',
    'predict_task_readout',
    'evaluate_categorical_task',
    'evaluate_temporal_task'
]
