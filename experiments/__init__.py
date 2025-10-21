# experiments/__init__.py
"""
Experiment coordination and execution modules.
"""

from .base_experiment import BaseExperiment
from .spontaneous_experiment import SpontaneousExperiment
from .stability_experiment import StabilityExperiment
from .encoding_experiment import EncodingExperiment
from .task_performance_experiment import TaskPerformanceExperiment

from .experiment_utils import (
    save_results,
    load_results,
    average_across_sessions_spontaneous,
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
    'SpontaneousExperiment',
    'StabilityExperiment',
    'EncodingExperiment',
    'TaskPerformanceExperiment',
    'save_results',
    'load_results',
    'average_across_sessions_spontaneous',
    'average_across_sessions_stability',
    'average_across_sessions_encoding',
    'apply_exponential_filter',
    'train_task_readout',
    'predict_task_readout',
    'evaluate_categorical_task',
    'evaluate_temporal_task'
]
