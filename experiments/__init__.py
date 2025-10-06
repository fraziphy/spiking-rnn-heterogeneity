# experiments/__init__.py
"""
Experiment coordination and execution modules.
"""

from .base_experiment import BaseExperiment
from .spontaneous_experiment import SpontaneousExperiment
from .stability_experiment import StabilityExperiment
from .encoding_experiment import EncodingExperiment

from .experiment_utils import (
    save_results,
    load_results,
    average_across_sessions_spontaneous,
    average_across_sessions_stability,
    average_across_sessions_encoding
)

__all__ = [
    'BaseExperiment',
    'SpontaneousExperiment',
    'StabilityExperiment',
    'EncodingExperiment',
    'save_results',
    'load_results',
    'average_across_sessions_spontaneous',
    'average_across_sessions_stability',
    'average_across_sessions_encoding'
]
