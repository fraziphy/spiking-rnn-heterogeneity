# experiments/base_experiment.py
"""
Base experiment class with shared functionality across all experiment types.
Eliminates code duplication while maintaining flexibility for experiment-specific needs.
"""

import numpy as np
import warnings
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    """Base class for all experiments with common utilities."""

    def __init__(self, n_neurons: int, dt: float = 0.1):
        """
        Initialize base experiment.

        Args:
            n_neurons: Number of neurons
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.dt = dt

    @staticmethod
    def create_parameter_grid(
        n_v_th_points: int = 10,
        n_g_points: int = 10,
        v_th_std_range: Tuple[float, float] = (0.0, 4.0),
        g_std_range: Tuple[float, float] = (0.0, 4.0),
        input_rate_range: Tuple[float, float] = (50.0, 1000.0),
        n_input_rates: int = 5,
        n_hd_points: int = None,
        hd_dim_range: Tuple[int, int] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create parameter grids for experiments.

        Returns different tuples based on experiment type:
        - Spontaneous/Stability: (v_th_stds, g_stds, static_input_rates)
        - Encoding: (v_th_stds, g_stds, hd_dims, static_input_rates)
        """
        v_th_stds = np.linspace(v_th_std_range[0], v_th_std_range[1], n_v_th_points)
        g_stds = np.linspace(g_std_range[0], g_std_range[1], n_g_points)
        static_input_rates = np.linspace(input_rate_range[0], input_rate_range[1], n_input_rates)

        if n_hd_points is not None and hd_dim_range is not None:
            hd_dims = np.linspace(hd_dim_range[0], hd_dim_range[1], n_hd_points, dtype=int)
            return v_th_stds, g_stds, hd_dims, static_input_rates

        return v_th_stds, g_stds, static_input_rates

    @staticmethod
    def compute_safe_mean(array: np.ndarray) -> float:
        """Compute mean suppressing empty slice warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                  message='Mean of empty slice')
            return float(np.nanmean(array))

    @staticmethod
    def compute_safe_std(array: np.ndarray) -> float:
        """Compute std suppressing degrees of freedom warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                  message='Degrees of freedom')
            return float(np.nanstd(array))

    def create_parameter_combinations(self, session_id: int,
                                     v_th_stds: np.ndarray,
                                     g_stds: np.ndarray,
                                     static_input_rates: np.ndarray,
                                     v_th_distribution: str = "normal",
                                     hd_dims: Optional[np.ndarray] = None,
                                     **extra_params) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations with randomization.

        Args:
            session_id: Session ID for reproducible randomization
            v_th_stds: Threshold std values
            g_stds: Weight std values
            static_input_rates: Input rate values
            v_th_distribution: Threshold distribution
            hd_dims: Optional HD dimensionalities (for encoding experiments)
            **extra_params: Additional experiment-specific parameters

        Returns:
            List of parameter combination dictionaries with combo_idx
        """
        param_combinations = []
        combo_idx = 0

        if hd_dims is not None:
            # Encoding experiment with HD dimensions
            for input_rate in static_input_rates:
                for hd_dim in hd_dims:
                    for v_th_std in v_th_stds:
                        for g_std in g_stds:
                            param_combinations.append({
                                'combo_idx': combo_idx,
                                'session_id': session_id,
                                'v_th_std': v_th_std,
                                'g_std': g_std,
                                'hd_dim': int(hd_dim),
                                'v_th_distribution': v_th_distribution,
                                'static_input_rate': input_rate,
                                **extra_params
                            })
                            combo_idx += 1
        else:
            # Spontaneous/Stability experiments without HD dimensions
            for input_rate in static_input_rates:
                for v_th_std in v_th_stds:
                    for g_std in g_stds:
                        param_combinations.append({
                            'combo_idx': combo_idx,
                            'session_id': session_id,
                            'v_th_std': v_th_std,
                            'g_std': g_std,
                            'v_th_distribution': v_th_distribution,
                            'static_input_rate': input_rate,
                            **extra_params
                        })
                        combo_idx += 1

        # Randomize with reproducible seed
        random.seed(session_id)
        random.shuffle(param_combinations)

        return param_combinations

    @abstractmethod
    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract arrays from trial results (experiment-specific).

        Must be implemented by subclasses.

        Args:
            trial_results: List of trial result dictionaries

        Returns:
            Dictionary of arrays with '_values' suffix
        """
        pass

    def compute_all_statistics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute mean and std for all _values arrays.

        Args:
            arrays: Dictionary of arrays with '_values' suffix

        Returns:
            Dictionary with '_mean' and '_std' for each array
        """
        stats = {}

        for key, array in arrays.items():
            if key.endswith('_values'):
                base_name = key[:-7]  # Remove '_values' suffix
                stats[f'{base_name}_mean'] = self.compute_safe_mean(array)
                stats[f'{base_name}_std'] = self.compute_safe_std(array)

        return stats
