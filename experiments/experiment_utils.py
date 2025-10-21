# experiments/experiment_utils.py - Extended with task training utilities
"""
Unified utilities for saving, loading, and averaging experimental results.
Extended with task-performance training functions.
"""

import numpy as np
import os
import pickle
from typing import List, Dict, Any
from .base_experiment import BaseExperiment
from sklearn.linear_model import Ridge
from analysis.common_utils import apply_exponential_filter


def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    """
    Save experimental results to pickle file.

    Args:
        results: List of result dictionaries
        filename: Output filename
        use_data_subdir: If True, save to results/data/ subdirectory
    """
    if not os.path.isabs(filename):
        if use_data_subdir:
            results_dir = os.path.join(os.getcwd(), "results", "data")
            full_path = os.path.join(results_dir, filename)
        else:
            full_path = os.path.join(os.getcwd(), filename)
    else:
        full_path = filename

    directory = os.path.dirname(full_path)
    os.makedirs(directory, exist_ok=True)

    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved: {full_path}")


def load_results(filename: str) -> List[Dict[str, Any]]:
    """
    Load experimental results from pickle file.

    Args:
        filename: Input filename

    Returns:
        List of result dictionaries
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded: {len(results)} combinations from {filename}")
    return results


# ==================== Task Training Utilities (REFACTORED) ====================


def train_task_readout(X_train: np.ndarray,
                       Y_train: np.ndarray,
                       lambda_reg: float = 1e-3) -> np.ndarray:
    """
    Train task readout weights with ridge regression.

    Args:
        X_train: Training features (n_train_trials, T, N)
        Y_train: Training targets (n_train_trials, T, n_outputs)
        lambda_reg: Regularization strength

    Returns:
        W_readout: Readout weights (N, n_outputs)
    """
    if X_train.ndim != 3 or Y_train.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got X: {X_train.shape}, Y: {Y_train.shape}")

    n_trials, T, N = X_train.shape
    _, _, n_outputs = Y_train.shape

    # Reshape to (n_trials * T, N) and (n_trials * T, n_outputs)
    X = X_train.reshape(n_trials * T, N)
    y = Y_train.reshape(n_trials * T, n_outputs)

    # # Exact Ridge regression: (X^T X + λI)^-1 X^T y
    # I = np.eye(N)
    # W_readout = np.linalg.solve(X.T @ X + lambda_reg * I, X.T @ y)

    # Approximate Ridge regression: (X^T X + λI)^-1 X^T y

    ridge = Ridge(alpha=lambda_reg, fit_intercept=False, solver='lsqr')  # ← Use iterative solver
    ridge.fit(X, y)
    W_readout = ridge.coef_.T  # Shape: (N, n_outputs)

    return W_readout


def predict_task_readout(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Make predictions using trained readout weights.

    Args:
        X: Input features (n_trials, T, N)
        W: Readout weights (N, n_outputs)

    Returns:
        Predictions (n_trials, T, n_outputs)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")

    n_trials, T, N = X.shape
    X_flat = X.reshape(n_trials * T, N)
    predictions_flat = X_flat @ W

    return predictions_flat.reshape(n_trials, T, -1)


def evaluate_categorical_task(Y_pred: np.ndarray,
                               Y_true: np.ndarray,
                               decision_window_steps: int) -> Dict[str, Any]:
    """
    Evaluate categorical classification performance.

    Args:
        Y_pred: Predicted values (n_trials, T, n_classes)
        Y_true: True class labels (n_trials, T, n_classes) - one-hot encoded
        decision_window_steps: Number of timesteps to average over

    Returns:
        Dictionary with accuracy, confusion matrix, per-class accuracy
    """
    # Use only last decision_window_steps for classification
    if decision_window_steps > Y_pred.shape[1]:
        decision_window_steps = Y_pred.shape[1]

    # Average over decision window
    Y_pred_decision = Y_pred[:, -decision_window_steps:, :]
    Y_pred_avg = Y_pred_decision.mean(axis=1)  # Shape: (n_trials, n_classes)

    # Get predicted class (argmax)
    predicted_class = np.argmax(Y_pred_avg, axis=1)

    # Get true class (ground truth is constant, so take first timestep)
    true_class = np.argmax(Y_true[:, 0, :], axis=1)

    # Overall accuracy
    accuracy = np.mean(predicted_class == true_class)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(true_class, predicted_class)

    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-10)

    return {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_accuracy': per_class_acc.tolist(),
        'predictions': predicted_class.tolist(),
        'targets': true_class.tolist()
    }


def evaluate_temporal_task(Y_pred: np.ndarray,
                           Y_true: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate temporal prediction performance.

    Args:
        Y_pred: Predicted outputs (n_trials, T, n_outputs)
        Y_true: True outputs (n_trials, T, n_outputs)

    Returns:
        Dictionary with RMSE, R², correlation metrics
    """
    n_trials, T, n_outputs = Y_pred.shape

    # Reshape to (n_trials * T, n_outputs) for metrics
    Y_pred_flat = Y_pred.reshape(n_trials * T, n_outputs)
    Y_true_flat = Y_true.reshape(n_trials * T, n_outputs)

    # RMSE per output dimension
    rmse_per_dim = np.sqrt(np.mean((Y_pred_flat - Y_true_flat)**2, axis=0))

    # R² per output dimension
    ss_res = np.sum((Y_pred_flat - Y_true_flat)**2, axis=0)
    ss_tot = np.sum((Y_true_flat - np.mean(Y_true_flat, axis=0))**2, axis=0)
    r2_per_dim = 1 - (ss_res / (ss_tot + 1e-10))

    # Correlation per output dimension
    corr_per_dim = []
    for i in range(n_outputs):
        if np.std(Y_pred_flat[:, i]) > 1e-10 and np.std(Y_true_flat[:, i]) > 1e-10:
            corr = np.corrcoef(Y_pred_flat[:, i], Y_true_flat[:, i])[0, 1]
            corr_per_dim.append(corr)
        else:
            corr_per_dim.append(np.nan)

    return {
        'rmse_mean': float(np.mean(rmse_per_dim)),
        'rmse_per_dim': rmse_per_dim.tolist(),
        'r2_mean': float(np.mean(r2_per_dim)),
        'r2_per_dim': r2_per_dim.tolist(),
        'correlation_mean': float(np.nanmean(corr_per_dim)),
        'correlation_per_dim': corr_per_dim
    }

# ==================== Original Averaging Functions ====================

def average_across_sessions_spontaneous(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average spontaneous activity results across sessions."""
    print(f"Averaging spontaneous results across {len(results_files)} sessions...")

    all_session_results = [load_results(f) for f in results_files]
    n_combinations = len(all_session_results[0])

    averaged_results = []

    for combo_idx in range(n_combinations):
        combo_results = [session_results[combo_idx] for session_results in all_session_results]
        first_result = combo_results[0]

        # Extract and concatenate arrays across sessions
        concatenated_arrays = {}
        array_keys = [k for k in first_result.keys() if k.endswith('_values')]

        for key in array_keys:
            all_values = np.concatenate([r[key] for r in combo_results if key in r])
            concatenated_arrays[key] = all_values

        # Create averaged result
        averaged_result = {
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'duration': first_result['duration'],
            'synaptic_mode': first_result['synaptic_mode'],
            'static_input_mode': first_result['static_input_mode'],
            'original_combination_index': first_result.get('original_combination_index', combo_idx),
            **{key.replace('_values', '_mean'): BaseExperiment.compute_safe_mean(array)
               for key, array in concatenated_arrays.items()},
            **{key.replace('_values', '_std'): BaseExperiment.compute_safe_std(array)
               for key, array in concatenated_arrays.items()},
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': len(concatenated_arrays[list(concatenated_arrays.keys())[0]]) if concatenated_arrays else 0,
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results]
        }

        averaged_results.append(averaged_result)

    print(f"Spontaneous averaging completed: {len(averaged_results)} combinations")
    return averaged_results


def average_across_sessions_stability(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average stability results across sessions."""
    print(f"Averaging stability results across {len(results_files)} sessions...")

    all_session_results = [load_results(f) for f in results_files]
    n_combinations = len(all_session_results[0])

    averaged_results = []

    for combo_idx in range(n_combinations):
        combo_results = [session_results[combo_idx] for session_results in all_session_results]
        first_result = combo_results[0]

        # Extract and concatenate arrays
        concatenated_arrays = {}
        array_keys = [k for k in first_result.keys() if k.endswith('_values')]

        for key in array_keys:
            all_values = np.concatenate([r[key] for r in combo_results if key in r])
            concatenated_arrays[key] = all_values

        # Create averaged result
        averaged_result = {
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'synaptic_mode': first_result['synaptic_mode'],
            'static_input_mode': first_result['static_input_mode'],
            'original_combination_index': first_result.get('original_combination_index', combo_idx),
            **{key.replace('_values', '_mean'): BaseExperiment.compute_safe_mean(array)
               for key, array in concatenated_arrays.items()},
            **{key.replace('_values', '_std'): BaseExperiment.compute_safe_std(array)
               for key, array in concatenated_arrays.items()},
            'settled_fraction': np.mean([r['settled_fraction'] for r in combo_results]),
            'settled_count': np.sum([r['settled_count'] for r in combo_results]),
            'settling_time_ms_mean': np.nanmean([r['settling_time_ms_mean'] for r in combo_results]),
            'settling_time_ms_std': np.nanstd([r['settling_time_ms_mean'] for r in combo_results]),
            'settling_time_median': np.nanmean([r.get('settling_time_median', np.nan) for r in combo_results
                                            if not np.isnan(r.get('settling_time_median', np.nan))])
                                   if any(not np.isnan(r.get('settling_time_median', np.nan)) for r in combo_results)
                                   else np.nan,
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': len(concatenated_arrays[list(concatenated_arrays.keys())[0]]) if concatenated_arrays else 0,
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results]
        }

        averaged_results.append(averaged_result)

    print(f"Stability averaging completed: {len(averaged_results)} combinations")
    return averaged_results


def average_across_sessions_encoding(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average encoding results across sessions with hierarchical statistics."""
    from analysis.statistics_utils import compute_hierarchical_stats

    print(f"Averaging encoding results across {len(results_files)} sessions...")

    all_session_results = [load_results(f) for f in results_files]
    n_combinations = len(all_session_results[0])

    averaged_results = []

    for combo_idx in range(n_combinations):
        combo_results = [session_results[combo_idx] for session_results in all_session_results]
        first_result = combo_results[0]

        # Check if this has neuron data
        has_neuron_data = first_result.get('saved_neuron_data', False)

        averaged_result = {
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'hd_dim': first_result['hd_dim'],
            'embed_dim': first_result['embed_dim'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'synaptic_mode': first_result['synaptic_mode'],
            'static_input_mode': first_result['static_input_mode'],
            'hd_input_mode': first_result['hd_input_mode'],
            'original_combination_index': first_result.get('original_combination_index', combo_idx),
            'n_sessions': len(combo_results),
            'saved_neuron_data': has_neuron_data
        }

        if has_neuron_data:
            # Aggregate neuron-level data across sessions
            all_weights = []
            all_jitter = []
            all_thresholds = []

            for session_result in combo_results:
                decoding = session_result['decoding']
                all_weights.extend(decoding['decoder_weights'])
                all_jitter.extend(decoding['spike_jitter_per_fold'])
                all_thresholds.append(decoding['spike_thresholds'])

            averaged_result['neuron_data'] = {
                'decoder_weights': all_weights,
                'spike_jitter': all_jitter,
                'spike_thresholds': all_thresholds
            }

        # Compute hierarchical stats for encoding metrics
        test_rmse_sessions = [r['decoding']['test_rmse_mean'] for r in combo_results]
        test_r2_sessions = [r['decoding']['test_r2_mean'] for r in combo_results]
        test_corr_sessions = [r['decoding']['test_correlation_mean'] for r in combo_results]

        averaged_result['encoding_metrics'] = {
            'test_rmse': compute_hierarchical_stats([np.array([v]) for v in test_rmse_sessions]),
            'test_r2': compute_hierarchical_stats([np.array([v]) for v in test_r2_sessions]),
            'test_correlation': compute_hierarchical_stats([np.array([v]) for v in test_corr_sessions])
        }

        # For high-dim, also average dimensionality metrics
        if not has_neuron_data:
            weight_pr_sessions = [r['decoding']['weight_participation_ratio_mean'] for r in combo_results]
            weight_ed_sessions = [r['decoding']['weight_effective_dim_mean'] for r in combo_results]
            decoded_pr_sessions = [r['decoding']['decoded_participation_ratio_mean'] for r in combo_results]
            decoded_ed_sessions = [r['decoding']['decoded_effective_dim_mean'] for r in combo_results]

            averaged_result['dimensionality_metrics'] = {
                'weight_participation_ratio': compute_hierarchical_stats([np.array([v]) for v in weight_pr_sessions]),
                'weight_effective_dim': compute_hierarchical_stats([np.array([v]) for v in weight_ed_sessions]),
                'decoded_participation_ratio': compute_hierarchical_stats([np.array([v]) for v in decoded_pr_sessions]),
                'decoded_effective_dim': compute_hierarchical_stats([np.array([v]) for v in decoded_ed_sessions])
            }

        averaged_results.append(averaged_result)

    print(f"Encoding averaging completed: {len(averaged_results)} combinations")
    return averaged_results
