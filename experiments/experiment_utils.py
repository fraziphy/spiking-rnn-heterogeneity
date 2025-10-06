# experiments/experiment_utils.py
"""
Unified utilities for saving, loading, and averaging experimental results.
Used by all three experiment types: spontaneous, stability, and encoding.
"""

import numpy as np
import os
import pickle
from typing import List, Dict, Any
from .base_experiment import BaseExperiment


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
