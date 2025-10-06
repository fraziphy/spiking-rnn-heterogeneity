# analysis/statistics_utils.py
import numpy as np
from typing import List, Tuple

def get_extreme_combinations(v_th_stds: np.ndarray, g_stds: np.ndarray) -> List[Tuple[float, float]]:
    """Get corner combinations of parameter space for detailed storage."""
    extremes = []
    for v_th in [v_th_stds[0], v_th_stds[-1]]:
        for g in [g_stds[0], g_stds[-1]]:
            extremes.append((v_th, g))
    return extremes

def is_extreme_combo(v_th_std: float, g_std: float,
                    extreme_combos: List[Tuple[float, float]]) -> bool:
    """Check if parameter combination is in extreme list."""
    for v_th, g in extreme_combos:
        if abs(v_th_std - v_th) < 1e-6 and abs(g_std - g) < 1e-6:
            return True
    return False

def compute_hierarchical_stats(session_arrays):
    """Compute mean and std across sessions."""
    all_values = np.concatenate(session_arrays)
    return {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'n_sessions': len(session_arrays),
        'n_total_values': len(all_values)
    }
