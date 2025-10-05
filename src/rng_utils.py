# src/rng_utils.py - Extended for HD input support
"""
Random Number Generator utilities for parameter-dependent network structure.
Extended to support HD input encoding experiments.
"""

import numpy as np
from typing import Dict

class HierarchicalRNG:
    """
    Manages hierarchical RNG with parameter-dependent network structure.

    Key principle: Network structure (spike thresholds, weights, connectivity)
    depends on session_id AND parameter combination. Only trial-varying
    processes (Poisson spikes, initial states) change with trial_id.

    Extended for HD inputs: hd_dim and embed_dim are used to generate
    different HD input structures.
    """

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self._rngs: Dict[str, np.random.Generator] = {}

    def get_rng(self, session_id: int, v_th_std: float, g_std: float, trial_id: int,
                component: str, time_step: int = 0, rate: float = 0.0,
                hd_dim: int = 0, embed_dim: int = 0) -> np.random.Generator:
        """Get RNG for specific component with parameter-dependent seeding.

        Args:
            session_id: Session ID for reproducibility across sessions
            v_th_std: Threshold standard deviation
            g_std: Weight standard deviation
            trial_id: Trial ID (for trial-varying processes)
            component: Component identifier string
            time_step: Time step (for Poisson processes)
            rate: Rate parameter (for Poisson processes)
            hd_dim: HD intrinsic dimensionality (for encoding experiments, default 0)
            embed_dim: HD embedding dimensionality (for encoding experiments, default 0)
        """

        # Convert float parameters to reproducible integers
        v_th_int = int(v_th_std * 10000)
        g_int = int(g_std * 10000)
        rate_int = int(rate * 1000000)  # Convert rate to integer (6 decimal precision)

        # TRIAL-VARYING: Include time_step and rate for Poisson processes
        if component in ['initial_state', 'static_poisson', 'dynamic_poisson_spikes', 'hd_input_noise']:
            seed_components = [self.base_seed, session_id, v_th_int, g_int, trial_id, time_step, rate_int, hd_dim, embed_dim]
        else:
            # Structure-determining: Fixed across trials
            seed_components = [self.base_seed, session_id, v_th_int, g_int, hd_dim, embed_dim]

        component_offset = abs(hash(component)) % 1000000
        seed_components.append(component_offset)

        seed_sequence = np.random.SeedSequence(seed_components)
        return np.random.default_rng(seed_sequence)


    def clear_cache(self):
        """Clear RNG cache to free memory."""
        self._rngs.clear()

    def reset_for_testing(self):
        """Reset state for testing to ensure independence."""
        self._rngs.clear()
        # Force garbage collection of any remaining RNG objects
        import gc
        gc.collect()

# Global instance
rng_manager = HierarchicalRNG()

def get_rng(session_id: int, v_th_std: float, g_std: float, trial_id: int,
           component: str, time_step: int = 0, rate: float = 0.0,
           hd_dim: int = 0, embed_dim: int = 0) -> np.random.Generator:
    """Convenience function to get parameter-dependent RNG.

    Extended for HD inputs with hd_dim and embed_dim parameters.
    """
    return rng_manager.get_rng(session_id, v_th_std, g_std, trial_id,
                               component, time_step, rate, hd_dim, embed_dim)
