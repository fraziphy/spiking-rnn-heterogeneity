# src/rng_utils.py - Extended for HD input support
"""
Random Number Generator utilities for parameter-dependent network structure.
Extended to support HD input encoding experiments.
"""

import numpy as np
from typing import Dict
import hashlib

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

        # Convert float parameters to reproducible integers
        v_th_int = int(v_th_std * 10000)
        g_int = int(g_std * 10000)
        rate_int = int(rate * 1000)

        # Build deterministic seed string that includes all relevant parameters
        # TRIAL-VARYING: Include time_step and rate for Poisson processes
        # Use startswith() to catch 'hd_input_noise_3', 'hd_input_noise_4', etc.
        if (component.startswith('hd_input_noise') or
            component in ['initial_state', 'static_poisson', 'dynamic_poisson_spikes']):
            seed_string = f"{self.base_seed}_{session_id}_{v_th_int}_{g_int}_{trial_id}_{time_step}_{rate_int}_{hd_dim}_{embed_dim}_{component}"
        else:
            # Structure-determining: Fixed across trials
            seed_string = f"{self.base_seed}_{session_id}_{v_th_int}_{g_int}_{hd_dim}_{embed_dim}_{component}"

        # Hash the string to get a DETERMINISTIC seed
        hash_obj = hashlib.sha256(seed_string.encode('utf-8'))
        hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder='big')

        # Create SeedSequence from the hashed integer
        seed_sequence = np.random.SeedSequence(hash_int)
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
