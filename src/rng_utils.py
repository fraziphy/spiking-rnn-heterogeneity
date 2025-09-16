# rng_utils.py
"""
Random Number Generator utilities for reproducible spiking RNN experiments.
Provides hierarchical seeding for session_id -> block_id -> trial_id structure.
"""

import numpy as np
from typing import Dict, Any

class HierarchicalRNG:
    """
    Manages hierarchical random number generation for reproducible experiments.

    Structure:
    - session_id: Changes all random generations across parameter sweeps
    - block_id: Changes RNG for different v_th_std and g_std combinations
    - trial_id: Changes only initial states and Poisson process, keeps network structure
    """

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self._rngs: Dict[str, np.random.Generator] = {}

    def get_rng(self, session_id: int, block_id: int, trial_id: int,
                component: str) -> np.random.Generator:
        """
        Get RNG for specific component with hierarchical seeding.

        Args:
            session_id: Session identifier
            block_id: Block identifier (for parameter combinations)
            trial_id: Trial identifier
            component: Component name ('network', 'initial_state', 'poisson', etc.)

        Returns:
            numpy random generator
        """
        # Create unique seed based on hierarchy
        if component in ['spike_threshold', 'synaptic_weights', 'connectivity']:
            # These depend only on session_id and block_id (fixed per parameter combo)
            seed_components = [self.base_seed, session_id, block_id]
        elif component in ['initial_state', 'poisson_process', 'perturbation']:
            # These change with trial_id
            seed_components = [self.base_seed, session_id, block_id, trial_id]
        else:
            # Default: all levels
            seed_components = [self.base_seed, session_id, block_id, trial_id]

        # Add component-specific offset
        component_offset = hash(component) % 1000000
        seed_components.append(component_offset)

        # Create unique key
        key = f"{session_id}_{block_id}_{trial_id}_{component}"

        if key not in self._rngs:
            seed_sequence = np.random.SeedSequence(seed_components)
            self._rngs[key] = np.random.default_rng(seed_sequence)

        return self._rngs[key]

    def clear_cache(self):
        """Clear RNG cache to free memory."""
        self._rngs.clear()

# Global instance
rng_manager = HierarchicalRNG()

def get_rng(session_id: int, block_id: int, trial_id: int,
           component: str) -> np.random.Generator:
    """Convenience function to get RNG."""
    return rng_manager.get_rng(session_id, block_id, trial_id, component)
