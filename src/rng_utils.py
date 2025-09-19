# src/rng_utils.py - Updated for fixed network structure across parameter combinations
"""
Random Number Generator utilities with fixed network structure.
Network topology depends only on session_id, not parameter values.
"""

import numpy as np
from typing import Dict, Any

class HierarchicalRNG:
    """
    Manages hierarchical RNG with fixed network structure across parameter sweeps.

    Key principle: Network structure (spike thresholds, weights, connectivity,
    perturbation targets) depends only on session_id. Only Poisson processes
    change with trial_id.
    """

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self._rngs: Dict[str, np.random.Generator] = {}

    def get_rng(self, session_id: int, block_id: int, trial_id: int,
                component: str) -> np.random.Generator:
        """Get RNG for specific component with controlled seeding hierarchy."""

        # TRIAL-VARYING: Poisson processes change with trial_id only
        if component in ['initial_state', 'static_poisson', 'dynamic_poisson_spikes']:
            seed_components = [self.base_seed, session_id, trial_id]
            key = f"session_{session_id}_trial_{trial_id}_comp_{component}"
        # Default fallback for other components
        else:
            seed_components = [self.base_seed, session_id, block_id, trial_id]
            key = f"session_{session_id}_block_{block_id}_trial_{trial_id}_comp_{component}"

        # Add component-specific offset (make sure it's non-negative)
        component_offset = abs(hash(component)) % 1000000
        seed_components.append(component_offset)

        if key not in self._rngs:
            seed_sequence = np.random.SeedSequence(seed_components)
            self._rngs[key] = np.random.default_rng(seed_sequence)

        return self._rngs[key]

    def generate_base_distributions(self, session_id: int, n_neurons: int,
                                n_input_channels: int = 20) -> Dict[str, np.ndarray]:
        """Generate base distributions that remain fixed across parameter combinations."""

        # Create fresh RNG instances each time (don't use cache for base distributions)
        # Use abs(hash()) to ensure non-negative integers
        v_th_seed = np.random.SeedSequence([self.base_seed, session_id, abs(hash('base_spike_thresholds'))])
        v_th_rng = np.random.default_rng(v_th_seed)

        g_seed = np.random.SeedSequence([self.base_seed, session_id, abs(hash('base_synaptic_weights'))])
        g_rng = np.random.default_rng(g_seed)

        conn_seed = np.random.SeedSequence([self.base_seed, session_id, abs(hash('connectivity'))])
        conn_rng = np.random.default_rng(conn_seed)

        input_conn_seed = np.random.SeedSequence([self.base_seed, session_id, abs(hash('dynamic_input_connectivity'))])
        input_conn_rng = np.random.default_rng(input_conn_seed)

        pert_seed = np.random.SeedSequence([self.base_seed, session_id, abs(hash('perturbation_targets'))])
        pert_rng = np.random.default_rng(pert_seed)

        # Generate distributions with fresh RNGs
        base_v_th = v_th_rng.normal(-55.0, 0.01, n_neurons)
        base_v_th = base_v_th - np.mean(base_v_th) + (-55.0)

        base_g = g_rng.normal(0.0, 0.01, (n_neurons, n_neurons))
        base_g = base_g - np.mean(base_g) + 0.0

        connectivity = conn_rng.random((n_neurons, n_neurons)) < 0.1
        np.fill_diagonal(connectivity, False)

        dynamic_connectivity = input_conn_rng.random((n_neurons, n_input_channels)) < 0.3

        sample_size = min(100, n_neurons)
        perturbation_neurons = pert_rng.choice(n_neurons, size=sample_size, replace=False)

        return {
            'base_v_th': base_v_th,
            'base_g': base_g,
            'connectivity': connectivity,
            'dynamic_connectivity': dynamic_connectivity,
            'perturbation_neurons': perturbation_neurons
        }

    def clear_cache(self):
        """Clear RNG cache to free memory."""
        self._rngs.clear()

# Global instance
rng_manager = HierarchicalRNG()

def get_rng(session_id: int, block_id: int, trial_id: int,
           component: str) -> np.random.Generator:
    """Convenience function to get RNG."""
    return rng_manager.get_rng(session_id, block_id, trial_id, component)

def generate_base_distributions(session_id: int, n_neurons: int,
                               n_input_channels: int = 20) -> Dict[str, np.ndarray]:
    """Convenience function to generate base distributions."""
    return rng_manager.generate_base_distributions(session_id, n_neurons, n_input_channels)
