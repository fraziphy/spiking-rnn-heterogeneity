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
        """
        Get RNG for specific component with controlled seeding hierarchy.

        Args:
            session_id: Session identifier
            block_id: Block identifier (for parameter combinations)
            trial_id: Trial identifier
            component: Component name

        Returns:
            numpy random generator
        """
        # FIXED NETWORK STRUCTURE: Only session_id affects these
        if component in ['base_spike_thresholds', 'base_synaptic_weights',
                        'connectivity', 'perturbation_targets',
                        'dynamic_input_connectivity']:
            seed_components = [self.base_seed, session_id]
            key_suffix = f"{session_id}_{component}"

        # TRIAL-VARYING: Poisson processes change with trial_id
        elif component in ['initial_state', 'static_poisson', 'dynamic_poisson_spikes']:
            seed_components = [self.base_seed, session_id, trial_id]
            key_suffix = f"{session_id}_{trial_id}_{component}"

        # Default fallback
        else:
            seed_components = [self.base_seed, session_id, block_id, trial_id]
            key_suffix = f"{session_id}_{block_id}_{trial_id}_{component}"

        # Add component-specific offset
        component_offset = hash(component) % 1000000
        seed_components.append(component_offset)

        # Create unique key
        key = f"{key_suffix}"

        if key not in self._rngs:
            seed_sequence = np.random.SeedSequence(seed_components)
            self._rngs[key] = np.random.default_rng(seed_sequence)

        return self._rngs[key]

    def generate_base_distributions(self, session_id: int, n_neurons: int,
                                   n_input_channels: int = 20) -> Dict[str, np.ndarray]:
        """
        Generate base distributions that remain fixed across parameter combinations.

        Args:
            session_id: Session ID (determines all structure)
            n_neurons: Number of neurons
            n_input_channels: Number of input channels

        Returns:
            Dictionary with base distributions
        """
        # Base spike thresholds: Normal(-55, 0.01) with exact mean -55
        v_th_rng = self.get_rng(session_id, 0, 0, 'base_spike_thresholds')
        base_v_th = v_th_rng.normal(-55.0, 0.01, n_neurons)

        # Force exact mean to be -55.0 (critical for scaling)
        base_v_th = base_v_th - np.mean(base_v_th) + (-55.0)

        # Base synaptic weights: Normal(0, 0.01) with exact mean 0
        g_rng = self.get_rng(session_id, 0, 0, 'base_synaptic_weights')

        # Generate weights for all possible connections (will be masked by connectivity)
        base_g = g_rng.normal(0.0, 0.01, (n_neurons, n_neurons))

        # Force exact mean to be 0.0 (critical for scaling)
        base_g = base_g - np.mean(base_g) + 0.0

        # Connectivity matrix (fixed across parameters)
        conn_rng = self.get_rng(session_id, 0, 0, 'connectivity')
        connectivity = conn_rng.random((n_neurons, n_neurons)) < 0.1
        np.fill_diagonal(connectivity, False)  # No self-connections

        # Dynamic input connectivity (fixed across parameters)
        input_conn_rng = self.get_rng(session_id, 0, 0, 'dynamic_input_connectivity')
        dynamic_connectivity = input_conn_rng.random((n_neurons, n_input_channels)) < 0.3

        # Perturbation target neurons (fixed list for all experiments)
        pert_rng = self.get_rng(session_id, 0, 0, 'perturbation_targets')
        perturbation_neurons = pert_rng.choice(n_neurons, size=100, replace=False)

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
