# src/__init__.py
"""
Core neural network modules for spiking RNN heterogeneity studies with random structure.

This package contains the fundamental building blocks:
- Random number generation utilities with parameter-dependent seeding
- LIF neuron models with mean-centered heterogeneity
- Synaptic connections with immediate vs dynamic modes
- Complete spiking RNN networks with random structure per parameter combination
"""

from .rng_utils import HierarchicalRNG, get_rng
from .lif_neuron import LIFNeuron
from .synaptic_model import ExponentialSynapses, StaticPoissonInput, DynamicPoissonInput, ReadoutLayer
from .spiking_network import SpikingRNN

__all__ = [
    'HierarchicalRNG',
    'get_rng',
    'LIFNeuron',
    'ExponentialSynapses',
    'StaticPoissonInput',
    'DynamicPoissonInput',
    'ReadoutLayer',
    'SpikingRNN'
]

__version__ = '2.0.0-random-structure'
