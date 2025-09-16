# src/__init__.py
"""
Core neural network modules for spiking RNN heterogeneity studies.

This package contains the fundamental building blocks:
- Random number generation utilities
- LIF neuron models
- Synaptic connections and Poisson inputs
- Complete spiking RNN networks
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

__version__ = '1.0.0'
