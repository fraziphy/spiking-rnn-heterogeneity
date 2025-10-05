# src/__init__.py
"""
Core neural network modules for spiking RNN heterogeneity studies with random structure.

This package contains the fundamental building blocks:
- Random number generation utilities with parameter-dependent seeding
- LIF neuron models with mean-centered heterogeneity
- Synaptic connections with immediate vs dynamic modes
- HD input generation for encoding experiments
- Complete spiking RNN networks with random structure per parameter combination
"""

from .rng_utils import HierarchicalRNG, get_rng
from .lif_neuron import LIFNeuron
from .synaptic_model import Synapse, StaticPoissonInput, HDDynamicInput, ReadoutLayer
from .hd_input_generator import HDInputGenerator, run_rate_rnn, make_embedding
from .spiking_network import SpikingRNN

__all__ = [
    'HierarchicalRNG',
    'get_rng',
    'LIFNeuron',
    'Synapse',
    'StaticPoissonInput',
    'HDDynamicInput',
    'ReadoutLayer',
    'HDInputGenerator',
    'run_rate_rnn',
    'make_embedding',
    'SpikingRNN'
]

__version__ = '2.1.0-hd-encoding'
