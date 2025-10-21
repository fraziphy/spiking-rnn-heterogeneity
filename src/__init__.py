# src/__init__.py
"""
Core neural network modules for spiking RNN heterogeneity studies.
"""

from .rng_utils import HierarchicalRNG, get_rng
from .lif_neuron import LIFNeuron
from .synaptic_model import Synapse, StaticPoissonInput, HDDynamicInput
from .hd_input import HDInputGenerator, run_rate_rnn, make_embedding  # FIXED: was hd_input_generator
from .spiking_network import SpikingRNN

__all__ = [
    'HierarchicalRNG',
    'get_rng',
    'LIFNeuron',
    'Synapse',
    'StaticPoissonInput',
    'HDDynamicInput',
    'HDInputGenerator',
    'run_rate_rnn',
    'make_embedding',
    'SpikingRNN'
]

__version__ = '2.1.0'
