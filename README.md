# Spiking RNN Heterogeneity Framework v5.1.0

A comprehensive framework for studying **spontaneous activity**, **network stability**, and **HD input encoding capacity** in heterogeneous spiking recurrent neural networks.

## Overview

This framework enables systematic investigation of how heterogeneity affects three key network properties:

1. **Spontaneous Activity**: Firing rates, dimensionality, silent neurons, Poisson statistics
2. **Network Stability**: Response to perturbations, complexity measures, settling dynamics
3. **HD Encoding Capacity**: How networks encode high-dimensional inputs with varying intrinsic dimensionality

## Major Updates in v5.1.0

### Code Refactoring: Zero Duplication

BREAKING CHANGES: Import paths updated, standalone functions removed

New Shared Modules:
- analysis/common_utils.py - Spike processing, dimensionality utilities
- analysis/statistics_utils.py - Extreme combinations, hierarchical statistics  
- experiments/base_experiment.py - Base class with shared functionality
- experiments/experiment_utils.py - Unified save/load/average functions
- src/hd_input.py - Merged HD generation + caching (was 2 separate files)

Files Deleted:
- experiments/param_grid_utils.py (moved to BaseExperiment)
- src/hd_input_generator.py (merged into hd_input.py)
- src/hd_signal_manager.py (merged into hd_input.py)

Benefits:
- Zero code duplication across experiments
- Single source of truth for all utilities
- Consistent 200ms transient time throughout codebase
- Cleaner import structure
- Better maintainability and extensibility
- ~500 lines of duplicate code eliminated

### From v5.0.0: HD Input Encoding System

HD Input Generation:
- d-dimensional signals embedded in k-dimensional space
- Generated from chaotic rate RNN dynamics via PCA
- Controlled intrinsic dimensionality (d) vs embedding dimensionality (k)
- Signal caching system: ~1000× storage savings

Linear Decoder with Full Analysis:
- Ridge regression with exponential kernel filtering
- Leave-one-out cross-validation (20 folds)
- Per-fold metrics: RMSE, R², Pearson correlation
- Weight dimensionality: SVD, singular values, participation ratio, effective dim (95%)
- Decoded dimensionality: PCA per trial analyzing if decoder discovers true d
- Spike jitter: Reliability analysis for weight-jitter correlations

Three HD Input Modes:
- independent: Each neuron gets independent Poisson from HD rates
- common_stochastic: Neurons share Poisson per channel, differs across channels
- common_tonic: Deterministic expected values (zero variance)

Smart Storage Logic:
- Low-dim (hd_dim ≤ 2, embed_dim ≤ 2) + extreme combos → Save full neuron data
- All other conditions → Summary statistics only
- Reduces storage by ~100× while preserving detailed analysis for key conditions

### From v4.0.0: Corrected Synaptic Architecture

- Fixed double filtering bug: Input classes now generate events only; synapses apply filtering
- Pulse vs filter synapses: Clear terminology for synaptic dynamics
- Three static input modes: independent, common_stochastic, common_tonic
- New stability measures: LZ column-wise, coincidence at 0.1ms

## Project Structure

spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter-dependent RNG
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Synapse + input generators + HDDynamicInput
│   ├── spiking_network.py         # Complete RNN
│   └── hd_input.py                # v5.1: Unified HD generation + caching
├── analysis/                      # Analysis modules  
│   ├── common_utils.py            # v5.1: Shared utilities
│   ├── statistics_utils.py        # v5.1: Extreme combos, hierarchical stats
│   ├── spontaneous_analysis.py    # Firing + dimensionality + Poisson
│   ├── stability_analysis.py      # Shannon + LZ + settling + coincidence
│   └── encoding_analysis.py       # Decoding + dimensionality
├── experiments/                   # Experiment coordination
│   ├── base_experiment.py         # v5.1: Base class with shared methods
│   ├── experiment_utils.py        # v5.1: Unified save/load/average
│   ├── spontaneous_experiment.py  # Inherits BaseExperiment
│   ├── stability_experiment.py    # Inherits BaseExperiment
│   └── encoding_experiment.py     # Inherits BaseExperiment + smart storage
├── runners/                       # Execution scripts
│   ├── mpi_utils.py               # Shared MPI utilities
│   ├── experiment_utils.sh        # Shared shell functions
│   ├── mpi_spontaneous_runner.py
│   ├── mpi_stability_runner.py
│   ├── mpi_encoding_runner.py
│   ├── run_spontaneous_experiment.sh
│   ├── run_stability_experiment.sh
│   └── run_encoding_experiment.sh
├── tests/                         # Testing framework (38 tests total)
│   ├── test_installation.py
│   ├── test_comprehensive_structure.py
│   └── test_encoding_implementation.py
├── hd_signals/                    # Cached HD signals
└── results/data/                  # Experiment outputs

## Quick Start

### 1. Setup Environment

# Install dependencies
pip install numpy scipy mpi4py psutil matplotlib scikit-learn

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation (38 comprehensive tests)
python tests/test_installation.py
python tests/test_comprehensive_structure.py
python tests/test_encoding_implementation.py

### 2. Run Sequential Pipeline (Recommended)

# Make pipeline executable
chmod +x pipeline.sh

# Run complete pipeline in background with logging
nohup ./pipeline.sh > pipeline.log 2>&1 &
disown

# Monitor progress
tail -f pipeline.log

# Check if still running
ps aux | grep pipeline

# Return later to check results
ls -la results/data/
tail -n 50 pipeline.log

### 3. Run Individual Experiments

# Encoding experiments
./runners/run_encoding_experiment.sh \
    --session_ids '1 2 3 4 5' \
    --n_v_th 10 --n_g 10 --n_hd 5 \
    --hd_dim_min 1 --hd_dim_max 10

# Spontaneous activity
./runners/run_spontaneous_experiment.sh \
    --duration 5 --session_ids "1 2 3" \
    --synaptic_mode filter \
    --static_input_mode independent

# Stability analysis
./runners/run_stability_experiment.sh \
    --session_ids "1 2 3" \
    --synaptic_mode filter \
    --static_input_mode common_tonic

## Code Organization (v5.1.0)

### Import Structure

Old (v5.0.0 - DEPRECATED):
from spontaneous_experiment import create_parameter_grid  # No longer exists
from spontaneous_experiment import save_results           # No longer exists

New (v5.1.0):
from experiments.base_experiment import BaseExperiment
from experiments.experiment_utils import save_results, load_results
from analysis.common_utils import spikes_to_binary, compute_participation_ratio
from analysis.statistics_utils import get_extreme_combinations

### BaseExperiment Class

All experiments now inherit from BaseExperiment:

class SpontaneousExperiment(BaseExperiment):
    def extract_trial_arrays(self, trial_results):
        # Experiment-specific extraction
        pass
    
    def compute_all_statistics(self, arrays_dict):
        # Experiment-specific statistics
        pass

# Shared methods available:
BaseExperiment.create_parameter_grid(...)  # Static method
self.create_parameter_combinations(...)     # Instance method
BaseExperiment.compute_safe_mean(...)      # Safe statistics

### Common Utils

Shared across all analysis modules:

from analysis.common_utils import (
    spikes_to_binary,              # Used by spontaneous + stability
    spikes_to_matrix,              # Used by encoding
    compute_participation_ratio,   # Used by all
    compute_dimensionality_from_covariance  # Unified computation
)

## Scientific Innovation

### HD Input Encoding (v5.0.0)

The Challenge: How does network heterogeneity affect encoding capacity for high-dimensional inputs?

HD Input Protocol:
1. Run chaotic rate RNN (g=1.2, 1000 neurons, 500ms)
2. Remove 200ms transient (standardized across all experiments)
3. PCA to extract k=10 principal components
4. Random rotation in k-space
5. Select d random components (intrinsic dimensionality)
6. Embed back into k-space via random orthogonal basis
7. Result: k channels spanning d-dimensional subspace

Smart Storage:
- Low-dim (d≤2, k≤2) + extreme corners → Full neuron data saved
- All others → Summary statistics only
- Enables detailed analysis where it matters most
- ~100× storage reduction

### Corrected Synaptic Filtering (v4.0.0)

The Problem: Input classes applied filtering, then synapses applied it again (double filtering).

The Solution:
# Input classes generate events only
class StaticPoissonInput:
    def generate_events(...) -> np.ndarray:
        return raw_events  # NO filtering

# Synapse class applies filtering
class Synapse:
    def apply_to_input(self, events: np.ndarray):
        if self.synaptic_mode == "filter":
            self.current *= exp(-dt/tau)  # Decay
            self.current += events         # Add new
        elif self.synaptic_mode == "pulse":
            self.current = events          # Replace

Impact: Single, consistent filtering path; correct pulse vs filter comparison.

### Static Input Modes (v4.0.0)

Three modes to probe network computation:

1. independent: Each neuron gets independent Poisson spikes (max variability)
2. common_stochastic: All neurons get identical Poisson spikes (synchronous drive)
3. common_tonic: Deterministic expected value (zero variance)

Pulse vs Filter with Tonic Input:
- Pulse: current = 0.05 (constant)
- Filter: current → 2.5 (50× due to integration)

## Data Analysis

### Encoding Results (v5.0.0)

import pickle
from experiments.experiment_utils import load_results

results = load_results('results/data/encoding_session_1_filter_independent_independent_normal_k10.pkl')

for result in results:
    hd_dim = result['hd_dim']
    
    # Performance
    test_rmse = result['decoding']['test_rmse_mean']
    test_r2 = result['decoding']['test_r2_mean']
    
    # Dimensionality
    weight_dims = [f['effective_dim_95'] for f in result['decoding']['weight_svd_analysis']]
    decoded_dims = [f['effective_dim_95'] for f in result['decoding']['decoded_pca_analysis']]
    
    print(f"HD={hd_dim}: RMSE={test_rmse:.4f}, Weight dim={np.mean(weight_dims):.1f}")

### Session Averaging

from experiments.experiment_utils import average_across_sessions_encoding

# Average across multiple sessions
averaged_results = average_across_sessions_encoding(
    session_files=[
        'results/data/encoding_session_1_filter_independent_independent_normal_k10.pkl',
        'results/data/encoding_session_2_filter_independent_independent_normal_k10.pkl',
    ]
)

# Access hierarchical statistics
for result in averaged_results:
    print(f"HD={result['hd_dim']}")
    print(f"  RMSE: {result['decoding']['test_rmse_mean']:.4f} ± {result['decoding']['test_rmse_std']:.4f}")

### Stability Results (v4.0.0)

filename = 'results/data/stability_session_1_filter_common_tonic_normal.pkl'
with open(filename, 'rb') as f:
    results = pickle.load(f)

for result in results:
    lz_spatial = result['lz_spatial_patterns_mean']
    lz_column = result['lz_column_wise_mean']
    kistler_01ms = result['kistler_delta_0.1ms_mean']
    settling_time = result['settling_time_ms_mean']

## System Requirements

- Python: 3.8+
- CPU: Multi-core (32+ cores recommended)
- Memory: 16GB+ (spontaneous), 32GB+ (stability), 64GB+ (encoding)
- Storage: 5GB+ per experiment, 10MB for HD signal cache

## Version History

- v5.1.0: CODE REFACTORING - Eliminated ALL code duplication
  - NEW: analysis/common_utils.py, analysis/statistics_utils.py
  - NEW: experiments/base_experiment.py, experiments/experiment_utils.py
  - MERGED: src/hd_input.py (was 2 files)
  - DELETED: experiments/param_grid_utils.py
  - All experiments inherit from BaseExperiment
  - Single source of truth for utilities
  - Updated all import paths
  - Standardized 200ms transient time throughout

- v5.0.0: ENCODING CAPACITY SYSTEM - HD input encoding experiments
  - HD input generation (d-dimensional signals in k-dimensional space)
  - Linear decoder with SVD/PCA dimensionality analysis
  - Spike jitter computation and weight-jitter correlations
  - HD signal caching system (~1000× storage savings)
  - Smart storage logic (low-dim + extremes only)

- v4.0.0: ARCHITECTURE REVOLUTION - Corrected synaptic filtering
  - Fixed double filtering bug
  - Three static/HD input modes
  - New stability measures (LZ column-wise, 0.1ms coincidence)

## Citation

@software{spiking_rnn_heterogeneity_v510,
  title = {Spiking RNN Heterogeneity Framework v5.1.0},
  author = {Your Name},
  year = {2025},
  version = {5.1.0},
  url = {https://github.com/yourusername/spiking-rnn-heterogeneity}
}