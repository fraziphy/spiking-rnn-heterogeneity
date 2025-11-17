# Spiking RNN Heterogeneity Framework v6.2.0

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MPI Support](https://img.shields.io/badge/MPI-supported-orange.svg)](https://www.mpi-forum.org/)

A comprehensive computational framework for investigating the effects of neural heterogeneity on spontaneous activity, network stability, high-dimensional (HD) encoding, and reservoir computing tasks in recurrent spiking neural networks.

## 🆕 What's New in v6.2.0

### Major Features
- **🔗 HD Connection Modes**: Two modes for controlling HD input connectivity
  - **Overlapping** (default): 30% random connectivity, ~9% overlap between channels
  - **Partitioned**: Equal neuron division, zero overlap between channels
  - Removes confound between dimensionality and connectivity structure
  - Flag-based selection: `--hd_connection_mode overlapping|partitioned`

- **📊 Empirical Dimensionality Tracking**: Automatic participation ratio computation
  - **Input/Output Patterns**: Track intrinsic dimensionality of HD signals
  - **Reconstructed Outputs**: Measure network's effective output dimensionality
  - **Per-Pattern Analysis**: Mean and std across test trials
  - Applies to all task experiments (categorical, temporal, autoencoding)

### Infrastructure Improvements
- Enhanced parameter tracking in result files
- Improved numerical stability for edge cases (k=1 handling)
- Organized results by HD connection mode in separate directories
- Extended sweep scripts with connection mode support

### Bug Fixes
- Fixed syntax error in `task_performance_experiment.py` (function placement)
- Robust handling of k=1 (single dimension) edge cases
- Improved test file imports for flexible execution

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Directory Structure](#directory-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Results & Analysis](#results--analysis)
- [Version History](#version-history)
- [Citation](#citation)

## ✨ Features

### Core Capabilities
- **Heterogeneity Modeling**: Threshold (v_th) and synaptic weight (g) variability
- **Multiple Input Modes**: Independent, common stochastic, common tonic inputs
- **HD Connection Modes**: Overlapping (30% random) or partitioned (equal division) ⭐ NEW
- **Synaptic Dynamics**: Pulse-based or filtered (exponential) transmission
- **Dimensionality Analysis**: Participation ratio for all HD patterns ⭐ NEW
- **MPI Parallelization**: Efficient distributed computing for large parameter sweeps
- **Automated Workflows**: Shell scripts for complete experiment pipelines

### Experiment Types

#### 1. **Spontaneous Activity** 
Analyze baseline network dynamics without structured input
- Firing rate statistics (mean, CV of ISI)
- Silent neuron detection
- Activity pattern characterization

#### 2. **Network Stability**
Measure attractor dynamics and settling behavior
- Lempel-Ziv complexity (spatial patterns, column-wise)
- Settling time analysis
- Trajectory convergence metrics

#### 3. **HD Encoding**
Evaluate continuous signal representation capacity
- Linear decoding (ridge regression)
- Dimensionality scaling analysis
- Cross-validation performance metrics

#### 4. **Categorical Classification** (v6.0+)
Multi-class pattern discrimination
- N-way classification with Bayesian decoder (primary) + time-averaged decoder (comparison)
- Pattern-based HD input generation with configurable connection modes ⭐ NEW
- Confidence and uncertainty metrics
- CV accuracy, confusion matrices, per-class performance
- Input dimensionality tracking ⭐ NEW

#### 5. **Temporal Transformation** (v6.0+)
Time-dependent signal mapping
- Continuous output trajectory decoding
- HD input → HD output transformation with connection mode control ⭐ NEW
- RMSE, R², and correlation metrics
- Input/output/reconstructed dimensionality tracking ⭐ NEW

#### 6. **Auto-Encoding** (v6.0+)
Input reconstruction and dimensionality characterization
- Self-supervised learning (input = output)
- Dimensionality metrics across different time scales
- Input and reconstructed output dimensionality analysis ⭐ NEW
- Network capacity characterization

## 🚀 Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# MPI implementation (OpenMPI or MPICH)
mpirun --version
```

### Option 1: pip install (recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/spiking-rnn-heterogeneity.git
cd spiking-rnn-heterogeneity

# Install with all dependencies
pip install -e ".[all]"

# Verify installation
spiking-rnn-test
```

### Option 2: Manual setup
```bash
# Clone repository
git clone https://github.com/yourusername/spiking-rnn-heterogeneity.git
cd spiking-rnn-heterogeneity

# Install core dependencies
pip install -r requirements.txt

# Verify MPI
python -c "from mpi4py import MPI; print(f'MPI working: {MPI.COMM_WORLD.Get_size()} process(es)')"
```

## 🎯 Quick Start

### Test Installation
```bash
# Run comprehensive structure test
spiking-rnn-structure-test

# Test HD connection modes ⭐ NEW
python tests/test_hd_connection_modes.py

# Test dimensionality computation ⭐ NEW
python tests/test_dimensionality_edge_cases.py
```

### Run Parameter Sweeps with Connection Modes ⭐ NEW
```bash
# Make sweep scripts executable
chmod +x sweep/*.sh

nohup ./sweep/run_sweep_categorical.sh > pipeline_categorical.log 2>&1 & disown

# Run categorical sweep with partitioned mode
HD_CONNECTION_MODE="partitioned" \
EMBED_DIM_INPUT=(2 4 6 8) \
N_PATTERNS=4 \
./sweep/run_sweep_categorical.sh

# Results organized by mode:
# results/categorical_sweep/partitioned/data/
```

## 📂 Directory Structure

```
spiking-rnn-heterogeneity/
├── src/                        # Core network implementation
│   ├── synaptic_model.py       # HD connection modes ⭐ NEW
│   ├── spiking_network.py
│   └── ...
├── experiments/
│   ├── task_performance_experiment.py  # Dimensionality tracking ⭐ NEW
│   └── ...
├── analysis/
│   ├── common_utils.py         # Participation ratio computation ⭐ NEW
│   └── ...
├── sweep/                      # Sweep orchestration
│   ├── generate_jobs.py        # Connection mode support ⭐ NEW
│   ├── run_sweep_categorical.sh
│   ├── run_sweep_transformation.sh
│   └── run_sweep_autoencoding.sh
├── tests/
│   ├── test_hd_connection_modes.py      ⭐ NEW
│   └── test_dimensionality_edge_cases.py ⭐ NEW
└── results/                    # Results by mode ⭐ NEW
    ├── categorical_sweep/
    │   ├── overlapping/data/
    │   └── partitioned/data/
    ├── temporal_sweep/
    │   ├── overlapping/data/
    │   └── partitioned/data/
    └── autoencoding_sweep/
        ├── overlapping/data/
        └── partitioned/data/
```

## 📊 Results & Analysis

### New Fields in v6.2.0 ⭐

#### All Task Experiments
```python
result = {
    'hd_connection_mode': 'partitioned',  # or 'overlapping'
    # ... existing fields ...
}
```

#### Categorical Task
```python
result = {
    'input_empirical_dims': [4.2, 3.9, 4.1, 4.0],  # Per pattern
    # ... existing fields ...
}
```

#### Temporal Task
```python
result = {
    'input_empirical_dims': [5.1, 4.9],  # Per input pattern
    'output_empirical_dims': [3.2, 3.0],  # Per output pattern
    'reconstructed_output_empirical_dim_means': [2.9, 2.8],  # Per pattern
    'reconstructed_output_empirical_dim_stds': [0.3, 0.4],  # Per pattern
    # ... existing fields ...
}
```

#### Auto-Encoding Task
```python
result = {
    'input_empirical_dim': 4.5,  # Single value (n_patterns=1)
    'reconstructed_output_empirical_dim_means': [4.1],  # Single-element list
    'reconstructed_output_empirical_dim_stds': [0.3],  # Single-element list
    # ... existing fields ...
}
```

### Loading and Comparing Results

```python
import pickle
import glob

# Load categorical result
with open('results/categorical_sweep/partitioned/data/task_categorical_*.pkl', 'rb') as f:
    result = pickle.load(f)[0]
    print(f"Mode: {result['hd_connection_mode']}")
    print(f"Input dims: {result['input_empirical_dims']}")
    print(f"Accuracy: {result['test_accuracy_bayesian_mean']:.3f}")

# Compare overlapping vs partitioned
overlapping_files = glob.glob('results/categorical_sweep/overlapping/data/*.pkl')
partitioned_files = glob.glob('results/categorical_sweep/partitioned/data/*.pkl')

# Analyze dimensionality scaling...
```

## 📜 Version History

- **v6.2.0** (Nov 2025): HD connection modes + empirical dimensionality tracking
- **v6.1.0** (Nov 2025): Infrastructure consolidation - sweep/ directory organization
- **v6.0.0** (Oct 2025): Reservoir computing tasks (categorical, temporal, auto-encoding)
- **v5.1.0**: Refactored codebase - eliminated duplication
- **v5.0.0**: HD encoding experiment
- **v4.0.0**: Network stability analysis
- **v3.0.0**: Spontaneous activity baseline

## 📚 Citation

```bibtex
@software{spiking_rnn_heterogeneity_2024,
  title={Spiking RNN Heterogeneity Framework},
  author={Computational Neuroscience Research Group},
  year={2024},
  version={6.2.0},
  url={https://github.com/yourusername/spiking-rnn-heterogeneity},
  note={HD connection modes and dimensionality tracking}
}
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Version**: 6.2.0  
**Release Date**: November 2025  
**Status**: Production/Stable