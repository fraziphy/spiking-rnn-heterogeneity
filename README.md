# Spiking RNN Heterogeneity Framework v7.1.0

[![Python Version](https://img.shields.io/badge/python-3.8\%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MPI Support](https://img.shields.io/badge/MPI-supported-orange.svg)](https://www.mpi-forum.org/)

A comprehensive computational framework for investigating the effects of neural heterogeneity on spontaneous activity, network stability, high-dimensional (HD) encoding, and reservoir computing tasks in recurrent spiking neural networks.

## 🆕 What's New in v7.1.0

### 🔧 Data Curation Improvements

**Renamed and refactored data curation scripts for clarity:**
- `data_curation_network_encoding.py` → `data_curation_network_autoencoding.py`
- New `data_curation_network_classification.py` for categorical task

**Fixed participation ratio (PR) computation:**
- PR now computed from **evoked spike patterns** (correct) instead of decoder weight matrix (incorrect)
- Uses same `compute_activity_dimensionality_multi_bin()` method as stability experiments
- Consistent 2ms bin size across all experiments

### 🔄 Failed Job Recovery System ⭐ NEW

New scripts to automatically retry only failed jobs from parameter sweeps:

```bash
# Interactive mode (shows failed jobs, asks for confirmation)
./sweep/rerun_failed.sh --task autoencoding
./sweep/rerun_failed.sh --task categorical
./sweep/rerun_failed.sh --task temporal
./sweep/rerun_failed.sh --task stability

# Override parallel job count
./sweep/rerun_failed.sh --task autoencoding --num_parallel 8

# Non-interactive mode (for nohup/background execution)
nohup ./sweep/rerun_failed_nohup.sh --task autoencoding > rerun_autoencoding.log 2>&1 & disown
```

**Features:**
- Reads configuration from existing `run_sweep_{task}.sh` scripts
- Uses `--resume-failed` flag in sweep engine
- Cleans joblog after successful retries (removes duplicate FAILED entries)
- Reports statistics on completion

### 📊 New Plotting Scripts

- `plot_main_figure_3.py`: Classification task results (accuracy vs d for different k)
- Updated `plot_main_figure_1.py`: Fixed data key compatibility
- Updated `plot_main_figure_2.py`: Support for both PR and R² curves

### 🐛 Bug Fixes

- Fixed `KeyError: 'duration'` in network dynamics plotting (key naming consistency)
- Fixed stability experiment to compute PR from spike patterns
- Improved sweep engine error handling

## 🆕 What's New in v7.0.0

### 🏗️ Major Architectural Refactoring

**This release introduces a fundamental restructuring of the simulation pipeline for correctness, efficiency, and reproducibility.**

#### 1. **Network Instance Consistency Across Tasks** ⭐ CRITICAL
- Network initialization (connectivity, weights, thresholds) now depends **ONLY** on:
  - `session_id`
  - `g_std` (weight heterogeneity)
  - `v_th_std` (threshold heterogeneity)
- Network is **IDENTICAL** across all tasks (stability, categorical, temporal, autoencoding)
- Network does **NOT** change based on:
  - Task type
  - HD dimensions (hd_dim, embed_dim)
  - Static input rate
  - HD connection mode (overlapping/partitioned)
- **Scientific Impact**: Ensures fair comparison across tasks and parameter sweeps

#### 2. **Transient State Caching** ⭐ PERFORMANCE
- Pre-compute 1000ms transient period once per parameter combination
- Cache final network state for all 100 trials
- All experiments (stability, classification, transformation, autoencoding) **load cached states**
- **Simulation time savings**: ~70% reduction (skip 1000ms transient per trial)
```bash
# Generate cached transient states
python experiments/transient_cache_experiment.py \
    --session-start 0 --session-end 20 \
    --g-std 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
    --v-th-std 0 \
    --static-rate 30 31 32 33 34 35
```

#### 3. **HD Signal Pre-generation** ⭐ REPRODUCIBILITY
- Generate all HD input and HD output signals **before** experiments
- Deterministic based on (session_id, hd_dim, embed_dim, pattern_id, signal_type)
- Cached in `results/hd_signals/` directory
- All tasks read from identical pre-generated signals
```bash
# Generate HD signals for all sessions and dimensions
python experiments/generate_hd_signals.py \
    --sessions 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --embed-dims 2 4 6 8 10 \
    --n-patterns 10
```

#### 4. **Evoked Spike Caching** ⭐ MAJOR EFFICIENCY GAIN
- Simulate network response to HD inputs **once**
- Cache spike times for reuse across all tasks
- Classification, transformation, and autoencoding **read identical cached spikes**
- Stored separately by `hd_connection_mode` (overlapping/partitioned)
- **Simulation time savings**: ~90% for multi-task experiments
```bash
# Generate cached evoked spikes
python experiments/evoked_spike_to_hd_input_cache_experiment.py \
    --session-start 0 --session-end 20 \
    --g-std 1.0 \
    --hd-dims 2 4 6 \
    --embed-dims 10 \
    --modes overlapping partitioned
```

### 📁 New Directory Structure
```
results/
├── cached_states/           # Transient network states (1000ms) ⭐ NEW
│   └── session_{s}_g_{g}_vth_{v}_rate_{r}_trial_states.pkl
├── cached_spikes/           # Evoked spike responses ⭐ NEW
│   ├── overlapping/
│   │   └── session_{s}_g_{g}_vth_{v}_rate_{r}_h_{h}_d_{d}_pattern_{p}_spikes.pkl
│   └── partitioned/
│       └── session_{s}_g_{g}_vth_{v}_rate_{r}_h_{h}_d_{d}_pattern_{p}_spikes.pkl
├── hd_signals/              # Pre-generated HD patterns ⭐ NEW
│   ├── hd_hd_input_session_{s}_hd_{h}_k_{k}_pattern_{p}.pkl
│   └── hd_hd_output_session_{s}_hd_{h}_k_{k}_pattern_{p}.pkl
└── data/                    # Task results (organized by mode/task)
    ├── overlapping/
    │   ├── categorical/
    │   ├── temporal/
    │   └── autoencoding/
    └── partitioned/
        ├── categorical/
        ├── temporal/
        └── autoencoding/
```

### 🔄 New Simulation Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Network Preparation (run once per parameter combo)    │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate HD signals (hd_input + hd_output)                  │
│     └── generate_hd_signals.py                                  │
│  2. Simulate 1000ms transient, cache final states               │
│     └── transient_cache_experiment.py                           │
│  3. Load states, simulate 300ms with HD input, cache spikes     │
│     └── evoked_spike_to_hd_input_cache_experiment.py            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Task Analysis (reads cached data, no re-simulation)   │
├─────────────────────────────────────────────────────────────────┤
│  • Stability: Load transient states → perturb → analyze         │
│  • Categorical: Load cached spikes → classify → evaluate        │
│  • Temporal: Load cached spikes → transform → evaluate          │
│  • Autoencoding: Load cached spikes → reconstruct → evaluate    │
└─────────────────────────────────────────────────────────────────┘
```

### 🎛️ New Command-Line Flags
```bash
# Task runner with cached spikes
mpirun -np 32 python runners/mpi_task_runner.py \
    --task_type categorical \
    --use_cached_spikes \
    --spike_cache_dir results/cached_spikes \
    --signal_cache_dir results/hd_signals \
    ...

# Stability runner with cached transients
mpirun -np 32 python runners/mpi_stability_runner.py \
    --use_cached_transients \
    --transient_cache_dir results/cached_states \
    ...
```

### 📊 New Result Fields
```python
result = {
    # Existing fields...
    'used_cached_spikes': True,        # ⭐ NEW: Whether cached spikes were used
    'used_cached_transients': True,    # ⭐ NEW: Whether cached transients were used
    # ... other fields unchanged
}
```

### Infrastructure Improvements
- Memory-efficient spike handling (NumPy structured arrays)
- Garbage collection optimization for large sweeps
- Debug output for cache hit/miss tracking
- Extended transient time: 500ms → 1000ms for better steady-state

### Bug Fixes
- Fixed network initialization dependency on task parameters
- Corrected HD connectivity to be independent of hd_dim/embed_dim
- Improved numerical stability in spike-to-trace conversion

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Simulation Pipeline](#simulation-pipeline)
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
- **HD Connection Modes**: Overlapping (30% random) or partitioned (equal division)
- **Synaptic Dynamics**: Pulse-based or filtered (exponential) transmission
- **Dimensionality Analysis**: Participation ratio for all HD patterns
- **MPI Parallelization**: Efficient distributed computing for large parameter sweeps
- **Automated Workflows**: Shell scripts for complete experiment pipelines
- **Caching System**: Transient states, HD signals, and evoked spikes ⭐ NEW

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
- **Now uses cached transient states** ⭐ NEW

#### 3. **HD Encoding**
Evaluate continuous signal representation capacity
- Linear decoding (ridge regression)
- Dimensionality scaling analysis
- Cross-validation performance metrics

#### 4. **Categorical Classification**
Multi-class pattern discrimination
- N-way classification with Bayesian decoder (primary) + time-averaged decoder (comparison)
- Pattern-based HD input generation with configurable connection modes
- Confidence and uncertainty metrics
- CV accuracy, confusion matrices, per-class performance
- **Now uses cached evoked spikes** ⭐ NEW

#### 5. **Temporal Transformation**
Time-dependent signal mapping
- Continuous output trajectory decoding
- HD input → HD output transformation with connection mode control
- RMSE, R², and correlation metrics
- **Now uses cached evoked spikes** ⭐ NEW

#### 6. **Auto-Encoding**
Input reconstruction and dimensionality characterization
- Self-supervised learning (input = output)
- Dimensionality metrics across different time scales
- Network capacity characterization
- **Now uses cached evoked spikes** ⭐ NEW

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

### Full Pipeline Example
```bash
# Step 1: Generate HD signals
python experiments/generate_hd_signals.py \
    --sessions 0 1 2 \
    --embed-dims 5 10 \
    --n-patterns 4

# Step 2: Generate cached transient states
python experiments/transient_cache_experiment.py \
    --session-start 0 --session-end 3 \
    --g-std 1.0 \
    --v-th-std 0 \
    --static-rate 30

# Step 3: Generate cached evoked spikes
python experiments/evoked_spike_to_hd_input_cache_experiment.py \
    --session-start 0 --session-end 3 \
    --g-std 1.0 \
    --hd-dims 2 \
    --embed-dims 5 \
    --modes overlapping

# Step 4: Run task experiment (uses cached data)
mpirun -np 4 python runners/mpi_task_runner.py \
    --task_type categorical \
    --session_id 0 \
    --use_cached_spikes \
    --spike_cache_dir results/cached_spikes \
    --signal_cache_dir results/hd_signals \
    --v_th_std 0 \
    --g_std 1.0 \
    --static_input_rate 30 \
    --input_hd_dim 2 \
    --output_hd_dim 2 \
    --hd_connection_mode overlapping
```

### Test Installation
```bash
# Run comprehensive structure test
spiking-rnn-structure-test

# Test caching system ⭐ NEW
python experiments/transient_cache_experiment.py --session-start 0 --session-end 1 --g-std 1.0

# Test HD connection modes
python tests/test_hd_connection_modes.py

# Test dimensionality computation
python tests/test_dimensionality_edge_cases.py
```

## 🔄 Simulation Pipeline

### Phase 1: Preparation (Run Once)
```bash
# 1. HD Signal Generation
python experiments/generate_hd_signals.py \
    --sessions 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --embed-dims 2 4 6 8 10 \
    --n-patterns 10

# 2. Transient State Caching (1000ms per trial)
python experiments/transient_cache_experiment.py \
    --session-start 0 --session-end 20 \
    --g-std 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
    --v-th-std 0 \
    --static-rate 30 31 32 33 34 35 \
    --n-trials 100

# 3. Evoked Spike Caching (300ms with HD input)
python experiments/evoked_spike_to_hd_input_cache_experiment.py \
    --session-start 0 --session-end 20 \
    --g-std 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
    --v-th-std 0 \
    --static-rate 30 31 32 33 34 35 \
    --hd-dims 1 2 3 4 5 6 7 8 9 10 \
    --embed-dims 10 \
    --pattern-ids 0 1 2 3 4 5 6 7 8 9 \
    --modes overlapping partitioned
```

### Phase 2: Task Analysis (Reads Cached Data)
```bash
# Stability experiment
mpirun -np 32 python runners/mpi_stability_runner.py \
    --session_id 0 \
    --use_cached_transients \
    --transient_cache_dir results/cached_states \
    --v_th_std 0 --g_std 1.0 --static_input_rate 30

# Categorical classification
mpirun -np 32 python runners/mpi_task_runner.py \
    --task_type categorical \
    --use_cached_spikes \
    --spike_cache_dir results/cached_spikes \
    --signal_cache_dir results/hd_signals \
    ...

# Temporal transformation
mpirun -np 32 python runners/mpi_task_runner.py \
    --task_type temporal \
    --use_cached_spikes \
    ...

# Autoencoding
mpirun -np 32 python runners/mpi_task_runner.py \
    --task_type autoencoding \
    --use_cached_spikes \
    ...
```

### Run Parameter Sweeps 
```bash
# Make sweep scripts executable
chmod +x sweep/*.sh
nohup ./sweep/run_sweep_categorical.sh > pipeline_categorical.log 2>&1 & disown

## 📂 Directory Structure
```
spiking-rnn-heterogeneity/
├── src/                        # Core network implementation
│   ├── spiking_network.py      # Network with state save/restore
│   ├── synaptic_model.py       # HD connection modes
│   ├── lif_neuron.py           # LIF neuron model
│   ├── hd_input.py             # HD signal generator with caching
│   └── rng_utils.py            # Reproducible RNG management
├── experiments/
│   ├── transient_cache_experiment.py    # ⭐ NEW: Transient caching
│   ├── evoked_spike_to_hd_input_cache_experiment.py  # ⭐ NEW: Spike caching
│   ├── generate_hd_signals.py           # ⭐ NEW: HD signal pre-generation
│   ├── task_performance_experiment.py   # Unified task experiment
│   ├── stability_experiment.py          # Stability with cached states
│   └── base_experiment.py
├── runners/
│   ├── mpi_task_runner.py      # Task runner with --use_cached_spikes
│   └── mpi_stability_runner.py # Stability runner with --use_cached_transients
├── analysis/
│   ├── common_utils.py         # Shared utilities
│   ├── encoding_analysis.py
│   ├── stability_analysis.py
│   └── spontaneous_analysis.py
├── sweep/                      # Sweep orchestration
│   ├── generate_jobs.py
│   ├── run_sweep_*.sh
│   ├── rerun_failed.sh         # ⭐ NEW: Retry failed jobs (interactive)
│   └── rerun_failed_nohup.sh   # ⭐ NEW: Retry failed jobs (non-interactive)
├── tests/
│   ├── test_hd_connection_modes.py
│   └── test_dimensionality_edge_cases.py
└── results/                    # Results and caches
    ├── cached_states/          # ⭐ NEW: Transient states
    ├── cached_spikes/          # ⭐ NEW: Evoked spikes
    │   ├── overlapping/
    │   └── partitioned/
    ├── hd_signals/             # ⭐ NEW: Pre-generated HD patterns
    └── data/                   # Task results
        ├── overlapping/
        │   ├── categorical/
        │   ├── temporal/
        │   └── autoencoding/
        └── partitioned/
            ├── categorical/
            ├── temporal/
            └── autoencoding/
```

## 📊 Results & Analysis

### Cache File Formats

#### Transient States (`cached_states/`)
```python
{
    'session_id': 0,
    'g_std': 1.0,
    'v_th_std': 0.0,
    'static_rate': 30.0,
    'n_trials': 100,
    'transient_duration': 1000.0,
    'trial_states': {
        0: {'current_time': 1000.0, 'neuron_v_membrane': ..., ...},
        1: {'current_time': 1000.0, 'neuron_v_membrane': ..., ...},
        ...
    }
}
```

#### Evoked Spikes (`cached_spikes/`)
```python
{
    'session_id': 0,
    'g_std': 1.0,
    'v_th_std': 0.0,
    'static_rate': 30.0,
    'hd_dim': 2,
    'embed_dim': 10,
    'pattern_id': 0,
    'hd_connection_mode': 'overlapping',
    'trial_spikes': {
        0: [(t1, n1), (t2, n2), ...],  # Trial 0 spikes
        1: [(t1, n1), (t2, n2), ...],  # Trial 1 spikes
        ...
    }
}
```

#### HD Signals (`hd_signals/`)
```python
{
    'Y_base': np.ndarray,  # (n_timesteps, embed_dim)
    'session_id': 0,
    'hd_dim': 2,
    'embed_dim': 10,
    'pattern_id': 0,
    'signal_type': 'hd_input',  # or 'hd_output'
    'statistics': {'mean': ..., 'std': ..., ...}
}
```

### Task Result Fields
```python
result = {
    # Task parameters
    'session_id': 0,
    'v_th_std': 0.0,
    'g_std': 1.0,
    'task_type': 'categorical',
    'hd_connection_mode': 'overlapping',
    
    # Caching info ⭐ NEW
    'used_cached_spikes': True,
    
    # Dimensionality tracking
    'input_empirical_dims': [4.2, 3.9, 4.1, 4.0],
    
    # Performance metrics
    'test_accuracy_bayesian_mean': 0.95,
    ...
}
```

## 📜 Version History

| Version | Date | Description |
|---------|------|-------------|
| **v7.1.0** | Dec 2025 | **Data curation fixes**: Correct PR computation from spikes, failed job recovery scripts, classification plotting |
| **v7.0.0** | Nov 2025 | **Major architectural refactoring**: Network consistency, transient caching, HD signal pre-generation, evoked spike caching |
| v6.2.0 | Nov 2025 | HD connection modes + empirical dimensionality tracking |
| v6.1.0 | Nov 2025 | Infrastructure consolidation - sweep/ directory organization |
| v6.0.0 | Oct 2025 | Reservoir computing tasks (categorical, temporal, auto-encoding) |
| v5.1.0 | - | Refactored codebase - eliminated duplication |
| v5.0.0 | - | HD encoding experiment |
| v4.0.0 | - | Network stability analysis |
| v3.0.0 | - | Spontaneous activity baseline |

## 📚 Citation
```bibtex
@software{spiking_rnn_heterogeneity_2024,
  title={Spiking RNN Heterogeneity Framework},
  author={Computational Neuroscience Research Group},
  year={2024},
  version={7.1.0},
  url={https://github.com/yourusername/spiking-rnn-heterogeneity},
  note={Caching architecture for efficient multi-task experiments}
}
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Version**: 7.1.0  
**Release Date**: December 2025  
**Status**: Production/Stable