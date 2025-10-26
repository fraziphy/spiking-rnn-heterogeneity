# Spiking RNN Heterogeneity Framework v6.0.0

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MPI Support](https://img.shields.io/badge/MPI-supported-orange.svg)](https://www.mpi-forum.org/)

A comprehensive computational framework for investigating the effects of neural heterogeneity on spontaneous activity, network stability, high-dimensional (HD) encoding, and reservoir computing tasks in recurrent spiking neural networks.

## 🆕 What's New in v6.0.0

### Major Features
- **🎯 Reservoir Computing Tasks**: Three new computational task experiments
  - **Categorical Classification**: Multi-class pattern classification with softmax readout
  - **Temporal Transformation**: Continuous signal transformation over time
  - **Auto-Encoding**: Input reconstruction with dimensionality analysis
- **🔧 Unified Task Infrastructure**: Single `TaskPerformanceExperiment` class handles all three tasks
- **📊 Dimensionality Analysis**: SVD-based metrics (participation ratio, effective/intrinsic dimensionality)
- **💾 Smart HD Signal Caching**: Task-specific pattern generation with intelligent reuse
- **🐛 Bug Fixes**: Fixed directory path duplication in all MPI runners

### Performance Improvements
- **Distributed/Centralized CV**: Toggle between distributed and centralized cross-validation
- **Parallel Trial Simulation**: MPI-based distribution of trial simulations
- **Memory Optimization**: Efficient spike time handling and cleanup

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Directory Structure](#directory-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Results & Analysis](#results--analysis)
- [Citation](#citation)
- [Contributing](#contributing)

## ✨ Features

### Core Capabilities
- **Heterogeneity Modeling**: Threshold (v_th) and synaptic weight (g) variability
- **Multiple Input Modes**: Independent, common stochastic, common tonic inputs
- **Synaptic Dynamics**: Pulse-based or filtered (exponential) transmission
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

#### 4. **Categorical Classification** ⭐ NEW
Multi-class pattern discrimination
- N-way softmax classification
- Pattern-based HD input generation
- CV accuracy and confusion analysis

#### 5. **Temporal Transformation** ⭐ NEW
Time-dependent signal mapping
- Continuous output trajectory decoding
- HD input → HD output transformation
- RMSE and R² performance metrics

#### 6. **Auto-Encoding** ⭐ NEW
Input reconstruction and dimensionality characterization
- Self-supervised learning (input = output)
- SVD dimensionality metrics across time scales
- Network capacity analysis

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

### Optional Components
```bash
# Development tools
pip install -e ".[dev]"

# Analysis tools only
pip install -e ".[analysis]"
```

## 🎯 Quick Start

### Test Installation
```bash
# Run comprehensive structure test
spiking-rnn-structure-test

# Test encoding implementation
spiking-rnn-encoding-test

# Test task performance (NEW in v6.0)
spiking-rnn-task-test
```

### Run a Single Experiment

#### Spontaneous Activity
```bash
# Make runner executable
chmod +x runners/run_spontaneous_experiment.sh

# Run single session with 10 MPI processes
./runners/run_spontaneous_experiment.sh \
    --n_sessions 1 \
    --n_processes 10 \
    --n_v_th_std 5 \
    --n_g_std 5
```

#### Categorical Task ⭐ NEW
```bash
chmod +x runners/run_categorical_experiment.sh

./runners/run_categorical_experiment.sh \
    --n_sessions 3 \
    --n_processes 10 \
    --n_input_patterns 10 \
    --n_v_th_std 3 \
    --n_g_std 3
```

#### Temporal Task ⭐ NEW
```bash
chmod +x runners/run_temporal_experiment.sh

./runners/run_temporal_experiment.sh \
    --n_sessions 3 \
    --n_processes 10 \
    --n_input_patterns 10 \
    --hd_dim_output_min 1 \
    --hd_dim_output_max 3
```

#### Auto-Encoding Task ⭐ NEW
```bash
chmod +x runners/run_autoencoding_experiment.sh

./runners/run_autoencoding_experiment.sh \
    --n_sessions 3 \
    --n_processes 10 \
    --n_input_patterns 10 \
    --hd_dim_input_min 1 \
    --hd_dim_input_max 5
```

### Run Sequential Pipeline (Recommended)
```bash
# Make pipeline executable
chmod +x pipeline.sh

# Run complete pipeline in background with logging
nohup ./pipeline.sh > pipeline.log 2>&1 & disown

# Monitor progress
tail -f pipeline.log

# Check if still running
ps aux | grep pipeline

# Return later to check results
ls -la results/spontaneous/data/
ls -la results/categorical/data/
tail -n 50 pipeline.log
```

## 📂 Directory Structure

```
spiking-rnn-heterogeneity/
├── network/
│   ├── neurons.py              # LIF neuron model
│   ├── connectivity.py         # Synaptic connectivity
│   └── network.py              # Network simulation engine
├── inputs/
│   ├── static_inputs.py        # Background/static inputs
│   ├── hd_inputs.py            # HD signal generation (encoding)
│   └── hd_pattern_generator.py # Pattern-based HD (tasks) ⭐ NEW
├── experiments/
│   ├── base_experiment.py      # Shared experiment infrastructure
│   ├── spontaneous_experiment.py
│   ├── stability_experiment.py
│   ├── encoding_experiment.py
│   └── task_performance_experiment.py  # Unified task framework ⭐ NEW
├── analysis/
│   ├── common_utils.py         # Shared analysis functions
│   ├── experiment_utils.py     # Result saving/loading
│   ├── spontaneous_analysis.py
│   ├── stability_analysis.py
│   ├── encoding_analysis.py
│   └── statistics_utils.py
├── runners/
│   ├── mpi_spontaneous_runner.py
│   ├── mpi_stability_runner.py
│   ├── mpi_encoding_runner.py
│   ├── mpi_task_runner.py      # Categorical & temporal ⭐ NEW
│   ├── mpi_autoencoding_runner.py  # Auto-encoding ⭐ NEW
│   ├── mpi_utils.py            # Shared MPI utilities
│   ├── linspace.py             # Parameter grid helper
│   ├── experiment_utils.sh     # Shell script utilities
│   ├── run_spontaneous_experiment.sh
│   ├── run_stability_experiment.sh
│   ├── run_encoding_experiment.sh
│   ├── run_categorical_experiment.sh   ⭐ NEW
│   ├── run_temporal_experiment.sh      ⭐ NEW
│   └── run_autoencoding_experiment.sh  ⭐ NEW
├── tests/
│   ├── test_installation.py
│   ├── test_comprehensive_structure.py
│   ├── test_encoding_implementation.py
│   └── test_task_performance.py        ⭐ NEW
├── results/                    # Output directory (organized by experiment)
│   ├── spontaneous/
│   │   ├── data/              # Individual session results
│   │   └── *.pkl              # Averaged results (if enabled)
│   ├── stability/
│   │   ├── data/
│   │   └── *.pkl
│   ├── encoding/
│   │   ├── data/
│   │   └── *.pkl
│   ├── categorical/           ⭐ NEW
│   │   ├── data/
│   │   └── *.pkl
│   ├── temporal/              ⭐ NEW
│   │   ├── data/
│   │   └── *.pkl
│   └── autoencoding/          ⭐ NEW
│       ├── data/
│       └── *.pkl
├── hd_signals/                # Cached HD signal patterns
│   ├── categorical/           ⭐ NEW
│   ├── temporal/              ⭐ NEW
│   └── autoencoding/          ⭐ NEW
├── logs/                      # Execution logs
│   ├── spontaneous/
│   ├── stability/
│   ├── encoding/
│   ├── categorical/           ⭐ NEW
│   ├── temporal/              ⭐ NEW
│   └── autoencoding/          ⭐ NEW
├── requirements.txt
├── setup.py
├── pipeline.sh                # Sequential experiment pipeline
└── README.md
```

## 📖 Usage Examples

### Parameter Sweep Configuration

All experiments support fine-grained control over parameter grids:

```bash
# Spontaneous activity with dense grid
./runners/run_spontaneous_experiment.sh \
    --n_sessions 10 \
    --n_v_th_std 15 \
    --n_g_std 15 \
    --v_th_std_min 0.01 \
    --v_th_std_max 4.0 \
    --g_std_min 0.01 \
    --g_std_max 4.0 \
    --static_input_rate_min 50.0 \
    --static_input_rate_max 1000.0 \
    --n_static_input_rates 10 \
    --duration 10.0

# Categorical task with multiple HD dimensions and rates
./runners/run_categorical_experiment.sh \
    --n_sessions 5 \
    --n_input_patterns 20 \
    --n_trials_per_pattern 100 \
    --hd_dim_input_min 1 \
    --hd_dim_input_max 10 \
    --n_hd_dim_input 5 \
    --static_input_rate_min 50.0 \
    --static_input_rate_max 500.0 \
    --n_static_input_rates 5 \
    --embed_dim_input 15

# Temporal task with output dimensionality sweep
./runners/run_temporal_experiment.sh \
    --n_sessions 5 \
    --hd_dim_input_min 3 \
    --hd_dim_input_max 3 \
    --n_hd_dim_input 1 \
    --hd_dim_output_min 1 \
    --hd_dim_output_max 5 \
    --n_hd_dim_output 5 \
    --embed_dim_input 10 \
    --embed_dim_output 5
```

### Network Mode Configuration

```bash
# Filter synapses with common tonic input
./runners/run_spontaneous_experiment.sh \
    --synaptic_mode filter \
    --static_input_mode common_tonic

# Pulse synapses with independent stochastic inputs
./runners/run_encoding_experiment.sh \
    --synaptic_mode pulse \
    --static_input_mode independent \
    --hd_input_mode independent

# Categorical task with common stochastic inputs
./runners/run_categorical_experiment.sh \
    --synaptic_mode filter \
    --static_input_mode common_stochastic \
    --hd_input_mode common_stochastic

# Uniform v_th distribution
./runners/run_stability_experiment.sh \
    --v_th_distribution uniform
```

### Distributed vs Centralized Cross-Validation

```bash
# Use distributed CV (higher memory but faster)
./runners/run_categorical_experiment.sh \
    --use_distributed_cv \
    --n_processes 20

# Use centralized CV (default, lower memory)
./runners/run_temporal_experiment.sh \
    --n_processes 10
```

### Session Averaging Control

```bash
# Run multiple sessions and average
./runners/run_encoding_experiment.sh \
    --n_sessions 10 \
    --session_start 0

# Skip averaging (keep individual sessions only)
./runners/run_spontaneous_experiment.sh \
    --n_sessions 5 \
    --no_average
```

## ⚙️ Configuration

### MPI Process Count Recommendations

| Experiment | Parameter Combinations | Recommended Processes |
|------------|------------------------|----------------------|
| Spontaneous | 10×10×5 = 500 | 50-100 |
| Stability | 10×10×5 = 500 | 50-100 |
| Encoding | 5×5×10×3 = 750 | 50-100 |
| Categorical | 5×5×5×3 = 375 | 10-50 |
| Temporal | 5×5×3×3×3 = 675 | 10-50 |
| Auto-Encoding | 5×5×5×3 = 375 | 10-50 |

**Note**: Task experiments (categorical, temporal, auto-encoding) use **sequential parameter combinations with parallel trial/CV distribution**, requiring fewer processes per combination but more combinations total.

### Memory Considerations

**Centralized CV (Default)**:
- Memory on rank 0: ~(n_trials × n_neurons × decision_window × 4 bytes)
- Example: 1000 trials × 1000 neurons × 3000 timesteps × 4 = ~12 GB
- Use for: Most cases, especially with limited RAM per node

**Distributed CV**:
- Memory per rank: ~(n_trials × n_neurons × decision_window × 4 bytes) / n_processes
- Example: 12 GB / 10 processes = ~1.2 GB per rank
- Use for: High-memory nodes, faster CV computation

### Time Estimates

| Experiment | Duration per Combination | 500 Combinations |
|------------|-------------------------|------------------|
| Spontaneous (5s) | ~30-60s | ~4-8 hours |
| Stability | ~90-120s | ~12-16 hours |
| Encoding | ~180-240s | ~25-33 hours |
| Categorical (400 trials) | ~300-600s | ~41-83 hours |
| Temporal (400 trials) | ~300-600s | ~41-83 hours |
| Auto-Encoding (400 trials) | ~300-600s | ~41-83 hours |

*Times scale with n_processes (more processes = faster)*

## 📊 Results & Analysis

### Output File Structure

#### Spontaneous Activity
```
results/spontaneous/data/spontaneous_session_0_filter_independent_normal_5.0s.pkl
```
Fields: `mean_firing_rate_mean`, `cv_isi_mean`, `percent_silent_mean`

#### Network Stability
```
results/stability/data/stability_session_0_filter_independent_normal.pkl
```
Fields: `lz_spatial_patterns_mean`, `lz_column_wise_mean`, `settling_time_ms_mean`

#### HD Encoding
```
results/encoding/data/encoding_session_0_filter_independent_independent_normal_k13.pkl
```
Fields: `decoding.test_rmse_mean`, `decoding.test_r2_mean`

#### Categorical Classification ⭐ NEW
```
results/categorical/data/task_categorical_session_0_vth_1.000_g_1.000_rate_200_hdin_5_embdin_10_npat_10.pkl
```
Fields: `test_accuracy_mean`, `test_accuracy_std`, `confusion_matrix`

#### Temporal Transformation ⭐ NEW
```
results/temporal/data/task_temporal_session_0_vth_1.000_g_1.000_rate_200_hdin_3_embdin_10_hdout_2_embdout_4_npat_10.pkl
```
Fields: `test_rmse_mean`, `test_r2_mean`, `test_correlation_mean`

#### Auto-Encoding ⭐ NEW
```
results/autoencoding/data/task_autoencoding_session_0_vth_1.000_g_1.000_rate_200_hdin_5_embd_10_npat_10.pkl
```
Fields: `test_rmse_mean`, `test_r2_mean`, `dimensionality_summary`

### Session-Averaged Results

When `--no_average` is NOT specified, averaged results are saved to parent directory:
```
results/spontaneous/spontaneous_averaged_filter_independent_normal_5.0s.pkl
results/categorical/task_categorical_averaged_filter_independent_independent_normal_k10.pkl
```

### Loading Results

```python
import pickle

# Load single session
with open('results/categorical/data/task_categorical_session_0_vth_1.000_g_1.000_rate_200_hdin_5_embdin_10_npat_10.pkl', 'rb') as f:
    results = pickle.load(f)

# Access metrics
print(f"Accuracy: {results[0]['test_accuracy_mean']:.3f}")
print(f"Std: {results[0]['test_accuracy_std']:.3f}")

# Load averaged results
with open('results/categorical/task_categorical_averaged_filter_independent_independent_normal_k10.pkl', 'rb') as f:
    avg_results = pickle.load(f)

print(f"Sessions averaged: {avg_results[0]['n_sessions_averaged']}")
print(f"Accuracy (across sessions): {avg_results[0]['test_accuracy_mean']:.3f}")
```

## 🔬 Advanced Topics

### Custom Parameter Grids

Modify shell scripts to use custom parameter arrays:

```bash
# In run_categorical_experiment.sh
V_TH_STDS=(0.0 0.5 1.0 2.0 4.0)
G_STDS=(0.0 0.3 1.0 2.0 3.0)
STATIC_RATES=(100.0 300.0 500.0)
HD_DIMS_INPUT=(1 3 5 10)
```

### HD Signal Caching

HD patterns are automatically cached for reuse:
```bash
# Categorical patterns
hd_signals/categorical/patterns_session_0_dim_5_npat_10.pkl

# Temporal patterns
hd_signals/temporal/patterns_session_0_dim_input_3_dim_output_2_npat_10.pkl

# Auto-encoding patterns
hd_signals/autoencoding/patterns_session_0_dim_5_npat_10.pkl
```

Clear cache to regenerate:
```bash
rm -rf hd_signals/categorical/*
rm -rf hd_signals/temporal/*
```

### Monitoring Long Runs

```bash
# Check progress in real-time
tail -f logs/categorical/session_0.log

# Count completed files
ls results/categorical/data/ | wc -l

# Check for errors
grep -i "error\|failed" logs/categorical/*.log

# Monitor system resources
watch -n 5 'ps aux | grep mpi'
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific component
python tests/test_task_performance.py

# Structure validation
spiking-rnn-structure-test

# Installation check
spiking-rnn-test
```

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{spiking_rnn_heterogeneity_2024,
  title={Spiking RNN Heterogeneity Framework: Reservoir Computing Tasks},
  author={Computational Neuroscience Research Group},
  year={2024},
  version={6.0.0},
  url={https://github.com/yourusername/spiking-rnn-heterogeneity}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linters
black .
flake8 .
mypy .

# Run tests
pytest tests/ -v
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/spiking-rnn-heterogeneity/wiki)
- **Issue Tracker**: [GitHub Issues](https://github.com/yourusername/spiking-rnn-heterogeneity/issues)
- **Changelog**: [Release Notes](https://github.com/yourusername/spiking-rnn-heterogeneity/releases)

## 💬 Support

For questions or issues:
- Open an [issue](https://github.com/yourusername/spiking-rnn-heterogeneity/issues)
- Email: research@example.com

## 🙏 Acknowledgments

This framework builds upon extensive research in:
- Spiking neural network dynamics
- Reservoir computing theory
- High-dimensional neural representations
- Heterogeneity in biological neural circuits

---

**Version**: 6.0.0  
**Last Updated**: 2024  
**Status**: Production/Stable