# Spiking RNN Heterogeneity Framework v6.0.0

A comprehensive framework for studying **spontaneous activity**, **network stability**, **HD input encoding**, and **reservoir computing tasks** in heterogeneous spiking recurrent neural networks.

## Overview

This framework enables systematic investigation of how heterogeneity affects four key network properties:

1. **Spontaneous Activity**: Firing rates, dimensionality, silent neurons, Poisson statistics
2. **Network Stability**: Response to perturbations, complexity measures, settling dynamics
3. **HD Encoding Capacity**: How networks encode high-dimensional inputs with varying intrinsic dimensionality
4. **Reservoir Computing**: Categorical classification, temporal transformation, and auto-encoding tasks

## Major Updates in v6.0.0

### 🎯 NEW: Reservoir Computing Tasks

**Three Task Types:**

1. **Categorical Classification** (`categorical`)
   - 10 input patterns → 10 one-hot output classes
   - Decision window-based readout (last 50ms for classification)
   - Metrics: Accuracy, confusion matrix, per-class performance
   - Output dim = number of input patterns (one-hot encoding)

2. **Temporal Transformation** (`temporal`)
   - d_in-dimensional input → d_out-dimensional output
   - Different HD signals with controlled dimensionality
   - Continuous time-varying transformation
   - Metrics: RMSE, R², Pearson correlation per channel

3. **Auto-Encoding** (`auto_encoding`)
   - Special case: input = output (d_in = d_out, k_in = k_out)
   - Network must reconstruct its own input signal
   - Tests representation capacity directly
   - Includes **dimensionality analysis** at 2ms, 10ms, 20ms bins

**Unified Infrastructure:**
- 100 trials per pattern (1000 total trials with 10 patterns)
- 20-fold stratified cross-validation (balanced across patterns)
- Parallel trial simulation via MPI
- Exponential synaptic filtering (τ = 5ms)
- Ridge regression readout (λ = 0.001)
- Pattern-based caching system for inputs/outputs
- Distributed or centralized CV modes

**Dimensionality Analysis (Auto-Encoding Only):**
- Computed per trial at multiple time scales (2ms, 10ms, 20ms)
- Participation Ratio: (Σλ)² / Σλ²
- Effective Dimensionality: exp(-Σ p_i log p_i)
- Intrinsic Dimensionality: Components for 95% variance
- Aggregated statistics: mean ± std across trials

### 🔧 Code Refactoring: Zero Duplication (v5.1.0)

**New Shared Modules:**
- `analysis/common_utils.py` - Spike processing, dimensionality, **exponential filtering**
- `analysis/statistics_utils.py` - Extreme combinations, hierarchical statistics  
- `experiments/base_experiment.py` - Base class with shared functionality
- `experiments/experiment_utils.py` - Unified save/load/average, **ridge regression**, **task evaluation**
- `experiments/task_performance_experiment.py` - **NEW**: Unified task infrastructure
- `src/hd_input.py` - Merged HD generation + caching (was 2 separate files)

**Files Deleted:**
- `experiments/param_grid_utils.py` (moved to BaseExperiment)
- `src/hd_input_generator.py` (merged into hd_input.py)
- `src/hd_signal_manager.py` (merged into hd_input.py)

**Benefits:**
- Zero code duplication across experiments
- Single source of truth for all utilities
- Consistent 200ms transient time throughout codebase
- Task experiments share 90% of code
- ~1000+ lines of duplicate code eliminated

### 🎨 Enhanced HD Input System (v5.0.0 → v6.0.0)

**Pattern-Based Generation:**
- `pattern_id` parameter added to all HD functions
- Different patterns from same (session, hd_dim, embed_dim)
- Used for multi-pattern tasks (10 patterns × 100 trials)
- Separate caching: `hd_signals/categorical/`, `hd_signals/temporal/`, `hd_signals/autoencoding/`

**Independence Guarantees:**
- HD patterns independent of (hd_dim, embed_dim) - controlled by session + pattern_id only
- Network structure independent of task parameters - controlled by session + v_th + g only
- Trial noise independent across all parameters - controlled by trial_id + all params

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter-dependent RNG (extended for HD params)
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Synapse + input generators + HDDynamicInput
│   ├── spiking_network.py         # Complete RNN
│   └── hd_input.py                # v6.0: Pattern-based HD generation + caching
├── analysis/                      # Analysis modules  
│   ├── common_utils.py            # v6.0: Added apply_exponential_filter + compute_dimensionality_svd
│   ├── statistics_utils.py        # Extreme combos, hierarchical stats
│   ├── spontaneous_analysis.py    # Firing + dimensionality + Poisson
│   ├── stability_analysis.py      # Shannon + LZ + settling + coincidence
│   └── encoding_analysis.py       # Decoding + dimensionality
├── experiments/                   # Experiment coordination
│   ├── base_experiment.py         # v5.1: Base class with shared methods
│   ├── experiment_utils.py        # v6.0: Added ridge regression + task evaluation
│   ├── task_performance_experiment.py  # v6.0: NEW - Unified task infrastructure
│   ├── spontaneous_experiment.py  # Inherits BaseExperiment
│   ├── stability_experiment.py    # Inherits BaseExperiment
│   └── encoding_experiment.py     # Inherits BaseExperiment + smart storage
├── runners/                       # Execution scripts
│   ├── mpi_utils.py               # Shared MPI utilities
│   ├── experiment_utils.sh        # Shared shell functions
│   ├── linspace.py                # v6.0: NEW - Parameter grid helper
│   ├── mpi_task_runner.py         # v6.0: NEW - Categorical/temporal tasks
│   ├── mpi_autoencoding_runner.py # v6.0: NEW - Auto-encoding task
│   ├── mpi_spontaneous_runner.py
│   ├── mpi_stability_runner.py
│   ├── mpi_encoding_runner.py
│   ├── run_categorical_task.sh    # v6.0: NEW
│   ├── run_temporal_task.sh       # v6.0: NEW
│   ├── run_autoencoding_task.sh   # v6.0: NEW
│   ├── run_spontaneous_experiment.sh
│   ├── run_stability_experiment.sh
│   └── run_encoding_experiment.sh
├── tests/                         # Testing framework (50+ tests total)
│   ├── test_installation.py       # v6.0: Updated for new imports
│   ├── test_comprehensive_structure.py
│   ├── test_encoding_implementation.py
│   └── test_task_performance.py   # v6.0: NEW - 17 task tests
├── hd_signals/                    # v6.0: Organized by task
│   ├── categorical/               # Categorical task patterns
│   ├── temporal/                  # Temporal task patterns (inputs + outputs)
│   ├── autoencoding/              # Auto-encoding patterns
│   └── encoding/                  # Original encoding experiment
└── results/data/                  # Experiment outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install numpy scipy mpi4py psutil matplotlib scikit-learn

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation (50+ comprehensive tests)
python tests/test_installation.py
python tests/test_comprehensive_structure.py
python tests/test_encoding_implementation.py
python tests/test_task_performance.py  # NEW in v6.0
```

### Run Sequential Pipeline (Recommended)

# Make pipeline executable
chmod +x pipeline.sh

# Run complete pipeline in background with logging
nohup ./pipeline.sh > pipeline.log 2>&1 & disown

# Monitor progress
tail -f pipeline.log

# Check if still running
ps aux | grep pipeline

# Return later to check results
ls -la results/data/
tail -n 50 pipeline.log

### 2. Run Reservoir Computing Tasks (NEW!)

```bash
# Categorical classification (10 patterns → 10 classes)
./runners/run_categorical_task.sh \
    --n_sessions 10 \
    --n_input_patterns 10 \
    --n_trials_per_pattern 100 \
    --n_processes 10 \
    --v_th_std_min 0.0 --v_th_std_max 4.0 --n_v_th_std 5 \
    --g_std_min 0.0 --g_std_max 4.0 --n_g_std 5 \
    --hd_dim_input_min 1 --hd_dim_input_max 5 --n_hd_dim_input 1

# Temporal transformation (d_in → d_out)
./runners/run_temporal_task.sh \
    --n_sessions 10 \
    --hd_dim_input_min 1 --hd_dim_input_max 5 \
    --hd_dim_output_min 1 --hd_dim_output_max 2 \
    --use_distributed_cv  # Optional: faster but uses more RAM

# Auto-encoding (input → input reconstruction)
./runners/run_autoencoding_task.sh \
    --n_sessions 10 \
    --input_hd_dim 3 \
    --embed_dim_input 10 \
    --n_processes 10
```

**Memory Management:**
- **Centralized CV** (default): Rank 0 does all CV, saves RAM on other ranks
- **Distributed CV** (`--use_distributed_cv`): All ranks do CV, faster but uses more RAM
- With 10 processes × 24GB/process ≈ 240GB total (safe for 251GB systems)

### 3. Run Original Experiments

```bash
# Encoding experiments (original HD encoding capacity)
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
```

## Code Organization (v6.0.0)

### Import Structure

```python
# Task experiments (NEW in v6.0)
from experiments.task_performance_experiment import TaskPerformanceExperiment
from experiments.experiment_utils import (
    save_results, load_results,
    apply_exponential_filter,  # NEW in v6.0
    train_task_readout,        # NEW in v6.0
    predict_task_readout,      # NEW in v6.0
    evaluate_categorical_task, # NEW in v6.0
    evaluate_temporal_task     # NEW in v6.0
)

# Original experiments
from experiments.base_experiment import BaseExperiment
from experiments.spontaneous_experiment import SpontaneousExperiment
from experiments.encoding_experiment import EncodingExperiment

# Analysis utilities
from analysis.common_utils import (
    spikes_to_binary, 
    spikes_to_matrix,
    compute_participation_ratio,
    compute_dimensionality_svd,    # NEW in v6.0
    apply_exponential_filter       # Moved from experiment_utils in v6.0
)

from analysis.statistics_utils import get_extreme_combinations
```

### TaskPerformanceExperiment Class (NEW)

```python
# Create categorical task
categorical_exp = TaskPerformanceExperiment(
    task_type='categorical',
    n_neurons=1000,
    n_input_patterns=10,
    input_dim_intrinsic=5,
    input_dim_embedding=10,
    n_trials_per_pattern=100,
    tau_syn=5.0,
    lambda_reg=0.001
)

# Create temporal task
temporal_exp = TaskPerformanceExperiment(
    task_type='temporal',
    output_dim_intrinsic=2,
    output_dim_embedding=4,
    # ... other params
)

# Create auto-encoding task (special case)
autoencoding_exp = TaskPerformanceExperiment(
    task_type='temporal',  # Uses temporal infrastructure
    output_dim_intrinsic=5,    # Same as input!
    output_dim_embedding=10,   # Same as input!
    # ... other params
)

# All share same methods:
exp.simulate_trials_parallel(...)   # Parallel trial simulation
exp.convert_spikes_to_traces(...)   # Exponential filtering
exp.cross_validate(...)             # Stratified CV with ridge regression
```

## Scientific Innovation

### Reservoir Computing Tasks (v6.0.0)

**Why Reservoir Computing?**
Networks process inputs without weight training. Only readout layer is trained (ridge regression). Tests computational capacity directly.

**Task Design:**

1. **Categorical**: Static pattern recognition
   - 10 unique HD patterns (d=5, k=10)
   - Network response → Last 50ms averaged → 10-class softmax
   - Measures: Can heterogeneity improve pattern separation?

2. **Temporal**: Dynamic signal transformation
   - Continuous d_in-dimensional input → d_out-dimensional output
   - Different intrinsic dimensionalities test capacity limits
   - Measures: Can networks transform across dimensionalities?

3. **Auto-Encoding**: Representation fidelity
   - Input = Output (d=d, k=k)
   - Network must maintain information through recurrent dynamics
   - **Plus dimensionality analysis**: Does network compress/expand representation?

**Key Innovation - Dimensionality Analysis in Auto-Encoding:**
- Unlike encoding (decoder dimensionality) or other tasks (only performance)
- Auto-encoding analyzes **network state dimensionality** at multiple timescales
- Reveals if network compresses (dim < d), preserves (dim ≈ d), or expands (dim > d)
- Critical for understanding representation strategies

**Pattern Independence:**
```python
# Same network structure for all patterns in a session
network = SpikingRNN(...)
network.initialize_network(session_id=1, v_th_std=1.0, g_std=1.0)
# → Thresholds and weights identical across all pattern presentations

# Different HD patterns from same parameters
pattern_A = HDInputGenerator.initialize_base_input(session_id=1, hd_dim=5, pattern_id=0)
pattern_B = HDInputGenerator.initialize_base_input(session_id=1, hd_dim=5, pattern_id=1)
# → Different chaotic trajectories, but both are 5-dimensional

# Different trial noise for each presentation
trial_1 = generate_trial_input(session_id=1, trial_id=1, pattern_id=0, ...)
trial_2 = generate_trial_input(session_id=1, trial_id=2, pattern_id=0, ...)
# → Same pattern, different noise realizations
```

### HD Input Encoding (v5.0.0 → v6.0.0)

**Enhanced with Pattern Support:**

```python
# Original encoding (single signal per session)
HDInputGenerator.initialize_base_input(session_id=1, hd_dim=5)

# Task experiments (multiple patterns per session)
for pattern_id in range(10):
    HDInputGenerator.initialize_base_input(session_id=1, hd_dim=5, pattern_id=pattern_id)
```

**HD Input Protocol:**
1. Run chaotic rate RNN (g=1.2, 100 neurons, 500ms) with pattern-specific seed
2. Remove 200ms transient (standardized across all experiments)
3. PCA to extract k=10 principal components
4. Random rotation in k-space (pattern-specific)
5. Select d random components (intrinsic dimensionality)
6. Embed back into k-space via random orthogonal basis
7. Result: k channels spanning d-dimensional subspace

**Caching System:**
```
hd_signals/
├── categorical/
│   └── hd_signal_session_1_hd_5_k_10_pattern_0.pkl
│   └── hd_signal_session_1_hd_5_k_10_pattern_1.pkl
├── temporal/
│   ├── inputs/
│   │   └── hd_signal_session_1_hd_5_k_10_pattern_0.pkl
│   └── outputs/
│       └── hd_signal_session_1_hd_2_k_4_pattern_0.pkl
└── autoencoding/
    └── hd_signal_session_1_hd_3_k_10_pattern_0.pkl
```

### Corrected Synaptic Filtering (v4.0.0)

**The Problem:** Input classes applied filtering, then synapses applied it again (double filtering).

**The Solution:**
```python
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
```

**Impact:** Single, consistent filtering path; correct pulse vs filter comparison.

## Data Analysis

### Task Performance Results (v6.0.0)

```python
import pickle
from experiments.experiment_utils import load_results

# Categorical task
results = load_results('results/data/task_categorical_session_1_vth_1.000_g_1.000_rate_200_hdin_5_hdout_10.pkl')

for result in results:
    print(f"Session {result['session_id']}")
    print(f"  Test Accuracy: {result['test_accuracy_mean']:.3f} ± {result['test_accuracy_std']:.3f}")
    print(f"  Confusion Matrix:\n{result['cv_confusion_matrices'][0]}")

# Temporal task
results = load_results('results/data/task_temporal_session_1_vth_1.000_g_1.000_rate_200_hdin_5_hdout_2.pkl')

for result in results:
    print(f"  Test RMSE: {result['test_rmse_mean']:.4f}")
    print(f"  Test R²: {result['test_r2_mean']:.4f}")
    print(f"  Test Correlation: {result['test_correlation_mean']:.4f}")

# Auto-encoding task (includes dimensionality!)
results = load_results('results/data/task_autoencoding_session_1_vth_1.000_g_1.000_rate_200_hd_3.pkl')

for result in results:
    print(f"  Test RMSE: {result['test_rmse_mean']:.4f}")
    
    # NEW: Dimensionality analysis
    for bin_size in ['2.0ms', '10.0ms', '20.0ms']:
        dim_key = f'bin_{bin_size}'
        pr_mean = result['dimensionality_summary'][dim_key]['participation_ratio_mean']
        ed_mean = result['dimensionality_summary'][dim_key]['effective_dimensionality_mean']
        print(f"  {bin_size}: PR={pr_mean:.2f}, ED={ed_mean:.2f}")
```

### Encoding Results (v5.0.0)

```python
results = load_results('results/data/encoding_session_1_filter_independent_independent_normal_k10.pkl')

for result in results:
    hd_dim = result['hd_dim']
    test_rmse = result['decoding']['test_rmse_mean']
    test_r2 = result['decoding']['test_r2_mean']
    
    weight_dims = [f['effective_dim_95'] for f in result['decoding']['weight_svd_analysis']]
    decoded_dims = [f['effective_dim_95'] for f in result['decoding']['decoded_pca_analysis']]
    
    print(f"HD={hd_dim}: RMSE={test_rmse:.4f}, Weight dim={np.mean(weight_dims):.1f}")
```

### Session Averaging

```python
from experiments.experiment_utils import average_across_sessions_encoding

averaged_results = average_across_sessions_encoding(
    session_files=[
        'results/data/encoding_session_1_filter_independent_independent_normal_k10.pkl',
        'results/data/encoding_session_2_filter_independent_independent_normal_k10.pkl',
    ]
)

for result in averaged_results:
    print(f"HD={result['hd_dim']}")
    print(f"  RMSE: {result['decoding']['test_rmse_mean']:.4f} ± {result['decoding']['test_rmse_std']:.4f}")
```

## System Requirements

- **Python**: 3.8+
- **CPU**: Multi-core (32+ cores recommended for MPI)
- **Memory**: 
  - Spontaneous: 16GB+
  - Stability: 32GB+
  - Encoding: 64GB+
  - Tasks: 240GB+ (10 processes × 24GB, centralized CV)
- **Storage**: 
  - ~5GB per traditional experiment
  - ~10MB for HD signal cache per experiment type
  - Task experiments: ~100MB per session (includes dimensionality for auto-encoding)

## Version History

- **v6.0.0**: RESERVOIR COMPUTING TASKS
  - **NEW**: Categorical classification, temporal transformation, auto-encoding
  - **NEW**: `TaskPerformanceExperiment` class (unified infrastructure)
  - **NEW**: Pattern-based HD input generation (`pattern_id` parameter)
  - **NEW**: Dimensionality analysis for auto-encoding (2ms, 10ms, 20ms bins)
  - **NEW**: Ridge regression readout with stratified 20-fold CV
  - **NEW**: Distributed and centralized CV modes
  - **NEW**: Task-specific signal caching directories
  - **NEW**: 17 comprehensive task tests in `test_task_performance.py`
  - MOVED: `apply_exponential_filter` from `experiment_utils` to `common_utils`
  - ADDED: `compute_dimensionality_svd` to `common_utils` (faster than covariance)
  - ADDED: Task evaluation functions to `experiment_utils`

- **v5.1.0**: CODE REFACTORING - Eliminated ALL code duplication
  - NEW: `analysis/common_utils.py`, `analysis/statistics_utils.py`
  - NEW: `experiments/base_experiment.py`, `experiments/experiment_utils.py`
  - MERGED: `src/hd_input.py` (was 2 files)
  - DELETED: `experiments/param_grid_utils.py`
  - All experiments inherit from BaseExperiment
  - Single source of truth for utilities
  - Standardized 200ms transient time throughout

- **v5.0.0**: ENCODING CAPACITY SYSTEM
  - HD input generation (d-dimensional signals in k-dimensional space)
  - Linear decoder with SVD/PCA dimensionality analysis
  - Spike jitter computation
  - HD signal caching system (~1000× storage savings)
  - Smart storage logic (low-dim + extremes only)

- **v4.0.0**: ARCHITECTURE REVOLUTION
  - Fixed double filtering bug
  - Three static/HD input modes
  - New stability measures (LZ column-wise, 0.1ms coincidence)

## Citation

```bibtex
@software{spiking_rnn_heterogeneity_v600,
  title = {Spiking RNN Heterogeneity Framework v6.0.0: Reservoir Computing Tasks},
  author = {Your Name},
  year = {2025},
  version = {6.0.0},
  note = {Categorical classification, temporal transformation, and auto-encoding tasks with dimensionality analysis},
  url = {https://github.com/yourusername/spiking-rnn-heterogeneity}
}
```

## License

MIT License - See LICENSE file for details