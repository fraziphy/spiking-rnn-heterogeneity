# Spiking RNN Heterogeneity Framework v5.0.0

A comprehensive framework for studying **spontaneous activity**, **network stability**, and **HD input encoding capacity** in heterogeneous spiking recurrent neural networks.

## Major Updates in v5.0.0

### 🧠 NEW: HD Input Encoding Experiments

Complete system for studying how networks encode high-dimensional inputs:

**HD Input Generation**:
- d-dimensional signals embedded in k-dimensional space
- Generated from chaotic rate RNN dynamics via PCA
- Controlled intrinsic dimensionality (d) vs embedding dimensionality (k)
- Signal caching: ~1000× storage savings

**Linear Decoder with Full Analysis**:
- Ridge regression with exponential kernel filtering
- Leave-one-out cross-validation (20 folds)
- Per-fold metrics: RMSE, R², Pearson correlation
- Weight dimensionality: SVD, singular values, participation ratio, effective dim (95%)
- Decoded dimensionality: PCA per trial analyzing if decoder discovers true d
- Spike jitter: Reliability analysis for weight-jitter correlations

**Three HD Input Modes**:
- `independent`: Each neuron gets independent Poisson from HD rates
- `common_stochastic`: Neurons share Poisson per channel, differs across channels
- `common_tonic`: Deterministic expected values (zero variance)

### 🔧 Code Refactoring: Eliminated Repetition

**New Shared Utilities**:
- `runners/mpi_utils.py`: Work distribution, health monitoring, recovery (used by all MPI runners)
- `runners/experiment_utils.sh`: Shell functions for logging, validation, averaging (used by all scripts)
- ~600 lines of duplicate code eliminated across runners

**New Modules**:
- `src/hd_input_generator.py`: Rate RNN simulation and PCA embedding
- `src/hd_signal_manager.py`: HD signal caching and loading
- `analysis/encoding_analysis.py`: Complete decoding pipeline
- `experiments/encoding_experiment.py`: Encoding experiment coordination

### 📊 From v4.0.0: Corrected Synaptic Architecture

- **Fixed double filtering bug**: Input classes now generate events only; synapses apply filtering
- **Pulse vs filter synapses**: Clear terminology for synaptic dynamics
- **Three static input modes**: independent, common_stochastic, common_tonic
- **New stability measures**: LZ column-wise, coincidence at 0.1ms

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter-dependent RNG (extended for HD)
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Synapse + input generators + HDDynamicInput
│   ├── spiking_network.py         # Complete RNN (encoding support)
│   ├── hd_input_generator.py      # NEW v5.0: HD signal generation
│   └── hd_signal_manager.py       # NEW v5.0: Signal caching
├── analysis/                      # Analysis modules  
│   ├── spontaneous_analysis.py    # Firing + dimensionality + Poisson
│   ├── stability_analysis.py      # Shannon + LZ + settling + coincidence
│   └── encoding_analysis.py       # NEW v5.0: Decoding + dimensionality
├── experiments/                   # Experiment coordination
│   ├── spontaneous_experiment.py  # Spontaneous activity
│   ├── stability_experiment.py    # Network stability
│   └── encoding_experiment.py     # NEW v5.0: HD encoding capacity
├── runners/                       # Execution scripts
│   ├── mpi_utils.py               # NEW v5.0: Shared MPI utilities
│   ├── experiment_utils.sh        # NEW v5.0: Shared shell functions
│   ├── mpi_spontaneous_runner.py  # Refactored with shared utils
│   ├── mpi_stability_runner.py    # Refactored with shared utils
│   ├── mpi_encoding_runner.py     # NEW v5.0: Encoding MPI runner
│   ├── run_spontaneous_experiment.sh
│   ├── run_stability_experiment.sh
│   └── run_encoding_experiment.sh # NEW v5.0: Encoding shell script
├── tests/                         # Testing framework
│   ├── test_installation.py
│   ├── test_comprehensive_structure.py
│   └── test_encoding_implementation.py  # NEW v5.0
├── hd_signals/                    # NEW v5.0: Cached HD signals
└── results/data/                  # Experiment outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies (added scikit-learn for encoding)
pip install numpy scipy mpi4py psutil matplotlib scikit-learn

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation
python tests/test_installation.py
python tests/test_encoding_implementation.py
```

### Run Sequential Pipeline (Recommended)
```bash
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


### 2. Run Encoding Experiments (NEW)

```bash
# Quick test
./runners/run_encoding_experiment.sh \
    --session_ids '1' \
    --n_v_th 3 --n_g 3 --n_hd 3 \
    --hd_dim_min 1 --hd_dim_max 5 \
    --embed_dim 10

# Full study
./runners/run_encoding_experiment.sh \
    --session_ids '1 2 3 4 5' \
    --n_v_th 10 --n_g 10 --n_hd 5 \
    --hd_dim_min 1 --hd_dim_max 10

# Compare HD input modes
./runners/run_encoding_experiment.sh --hd_input_mode independent
./runners/run_encoding_experiment.sh --hd_input_mode common_stochastic
./runners/run_encoding_experiment.sh --hd_input_mode common_tonic
```

### 3. Run Spontaneous Activity

```bash
./runners/run_spontaneous_experiment.sh \
    --duration 5 --session_ids "1 2 3" \
    --synaptic_mode filter \
    --static_input_mode independent
```

### 4. Run Stability Analysis

```bash
./runners/run_stability_experiment.sh \
    --session_ids "1 2 3" \
    --synaptic_mode filter \
    --static_input_mode common_tonic
```

## Scientific Innovation

### HD Input Encoding (v5.0.0)

**The Challenge**: How does network heterogeneity affect encoding capacity for high-dimensional inputs?

**HD Input Protocol**:
```python
# Generate d-dimensional signal in k-dimensional space
1. Run chaotic rate RNN (g=1.2, 1000 neurons, 350ms)
2. PCA to extract k=10 principal components
3. Random rotation in k-space
4. Select d random components (intrinsic dimensionality)
5. Embed back into k-space via random orthogonal basis
6. Result: k channels spanning d-dimensional subspace
```

**Key Design**:
- All k channels active (equal drive across HD values)
- Intrinsic dimensionality d hidden from decoder
- Per-trial Gaussian noise added to base signal
- Fair comparison: decoder complexity doesn't scale with d

**Decoding Analysis**:
```python
# Per cross-validation fold:
1. Train linear decoder on training trials
2. Analyze weight matrix W (n_neurons × k):
   - SVD: singular values, explained variance
   - Effective dimensionality: components for 95% variance
   - Participation ratio: (Σλ)² / Σλ²
3. Decode test trials
4. PCA on decoded output (per trial):
   - Does decoded signal span d-dimensional space?
5. Compute spike jitter for training trials:
   - Which neurons are reliable?
   - Correlate with decoder weights
```

**Scientific Questions**:
- Does encoding capacity decrease with higher d?
- Do heterogeneous networks encode better?
- Does decoder discover true dimensionality d?
- Are reliable neurons (low jitter) more important for decoding?

### Corrected Synaptic Filtering (v4.0.0)

**The Problem**: Input classes applied filtering, then synapses applied it again (double filtering).

**The Solution**:
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

**Impact**: Single, consistent filtering path; correct pulse vs filter comparison.

### Static Input Modes (v4.0.0)

**Three modes to probe network computation**:

1. **independent**: Each neuron gets independent Poisson spikes (max variability)
2. **common_stochastic**: All neurons get identical Poisson spikes (synchronous drive)
3. **common_tonic**: Deterministic expected value (zero variance)

**Pulse vs Filter with Tonic Input**:
- Pulse: `current = 0.05` (constant)
- Filter: `current → 2.5` (50× due to integration)

## Data Analysis

### Encoding Results (v5.0.0)

```python
import pickle

filename = 'results/data/encoding_session_1_filter_independent_independent_normal_k10.pkl'
with open(filename, 'rb') as f:
    results = pickle.load(f)

for result in results:
    hd_dim = result['hd_dim']
    embed_dim = result['embed_dim']
    
    # Performance metrics
    test_rmse = result['decoding']['test_rmse_mean']
    test_r2 = result['decoding']['test_r2_mean']
    test_corr = result['decoding']['test_correlation_mean']
    
    # Weight dimensionality (per fold)
    weight_svd = result['decoding']['weight_svd_analysis']
    weight_dims = [f['effective_dim_95'] for f in weight_svd]
    weight_pr = [f['participation_ratio'] for f in weight_svd]
    
    # Decoded dimensionality (per fold, averaged over trials)
    decoded_pca = result['decoding']['decoded_pca_analysis']
    decoded_dims = [f['effective_dim_95'] for f in decoded_pca]
    
    # Spike jitter (per fold)
    jitter_folds = result['decoding']['spike_jitter_per_fold']
    
    print(f"HD={hd_dim}, k={embed_dim}")
    print(f"  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    print(f"  Weight dim: {np.mean(weight_dims):.1f}")
    print(f"  Decoded dim: {np.mean(decoded_dims):.1f}")
```

### Stability Results (v4.0.0)

```python
filename = 'results/data/stability_session_1_filter_common_tonic_normal.pkl'
with open(filename, 'rb') as f:
    results = pickle.load(f)

for result in results:
    lz_spatial = result['lz_spatial_patterns_mean']
    lz_column = result['lz_column_wise_mean']
    kistler_01ms = result['kistler_delta_0.1ms_mean']
    settling_time = result['settling_time_ms_mean']
```

## System Requirements

- **Python**: 3.8+
- **CPU**: Multi-core (32+ cores recommended)
- **Memory**: 16GB+ (spontaneous), 32GB+ (stability), 64GB+ (encoding)
- **Storage**: 5GB+ per experiment, 10MB for HD signal cache

## Version History

- **v5.0.0**: **ENCODING CAPACITY SYSTEM** - HD input encoding experiments with comprehensive decoding analysis
  - NEW: HD input generation (d-dimensional signals in k-dimensional space)
  - NEW: Linear decoder with SVD/PCA dimensionality analysis
  - NEW: Spike jitter computation and weight-jitter correlations
  - NEW: HD signal caching system (~1000× storage savings)
  - Code refactoring: Shared utilities (mpi_utils.py, experiment_utils.sh)
  - ~600 lines of duplicate code eliminated
  - Added: hd_input_generator.py, hd_signal_manager.py, encoding_analysis.py
  - Added: encoding_experiment.py, mpi_encoding_runner.py, run_encoding_experiment.sh

- **v4.0.0**: **ARCHITECTURE REVOLUTION** - Corrected synaptic filtering, pulse/filter terminology
  - Fixed double filtering bug
  - Three static input modes: independent, common_stochastic, common_tonic
  - New stability measures: lz_column_wise, coincidence at 0.1ms

- **v3.5.1**: Critical bug fixes - Shannon entropy, settling time
- **v3.5.0**: Full-simulation stability, Shannon entropy, settling time
- **v3.0.0**: Split architecture (spontaneous + stability)

## Citation

```bibtex
@software{spiking_rnn_heterogeneity_v500,
  title = {Spiking RNN Heterogeneity Framework v5.0.0},
  author = {Your Name},
  year = {2025},
  version = {5.0.0},
  url = {https://github.com/yourusername/spiking-rnn-heterogeneity}
}
```

---

**Key Innovation v5.0.0**: Complete HD input encoding system with linear decoder, dimensionality analysis (SVD/PCA), spike jitter, and weight-jitter correlations. Addresses how network heterogeneity affects encoding capacity for high-dimensional inputs.