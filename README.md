# Spiking RNN Heterogeneity Framework - Synaptic Dynamics Comparison

A comprehensive framework for studying chaos and network dynamics in heterogeneous spiking recurrent neural networks with **synaptic mode comparison** (immediate vs dynamic synapses) and **session averaging** for robust statistics.

## Key Architecture Features

### Random Structure with Parameter Dependence
- **Network topology depends on `session_id` AND parameter values**
- Different connectivity patterns for each (session, v_th_std, g_std) combination
- Enables systematic study across multiple network realizations

### Mean-Centered Heterogeneity
- **Direct heterogeneity values**: `v_th_std` and `g_std` (0.0-4.0 range)
- **Exact mean preservation**: -55mV spike thresholds, 0 synaptic weights
- **Distribution flexibility**: Normal and uniform threshold distributions

### Fair Synaptic Mode Comparison
- **Dynamic synapses**: Exponential decay with τ_syn = 5ms (realistic)
- **Immediate synapses**: Instantaneous coupling (like previous studies)
- **Impact normalization**: Immediate weights scaled by τ_syn/dt for fair comparison

### Session Averaging for Robustness
- **Single session execution**: Efficient MPI parallelization
- **Multi-session studies**: Average results across network realizations
- **Statistical robustness**: 100 trials × N sessions per parameter combination

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter-dependent RNG
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Immediate vs dynamic synapses
│   └── spiking_network.py         # Complete RNN with mode selection
├── analysis/                      # Analysis and measurement tools  
│   └── spike_analysis.py          # Enhanced chaos quantification
├── experiments/                   # Experiment coordination
│   └── chaos_experiment.py        # Single session + averaging
├── runners/                       # Execution scripts
│   ├── mpi_chaos_runner.py        # MPI single session runner
│   └── run_chaos_experiment.sh    # Session coordination script
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation verification
│   └── test_random_structure.py   # Random structure validation
└── results/data/                  # Experiment outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install numpy scipy mpi4py psutil scikit-learn

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation
python tests/test_installation.py
python tests/test_random_structure.py
```

### 2. Run Experiments

**Quick test (single session):**
```bash
chmod +x runners/run_chaos_experiment.sh
./runners/run_chaos_experiment.sh --session_ids "1" --n_v_th 3 --n_g 3 --no_average --nproc 4
```

**Compare synaptic modes:**
```bash
# Test immediate synapses
./runners/run_chaos_experiment.sh --synaptic_mode immediate --session_ids "1 2 3" --n_v_th 5 --n_g 5

# Test dynamic synapses  
./runners/run_chaos_experiment.sh --synaptic_mode dynamic --session_ids "1 2 3" --n_v_th 5 --n_g 5
```

**Full heterogeneity study:**
```bash
./runners/run_chaos_experiment.sh --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20 --v_th_std_max 2.0 --g_std_max 2.0 --nproc 50
```

**Test different threshold distributions:**
```bash
./runners/run_chaos_experiment.sh --v_th_distributions "normal uniform" --session_ids "1 2 3"
```

### 3. Monitor Progress

```bash
# Follow experiment progress
tail -f output_run_chaos_experiment.log

# Check MPI processes
htop  # Look for python/mpirun processes
```

## Parameter Specification

### Core Parameters
- `--session_ids`: Space-separated session IDs for averaging (e.g., "1 2 3")
- `--n_v_th`: Number of spike threshold heterogeneity values (default: 10)
- `--n_g`: Number of synaptic weight heterogeneity values (default: 10)
- `--nproc`: Number of MPI processes (default: 50)

### Heterogeneity Control
- `--v_th_std_min/max`: Threshold heterogeneity range (default: 0.0-4.0)
- `--g_std_min/max`: Weight heterogeneity range (default: 0.0-4.0)
- `--v_th_distributions`: "normal", "uniform", or "normal uniform"

### Synaptic Mode Comparison
- `--synaptic_mode`: "immediate" or "dynamic" (critical parameter!)
- Impact normalization ensures fair comparison between modes

### System Configuration
- `--n_neurons`: Network size (default: 1000)
- `--n_input_rates`: Background input modulation levels (default: 5)
- `--output`: Results directory (default: results)

## Scientific Innovation

### Fair Synaptic Comparison
The key innovation is **impact normalization** between synaptic modes:

- **Dynamic synapses**: Standard exponential decay (τ_syn = 5ms)
- **Immediate synapses**: Weights scaled by τ_syn/dt ≈ 50 to match total impact

This allows clean comparison of temporal dynamics effects while controlling for synaptic strength.

### Session Averaging Strategy
- **Single session execution**: Each MPI job processes one session efficiently
- **Automatic averaging**: Script combines results across sessions
- **Statistical robustness**: 100 trials × N sessions per parameter combination

### Mean-Centered Distributions
All parameters are explicitly centered after sampling:
```python
# Thresholds: exact -55mV mean
thresholds = thresholds - np.mean(thresholds) + (-55.0)

# Weights: exact 0 mean  
weights = weights - np.mean(weights) + 0.0
```

## Data Analysis

### Loading Results

```python
import pickle
import numpy as np

# Load single session results
with open('results/data/chaos_session_1_dynamic.pkl', 'rb') as f:
    session_results = pickle.load(f)

# Load averaged results
with open('results/data/chaos_averaged_dynamic_sessions_1_2_3.pkl', 'rb') as f:
    averaged_results = pickle.load(f)

print(f"Session 1: {len(session_results)} combinations")
print(f"Averaged: {len(averaged_results)} combinations")
```

### Synaptic Mode Comparison

```python
# Load both synaptic modes
with open('results/data/chaos_averaged_immediate_sessions_1_2_3.pkl', 'rb') as f:
    immediate_results = pickle.load(f)

with open('results/data/chaos_averaged_dynamic_sessions_1_2_3.pkl', 'rb') as f:
    dynamic_results = pickle.load(f)

# Compare chaos measures
for i, d in zip(immediate_results, dynamic_results):
    v_th = i['v_th_std']
    g_std = i['g_std']
    
    lz_immediate = i['lz_mean']
    lz_dynamic = d['lz_mean']
    
    print(f"v_th={v_th:.1f}, g={g_std:.1f}: LZ_immediate={lz_immediate:.1f}, LZ_dynamic={lz_dynamic:.1f}")
```

### Enhanced Metrics Access

```python
for result in averaged_results:
    # Parameter values
    v_th_std = result['v_th_std']
    g_std = result['g_std'] 
    synaptic_mode = result['synaptic_mode']
    
    # Chaos measures (averaged across sessions)
    lz_complexity = result['lz_mean']
    hamming_slope = result['hamming_mean']
    
    # Enhanced measures
    spike_differences = result['spike_diff_mean']
    network_dimensionality = result['effective_dim_mean']
    temporal_precision = result['gamma_coincidence_mean']
    
    # Statistical info
    n_sessions = result['n_sessions']
    total_trials = result['total_trials']  # 100 × n_sessions
```

## System Requirements

### Computational Resources
- **CPU**: Multi-core system (20+ cores recommended)
- **Memory**: 16GB+ for large parameter sweeps
- **Storage**: 5GB+ per major experiment
- **Time**: ~2 minutes per parameter combination per session

### Software Dependencies
```bash
pip install numpy scipy mpi4py psutil scikit-learn matplotlib
```

## Troubleshooting

### Common Issues
1. **MPI not found**: Install OpenMPI or MPICH
2. **Memory errors**: Reduce `--n_neurons` or increase system RAM
3. **File permissions**: Ensure scripts are executable (`chmod +x`)
4. **Missing results**: Check MPI process completion in logs

### Health Monitoring
Built-in system monitoring with recovery breaks:
- **Temperature**: Pauses if CPU > 90°C
- **Memory**: Recovery if usage > 95%
- **CPU**: Throttling if utilization > 98%

### Testing
```bash
# Verify random structure implementation
python tests/test_random_structure.py

# Should confirm:
# ✓ Network structure depends on session_id AND parameters
# ✓ Mean centering works for normal and uniform distributions  
# ✓ Synaptic modes have fair impact normalization
# ✓ Session averaging combines results correctly
```

## Expected Results

### Synaptic Mode Differences
You should observe different chaos landscapes between immediate and dynamic synapses:

- **Dynamic synapses**: Smoother parameter dependence due to temporal filtering
- **Immediate synapses**: Sharper transitions, potentially higher sensitivity

### Heterogeneity Effects
- **Low heterogeneity**: More synchronized, lower dimensional dynamics
- **High heterogeneity**: More chaotic, higher dimensional dynamics
- **Distribution shape**: Normal vs uniform may show different sensitivity

### Session Variability
Session averaging should reduce noise while preserving systematic parameter effects.

## Version History

- **v2.0.0-random-structure**: Random structure with synaptic mode comparison
- **v1.0.0-fixed-structure**: Fixed topology with multiplier scaling  
- **v0.x**: Initial development and testing

---

**Key Innovation**: This framework provides the first systematic comparison of immediate vs. dynamic synaptic coupling effects on network chaos with proper impact normalization, enabling clean separation of temporal dynamics from coupling strength effects.