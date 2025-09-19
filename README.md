# Spiking RNN Heterogeneity Framework with Fixed Network Structure

A comprehensive framework for studying chaos and network dynamics in heterogeneous spiking recurrent neural networks with **fixed network topology** across parameter combinations.

## Key Architecture Features

### Fixed Network Structure
- **Network topology depends ONLY on `session_id`**
- Same connectivity patterns, perturbation targets, and input channels across all parameter combinations
- Enables pure heterogeneity effect studies without topology confounds

### Multiplier-Based Heterogeneity Scaling
- **Base heterogeneities**: `v_th_std = 0.01`, `g_std = 0.01` (fixed)
- **Multiplier scaling**: 1-100 → actual heterogeneity 0.01-1.0
- **Exact mean preservation**: -55mV spike thresholds, 0 synaptic weights
- **Relative structure preserved** across all heterogeneity levels

### Enhanced Analysis Metrics
- **Original chaos measures**: Lempel-Ziv complexity, Hamming distance slopes
- **Network dimensionality**: Intrinsic and effective dimensions via PCA
- **Spike divergence**: Total difference magnitude between conditions
- **Gamma coincidence**: Temporal precision metrics (5ms window)

### Robust Statistical Sampling
- **100 trials per parameter combination** (increased from 20)
- **Extended analysis window**: 300ms post-perturbation
- **Individual rank recovery**: No coordinated system breaks
- **Relaxed health monitoring**: 90°C, 98% CPU, 95% memory thresholds

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Hierarchical RNG with fixed base distributions
│   ├── lif_neuron.py              # LIF neurons with multiplier scaling
│   ├── synaptic_model.py          # Fixed connectivity with scaled weights
│   └── spiking_network.py         # Complete RNN with multiplier interface
├── analysis/                      # Analysis and measurement tools  
│   └── spike_analysis.py          # Enhanced chaos quantification
├── experiments/                   # Experiment coordination
│   └── chaos_experiment.py        # Fixed-structure chaos experiments
├── runners/                       # Execution scripts
│   ├── mpi_chaos_runner.py        # MPI-parallelized execution
│   └── run_chaos_experiment.sh    # Shell script launcher
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation verification
│   └── test_fixed_structure.py    # Fixed structure validation
└── results/data/                  # Experiment outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd spiking_rnn_heterogeneity

# Install dependencies
pip install -r requirements.txt

# Install MPI (if needed)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation
python tests/test_installation.py
python tests/test_fixed_structure.py
```

### 2. Define nohup_bash Function

Add this to your `~/.bashrc` or run in your terminal:

```bash
nohup_bash () {
   script_name=$1
   script_basename=$(basename "$script_name" .sh)  # Get just the filename without path or extension
   shift
   nohup ./${script_name} "$@" > output_${script_basename}.log 2>&1 & 
   echo "Process started. Monitor with: tail -f output_${script_basename}.log"
   disown
}
```

### 3. Run Experiments

**Small test experiment:**
```bash
nohup_bash runners/run_chaos_experiment.sh --session 1 --n_v_th 2 --n_g 2 --multiplier_min 1 --multiplier_max 10 --n_input_rates 2 --nproc 8
```

**Production experiment:**
```bash
nohup_bash runners/run_chaos_experiment.sh --session 1 --n_v_th 20 --n_g 20 --n_input_rates 10 --input_rate_min 100.0 --input_rate_max 200.0 --multiplier_min 1.0 --multiplier_max 100.0 --nproc 50
```

**Monitor progress:**
```bash
tail -f output_run_chaos_experiment.log
```

## Parameter Specification

### Core Parameters
- `--session`: Session ID (determines fixed network structure)
- `--n_v_th`: Number of spike threshold multiplier values (default: 10)
- `--n_g`: Number of synaptic weight multiplier values (default: 10)
- `--nproc`: Number of MPI processes (default: 50)

### Heterogeneity Scaling
- `--multiplier_min`: Minimum multiplier (default: 1.0 → 0.01 actual heterogeneity)
- `--multiplier_max`: Maximum multiplier (default: 100.0 → 1.0 actual heterogeneity)

### Input Rate Modulation
- `--n_input_rates`: Number of input rate values (default: 5)
- `--input_rate_min`: Minimum background rate Hz (default: 50.0)
- `--input_rate_max`: Maximum background rate Hz (default: 500.0)

### System Configuration
- `--n_neurons`: Network size (default: 1000)
- `--output`: Output directory (default: results)

## Scientific Benefits

### Fixed Structure Architecture
- **Pure heterogeneity effects**: Network topology held constant
- **Systematic scaling**: Relative structure preserved at all levels
- **Perfect reproducibility**: Same connectivity across sessions
- **Enhanced comparability**: Identical perturbation targets

### Multiplier Scaling System
- **Continuous parameter space**: Smooth scaling from 0.01 to 1.0 heterogeneity
- **Mean preservation**: Exact -55mV thresholds, 0 weights at all scales
- **Biological relevance**: Covers realistic heterogeneity ranges
- **Mathematical control**: Known scaling relationships

## Data Analysis

### Loading Results
```python
import pickle
import numpy as np

# Load experiment results
with open('results/data/chaos_fixed_structure_session_2.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"Total combinations: {len(results)}")
print(f"First result keys: {list(results[0].keys())}")
```

### Accessing Enhanced Metrics
```python
# Extract key metrics
for result in results:
    v_th_mult = result['v_th_multiplier']           # Multiplier used (1-100)
    actual_v_th = result['v_th_std']                # Actual heterogeneity (0.01-1.0)
    
    # Original chaos measures
    lz_complexity = result['lz_mean']               # Lempel-Ziv complexity
    hamming_slope = result['hamming_mean']          # Hamming distance slope
    
    # Enhanced measures
    spike_diffs = result['spike_diff_mean']         # Spike divergence magnitude
    dimensions = result['effective_dim_mean']       # Network dimensionality
    gamma_coinc = result['gamma_coincidence_mean']  # Temporal precision
    
    # 100 individual trial results available in arrays:
    trial_lz = result['lz_complexities']            # Shape: (100,)
    trial_dims = result['effective_dimensionalities'] # Shape: (100,)
```

### Visualization Examples
```python
import matplotlib.pyplot as plt

# Extract parameter grids
v_th_mults = sorted(set(r['v_th_multiplier'] for r in results))
g_mults = sorted(set(r['g_multiplier'] for r in results))

# Create 2D heatmap of chaos measures
lz_matrix = np.zeros((len(v_th_mults), len(g_mults)))
for result in results:
    i = v_th_mults.index(result['v_th_multiplier'])
    j = g_mults.index(result['g_multiplier'])
    lz_matrix[i, j] = result['lz_mean']

plt.figure(figsize=(10, 8))
plt.imshow(lz_matrix, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='LZ Complexity')
plt.xlabel('g_multiplier')
plt.ylabel('v_th_multiplier')
plt.title('Chaos Landscape with Fixed Network Structure')
plt.show()
```

## System Requirements

### Computational Resources
- **CPU**: Multi-core system (recommended: 20+ cores for parallel execution)
- **Memory**: 8GB minimum, 32GB+ recommended for large experiments
- **Storage**: 10GB+ for comprehensive parameter sweeps
- **Time**: ~5 minutes per parameter combination (100 trials)

### Software Dependencies
- Python 3.8+
- NumPy, SciPy
- MPI4Py (for parallel execution)
- PSUtil (for health monitoring)

## Troubleshooting

### Common Issues
1. **Missing files**: Ensure you're running from project root directory
2. **MPI errors**: Check MPI installation and process limits
3. **Memory issues**: Reduce `--n_neurons` or increase system memory
4. **Long experiments**: Start with small parameter grids for testing

### Health Monitoring
The system includes built-in health monitoring with automatic recovery:
- **Temperature monitoring**: Pauses execution if CPU > 90°C
- **Memory management**: Recovery breaks for memory pressure > 95%
- **Individual rank recovery**: Only affected processes pause, others continue

### Testing Fixed Structure
```bash
# Verify fixed structure requirements
python tests/test_fixed_structure.py

# Test should confirm:
# ✓ Network structure depends only on session_id
# ✓ Base distributions have exact means (-55.0, 0.0)
# ✓ Multiplier scaling preserves means exactly
# ✓ Relative network structure preserved across multipliers
```

## Version History

- **v2.0.0-fixed-structure**: Fixed network topology with multiplier scaling
- **v1.0.0-stable**: Enhanced analysis with production sampling
- **v0.x**: Initial implementations and testing

---

**Note**: This framework provides unprecedented experimental control for studying heterogeneity effects in spiking neural networks by maintaining identical network structure across all parameter combinations while systematically scaling heterogeneity magnitude.