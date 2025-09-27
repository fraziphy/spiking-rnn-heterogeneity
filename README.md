# Spiking RNN Heterogeneity Framework - Split Experiments Architecture v3.1.0

A comprehensive framework for studying **spontaneous activity** and **network stability** in heterogeneous spiking recurrent neural networks with **optimized coincidence analysis**, **enhanced connectivity strength**, and **randomized job distribution for CPU load balancing**.

## Key Architecture Features

### Split Experiments Design
- **Spontaneous Activity Analysis**: Firing rates, dimensionality (6 bin sizes), silent neurons
- **Network Stability Analysis**: Perturbation response, LZ spatial complexity, coincidence measures
- **Enhanced static Poisson connectivity**: Strength 25 (up from 1)
- **Optimized coincidence calculation**: Single loop for both Kistler and Gamma measures

### Random Structure with Parameter Dependence
- **Network topology depends on `session_id` AND parameter values**
- Different connectivity patterns for each (session, v_th_std, g_std) combination
- **Randomized job distribution**: Prevents CPU load imbalance

### Mean-Centered Heterogeneity
- **Direct heterogeneity values**: `v_th_std` and `g_std` (0.0-4.0 range)
- **Exact mean preservation**: -55mV spike thresholds, 0 synaptic weights
- **Distribution flexibility**: Normal and uniform threshold distributions

### Fair Synaptic Mode Comparison
- **Dynamic synapses**: Exponential decay with τ_syn = 5ms (realistic)
- **Immediate synapses**: Instantaneous coupling (like previous studies)
- **Impact normalization**: Immediate weights scaled by τ_syn/dt for fair comparison

### Optimized Analysis Features
- **Unified coincidence calculation**: Single loop computes both Kistler and Gamma
- **No PCI measures**: Removed pci_raw, pci_normalized, pci_with_threshold
- **No lz_matrix_flattened**: Streamlined to LZ spatial patterns only
- **Extended dimensionality bins**: 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter-dependent RNG
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Enhanced connectivity strength (25)
│   └── spiking_network.py         # Complete RNN with mode selection
├── analysis/                      # Split analysis modules  
│   ├── spontaneous_analysis.py    # Firing rates + dimensionality (6 bins)
│   └── stability_analysis.py      # LZ spatial + optimized coincidence
├── experiments/                   # Split experiment coordination
│   ├── spontaneous_experiment.py  # Duration parameter + firing analysis
│   └── stability_experiment.py    # Perturbation response + stability
├── runners/                       # Execution scripts
│   ├── mpi_spontaneous_runner.py  # MPI spontaneous activity runner
│   ├── mpi_stability_runner.py    # MPI network stability runner
│   ├── run_spontaneous_experiment.sh    # Spontaneous activity script
│   └── run_stability_experiment.sh      # Network stability script
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation verification
│   └── test_comprehensive_structure.py  # Structure validation
└── results/data/                  # Experiment outputs
```

## Split Analysis Features

### Spontaneous Activity Analysis
**Duration-based simulation with comprehensive statistics:**
- **Firing rate analysis**: Mean, std, min, max firing rates
- **Silent neuron analysis**: Percentage of silent/active neurons
- **6-bin dimensionality analysis**: 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms temporal resolutions
- **Participation ratio**: Network-wide activity distribution
- **10 trials per combination**: Efficient for steady-state measures

### Network Stability Analysis  
**Perturbation response with optimized computation:**
- **LZ spatial patterns**: Complexity of spatial pattern sequences only
- **Optimized coincidence**: Single-loop calculation for Kistler + Gamma measures
- **Hamming distance slopes**: Perturbation divergence analysis
- **Pattern stability detection**: Identifies repeating spatiotemporal patterns
- **100 trials per combination**: Comprehensive sampling for dynamics

### Enhanced Connectivity
```python
# Enhanced static Poisson connectivity
static_input_strength = 25.0  # Up from 1.0
```

### Optimized Coincidence Calculation
```python
# Single loop computes both measures efficiently
kistler_factor, gamma_factor = unified_coincidence_factor(
    spikes1, spikes2, delta=2.0, duration=100.0
)
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install numpy scipy mpi4py psutil

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test split experiments installation
python tests/test_installation.py
python tests/test_comprehensive_structure.py
```

### 2. Run Split Experiments

**Spontaneous Activity Analysis (with duration parameter):**
```bash
chmod +x runners/run_spontaneous_experiment.sh

# Quick 2-second spontaneous activity test
./runners/run_spontaneous_experiment.sh --duration 2 --session_ids "1" --n_v_th 3 --n_g 3 --nproc 4

# Long 10-second analysis for detailed statistics  
./runners/run_spontaneous_experiment.sh --duration 10 --session_ids "1 2 3" --n_v_th 10 --n_g 10
```

**Network Stability Analysis (perturbation response):**
```bash
chmod +x runners/run_stability_experiment.sh

# Quick stability test with randomized jobs
./runners/run_stability_experiment.sh --session_ids "1" --n_v_th 3 --n_g 3 --nproc 4

# Full stability study with session averaging
./runners/run_stability_experiment.sh --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20
```

**Compare Synaptic Modes:**
```bash
# Test immediate synapses for both experiments
./runners/run_spontaneous_experiment.sh --synaptic_mode immediate --duration 5
./runners/run_stability_experiment.sh --synaptic_mode immediate

# Test dynamic synapses for both experiments  
./runners/run_spontaneous_experiment.sh --synaptic_mode dynamic --duration 5
./runners/run_stability_experiment.sh --synaptic_mode dynamic
```

### 3. Monitor Experiment Progress

**Spontaneous Activity Progress:**
```bash
# Monitor spontaneous activity analysis
tail -f output_spontaneous.log

# Shows:
# Mean firing rate: 15.2±2.3 Hz
# Silent neurons: 23.5±4.1%
# Dimensionality (5ms): 12.3±1.8
# Total spikes: 8,450±1,200
```

**Network Stability Progress:**
```bash  
# Monitor stability analysis with optimized measures
tail -f output_stability.log

# Shows:
# LZ (spatial): 23.4±2.8  
# Hamming slope: 0.0034±0.0012
# Kistler (2ms): 0.34±0.08 (optimized calculation)
# Stable patterns: 0.15±0.05
```

## Split Experiment Parameters

### Spontaneous Activity Parameters
- `--duration`: Simulation duration in seconds (auto-converts to milliseconds)
- `--session_ids`: Sessions for network averaging (e.g., "1 2 3")
- **6 dimensionality bin sizes**: Automatic analysis at all temporal resolutions
- **10 trials per combination**: Efficient for steady-state measures

### Network Stability Parameters  
- `--session_ids`: Sessions for perturbation averaging (e.g., "1 2 3")
- **100 trials per combination**: Comprehensive perturbation sampling
- **Optimized coincidence**: Single-loop Kistler + Gamma calculation
- **Randomized job distribution**: Automatic CPU load balancing

### Common Parameters
- `--n_v_th/--n_g`: Heterogeneity grid sizes (default: 10x10)
- `--synaptic_mode`: "immediate" or "dynamic" synaptic coupling
- `--input_rate_min/max`: Background input range (enhanced: 50-1000 Hz)
- `--nproc`: MPI processes (default: 50)

## Scientific Innovation

### CPU Load Balancing
**Problem solved**: Easy parameter combinations (low rates, low heterogeneity) finish much faster than hard ones, causing CPU imbalance.

**Solution**: Jobs are randomized before MPI distribution, ensuring each CPU gets a mix of easy/hard combinations.

```python
# Randomize job order for better CPU load balancing  
random.shuffle(all_combinations)
print(f"Job order: RANDOMIZED for load balancing")
```

### Enhanced Connectivity Analysis
**Static Poisson strength increased from 1.0 to 25.0:**
- More realistic background drive
- Better network activation across parameter ranges
- Clearer differentiation between heterogeneity effects

### Optimized Coincidence Computation
**Before**: Two separate loops for Kistler and Gamma coincidence
**Now**: Single unified loop computes both measures simultaneously

```python
# Old approach - duplicate computation
kistler = kistler_coincidence_factor(spikes1, spikes2, delta=2.0)
gamma = gamma_coincidence(spikes1, spikes2, window=2.0)

# New approach - unified computation  
kistler, gamma = unified_coincidence_factor(spikes1, spikes2, delta=2.0)
```

### Extended Dimensionality Analysis
**6 temporal bin sizes** for comprehensive multi-scale analysis:
- **0.1ms**: Individual spike precision
- **2ms**: Refractory period scale  
- **5ms**: Synaptic time scale
- **20ms**: Population synchronization
- **50ms**: Slow population dynamics
- **100ms**: Network-wide integration

## Data Analysis

### Loading Split Experiment Results

**Spontaneous Activity Results:**
```python
import pickle

# Load spontaneous activity analysis
with open('results/data/spontaneous_session_1_dynamic.pkl', 'rb') as f:
    spontaneous_results = pickle.load(f)

for result in spontaneous_results:
    # Duration and firing statistics
    duration = result['duration']  # Duration in milliseconds
    mean_rate = result['mean_firing_rate_mean']
    silent_pct = result['percent_silent_mean']
    
    # Multi-bin dimensionality  
    dim_0_1ms = result['effective_dimensionality_bin_0.1ms_mean']
    dim_2ms = result['effective_dimensionality_bin_2.0ms_mean']
    dim_5ms = result['effective_dimensionality_bin_5.0ms_mean']
    dim_20ms = result['effective_dimensionality_bin_20.0ms_mean']
    dim_50ms = result['effective_dimensionality_bin_50.0ms_mean']
    dim_100ms = result['effective_dimensionality_bin_100.0ms_mean']
    
    print(f"Duration {duration}ms: {mean_rate:.1f}Hz, {silent_pct:.1f}% silent")
    print(f"Dimensionality: {dim_0_1ms:.1f} → {dim_5ms:.1f} → {dim_50ms:.1f}")
```

**Network Stability Results:**
```python
# Load network stability analysis  
with open('results/data/stability_session_1_dynamic.pkl', 'rb') as f:
    stability_results = pickle.load(f)

for result in stability_results:
    # Optimized stability measures (no PCI, no lz_matrix_flattened)
    lz_spatial = result['lz_spatial_patterns_mean']
    hamming_slope = result['hamming_slope_mean']  # Note: singular, not plural
    
    # Optimized coincidence measures
    kistler_2ms = result['kistler_delta_2ms_mean']
    kistler_5ms = result['kistler_delta_5ms_mean']  
    gamma_2ms = result['gamma_window_2ms_mean']
    gamma_5ms = result['gamma_window_5ms_mean']
    
    # Pattern stability
    stable_fraction = result['stable_pattern_fraction']
    
    print(f"LZ spatial: {lz_spatial:.1f}, Hamming: {hamming_slope:.4f}")
    print(f"Kistler: {kistler_2ms:.3f}, Gamma: {gamma_2ms:.3f}")
    print(f"Stable patterns: {stable_fraction:.3f}")
```

### Cross-Experiment Comparison

```python
# Compare spontaneous vs stability measures
for spont, stab in zip(spontaneous_results, stability_results):
    v_th = spont['v_th_std']
    g = spont['g_std']
    
    # Spontaneous measures
    firing_rate = spont['mean_firing_rate_mean']
    dimensionality = spont['effective_dimensionality_bin_5.0ms_mean']
    
    # Stability measures  
    lz_complexity = stab['lz_spatial_patterns_mean']
    coincidence = stab['kistler_delta_2ms_mean']
    
    print(f"v_th={v_th:.1f}, g={g:.1f}:")
    print(f"  Spontaneous: {firing_rate:.1f}Hz, {dimensionality:.1f}D")
    print(f"  Stability: LZ={lz_complexity:.1f}, Γ={coincidence:.3f}")
```

## System Requirements

### Computational Resources
- **CPU**: Multi-core system (32+ cores recommended)
- **Memory**: 16GB+ (spontaneous), 32GB+ (stability with 100 trials)
- **Storage**: 5GB+ per major experiment
- **Time**: 
  - Spontaneous: ~30s per combination per session
  - Stability: ~2min per combination per session (optimized)

### Load Balancing Benefits
- **Old approach**: Some CPUs finish 10x faster than others
- **New approach**: Randomized jobs ensure even CPU utilization
- **Speedup**: 2-3x faster completion for mixed parameter ranges

## Enhanced Troubleshooting

### Split Experiment Issues
1. **Duration conversion**: Automatically converts seconds to milliseconds
2. **Field name consistency**: All arrays use `_values` suffix, stats use `_mean`
3. **Static connectivity**: Should show 25.0, not 1.0
4. **Optimized coincidence**: Single loop computes both Kistler and Gamma

### Testing Split Framework
```bash
# Verify split experiments implementation
python tests/test_installation.py
python tests/test_comprehensive_structure.py

# Should confirm:
# ✓ Spontaneous analysis with 6 bin sizes
# ✓ Stability analysis without PCI measures
# ✓ Enhanced static connectivity (25.0)  
# ✓ Optimized coincidence calculation
# ✓ Randomized job distribution
```

## Expected Results

### Spontaneous Activity Patterns
- **Low heterogeneity**: High firing rates, low dimensionality
- **High heterogeneity**: Variable firing rates, higher dimensionality
- **Multi-bin trends**: Dimensionality decreases with larger bin sizes
- **Duration effects**: Longer simulations give more stable statistics

### Network Stability Signatures  
- **Low heterogeneity**: Simple spatial patterns, high coincidence
- **High heterogeneity**: Complex spatial patterns, low coincidence  
- **Dynamic vs immediate**: Dynamic shows more temporal structure
- **Pattern stability**: More stable at low heterogeneity

### CPU Load Balancing Effects
- **Randomized jobs**: Even CPU utilization across processes
- **Completion time**: 2-3x faster than ordered job distribution
- **Resource usage**: More consistent memory and CPU usage patterns

## Version History

- **v3.0.0-split-experiments**: Split architecture (spontaneous + stability), enhanced connectivity (25), optimized coincidence, randomized jobs
- **v2.1.0-enhanced-analysis**: 4 LZ measures, Kistler coincidence, pattern stability, multi-bin analysis  
- **v2.0.0-random-structure**: Random structure with synaptic mode comparison
- **v1.0.0-fixed-structure**: Fixed topology with multiplier scaling

---

**Key Innovation**: Split experiments architecture separates spontaneous activity analysis (firing rates, 6-bin dimensionality) from network stability analysis (perturbation response, optimized coincidence), with enhanced connectivity strength (25), unified coincidence calculation, and randomized job distribution for optimal CPU load balancing.