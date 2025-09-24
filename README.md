# Spiking RNN Heterogeneity Framework - Enhanced Complexity Analysis

A comprehensive framework for studying chaos and network dynamics in heterogeneous spiking recurrent neural networks with **enhanced complexity measures**, **Kistler coincidence analysis**, **pattern stability detection**, and **multi-dimensional analysis**.

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

### Enhanced Complexity Analysis
- **4 LZ-based measures**: Matrix flattened, spatial patterns, PCI variants
- **Kistler coincidence factor**: Official formula with multiple precision windows
- **Pattern stability detection**: Identifies repeating spatiotemporal patterns
- **Multi-bin dimensionality**: Participation ratio at 2ms, 5ms, 20ms resolutions

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
├── analysis/                      # Enhanced analysis tools  
│   └── spike_analysis.py          # 4 LZ measures + Kistler + PCI
├── experiments/                   # Experiment coordination
│   └── chaos_experiment.py        # Single session + averaging
├── runners/                       # Execution scripts
│   ├── mpi_chaos_runner.py        # MPI enhanced runner
│   └── run_chaos_experiment.sh    # Session coordination script
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation verification
│   └── test_comprehensive_structure.py  # Structure validation
└── results/data/                  # Experiment outputs
```

## Enhanced Analysis Features

### Four LZ-based Complexity Measures
1. **LZ Matrix Flattened**: Original approach using flattened difference matrix
2. **LZ Spatial Patterns**: Complexity of spatial pattern sequences
3. **PCI Raw**: Perturbational Complexity Index without normalization
4. **PCI Normalized**: PCI with entropy normalization (Casali et al.)

### Coincidence Analysis
- **Kistler Coincidence Factor**: Official Γ formula with 2ms and 5ms precision windows
- **Enhanced Gamma Coincidence**: Multiple window sizes (5ms, 10ms) for comparison
- **Network-wide averaging**: Statistics across all neurons

### Pattern Stability Detection
```python
# Identifies repeating patterns in neural dynamics
stable_info = {
    'period': 3,           # Pattern repeats every 3 time steps
    'repeats': 5,          # Pattern repeats 5 times
    'onset_time': 120,     # Pattern starts at t=120ms
    'pattern': [1, 0, 1]   # The repeating pattern
}
```

### Multi-Resolution Dimensionality
- **2ms bins**: High temporal resolution for fast dynamics
- **5ms bins**: Intermediate resolution matching refractory period
- **20ms bins**: Coarse resolution for slow population dynamics

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install numpy scipy mpi4py psutil

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test enhanced installation
python tests/test_installation.py
python tests/test_comprehensive_structure.py
```

### 2. Run Enhanced Experiments

**Quick test with enhanced analysis:**
```bash
chmod +x runners/run_chaos_experiment.sh
./runners/run_chaos_experiment.sh --session_ids "1" --n_v_th 3 --n_g 3 --no_average --nproc 4
```

**Compare synaptic modes with full analysis:**
```bash
# Test immediate synapses
./runners/run_chaos_experiment.sh --synaptic_mode immediate --session_ids "1 2 3" --n_v_th 5 --n_g 5 --input_rate_max 1000

# Test dynamic synapses with extended rates
./runners/run_chaos_experiment.sh --synaptic_mode dynamic --session_ids "1 2 3" --n_v_th 5 --n_g 5 --input_rate_max 1000
```

**Full enhanced heterogeneity study:**
```bash
./runners/run_chaos_experiment.sh --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20 --v_th_std_max 2.0 --g_std_max 2.0 --input_rate_max 1000 --nproc 50
```

**Test different threshold distributions:**
```bash
./runners/run_chaos_experiment.sh --v_th_distributions "normal uniform" --session_ids "1 2 3"
```

### 3. Enhanced Progress Monitoring

```bash
# Follow enhanced experiment progress with detailed metrics
tail -f output_run_chaos_experiment.log

# Monitor shows all new measures:
# LZ (flattened): 45.2±3.1
# LZ (spatial): 23.4±2.8  
# PCI (normalized): 0.67±0.12
# Kistler (2ms): 0.34±0.08
# Silent neurons: 23.5%
# Stable patterns: 0.15
```

## Enhanced Parameter Specification

### Core Parameters
- `--session_ids`: Space-separated session IDs for averaging (e.g., "1 2 3")
- `--n_v_th`: Number of spike threshold heterogeneity values (default: 10)
- `--n_g`: Number of synaptic weight heterogeneity values (default: 10)
- `--nproc`: Number of MPI processes (default: 50)

### Extended Input Range
- `--input_rate_min/max`: Background input range (default: 50-1000 Hz for dynamic mode)
- **Dynamic synapses**: Tested up to 1000 Hz for high-activity regimes
- **Immediate synapses**: Tested up to 500 Hz for stability

### Enhanced Analysis Control
- **2ms binning**: Default for all spike matrices (matches refractory period)
- **Multiple coincidence windows**: Automatic testing of different temporal precisions
- **Pattern stability**: Automatic detection with configurable minimum repeats

## Scientific Innovation

### Enhanced Complexity Quantification
The framework now implements the official **Perturbational Complexity Index (PCI)** from Casali et al.:

```
PCI = c_L(t = L₂) × log₂ L / (L × H(L))
```

Where:
- `c_L`: Lempel-Ziv complexity of spatial patterns
- `L`: Total spatiotemporal samples
- `H(L)`: Source entropy of the binary matrix
- **Activation threshold**: Only computed when >1% of samples are active

### Kistler Coincidence Factor
Implementation of the official coincidence factor from Kistler et al. (1997):

```
Γ = (N_coinc - ⟨N_coinc⟩) / (½(N_data + N_SRM) × N)
```

Where:
- `N_coinc`: Observed coincidences within precision Δ
- `⟨N_coinc⟩`: Expected coincidences for Poisson process
- `N`: Normalization factor (1 - 2νΔ)

### Pattern Stability Analysis
Novel detection of stable spatiotemporal patterns:
- **High complexity + stable patterns**: Initial complexity that settles
- **Low complexity + stable patterns**: Network quickly reaches steady state
- **High complexity + no patterns**: True ongoing complexity
- **Low complexity + no patterns**: Minimal response to perturbation

### Multi-Resolution Dimensionality
Analysis across temporal scales reveals:
- **Fine-scale (2ms)**: Individual spike precision
- **Medium-scale (5ms)**: Population synchronization
- **Coarse-scale (20ms)**: Slow population dynamics

## Enhanced Data Analysis

### Loading Enhanced Results

```python
import pickle
import numpy as np

# Load enhanced results
with open('results/data/chaos_enhanced_session_1_dynamic.pkl', 'rb') as f:
    results = pickle.load(f)

for result in results:
    # Enhanced complexity measures
    lz_flattened = result['lz_matrix_flattened_mean']
    lz_spatial = result['lz_spatial_patterns_mean']
    pci_normalized = result['pci_normalized_mean']
    pci_threshold = result['pci_with_threshold_mean']
    
    # Coincidence measures  
    kistler_2ms = result['kistler_delta_2ms_mean']
    kistler_5ms = result['kistler_delta_5ms_mean']
    gamma_5ms = result['gamma_window_5ms_mean']
    gamma_10ms = result['gamma_window_10ms_mean']
    
    # Pattern stability
    stable_fraction = result['stable_pattern_fraction']
    stable_period = result['stable_period_mean']
    
    # Multi-bin dimensionality
    participation_2ms = result['participation_ratio_bin_2ms_mean']
    participation_5ms = result['participation_ratio_bin_5ms_mean']  
    participation_20ms = result['participation_ratio_bin_20ms_mean']
    
    # Firing rate analysis
    silent_percent = result['control_percent_silent_mean']
    mean_rate = result['control_mean_firing_rate_mean']
```

### Enhanced Synaptic Mode Comparison

```python
# Compare all enhanced measures between modes
for param_combo in zip(immediate_results, dynamic_results):
    imm, dyn = param_combo
    
    print(f"v_th={imm['v_th_std']:.1f}, g={imm['g_std']:.1f}")
    print(f"  LZ spatial:    {imm['lz_spatial_patterns_mean']:.1f} vs {dyn['lz_spatial_patterns_mean']:.1f}")
    print(f"  PCI:           {imm['pci_normalized_mean']:.3f} vs {dyn['pci_normalized_mean']:.3f}")
    print(f"  Kistler (2ms): {imm['kistler_delta_2ms_mean']:.3f} vs {dyn['kistler_delta_2ms_mean']:.3f}")
    print(f"  Stable patterns: {imm['stable_pattern_fraction']:.3f} vs {dyn['stable_pattern_fraction']:.3f}")
    print()
```

### Firing Rate Onset Analysis

```python
# Analyze network activation thresholds
for result in results:
    input_rate = result['static_input_rate']
    silent_pct = result['control_percent_silent_mean']
    mean_rate = result['control_mean_firing_rate_mean']
    
    print(f"Input {input_rate:.0f}Hz: {silent_pct:.1f}% silent, {mean_rate:.2f}Hz mean")
```

## Enhanced System Requirements

### Computational Resources
- **CPU**: Multi-core system (32+ cores recommended for full analysis)
- **Memory**: 32GB+ for enhanced analysis with large parameter sweeps
- **Storage**: 10GB+ per major experiment (enhanced data)
- **Time**: ~3-4 minutes per parameter combination per session (enhanced analysis)

### Extended Analysis Time
Enhanced analysis includes:
- 4 LZ-based complexity computations
- Multiple coincidence measure calculations  
- Pattern stability detection across 100 trials
- Multi-resolution dimensionality analysis
- Comprehensive firing rate statistics

## Enhanced Troubleshooting

### New Analysis-Specific Issues
1. **PCI computation errors**: Check activation threshold (>1% activity required)
2. **Pattern stability false positives**: Adjust `min_repeats` parameter
3. **Dimensionality matrix errors**: Handled automatically with safe eigenvalue computation
4. **High rate analysis**: Dynamic mode tested up to 1000Hz, immediate mode up to 500Hz

### Enhanced Health Monitoring
Extended monitoring for intensive computations:
- **Complexity computation**: Memory usage tracking during LZ analysis
- **Pattern detection**: CPU monitoring during stability search
- **Dimensionality analysis**: Safe eigenvalue computation with fallbacks

### Enhanced Testing
```bash
# Verify enhanced implementation
python tests/test_comprehensive_structure.py

# Should confirm:
# ✓ All 4 LZ measures computed correctly
# ✓ Kistler coincidence matches reference implementation  
# ✓ Pattern stability detection functional
# ✓ Multi-bin dimensionality analysis working
# ✓ Extended firing rate analysis comprehensive
```

## Expected Enhanced Results

### Complexity Measure Relationships
You should observe systematic relationships between measures:
- **LZ spatial < LZ flattened**: Spatial patterns more compressible
- **PCI normalized ≈ 0.5-0.8**: For chaotic regimes
- **PCI threshold < PCI normalized**: Activation filtering effect

### Coincidence Analysis Insights
- **Kistler (2ms) > Kistler (5ms)**: Precision window effects
- **Dynamic mode**: Generally higher temporal precision
- **Immediate mode**: More variable coincidence patterns

### Pattern Stability Signatures
- **Low heterogeneity**: Higher stable pattern fraction
- **High input rates**: Reduced pattern stability
- **Dynamic synapses**: More stable long-term patterns

### Multi-Resolution Dimensionality
- **Fine bins (2ms)**: Highest dimensionality
- **Medium bins (5ms)**: Intermediate dimensionality  
- **Coarse bins (20ms)**: Population-level dimensionality

### Extended Rate Range Effects
- **50-200 Hz**: Linear firing rate increase
- **200-500 Hz**: Saturation effects appear
- **500-1000 Hz**: Dynamic mode maintains responsivity, immediate mode saturates

## Version History

- **v2.1.0-enhanced-analysis**: 4 LZ measures, Kistler coincidence, pattern stability, multi-bin analysis
- **v2.0.0-random-structure**: Random structure with synaptic mode comparison
- **v1.0.0-fixed-structure**: Fixed topology with multiplier scaling  
- **v0.x**: Initial development and testing

---

**Key Innovation**: This framework provides comprehensive complexity quantification with official PCI implementation, Kistler coincidence analysis, pattern stability detection, and multi-resolution dimensionality analysis, enabling deep characterization of chaotic dynamics in heterogeneous spiking networks across temporal scales.