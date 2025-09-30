# Spiking RNN Heterogeneity Framework - Split Experiments Architecture v3.5.0

A comprehensive framework for studying **spontaneous activity** and **network stability** in heterogeneous spiking recurrent neural networks with **Poisson process validation**, **Shannon entropy**, **settling time analysis**, and **full-simulation stability measures**.

## Key Architecture Features

### Split Experiments Design
- **Spontaneous Activity Analysis**: Firing rates, dimensionality (6 bin sizes), silent neurons, **Poisson validation**
- **Network Stability Analysis**: Full-simulation perturbation response, **Shannon entropy**, **settling time**, LZ complexity, coincidence measures
- **Enhanced static Poisson connectivity**: Strength 10 (up from 1)
- **Optimized coincidence calculation**: Single loop for both Kistler and Gamma measures

### Network Stability Revolution (NEW in v3.5.0)
- **Full-simulation analysis**: Spike differences computed from t=0 (not just post-perturbation)
- **Shannon entropy measures**: Symbol sequence entropy and spike difference entropy
- **Settling time detection**: Time to return to baseline (50ms of zeros), searched backwards
- **Pattern diversity**: Unique spatial pattern counts across full simulation
- **Removed measures**: Hamming slopes, stable periods, old spatial entropy (cleaner, focused analysis)

### Poisson Process Validation (v3.2.0)
- **Statistical tests**: Kolmogorov-Smirnov (ISI exponential), Chi-square (count Poisson)
- **Coefficient of Variation (CV)**: Expected ~1.0 for Poisson processes
- **Fano Factor**: Variance/mean ratio, expected ~1.0 for Poisson
- **Population statistics**: Fraction of neurons showing Poisson-like behavior
- **Automatic transient removal**: First 25ms excluded from analysis

### Fixed RNG Bugs (v3.2.0)
- **Time-step dependent**: Each time step gets unique random numbers
- **Rate-dependent**: Each input rate gets independent random sequences
- **Proper seeding**: `seed = [session_id, v_th_std, g_std, trial_id, time_step, rate]`

### Random Structure with Parameter Dependence
- **Network topology depends on `session_id` AND parameter values**
- Different connectivity patterns for each (session, v_th_std, g_std) combination
- **Randomized job distribution**: Prevents CPU load imbalance

### Mean-Centered Heterogeneity
- **Direct heterogeneity values**: `v_th_std` and `g_std` (0.01-1.0 range)
- **Exact mean preservation**: -55mV spike thresholds, 0 synaptic weights
- **Distribution flexibility**: Normal and uniform threshold distributions

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Rate & time-dependent RNG
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Enhanced connectivity + time-step RNG
│   └── spiking_network.py         # Complete RNN with time tracking
├── analysis/                      # Split analysis modules  
│   ├── spontaneous_analysis.py    # Firing + dimensionality + Poisson tests
│   └── stability_analysis.py      # Shannon entropy + settling time + LZ (UPDATED v3.5.0)
├── experiments/                   # Split experiment coordination
│   ├── spontaneous_experiment.py  # Duration parameter + Poisson analysis
│   └── stability_experiment.py    # Full-simulation stability (UPDATED v3.5.0)
├── runners/                       # Execution scripts
│   ├── mpi_spontaneous_runner.py  # MPI spontaneous activity runner
│   ├── mpi_stability_runner.py    # MPI stability runner (UPDATED v3.5.0)
│   ├── run_spontaneous_experiment.sh    # Spontaneous activity script
│   └── run_stability_experiment.sh      # Stability script (UPDATED v3.5.0)
├── pipeline.sh                    # Sequential pipeline execution
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation verification (UPDATED v3.5.0)
│   └── test_comprehensive_structure.py  # Structure validation (UPDATED v3.5.0)
└── results/data/                  # Experiment outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install numpy scipy mpi4py psutil

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation
python tests/test_installation.py
python tests/test_comprehensive_structure.py
```

### 2. Run Sequential Pipeline (Recommended)

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
```

The pipeline automatically runs:
1. Spontaneous activity analysis (all input rates)
2. Network stability analysis (after spontaneous completes)
3. Session averaging (if multiple sessions)
4. Comprehensive logging to `pipeline.log`

### 3. Run Individual Experiments

**Spontaneous Activity Analysis (with Poisson validation):**
```bash
chmod +x runners/run_spontaneous_experiment.sh

# Quick 2-second test with Poisson analysis
./runners/run_spontaneous_experiment.sh --duration 2 --session_ids "1" --n_v_th 2 --n_g 2 --nproc 4

# Full analysis with multiple sessions
./runners/run_spontaneous_experiment.sh --duration 5 --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20

# Background execution
nohup bash runners/run_spontaneous_experiment.sh --session_ids "1 2 3" --n_input_rates 10 > spontaneous.log 2>&1 &
disown
```

**Network Stability Analysis (with new measures):**
```bash
chmod +x runners/run_stability_experiment.sh

# Quick test
./runners/run_stability_experiment.sh --session_ids "1" --n_v_th 2 --n_g 2 --nproc 4

# Full study
./runners/run_stability_experiment.sh --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20

# Background execution
nohup bash runners/run_stability_experiment.sh --session_ids "1 2 3" --n_input_rates 10 > stability.log 2>&1 &
disown
```

### 4. Monitor Background Jobs

```bash
# Check if jobs are running
jobs
ps aux | grep run_spontaneous
ps aux | grep run_stability

# Monitor progress
tail -f spontaneous.log
tail -f stability.log
tail -f pipeline.log

# Watch system resources
htop
free -h
df -h results/

# Safe to logout - jobs continue with nohup + disown
exit
```

## Split Experiment Parameters

### Spontaneous Activity Parameters
- `--duration`: Simulation duration in seconds (auto-converts to milliseconds)
- `--session_ids`: Sessions for network averaging (e.g., "1 2 3")
- **Automatic transient removal**: First 25ms excluded
- **6 dimensionality bin sizes**: 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms
- **Poisson validation**: CV, Fano Factor, KS/Chi-square tests
- **10 trials per combination**: Efficient for steady-state measures

### Network Stability Parameters (Updated v3.5.0)
- `--session_ids`: Sessions for perturbation averaging (e.g., "1 2 3")
- **100 trials per combination**: Comprehensive perturbation sampling (different neuron each trial)
- **Full-simulation analysis**: Spike differences from t=0 to end
- **Shannon entropy**: Symbol sequence and spike difference entropies
- **Settling time**: Backward search for 50ms baseline return
- **Optimized coincidence**: Single-loop Kistler + Gamma calculation

### Common Parameters
- `--n_v_th/--n_g`: Heterogeneity grid sizes (default: 20x20)
- `--synaptic_mode`: "immediate" or "dynamic" synaptic coupling
- `--input_rate_min/max`: Background input range (default: 0.1-50 Hz)
- `--n_input_rates`: Number of input rate points (default: 15)
- `--nproc`: MPI processes (default: 50)

## Scientific Innovation

### Full-Simulation Stability Analysis (v3.5.0)

**Revolution**: Analysis now uses **entire simulation** instead of post-perturbation only.

**New workflow**:
1. Convert full spike trains to binary matrices (t=0 to end, dt=0.1ms bins)
2. Compute spike difference matrix: `spike_diff = (control != perturbed)`
3. Extract spatial patterns column by column → create symbol sequence
4. **Baseline is symbol=0** (identical networks before perturbation)
5. Analyze post-perturbation symbols for complexity and entropy

**Why this matters**:
- More accurate LZ complexity (considers full pattern evolution)
- Settling time detection (can network "forget" the perturbation?)
- Shannon entropy captures true pattern diversity
- Cleaner interpretation (baseline = 0 state)

### New Stability Measures

**1. Shannon Entropy (Symbols)**
```python
shannon_entropy_symbols = entropy(symbol_sequence[pert_bin:], base=2)
# Measures: Pattern diversity after perturbation
# High values: Complex, diverse patterns
# Low values: Simple, repetitive patterns
```

**2. Shannon Entropy (Spikes)**
```python
shannon_entropy_spikes = entropy(spike_diff_full.flatten(), base=2)
# Measures: Overall spike difference distribution
# Binary entropy on 0s and 1s
```

**3. Settling Time**
```python
settling_time_ms = find_settling_time(symbol_seq, pert_bin, bin_size=0.1, 
                                     min_zero_duration_ms=50.0)
# Searches BACKWARDS for 50ms of consecutive zeros
# Returns: Time in ms from perturbation to settling
# NaN if network never returns to baseline
```

**4. Pattern Diversity**
```python
unique_patterns_count = len(pattern_dict)  # Total unique spatial patterns
post_pert_symbol_sum = sum(symbol_seq[pert_bin:])  # Deviation from baseline
```

### Removed Measures (Cleaner Analysis)

**Eliminated complexity**:
- ❌ `hamming_slope` - Linear Hamming distance growth
- ❌ `stable_period` - Repeating pattern detection
- ❌ `spatial_entropy` - Old coarse-binned entropy
- ❌ `pattern_fraction` - Active neuron fraction

**Result**: Simpler, more interpretable stability metrics focused on:
- Information content (Shannon entropy)
- Pattern complexity (LZ, unique patterns)
- Recovery dynamics (settling time)
- Spike synchrony (coincidence measures)

### Poisson Process Validation (v3.2.0)

**Question**: Networks receive Poisson input, but do neurons output Poisson spike trains?

**Tests implemented**:
1. **Inter-Spike Interval (ISI) Analysis**:
   - Kolmogorov-Smirnov test for exponential distribution
   - CV of ISI (expected: 1.0 for Poisson)
   
2. **Spike Count Analysis**:
   - Chi-square test for Poisson distribution in time bins
   - Fano Factor = var/mean (expected: 1.0 for Poisson)

3. **Population Statistics**:
   - Fraction of neurons passing statistical tests
   - Mean CV and Fano Factor across active neurons

```python
# Example Poisson analysis results
poisson_stats = {
    'mean_cv_isi': 1.02,              # Close to 1.0 ✓
    'mean_fano_factor': 0.98,         # Close to 1.0 ✓
    'poisson_isi_fraction': 0.85,     # 85% pass KS test
    'poisson_count_fraction': 0.79    # 79% pass Chi-square test
}
```

## Data Analysis

### Loading Results with New Stability Measures

**Spontaneous Activity Results:**
```python
import pickle

# Load session-averaged results
with open('results/data/spontaneous_averaged_dynamic_sessions_1_2_3.pkl', 'rb') as f:
    results = pickle.load(f)

for result in results:
    # Standard measures
    firing_rate = result['mean_firing_rate_mean']
    dimensionality = result['effective_dimensionality_bin_5.0ms_mean']
    
    # Poisson process measures
    cv_isi = result['mean_cv_isi_mean']
    fano_factor = result['mean_fano_factor_mean']
    poisson_fraction = result['poisson_isi_fraction_mean']
    
    # Transient information
    transient_time = result['transient_time']  # 25.0 ms
    steady_spikes = result['steady_state_spikes_mean']
    
    print(f"Firing: {firing_rate:.1f}Hz, CV ISI: {cv_isi:.3f}, Fano: {fano_factor:.3f}")
    print(f"Poisson-like: {poisson_fraction:.1%}")
```

**Network Stability Results (Updated v3.5.0):**
```python
with open('results/data/stability_session_1_dynamic.pkl', 'rb') as f:
    stability_results = pickle.load(f)

for result in stability_results:
    # NEW: Shannon entropy measures
    shannon_symbols = result['shannon_entropy_symbols_mean']
    shannon_spikes = result['shannon_entropy_spikes_mean']
    
    # NEW: Settling dynamics
    settling_time = result['settling_time_mean']  # ms from perturbation
    settled_fraction = result['settled_fraction']  # fraction that settled
    
    # Pattern complexity
    lz_spatial = result['lz_spatial_patterns_mean']
    unique_patterns = result['unique_patterns_count_mean']
    
    # Optimized coincidence measures
    kistler_2ms = result['kistler_delta_2ms_mean']
    gamma_2ms = result['gamma_window_2ms_mean']
    
    print(f"Shannon (symbols): {shannon_symbols:.3f}, (spikes): {shannon_spikes:.3f}")
    print(f"Settling: {settling_time:.1f}ms, Fraction: {settled_fraction:.2%}")
    print(f"LZ: {lz_spatial:.1f}, Patterns: {unique_patterns:.0f}")
    print(f"Kistler: {kistler_2ms:.3f}, Settled: {settled_fraction:.2%}")
```

**Key Differences from v3.4.0:**
```python
# REMOVED (no longer in results):
# - hamming_slope
# - stable_period, stable_period_mean, stable_pattern_fraction
# - spatial_entropy (old version)
# - pattern_fraction

# ADDED (new in v3.5.0):
# - shannon_entropy_symbols
# - shannon_entropy_spikes
# - settling_time_ms
# - settled_fraction, settled_count
# - unique_patterns_count
# - post_pert_symbol_sum
```

## System Requirements

### Computational Resources
- **CPU**: Multi-core system (32+ cores recommended)
- **Memory**: 16GB+ (spontaneous), 32GB+ (stability with 100 trials)
- **Storage**: 5GB+ per major experiment
- **Time**: 
  - Spontaneous: ~30s per combination per session
  - Stability: ~2min per combination per session

### Pipeline Execution
- **Sequential pipeline**: Spontaneous → Stability (automatic)
- **Background jobs**: Use `nohup ... & disown` for safe logout
- **Process management**: Use `screen`, `tmux`, or `nohup`
- **Monitoring**: Check logs with `tail -f` and `htop`

## Troubleshooting

### Stability Analysis (v3.5.0)
- **Settling time = NaN**: Network never returned to baseline (50ms zeros)
- **High Shannon entropy**: Complex, diverse patterns after perturbation
- **Low unique patterns**: Network locked into few spatial configurations
- **Short settling times**: Quick recovery indicates high stability

### RNG Issues (Fixed in v3.2.0)
- ✓ Time-step bug: Each timestep now gets unique random numbers
- ✓ Rate-dependent bug: Each input rate gets independent sequences
- ✓ Proper seeding: Includes session, parameters, trial, timestep, and rate

### Poisson Analysis Issues
- **Chi-square test errors**: Handled with robust error checking and normalization
- **Insufficient spikes**: Minimum 10 spikes required for reliable tests
- **CV and Fano interpretation**: Values ≈1 indicate Poisson-like behavior

### Background Job Management
- **Job monitoring**: Use `ps`, `jobs`, and log files to track progress
- **Resource exhaustion**: Monitor memory and storage usage with `htop` and `df -h`
- **Network disconnection**: Jobs continue with `nohup ... & disown`
- **Result verification**: Check output files after completion

### Pipeline Execution
- **Sequential execution**: `pipeline.sh` ensures spontaneous completes before stability
- **Error handling**: Pipeline stops on first error (`set -e`)
- **Logging**: All output captured in `pipeline.log`

## Version History

- **v3.5.0**: **STABILITY REVOLUTION** - Full-simulation analysis, Shannon entropy, settling time
  - Complete rewrite of stability analysis using full spike trains (t=0 to end)
  - Shannon entropy measures for pattern diversity (symbols & spike differences)
  - Settling time detection with backward search for baseline return (50ms zeros)
  - Pattern diversity tracking (unique spatial patterns)
  - Removed: Hamming slopes, stable periods, old spatial entropy (cleaner analysis)
  - Updated: All downstream files (experiment, runner, tests, __init__.py)

- **v3.4.0**: **CRITICAL BUG FIX** - Synaptic input timing corrected, weight variance scaling
  - Fixed synaptic current to use previous timestep spikes
  - Added weight variance scaling 1/(N*p) to prevent synchronization

- **v3.3.0**: Logarithmic input rate sampling (geomspace), sequential pipeline automation

- **v3.2.0**: Poisson validation, fixed RNG bugs (time-step, rate), transient removal

- **v3.1.0**: Enhanced connectivity (10), optimized coincidence, randomized jobs

- **v3.0.0**: Split architecture (spontaneous + stability)

- **v2.1.0**: Multi-bin dimensionality, pattern stability  

- **v2.0.0**: Random structure with synaptic mode comparison

- **v1.0.0**: Fixed topology framework

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{spiking_rnn_heterogeneity_v350,
  title = {Spiking RNN Heterogeneity Framework v3.5.0},
  author = {Your Name},
  year = {2025},
  version = {3.5.0},
  url = {https://github.com/yourusername/spiking-rnn-heterogeneity}
}
```

---

**Key Innovation v3.5.0**: Revolutionary **full-simulation stability analysis** with **Shannon entropy** (pattern diversity), **settling time detection** (baseline recovery), and **cleaner metrics** (removed Hamming slopes, stable periods). Comprehensive framework for analyzing spontaneous activity and network stability in heterogeneous spiking networks with Poisson validation and information-theoretic measures.