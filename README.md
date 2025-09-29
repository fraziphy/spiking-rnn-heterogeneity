# Spiking RNN Heterogeneity Framework - Split Experiments Architecture v3.3.0

A comprehensive framework for studying **spontaneous activity** and **network stability** in heterogeneous spiking recurrent neural networks with **Poisson process validation**, **rate-dependent randomization**, **transient removal**, and **automated pipeline execution**.

## Key Architecture Features

### Split Experiments Design
- **Spontaneous Activity Analysis**: Firing rates, dimensionality (6 bin sizes), silent neurons, **Poisson validation**
- **Network Stability Analysis**: Perturbation response, LZ spatial complexity, coincidence measures
- **Enhanced static Poisson connectivity**: Strength 10 (up from 1)
- **Optimized coincidence calculation**: Single loop for both Kistler and Gamma measures

### Poisson Process Validation (NEW in v3.2.0)
- **Statistical tests**: Kolmogorov-Smirnov (ISI exponential), Chi-square (count Poisson)
- **Coefficient of Variation (CV)**: Expected ~1.0 for Poisson processes
- **Fano Factor**: Variance/mean ratio, expected ~1.0 for Poisson
- **Population statistics**: Fraction of neurons showing Poisson-like behavior
- **Automatic transient removal**: First 25ms excluded from analysis

### Fixed RNG Bugs (v3.2.0)
- **Time-step dependent**: Each time step gets unique random numbers (was: identical at each step)
- **Rate-dependent**: Each input rate gets independent random sequences (was: same for all rates)
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
│   ├── rng_utils.py               # Rate & time-dependent RNG (FIXED)
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Enhanced connectivity + time-step RNG
│   └── spiking_network.py         # Complete RNN with time tracking
├── analysis/                      # Split analysis modules  
│   ├── spontaneous_analysis.py    # Firing + dimensionality + Poisson tests (NEW)
│   └── stability_analysis.py      # LZ spatial + optimized coincidence
├── experiments/                   # Split experiment coordination
│   ├── spontaneous_experiment.py  # Duration parameter + Poisson analysis (UPDATED)
│   └── stability_experiment.py    # Perturbation response + stability (UPDATED)
├── runners/                       # Execution scripts
│   ├── mpi_spontaneous_runner.py  # MPI spontaneous activity runner
│   ├── mpi_stability_runner.py    # MPI network stability runner
│   ├── run_spontaneous_experiment.sh    # Spontaneous activity script
│   └── run_stability_experiment.sh      # Network stability script
├── pipeline.sh                    # Sequential pipeline execution (NEW)
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation verification
│   └── test_comprehensive_structure.py  # Structure validation
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

**NEW in v3.2.0**: Automated sequential execution of both experiments.

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
./runners/run_spontaneous_experiment.sh --duration 2 --session_ids "1" --n_v_th 2--n_g 2 --nproc 4

# Full analysis with multiple sessions
./runners/run_spontaneous_experiment.sh --duration 5 --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20

# Background execution
nohup bash runners/run_spontaneous_experiment.sh --session_ids "1 2 3" --n_input_rates 10 > spontaneous.log 2>&1 &
disown
```

**Network Stability Analysis:**
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
- **Automatic transient removal**: First 25ms excluded (NEW)
- **6 dimensionality bin sizes**: 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms
- **Poisson validation**: CV, Fano Factor, KS/Chi-square tests (NEW)
- **10 trials per combination**: Efficient for steady-state measures

### Network Stability Parameters  
- `--session_ids`: Sessions for perturbation averaging (e.g., "1 2 3")
- **100 trials per combination**: Comprehensive perturbation sampling
- **Optimized coincidence**: Single-loop Kistler + Gamma calculation
- **Rate-dependent seeds**: Each input rate gets unique random patterns (FIXED)

### Common Parameters
- `--n_v_th/--n_g`: Heterogeneity grid sizes (default: 20x20)
- `--synaptic_mode`: "immediate" or "dynamic" synaptic coupling
- `--input_rate_min/max`: Background input range (default: 10-500 Hz)
- `--n_input_rates`: Number of input rate points (supports decimals)
- `--nproc`: MPI processes (default: 50)

## Scientific Innovation

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

### Fixed RNG Bugs (v3.2.0)

**Problem 1**: Time-step bug caused identical random numbers at every time step
```python
# OLD BUG: Same random sequence every timestep
rng = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson')
# Result: All neurons got correlated input patterns

# NEW FIX: Each timestep gets unique random numbers
rng = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson', time_step)
# Result: Proper independent Poisson processes
```

**Problem 2**: Rate-independent seeds caused same patterns for all input rates
```python
# OLD BUG: 100 Hz and 200 Hz used same random sequence
rng = get_rng(session_id, v_th_std, g_std, trial_id, component)

# NEW FIX: Each rate gets independent random sequence
rng = get_rng(session_id, v_th_std, g_std, trial_id, component, time_step, rate)
# Result: True independence across input rates
```

### Transient Removal (v3.2.0)

**Problem**: Network initialization affects early dynamics

**Solution**: Automatic 25ms transient exclusion
```python
# All spontaneous analysis now excludes first 25ms
steady_spikes = [(t - 25.0, n) for t, n in spikes if t >= 25.0]
steady_duration = duration - 25.0
# Analysis uses only steady-state data
```

### Enhanced Connectivity Analysis
**Static Poisson strength increased from 1.0 to 25.0:**
- More realistic background drive
- Better network activation across parameter ranges
- Clearer differentiation between heterogeneity effects

## Data Analysis

### Loading Results with Poisson Analysis

**Spontaneous Activity Results:**
```python
import pickle

# Load session-averaged results (if multiple sessions)
with open('results/data/spontaneous_averaged_dynamic_sessions_1_2_3.pkl', 'rb') as f:
    results = pickle.load(f)

# Or load single session
with open('results/data/spontaneous_session_1_dynamic.pkl', 'rb') as f:
    results = pickle.load(f)

for result in results:
    # Standard measures
    firing_rate = result['mean_firing_rate_mean']
    dimensionality = result['effective_dimensionality_bin_5.0ms_mean']
    
    # NEW: Poisson process measures
    cv_isi = result['mean_cv_isi_mean']
    fano_factor = result['mean_fano_factor_mean']
    poisson_fraction = result['poisson_isi_fraction_mean']
    
    # NEW: Transient information
    transient_time = result['transient_time']  # 25.0 ms
    steady_spikes = result['steady_state_spikes_mean']
    total_spikes = result['total_spikes_mean']
    
    print(f"Firing: {firing_rate:.1f}Hz, CV ISI: {cv_isi:.3f}, Fano: {fano_factor:.3f}")
    print(f"Poisson-like: {poisson_fraction:.1%}")
    print(f"Spikes: {steady_spikes:.0f}/{total_spikes:.0f} (after transient)")
```

**Network Stability Results:**
```python
with open('results/data/stability_session_1_dynamic.pkl', 'rb') as f:
    stability_results = pickle.load(f)

for result in stability_results:
    # Stability measures (optimized, no PCI)
    lz_spatial = result['lz_spatial_patterns_mean']
    hamming_slope = result['hamming_slope_mean']
    
    # Optimized coincidence measures
    kistler_2ms = result['kistler_delta_2ms_mean']
    gamma_2ms = result['gamma_window_2ms_mean']
    
    # Pattern stability
    stable_fraction = result['stable_pattern_fraction']
    
    print(f"LZ: {lz_spatial:.1f}, Hamming: {hamming_slope:.4f}")
    print(f"Kistler: {kistler_2ms:.3f}, Stable: {stable_fraction:.3f}")
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

- **v3.3.0**: Logarithmic input rate sampling (geomspace), sequential pipeline automation
- **v3.2.0**: Poisson validation, fixed RNG bugs (time-step, rate), transient removal, pipeline automation
- **v3.1.0**: Enhanced connectivity (25), optimized coincidence, randomized jobs
- **v3.0.0**: Split architecture (spontaneous + stability)
- **v2.1.0**: Multi-bin dimensionality, pattern stability  
- **v2.0.0**: Random structure with synaptic mode comparison
- **v1.0.0**: Fixed topology framework

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{spiking_rnn_heterogeneity_v330,
  title = {Spiking RNN Heterogeneity Framework v3.3.0},
  author = {Your Name},
  year = {2025},
  version = {3.3.0},
  url = {https://github.com/yourusername/spiking-rnn-heterogeneity}
}
```

---

**Key Innovation v3.2.0**: Comprehensive **Poisson process validation** with statistical tests (CV, Fano Factor, KS, Chi-square), **fixed critical RNG bugs** (time-step and rate dependencies), **automatic transient removal** (25ms), and **sequential pipeline automation** for robust analysis of spontaneous activity and network stability in heterogeneous spiking networks.