# Spiking RNN Heterogeneity Framework - Split Experiments Architecture v3.2.0

A comprehensive framework for studying **spontaneous activity** and **network stability** in heterogeneous spiking recurrent neural networks with **Poisson process analysis**, **rate-dependent random seeds**, **transient removal**, and **optimized coincidence analysis**.

## Key Architecture Features

### Split Experiments Design
- **Spontaneous Activity Analysis**: Firing rates, dimensionality (6 bin sizes), silent neurons, **Poisson process tests**
- **Network Stability Analysis**: Perturbation response, LZ spatial complexity, coincidence measures
- **Enhanced static Poisson connectivity**: Strength 10 (up from 1)
- **Rate-dependent randomization**: Different random seeds for different input rates
- **Transient removal**: Automatic 25ms transient period exclusion

### Random Structure with Parameter Dependence
- **Network topology depends on `session_id` AND parameter values**
- Different connectivity patterns for each (session, v_th_std, g_std) combination
- **Time-step dependent Poisson processes**: Fixed RNG bug for independent input
- **Rate-dependent seeds**: Each input rate gets unique random sequences

### Poisson Process Analysis
- **Inter-Spike Interval (ISI) tests**: Exponential distribution analysis
- **Spike count tests**: Poisson distribution in time bins
- **Coefficient of Variation**: CV ≈ 1 for Poisson processes
- **Fano Factor**: Variance/mean ratio ≈ 1 for Poisson
- **Population statistics**: Fraction of neurons showing Poisson-like behavior

### Mean-Centered Heterogeneity
- **Direct heterogeneity values**: `v_th_std` and `g_std` (0.01-1.0 range)
- **Exact mean preservation**: -55mV spike thresholds, 0 synaptic weights
- **Distribution flexibility**: Normal and uniform threshold distributions

### Fair Synaptic Mode Comparison
- **Dynamic synapses**: Exponential decay with τ_syn = 5ms (realistic)
- **Immediate synapses**: Instantaneous coupling (like previous studies)
- **Impact normalization**: Immediate weights scaled by τ_syn/dt for fair comparison

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter and rate-dependent RNG
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Enhanced connectivity strength (25)
│   └── spiking_network.py         # Complete RNN with mode selection
├── analysis/                      # Split analysis modules  
│   ├── spontaneous_analysis.py    # Firing rates + dimensionality + Poisson tests
│   └── stability_analysis.py      # LZ spatial + optimized coincidence
├── experiments/                   # Split experiment coordination
│   ├── spontaneous_experiment.py  # Duration parameter + Poisson analysis
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

## Enhanced Analysis Features

### Spontaneous Activity Analysis
**Duration-based simulation with Poisson process validation:**
- **Firing rate analysis**: Mean, std, min, max firing rates
- **Silent neuron analysis**: Percentage of silent/active neurons
- **6-bin dimensionality analysis**: 0.1ms, 2ms, 5ms, 20ms, 50ms, 100ms temporal resolutions
- **Participation ratio**: Network-wide activity distribution
- **NEW: Poisson process tests**: ISI exponential, count Poisson, CV and Fano Factor analysis
- **NEW: Transient removal**: Automatic 25ms exclusion for steady-state analysis
- **10 trials per combination**: Efficient for steady-state measures

### Network Stability Analysis  
**Perturbation response with optimized computation:**
- **LZ spatial patterns**: Complexity of spatial pattern sequences only
- **Optimized coincidence**: Single-loop calculation for Kistler + Gamma measures
- **Hamming distance slopes**: Perturbation divergence analysis
- **Pattern stability detection**: Identifies repeating spatiotemporal patterns
- **100 trials per combination**: Comprehensive sampling for dynamics

### Poisson Process Validation
```python
# New Poisson analysis results
poisson_stats = {
    'mean_cv_isi': 1.02,           # Should be ≈1 for Poisson
    'mean_fano_factor': 0.98,      # Should be ≈1 for Poisson  
    'poisson_isi_fraction': 0.85,  # 85% of neurons show exponential ISI
    'poisson_count_fraction': 0.79 # 79% show Poisson spike counts
}
```

### Rate-Dependent Randomization
```python
# Fixed: Different rates now get different random sequences
rng = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson', 
              time_step, rate=100.5)  # Unique for rate 100.5 Hz
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

**Spontaneous Activity Analysis (with Poisson validation):**
```bash
chmod +x runners/run_spontaneous_experiment.sh

# Quick 2-second spontaneous activity test with Poisson analysis
./runners/run_spontaneous_experiment.sh --duration 2 --session_ids "1" --n_v_th 2 --n_g 2 --nproc 4

# Long 10-second analysis for detailed Poisson statistics  
./runners/run_spontaneous_experiment.sh --duration 10 --session_ids "1 2 3" --n_v_th 10 --n_g 10
```

**Network Stability Analysis (perturbation response):**
```bash
chmod +x runners/run_stability_experiment.sh

# Quick stability test
./runners/run_stability_experiment.sh --session_ids "1" --n_v_th 2 --n_g 2 --nproc 4

# Full stability study with session averaging
./runners/run_stability_experiment.sh --session_ids "1 2 3 4 5" --n_v_th 20 --n_g 20
```

### 3. Long-Running Experiments with nohup

**Run experiments in background and logout safely:**
```bash
# Start spontaneous activity analysis
nohup bash runners/run_spontaneous_experiment.sh --session_ids "1" --n_input_rates 10 > spontaneous.log 2>&1 &

# Check if first job is running, then start stability analysis
nohup bash runners/run_stability_experiment.sh --session_ids "1" --n_input_rates 10 > stability.log 2>&1 &

# Monitor progress
tail -f spontaneous.log
tail -f stability.log

# Safe to logout - jobs continue running
exit
```

**Return next day to check results:**
```bash
# Check if jobs are still running
ps aux | grep run_spontaneous
ps aux | grep run_stability

# Check progress
tail -n 50 spontaneous.log
tail -n 50 stability.log

# Check results
ls -la results/data/
```

### 4. Monitor Experiment Progress

**Spontaneous Activity Progress (with Poisson metrics):**
```bash
tail -f spontaneous.log

# Shows:
# Mean firing rate: 15.2±2.3 Hz
# Silent neurons: 23.5±4.1%
# Dimensionality (5ms): 12.3±1.8
# CV ISI: 1.02±0.15 (NEW)
# Poisson-like: 85.3% (NEW)
# Total spikes: 8,450±1,200
```

**Network Stability Progress:**
```bash  
tail -f stability.log

# Shows:
# LZ (spatial): 23.4±2.8  
# Hamming slope: 0.0034±0.0012
# Kistler (2ms): 0.34±0.08
# Stable patterns: 0.15±0.05
```

## Split Experiment Parameters

### Spontaneous Activity Parameters
- `--duration`: Simulation duration in seconds (auto-converts to milliseconds)
- `--session_ids`: Sessions for network averaging (e.g., "1 2 3")
- **Automatic transient removal**: First 25ms excluded from analysis
- **6 dimensionality bin sizes**: Automatic analysis at all temporal resolutions
- **Poisson process validation**: Automatic CV, Fano Factor, and distribution tests
- **10 trials per combination**: Efficient for steady-state measures

### Network Stability Parameters  
- `--session_ids`: Sessions for perturbation averaging (e.g., "1 2 3")
- **100 trials per combination**: Comprehensive perturbation sampling
- **Optimized coincidence**: Single-loop Kistler + Gamma calculation
- **Rate-dependent seeds**: Each input rate gets unique random patterns

### Common Parameters
- `--n_v_th/--n_g`: Heterogeneity grid sizes (default: 10x10)
- `--synaptic_mode`: "immediate" or "dynamic" synaptic coupling
- `--input_rate_min/max`: Background input range (enhanced: 50-1000 Hz)
- `--n_input_rates`: Number of input rate points (NEW: supports decimals)
- `--nproc`: MPI processes (default: 50)

## Scientific Innovation

### Poisson Process Validation
**Problem**: Networks receive Poisson input, but do neurons output Poisson spike trains?

**Solution**: Comprehensive statistical tests validate if spike trains maintain Poisson properties:

```python
# Test if neurons preserve Poisson statistics
isi_test = test_exponential_isi_distribution(spike_train)
count_test = test_count_poisson_distribution(spike_train, duration)

print(f"CV ISI: {cv_isi:.3f} (expect 1.0 for Poisson)")
print(f"Fano Factor: {fano:.3f} (expect 1.0 for Poisson)")
print(f"ISI exponential: {isi_test['is_exponential']}")
print(f"Count Poisson: {count_test['is_poisson']}")
```

### Rate-Dependent Randomization
**Problem**: Same random sequence used for all input rates, creating artificial correlations.

**Solution**: Each input rate gets unique random seed:
```python
# OLD: Same random sequence for 100Hz and 200Hz
rng = get_rng(session_id, v_th_std, g_std, trial_id, component)

# NEW: Different sequences for different rates  
rng = get_rng(session_id, v_th_std, g_std, trial_id, component, time_step, rate)
```

### Transient Period Removal
**Problem**: Network initialization affects early dynamics.

**Solution**: Automatic 25ms transient removal:
```python
# All spontaneous analysis now excludes first 25ms
steady_spikes = [(t - 25.0, n) for t, n in spikes if t >= 25.0]
```

### Fixed Time-Step RNG Bug
**Problem**: Poisson processes generated identical random numbers at each time step.

**Solution**: Time-step dependent seeding:
```python
# Each time step gets different random numbers
rng = get_rng(session_id, v_th_std, g_std, trial_id, component, time_step)
```

## Data Analysis

### Loading Enhanced Results

**Spontaneous Activity with Poisson Analysis:**
```python
import pickle

with open('results/data/spontaneous_session_1_dynamic.pkl', 'rb') as f:
    results = pickle.load(f)

for result in results:
    # Standard measures
    firing_rate = result['mean_firing_rate_mean']
    silent_pct = result['percent_silent_mean']
    dimensionality = result['effective_dimensionality_bin_5.0ms_mean']
    
    # NEW: Poisson process measures
    cv_isi = result['mean_cv_isi_mean']
    fano_factor = result['mean_fano_factor_mean'] 
    poisson_isi_fraction = result['poisson_isi_fraction_mean']
    poisson_count_fraction = result['poisson_count_fraction_mean']
    
    # NEW: Transient information
    total_spikes = result['total_spikes_mean']
    steady_spikes = result['steady_state_spikes_mean']
    transient_time = result['transient_time']
    
    print(f"Firing: {firing_rate:.1f}Hz, Silent: {silent_pct:.1f}%")
    print(f"CV ISI: {cv_isi:.3f}, Fano: {fano_factor:.3f}")
    print(f"Poisson-like: ISI {poisson_isi_fraction:.1%}, Count {poisson_count_fraction:.1%}")
    print(f"Spikes: {steady_spikes:.0f}/{total_spikes:.0f} (after {transient_time}ms)")
```

### Background Job Management

**Start experiments in background:**
```bash
# Method 1: Simple nohup
nohup bash runners/run_spontaneous_experiment.sh --session_ids "1" > spont.log 2>&1 &
echo $! > spont.pid  # Save process ID

# Method 2: Screen/tmux session (recommended)
screen -S experiments
bash runners/run_spontaneous_experiment.sh --session_ids "1"
# Ctrl+A, D to detach

# Return later and reattach
screen -r experiments
```

**Monitor running jobs:**
```bash
# Check if jobs are running
jobs
ps aux | grep run_spontaneous

# Monitor progress
tail -f spont.log
watch -n 30 "tail -n 10 spont.log"

# Check system resources
htop
free -h
df -h results/
```

## System Requirements

### Computational Resources
- **CPU**: Multi-core system (32+ cores recommended)
- **Memory**: 16GB+ (spontaneous), 32GB+ (stability with 100 trials)
- **Storage**: 5GB+ per major experiment
- **Time**: 
  - Spontaneous: ~30s per combination per session
  - Stability: ~2min per combination per session

### Background Job Considerations
- **Process management**: Use `screen`, `tmux`, or `nohup` for long jobs
- **Storage monitoring**: Check disk space regularly for large experiments
- **Network stability**: Ensure reliable connection for remote systems
- **Resource limits**: Monitor CPU and memory usage

## Enhanced Troubleshooting

### Poisson Analysis Issues
1. **Chi-square test errors**: Handled with robust error checking and normalization
2. **Insufficient spikes**: Minimum 10 spikes required for reliable tests
3. **CV and Fano interpretation**: Values ≈1 indicate Poisson-like behavior

### Rate-Dependent RNG Issues  
1. **Floating point precision**: Rates converted to integers with 6 decimal precision
2. **Large rate values**: Modular arithmetic prevents integer overflow
3. **Seed uniqueness**: Each rate gets unique random sequence

### Background Job Management
1. **Job monitoring**: Use `ps`, `jobs`, and log files to track progress
2. **Resource exhaustion**: Monitor memory and storage usage
3. **Network disconnection**: Jobs continue with `nohup` or `screen`
4. **Result verification**: Check output files after completion

## Version History

- **v3.2.0**: Poisson process analysis, rate-dependent RNG, transient removal, fixed time-step bug
- **v3.1.0**: Enhanced connectivity (25), optimized coincidence, randomized jobs
- **v3.0.0**: Split architecture (spontaneous + stability)
- **v2.1.0**: Multi-bin dimensionality, pattern stability  
- **v2.0.0**: Random structure with synaptic mode comparison
- **v1.0.0**: Fixed topology framework

---

**Key Innovation**: Enhanced split experiments framework with **Poisson process validation**, **rate-dependent randomization**, **automatic transient removal**, and **fixed time-step RNG** for rigorous analysis of spontaneous activity and network stability in heterogeneous spiking networks.