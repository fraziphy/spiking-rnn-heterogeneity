# Spiking RNN Heterogeneity Framework v4.0.0

A comprehensive framework for studying **spontaneous activity** and **network stability** in heterogeneous spiking recurrent neural networks with **pulse/filter synapses**, **three static input modes**, and **corrected synaptic filtering architecture**.

## Major Updates in v4.0.0

### 🔧 Architecture Fix: Corrected Synaptic Filtering
**Problem Fixed**: Input classes were applying exponential filtering, then synapses applied it again (double filtering!)

**New Architecture**:
- **Input classes generate events only** (no filtering)
- **Synapse class applies filtering** (pulse or filter mode)
- **Single, consistent filtering path** for all inputs

```python
# Before (WRONG):
Poisson → [exp decay in input] → [exp decay in synapse] → neuron  # Double filtering!

# After (CORRECT):
Poisson → Synapse [exp decay once] → neuron  # Single filtering
```

### 🔄 New Terminology: Pulse vs Filter Synapses
- **OLD**: "immediate" synapses → **NEW**: "pulse" synapses
- **OLD**: "dynamic" synapses → **NEW**: "filter" synapses

Clearer, more descriptive terminology for synaptic dynamics.

### 🎯 Three Static Input Modes

Control how background Poisson input is delivered to neurons:

1. **`independent`** (default): Each neuron receives independent Poisson spikes
   - Different random spike trains per neuron
   - Maximum variability across population

2. **`common_stochastic`**: All neurons receive identical Poisson spike trains
   - Same stochastic process broadcast to all neurons
   - Trial-dependent (varies with RNG seed)
   - Tests network response to common fluctuations

3. **`common_tonic`**: Deterministic constant input (zero variance)
   - Expected value: `input_strength × spike_probability`
   - No stochasticity, pure DC drive
   - **Pulse synapses**: Constant input each timestep
   - **Filter synapses**: Builds up to steady-state via temporal integration

### 📊 New Stability Measures

**LZ Column-Wise Complexity**:
- Sorts neurons by post-perturbation activity
- Flattens matrix column-by-column (reads neuron-by-neuron temporally)
- Captures temporal structure in activity-sorted neurons

**Coincidence at 0.1ms Resolution**:
- Now measures: `delta = [0.1ms, 2ms, 5ms]`
- Ultra-fine temporal precision for spike synchrony
- Kistler and Gamma coincidence at all three windows

## Key Architecture Features

### Corrected Synapse Model
- **`Synapse` class** (renamed from `ExponentialSynapses`): Handles ALL synaptic filtering
- **Three synapse instances per network**:
  - `recurrent_synapses`: Neuron → neuron connections
  - `static_input_synapses`: Static Poisson → neurons
  - `dynamic_input_synapses`: Dynamic Poisson → neurons

### Split Experiments Design
- **Spontaneous Activity Analysis**: Firing rates, dimensionality (6 bin sizes), Poisson validation
- **Network Stability Analysis**: Full-simulation analysis, Shannon entropy, settling time, LZ complexity
- **Input mode control**: Switch between independent, common_stochastic, common_tonic
- **Synaptic mode control**: Switch between pulse and filter dynamics

### Mean-Centered Heterogeneity
- **Direct heterogeneity values**: `v_th_std` and `g_std` (0.01-1.0 range)
- **Exact mean preservation**: -55mV spike thresholds, 0 synaptic weights
- **Distribution flexibility**: Normal and uniform threshold distributions

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── rng_utils.py               # Parameter-dependent RNG
│   ├── lif_neuron.py              # Mean-centered LIF neurons
│   ├── synaptic_model.py          # Synapse class + input generators (UPDATED v4.0.0)
│   └── spiking_network.py         # Complete RNN (UPDATED v4.0.0)
├── analysis/                      # Split analysis modules  
│   ├── spontaneous_analysis.py    # Firing + dimensionality + Poisson
│   └── stability_analysis.py      # Shannon + LZ + settling + coincidence (UPDATED v4.0.0)
├── experiments/                   # Split experiment coordination
│   ├── spontaneous_experiment.py  # Duration + Poisson + input modes (UPDATED v4.0.0)
│   └── stability_experiment.py    # Full-simulation stability + input modes (UPDATED v4.0.0)
├── runners/                       # Execution scripts
│   ├── mpi_spontaneous_runner.py  # MPI runner (UPDATED v4.0.0)
│   ├── mpi_stability_runner.py    # MPI runner (UPDATED v4.0.0)
│   ├── run_spontaneous_experiment.sh    # Shell script (UPDATED v4.0.0)
│   └── run_stability_experiment.sh      # Shell script (UPDATED v4.0.0)
├── tests/                         # Testing framework
│   ├── test_installation.py       # Installation tests (UPDATED v4.0.0)
│   └── test_comprehensive_structure.py  # Structure tests (UPDATED v4.0.0)
└── results/data/                  # Experiment outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install numpy scipy mpi4py psutil matplotlib

# Install MPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-dev

# Test installation with new features
python tests/test_installation.py
python tests/test_comprehensive_structure.py
```

### 2. Run Experiments with New Features

**Spontaneous Activity with Different Input Modes:**
```bash
# Independent input (default)
./runners/run_spontaneous_experiment.sh \
    --duration 5 \
    --synaptic_mode filter \
    --static_input_mode independent \
    --v_th_distribution normal \
    --session_ids "1 2 3"

# Common stochastic input
./runners/run_spontaneous_experiment.sh \
    --duration 5 \
    --synaptic_mode filter \
    --static_input_mode common_stochastic \
    --v_th_distribution normal \
    --session_ids "1 2 3"

# Common tonic input
./runners/run_spontaneous_experiment.sh \
    --duration 5 \
    --synaptic_mode filter \
    --static_input_mode common_tonic \
    --v_th_distribution normal \
    --session_ids "1 2 3"
```

**Test Pulse vs Filter Synapses:**
```bash
# Pulse synapses with tonic input
./runners/run_stability_experiment.sh \
    --synaptic_mode pulse \
    --static_input_mode common_tonic \
    --v_th_distribution normal \
    --session_ids "1 2"

# Filter synapses with tonic input
./runners/run_stability_experiment.sh \
    --synaptic_mode filter \
    --static_input_mode common_tonic \
    --v_th_distribution normal \
    --session_ids "1 2"
```

### 3. New Command-Line Parameters

**Both experiments now accept:**
- `--synaptic_mode`: `pulse` or `filter` (default: `filter`)
- `--static_input_mode`: `independent`, `common_stochastic`, or `common_tonic` (default: `independent`)
- `--v_th_distribution`: `normal` or `uniform` (default: `normal`)

**Note**: Files now include ALL parameters in the name:
```
spontaneous_session_1_filter_independent_normal_5.0s.pkl
stability_session_1_pulse_common_tonic_uniform.pkl
```

## Scientific Innovation

### Corrected Synaptic Filtering (v4.0.0)

**The Problem**:
Previous versions had input classes (`StaticPoissonInput`, `DynamicPoissonInput`) applying exponential synaptic filtering internally, then the network's synapses applied filtering again. This resulted in:
- Double filtering of input currents
- Inconsistent dynamics between input and recurrent pathways
- Confusion about pulse vs filter behavior

**The Solution**:
```python
# Input classes now generate events only
class StaticPoissonInput:
    def generate_events(...) -> np.ndarray:
        # Returns raw spike events or tonic values
        # NO filtering applied here
        
# Synapse class applies filtering
class Synapse:
    def apply_to_input(self, events: np.ndarray) -> np.ndarray:
        if self.synaptic_mode == "filter":
            self.current *= exp(-dt/tau)  # Decay
            self.current += events         # Add new
        elif self.synaptic_mode == "pulse":
            self.current = events          # Replace
```

**Impact**:
- ✅ Single, consistent filtering path
- ✅ True pulse synapses (no temporal integration)
- ✅ True filter synapses (exponential accumulation)
- ✅ Correct comparison between pulse and filter modes

### Static Input Modes (v4.0.0)

**Three modes to probe network computation:**

**1. Independent Stochastic**
```python
# Each neuron: independent Poisson process
for neuron in range(n_neurons):
    if random() < spike_prob:
        events[neuron] = input_strength
```
- **Use case**: Natural variability, asynchronous input
- **Network sees**: Independent fluctuations per neuron

**2. Common Stochastic**
```python
# All neurons: same Poisson process
single_spike = random() < spike_prob
if single_spike:
    events[:] = input_strength  # Broadcast to all
```
- **Use case**: Common drive, synchronous fluctuations
- **Network sees**: Coordinated input events
- **Trial-dependent**: Different pattern each trial

**3. Common Tonic**
```python
# All neurons: deterministic expected value
events[:] = input_strength * spike_prob
```
- **Use case**: Zero-variance drive, pure DC input
- **Network sees**: Constant, predictable input
- **With filter synapses**: Builds to steady-state (~50× input due to integration)
- **With pulse synapses**: Stays constant each timestep

### Pulse vs Filter with Tonic Input

**Critical insight**: Tonic input reveals synapse dynamics clearly.

**Pulse Synapses + Tonic**:
```
Input: 1.0 × 0.05 = 0.05 every timestep
Synapse: current = 0.05 (no accumulation)
Result: Constant DC of 0.05
```

**Filter Synapses + Tonic**:
```
Input: 1.0 × 0.05 = 0.05 every timestep
Synapse: current *= exp(-dt/tau); current += 0.05
Result: Builds to steady-state ≈ 2.5 (50× higher!)
```

This **50× difference** comes from temporal integration in filter synapses. The steady-state is:
```
steady_state = input / (1 - exp(-dt/tau))
             = 0.05 / (1 - exp(-0.1/5.0))
             ≈ 2.5
```

### New Stability Measures (v4.0.0)

**LZ Column-Wise**:
```python
# Sort neurons by activity level
activity = spike_diff_matrix.sum(axis=1)
sorted_indices = np.argsort(activity)
matrix_sorted = spike_diff_matrix[sorted_indices, :]

# Flatten column-by-column (read each neuron's temporal pattern)
lz_column_wise = lempel_ziv_complexity(matrix_sorted.flatten(order='F'))
```
- Captures temporal structure when neurons ordered by activity
- Different from spatial LZ (row-by-row reading)

**Coincidence at 0.1ms**:
```python
# Now computed at three timescales
delta_values = [0.1, 2.0, 5.0]  # milliseconds

# Ultra-fine temporal resolution
kistler_0.1ms, gamma_0.1ms  # Sub-millisecond synchrony
kistler_2.0ms, gamma_2.0ms  # Standard precision
kistler_5.0ms, gamma_5.0ms  # Coarse precision
```

## Data Analysis

### Loading Results with New Features

```python
import pickle

# Load results - note new filename format!
filename = 'results/data/stability_session_1_filter_common_tonic_normal.pkl'
with open(filename, 'rb') as f:
    results = pickle.load(f)

for result in results:
    # Parameters now include input mode
    synaptic_mode = result['synaptic_mode']  # 'pulse' or 'filter'
    static_input_mode = result['static_input_mode']  # 'independent', 'common_stochastic', 'common_tonic'
    v_th_distribution = result['v_th_distribution']  # 'normal' or 'uniform'
    
    # New stability measures
    lz_column_wise = result['lz_column_wise_mean']
    kistler_01ms = result['kistler_delta_0.1ms_mean']
    
    # Existing measures
    lz_spatial = result['lz_spatial_patterns_mean']
    shannon_symbols = result['shannon_entropy_symbols_mean']
    settling_time = result['settling_time_ms_mean']
    
    print(f"Mode: {synaptic_mode}, Input: {static_input_mode}")
    print(f"LZ spatial: {lz_spatial:.1f}, LZ column: {lz_column_wise:.1f}")
    print(f"Kistler 0.1ms: {kistler_01ms:.3f}, Settling: {settling_time:.1f}ms")
```

## System Requirements

- **Python**: 3.8+
- **CPU**: Multi-core (32+ cores recommended)
- **Memory**: 16GB+ (spontaneous), 32GB+ (stability)
- **Storage**: 5GB+ per experiment

## Version History

- **v4.0.0**: **ARCHITECTURE REVOLUTION** - Corrected synaptic filtering, pulse/filter terminology, three static input modes
  - Fixed double filtering bug (inputs + synapses both filtered)
  - Renamed: `ExponentialSynapses` → `Synapse`
  - New terminology: pulse/filter instead of immediate/dynamic
  - Three static input modes: independent, common_stochastic, common_tonic
  - New measures: lz_column_wise, coincidence at 0.1ms
  - Updated ALL files: src, analysis, experiments, runners, tests

- **v3.5.1**: Critical bug fixes - Shannon entropy, settling time
- **v3.5.0**: Full-simulation stability, Shannon entropy, settling time
- **v3.4.0**: Synaptic timing fix, weight variance scaling
- **v3.3.0**: Logarithmic input rates, pipeline automation
- **v3.2.0**: Poisson validation, RNG bug fixes
- **v3.0.0**: Split architecture (spontaneous + stability)

## Citation

```bibtex
@software{spiking_rnn_heterogeneity_v400,
  title = {Spiking RNN Heterogeneity Framework v4.0.0},
  author = {Your Name},
  year = {2025},
  version = {4.0.0},
  url = {https://github.com/yourusername/spiking-rnn-heterogeneity}
}
```

---

**Key Innovation v4.0.0**: Fixed synaptic filtering architecture, introduced pulse/filter terminology and three static input modes (independent, common_stochastic, common_tonic) for comprehensive network analysis.