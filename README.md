# Spiking RNN Heterogeneity Study

**Version:** 6.1.0  
**Status:** Production-ready infrastructure with sweep/ reorganization  
**Next:** v6.2.0 will add partitioned HD connection mode

---

## Project Overview

This project investigates how heterogeneity in spiking recurrent neural networks affects spontaneous dynamics, stability, encoding capacity, and task performance. The framework implements a biologically-inspired leaky integrate-and-fire (LIF) network with direct parameter heterogeneity and comprehensive reproducibility guarantees.

### Six Experiment Types

1. **Spontaneous Activity** - Baseline dynamics without external input
2. **Network Stability** - Response to perturbations  
3. **Encoding Capacity** - High-dimensional input representation
4. **Categorical Classification** - Pattern recognition (reservoir computing)
5. **Temporal Transformation** - Temporal pattern mapping (reservoir computing)
6. **Auto-encoding** - Input reconstruction with dimensionality analysis

---

## Project Structure (v6.1.0)

```
spiking-rnn-heterogeneity/
├── src/                          # Core neural network modules
│   ├── lif_neuron.py            # LIF neuron model
│   ├── synaptic_model.py        # Synapses + input generators
│   ├── spiking_network.py       # Network integration
│   ├── hd_input.py              # High-dimensional input generation
│   └── rng_utils.py             # Hierarchical RNG system
│
├── experiments/                  # Experiment classes
│   ├── base_experiment.py       # Shared utilities
│   ├── spontaneous_experiment.py
│   ├── stability_experiment.py
│   ├── encoding_experiment.py
│   ├── task_performance_experiment.py  # Categorical/temporal/autoencoding
│   └── experiment_utils.py      # Readout training & evaluation
│
├── analysis/                     # Analysis utilities
│   ├── common_utils.py          # Shared analysis functions
│   ├── spontaneous_analysis.py
│   ├── stability_analysis.py
│   ├── encoding_analysis.py
│   └── statistics_utils.py
│
├── runners/                      # Single-job MPI runners
│   ├── mpi_spontaneous_runner.py
│   ├── mpi_stability_runner.py
│   ├── mpi_encoding_runner.py
│   ├── mpi_task_runner.py       # Categorical + temporal
│   ├── mpi_autoencoding_runner.py
│   ├── mpi_utils.py             # Shared MPI utilities
│   └── run_*_experiment.sh      # Individual experiment launchers
│
├── sweep/                        # NEW: Batch sweep orchestration
│   ├── generate_jobs.py         # Parameter grid generation
│   ├── run_sweep_engine.sh      # Common sweep execution with resume
│   ├── run_sweep_spontaneous.sh
│   ├── run_sweep_stability.sh
│   ├── run_sweep_categorical.sh
│   ├── run_sweep_transformation.sh  # Temporal task
│   ├── run_sweep_autoencoding.sh
│   ├── check_system_thresholds.py   # System monitoring config
│   └── logs_*/                  # Sweep execution logs
│
├── results/                      # Experimental results
│   ├── spontaneous_sweep/data/
│   ├── stability_sweep/data/
│   ├── encoding_sweep/data/
│   ├── categorical_sweep/data/
│   ├── temporal_sweep/data/
│   └── autoencoding_sweep/data/
│
├── hd_signals/                   # Cached HD input patterns
│   ├── categorical_sweep/
│   ├── temporal_sweep/
│   └── autoencoding_sweep/
│
├── tests/                        # Comprehensive test suite
│   ├── test_installation.py
│   ├── test_comprehensive_structure.py
│   ├── test_encoding_implementation.py
│   └── test_task_performance.py
│
├── plots/                        # Figure generation scripts
│   ├── plot_main_figure_1.py
│   ├── plot_main_figure_2.py
│   ├── plot_supplementary_figure_1.py
│   └── plot_supplementary_figure_2.py
│
├── data_curation/                # Data processing for analysis
│   ├── data_curation_network_dynamics.py
│   └── data_curation_network_encoding.py
│
├── setup.py                      # Package installation
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Key Features (v6.1.0)

### HD Input Connectivity
**Current Implementation:** Overlapping mode
- Each HD channel connects to random 30% of neurons
- Natural overlap between channels (~9%)
- Used in all encoding and task experiments

**Coming in v6.2.0:** Partitioned mode option
- Equal division of neurons across channels
- Zero overlap (each neuron receives exactly one feature)
- Removes confound between dimensionality and connectivity

### Infrastructure Improvements
✅ **Sweep directory reorganization** - Better separation of concerns  
✅ **Resume support** - Handle system reboots gracefully  
✅ **System health monitoring** - Configurable thresholds  
✅ **Enhanced logging** - Per-experiment log directories  
✅ **GNU parallel integration** - Efficient job distribution  

### Reproducibility Guarantees
- **Network structure** ⊥ (session_id, v_th_std, g_std)
- **Trial dynamics** ⊥ trial_id
- **HD patterns** ⊥ (session_id, hd_dim, embed_dim, pattern_id)
- **Trial noise** ⊥ all parameters
- Deterministic RNG with hierarchical seeding

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd spiking-rnn-heterogeneity

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run tests
pytest tests/
```

### Dependencies
- Python 3.8+
- NumPy, SciPy, scikit-learn
- MPI4Py (for parallel execution)
- matplotlib, seaborn (for plotting)

---

## Quick Start

### Running Single Experiments

```bash
# Spontaneous activity
cd runners
./run_spontaneous_experiment.sh

# Network stability
./run_stability_experiment.sh

# Encoding capacity
./run_encoding_experiment.sh

# Categorical task
./run_categorical_task.sh

# Temporal task
./run_temporal_task.sh

# Auto-encoding
./run_autoencoding_task.sh
```

### Running Parameter Sweeps

```bash
cd sweep

# Edit parameters in sweep script (e.g., run_sweep_categorical.sh)
V_TH_VALUES=(0 0.5 1.0)
G_VALUES=(0.5 1.0 1.5)
# ... etc

# Run sweep
./run_sweep_categorical.sh

# Resume after interruption (automatic)
./run_sweep_categorical.sh  # Detects existing joblog and resumes
```

---

## Experiment Details

### 1. Spontaneous Activity
**Purpose:** Baseline network dynamics  
**Duration:** 800ms (500ms transient + 300ms analysis)  
**Measures:** Firing rates, silence, ISI statistics, dimensionality

### 2. Network Stability  
**Purpose:** Perturbation response  
**Protocol:** 500ms baseline → auxiliary spike → 300ms recovery  
**Measures:** Settling time, coincidence factors, entropy

### 3. Encoding Capacity
**Purpose:** HD input representation  
**Input:** d-dimensional signal embedded in k-dimensional space  
**Duration:** 500ms transient + 300ms encoding  
**Measures:** Decoding accuracy (RMSE, R², correlation), dimensionality

### 4. Categorical Classification
**Purpose:** Pattern recognition (reservoir computing)  
**Input:** n patterns (default: 4)  
**Output:** Class labels (1-hot encoding)  
**Measures:** Accuracy, F1-score, confusion matrix

### 5. Temporal Transformation
**Purpose:** Temporal pattern mapping (reservoir computing)  
**Input:** d_in-dimensional temporal patterns  
**Output:** d_out-dimensional target trajectories  
**Measures:** RMSE, R², correlation per dimension

### 6. Auto-encoding
**Purpose:** Input reconstruction with dimensionality analysis  
**Input:** d-dimensional patterns  
**Output:** Same d-dimensional patterns (reconstruction)  
**Measures:** RMSE, R², dimensionality at multiple timescales (2ms/10ms/20ms)

---

## Parameter Sweeps

All sweeps use `sweep/generate_jobs.py` and `sweep/run_sweep_engine.sh` for consistent execution.

### Common Parameters
- `v_th_std`: Threshold heterogeneity (e.g., 0, 0.5, 1.0)
- `g_std`: Weight heterogeneity (e.g., 0.5, 1.0, 1.5)
- `static_input_rate`: Background input rate (e.g., 20, 30, 40 Hz)

### Task-Specific Parameters
- `embed_dim_input`: Input embedding dimension (1-5)
- `embed_dim_output`: Output embedding dimension (1-5, temporal only)
- `n_patterns`: Number of patterns (categorical: 4, temporal: 2, autoencoding: 1)

### Sweep Features
- **Automatic job generation** from parameter grids
- **Resume support** via joblog tracking
- **Progress milestones** (25%, 50%, 75%, 100%)
- **System health monitoring** with configurable thresholds
- **Parallel execution** with GNU parallel

---

## Results Organization

### Directory Structure
```
results/
├── {experiment}_sweep/
│   └── data/
│       └── {experiment}_session_{id}_vth_{v}_g_{g}_*.pkl
```

### Result File Contents
Each `.pkl` file contains:
- **Parameters:** session_id, v_th_std, g_std, etc.
- **Network info:** connectivity stats, threshold distributions
- **Experiment metrics:** task-specific performance measures
- **Metadata:** computation time, number of trials

### HD Signal Caching
Deterministically generated patterns cached in:
```
hd_signals/
├── {experiment}_sweep/
│   └── hd_signal_session_{id}_hd_{d}_k_{k}_pattern_{p}.pkl
```

---

## System Configuration

### MPI Configuration
```bash
# Example: 20 cores for sweep
export OMP_NUM_THREADS=1
mpirun -n 20 python runners/mpi_task_runner.py ...
```

### Health Monitoring
Configure thresholds in `sweep/check_system_thresholds.py` or via environment:
```bash
export TEMP_THRESHOLD=100      # °C
export CPU_THRESHOLD=98        # %
export MEMORY_THRESHOLD=95     # %
```

### Resume Support
Sweeps automatically resume from last successful job:
```bash
# Run sweep
./sweep/run_sweep_categorical.sh

# System reboots...

# Resume automatically
./sweep/run_sweep_categorical.sh  # Picks up where it left off
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Specific test suites
pytest tests/test_installation.py           # Installation check
pytest tests/test_comprehensive_structure.py # Network structure
pytest tests/test_encoding_implementation.py # Encoding experiments
pytest tests/test_task_performance.py       # Task experiments
```

**Current Status:** 50+ tests passing ✅

---

## Version History

### v6.1.0 (Current - Infrastructure Consolidation)
- Reorganized sweep infrastructure (dedicated sweep/ directory)
- Enhanced run_sweep_engine.sh with resume support
- Added system health monitoring configuration
- Improved logging and job tracking
- All 6 experiment types stable and production-ready

### v6.0.0 (Task-Performance Experiments)
- Added categorical classification
- Added temporal transformation
- Added auto-encoding with dimensionality analysis
- Unified TaskPerformanceExperiment infrastructure
- Pattern-based HD input generation
- Fixed directory path duplication bug

### v5.1.0 (Encoding Capacity)
- Encoding experiment with HD inputs
- HD input generator with caching
- Decoding analysis infrastructure

### v5.0.0 (Network Stability)
- Stability experiment with perturbation analysis
- Settling time and coincidence factor metrics

### v4.0.0 (Spontaneous Activity)
- Spontaneous activity experiment
- Comprehensive firing rate and ISI analysis

---

## Roadmap

### v6.2.0 (Next - HD Connection Modes)
- [ ] Add partitioned HD connection mode
- [ ] Flag-based mode selection (--hd_connection_mode)
- [ ] Automatic result organization by mode
- [ ] Backward compatible (overlapping as default)
- [ ] Documentation and testing

### Future
- [ ] Additional task types (delay tasks, working memory)
- [ ] Enhanced visualization tools
- [ ] Performance optimization for large-scale sweeps
- [ ] Analysis pipelines for publication figures

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{spiking_rnn_heterogeneity_2025,
  title = {Spiking RNN Heterogeneity Study},
  author = {[Your Name]},
  year = {2025},
  version = {6.1.0},
  url = {[repository-url]}
}
```

---

## License

[Your License Here]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email]

---

## Acknowledgments

Built with:
- MPI4Py for parallel computing
- NumPy/SciPy for numerical computation
- scikit-learn for machine learning utilities
- GNU Parallel for job orchestration

---

**Last Updated:** November 2025  
**Maintainer:** [Your Name]  
**Status:** ✅ Production-ready, actively maintained