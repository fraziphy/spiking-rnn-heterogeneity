# Spiking RNN Heterogeneity Studies

A comprehensive framework for studying the effects of spike threshold heterogeneity and synaptic strength heterogeneity on network dynamics, coding capacity, and task performance in spiking recurrent neural networks (RNNs).

## Project Structure

```
spiking_rnn_heterogeneity/
├── src/                           # Core neural network modules
│   ├── __init__.py                # Package initialization
│   ├── rng_utils.py               # Hierarchical random number generation
│   ├── lif_neuron.py              # Leaky Integrate-and-Fire neuron model
│   ├── synaptic_model.py          # Synaptic connections and Poisson inputs
│   └── spiking_network.py         # Complete spiking RNN network class
├── analysis/                      # Analysis and measurement tools  
│   ├── __init__.py                # Package initialization
│   └── spike_analysis.py          # Chaos quantification functions
├── experiments/                   # Experiment coordination
│   ├── __init__.py                # Package initialization
│   ├── chaos_experiment.py        # Network dynamics & chaos analysis
│   ├── encoding_experiment.py     # Encoding capacity studies (future)
│   └── task_performance_experiment.py # Task performance evaluation (future)
├── runners/                       # Execution scripts
│   ├── mpi_chaos_runner.py        # MPI-parallelized chaos experiment
│   └── run_chaos_experiment.sh    # Bash script to run experiments
├── tests/                         # Testing framework
│   ├── __init__.py                # Package initialization
│   └── test_installation.py       # Installation verification
├── results/                       # Output directory
│   └── data/                      # Experiment data files
├── setup.py                       # Package installation configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Setup Project Structure

If you have files in a single directory, organize them as follows:

```bash
# Create the main project directory
mkdir spiking_rnn_heterogeneity
cd spiking_rnn_heterogeneity

# Create subdirectories
mkdir src analysis experiments runners tests results/data

# Move existing files to proper locations
mv rng_utils.py src/
mv lif_neuron.py src/
mv synaptic_model.py src/
mv spiking_network.py src/
mv spike_analysis.py analysis/
mv chaos_experiment.py experiments/
mv mpi_chaos_runner.py runners/
mv run_chaos_experiment.sh runners/
mv test_installation.py tests/
```

### 2. Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install the package in development mode (enables cross-directory imports)
pip install -e .

# Make scripts executable
chmod +x runners/run_chaos_experiment.sh
```

### 3. Verify Installation

```bash
python tests/test_installation.py
```

### 4. Setup Enhanced nohup Function (Optional)

For long-running experiments, add this to your `~/.bashrc`:

```bash
nohup_bash () {
   script_name=$1
   shift  # Remove first argument (script name)
   nohup ./${script_name} "$@" > output_${script_name%.*}.log 2>&1 & disown;clear
}
```

Then reload: `source ~/.bashrc`

## Usage

### Quick Test (Recommended First Step)
```bash
# From project root directory
runners/run_chaos_experiment.sh --session 1 --n_v_th 3 --n_g 3 --nproc 4
```

### Full Scale Experiments

```bash
# Medium experiment (~30 minutes)
runners/run_chaos_experiment.sh --session 1 --n_v_th 10 --n_g 10 --nproc 25

# Full experiment (~2 hours)
runners/run_chaos_experiment.sh --session 1 --n_v_th 20 --n_g 20 --nproc 50
```

### Using with nohup for Long Experiments

```bash
# Quick test with nohup
nohup_bash runners/run_chaos_experiment.sh --session 1 --n_v_th 3 --n_g 3 --nproc 4

# Full experiment with nohup (recommended)
nohup_bash runners/run_chaos_experiment.sh --session 1 --n_v_th 20 --n_g 20 --nproc 50

# Monitor progress
tail -f output_run_chaos_experiment.log
```

### Command Line Options

```bash
runners/run_chaos_experiment.sh [OPTIONS]

Options:
  -n, --nproc N_PROCESSES    Number of MPI processes (default: 50)
  -s, --session SESSION_ID   Session ID for reproducibility (default: 1)
  --n_v_th N_V_TH           Number of v_th_std values (default: 20)
  --n_g N_G                 Number of g_std values (default: 20)
  --n_neurons N_NEURONS     Number of neurons (default: 1000)
  --max_cores MAX_CORES     Maximum CPU cores to use (default: 50)
  -o, --output OUTPUT_DIR    Output directory (default: results)
  -h, --help                 Show detailed help message
```

## Features

### Network Architecture
- **LIF Neurons**: Biologically realistic neurons with heterogeneous spike thresholds (-55mV ± 0.05-0.5mV)
- **Exponential Synapses**: Recurrent connections with heterogeneous weights (mean=0, std=0.05-0.5)
- **Static Poisson Input**: Independent background noise for each neuron (200Hz default)
- **Dynamic Poisson Input**: 20 channels with 30% random connectivity (~9% overlap between channels)
- **Readout Layer**: 10 neurons for task performance evaluation (future use)

### Random Number Generation
Hierarchical seeding system ensures reproducibility:
- **Session ID**: Controls all random generations across parameter sweeps
- **Block ID**: Changes network structure for different (v_th_std, g_std) combinations
- **Trial ID**: Varies initial conditions and dynamics while keeping network structure fixed

### Chaos Analysis Framework

#### Current Implementation: Network Dynamics Study
- **Protocol**: 50ms baseline → auxiliary spike perturbation → 500ms observation
- **Analysis**: Two complementary chaos measures:
  - **Lempel-Ziv Complexity**: Algorithmic complexity of spike pattern differences
  - **Robust Hamming Distance Slope**: Perturbation divergence rate (avoids saturation bias)
- **Replication**: 20 different neurons perturbed per parameter combination
- **Parameter Space**: 20×20 grid of (v_th_std, g_std) values = 400 combinations

#### Future Extensions
- **Encoding Capacity**: 20-channel dynamic input patterns with information-theoretic analysis
- **Task Performance**: Classification, regression, and working memory tasks with 10-neuron readout

### Advanced Safety Features
- **Temperature Monitoring**: Automatic shutdown at 85°C, warnings at 75°C
- **Resource Management**: CPU/memory usage monitoring with configurable core limits
- **Cooling Breaks**: Mandatory 5-minute breaks every hour for experiments >2 hours
- **Colleague-Friendly**: Default 50/64 core usage leaves resources for others
- **Data Safety**: Intermediate saves every 10 combinations prevent data loss

## Experimental Parameters

### Parameter Space
- **Spike Threshold Heterogeneity (v_th_std)**: 0.05 to 0.5 mV
- **Synaptic Weight Heterogeneity (g_std)**: 0.05 to 0.5 (normalized units)
- **Default Grid**: 20 × 20 = 400 parameter combinations

### Network Specifications
- **RNN Size**: 1000 LIF neurons (configurable)
- **Background Activity**: 200Hz static Poisson input per neuron
- **Recurrent Connectivity**: 10% connection probability
- **Simulation Duration**: 550ms (50ms baseline + 500ms post-perturbation)
- **Time Resolution**: 0.1ms time steps

### Performance Specifications

#### Computational Requirements
- **Memory**: ~100MB per parameter combination
- **Time**: ~60 seconds per combination (1000 neurons, 20 perturbation trials)
- **Full Experiment**: ~2 hours on 50 cores (400 combinations)

#### Scaling Examples
| Grid Size | Combinations | Time (50 cores) | Use Case |
|-----------|-------------|-----------------|----------|
| 3×3       | 9           | ~3 minutes      | Quick test |
| 5×5       | 25          | ~8 minutes      | Development |
| 10×10     | 100         | ~30 minutes     | Pilot study |
| 20×20     | **400**     | **~2 hours**    | **Full study** |

## Output Files and Data Management

### Directory Structure
```
results/
└── data/                           # All experimental data
    ├── chaos_results_session_1.pkl           # Main results file
    ├── experiment_summary_session_1.txt      # Human-readable summary
    └── intermediate_session_1_rank_0.pkl     # Backup saves (if applicable)

output_run_chaos_experiment.log    # nohup execution log (if using nohup)
```

### Result Data Structure
Each experiment produces comprehensive results:

```python
# Load results
import pickle
with open('results/data/chaos_results_session_1.pkl', 'rb') as f:
    results = pickle.load(f)

# Each result dictionary contains:
result = {
    'v_th_std': float,              # Parameter values
    'g_std': float,
    'lz_complexities': array,       # Raw measurements (20 trials)
    'hamming_slopes': array,
    'lz_mean': float,               # Summary statistics
    'lz_std': float,
    'hamming_mean': float,
    'hamming_std': float,
    'n_trials': int,                # Number of perturbation trials
    'computation_time': float,      # Execution time
    'block_id': int,                # Parameter combination ID
    'perturbation_neurons': list    # Which neurons were perturbed
}
```

## Analysis Examples

### Load and Visualize Results
```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load experimental results
with open('results/data/chaos_results_session_1.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract parameter space
v_th_std = np.array([r['v_th_std'] for r in results])
g_std = np.array([r['g_std'] for r in results])
lz_complexity = np.array([r['lz_mean'] for r in results])
hamming_slope = np.array([r['hamming_mean'] for r in results])

# Reshape for heatmap visualization
n_v_th = len(np.unique(v_th_std))
n_g = len(np.unique(g_std))

lz_grid = lz_complexity.reshape(n_v_th, n_g)
hamming_grid = hamming_slope.reshape(n_v_th, n_g)

# Create chaos landscape visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# LZ Complexity heatmap
im1 = ax1.imshow(lz_grid, aspect='auto', origin='lower', cmap='viridis')
ax1.set_title('LZ Complexity (Algorithmic Chaos)')
ax1.set_xlabel('Synaptic Weight Heterogeneity (g_std)')
ax1.set_ylabel('Spike Threshold Heterogeneity (v_th_std)')
plt.colorbar(im1, ax=ax1)

# Hamming Slope heatmap
im2 = ax2.imshow(hamming_grid, aspect='auto', origin='lower', cmap='plasma')
ax2.set_title('Hamming Slope (Perturbation Growth Rate)')
ax2.set_xlabel('Synaptic Weight Heterogeneity (g_std)')
ax2.set_ylabel('Spike Threshold Heterogeneity (v_th_std)')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('chaos_landscape.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Statistical Analysis
```python
# Correlation between heterogeneity parameters and chaos measures
from scipy.stats import pearsonr

# Test scientific hypotheses
v_th_lz_corr, v_th_lz_p = pearsonr(v_th_std, lz_complexity)
g_lz_corr, g_lz_p = pearsonr(g_std, lz_complexity)

print("Scientific Results:")
print(f"Spike threshold heterogeneity vs LZ complexity: r={v_th_lz_corr:.3f}, p={v_th_lz_p:.3e}")
print(f"Synaptic weight heterogeneity vs LZ complexity: r={g_lz_corr:.3f}, p={g_lz_p:.3e}")

# Find optimal parameter regions
max_chaos_idx = np.argmax(lz_complexity)
optimal_params = results[max_chaos_idx]
print(f"Highest chaos at: v_th_std={optimal_params['v_th_std']:.3f}, g_std={optimal_params['g_std']:.3f}")
```

## Python API Usage

For custom experiments or analysis:

```python
# Import main components
from experiments.chaos_experiment import ChaosExperiment, create_parameter_grid
from src.spiking_network import SpikingRNN
from analysis.spike_analysis import analyze_perturbation_response

# Initialize experiment
experiment = ChaosExperiment(n_neurons=1000)

# Create parameter grid
v_th_std_values, g_std_values = create_parameter_grid(n_points=20)

# Run single parameter combination
result = experiment.run_parameter_combination(
    session_id=1, block_id=0, v_th_std=0.1, g_std=0.1
)

# Run complete experiment
results = experiment.run_full_experiment(
    session_id=1, v_th_std_values, g_std_values
)
```

## Cluster Usage and Best Practices

### Resource Management
- **Default Configuration**: 50 cores (colleague-friendly on 64-core systems)
- **Memory Usage**: ~5GB total for full experiment
- **Storage**: ~50MB for complete results
- **Network I/O**: Minimal (all computation local)

### Monitoring Long Experiments
```bash
# Check experiment progress
tail -f output_run_chaos_experiment.log

# Monitor system resources
htop

# Check intermediate results
ls -la results/data/intermediate*

# Monitor disk space
df -h results/
```

### Safety Features for Cluster Use
- **Temperature monitoring** prevents hardware damage
- **Resource limits** respect shared computing environments  
- **Automatic cooling breaks** prevent system overheating
- **Graceful shutdown** on resource exhaustion
- **Progress logging** enables restart from intermediate saves

## Troubleshooting

### Common Issues and Solutions

**Installation Problems:**
```bash
# Missing MPI
sudo apt-get install openmpi-bin openmpi-dev

# Missing Python packages  
pip install -r requirements.txt

# Package not found errors
pip install -e .

# Permission issues
chmod +x runners/run_chaos_experiment.sh
```

**Runtime Problems:**
```bash
# Test with minimal configuration
runners/run_chaos_experiment.sh --n_v_th 2 --n_g 2 --nproc 2

# Check system resources
python tests/test_installation.py

# Verify file integrity
python -c "import experiments.chaos_experiment"
```

**Import Errors:**
```bash
# If you get import errors, make sure you're in the right directory
cd spiking_rnn_heterogeneity/

# And that the package is installed
pip install -e .
```

**Performance Issues:**
- Reduce `--n_neurons` for faster execution
- Reduce `--nproc` if system overloading
- Use `--max_cores` to limit CPU usage
- Check available memory with `free -h`

### Debug Mode
```bash
# Single-process debugging
python experiments/chaos_experiment.py

# Small MPI test
mpirun -n 2 python runners/mpi_chaos_runner.py --n_v_th 2 --n_g 2
```

## Scientific Applications

### Research Questions Addressed
1. **How does spike threshold heterogeneity affect network chaos?**
2. **How does synaptic weight heterogeneity influence dynamical stability?**
3. **What is the interaction between these two heterogeneity sources?**
4. **Which parameter regimes optimize information processing vs. stability?**

### Experimental Design Features
- **Controlled perturbations**: Single auxiliary spike injection
- **Statistical robustness**: 20 independent perturbation trials
- **Comprehensive measurement**: Two complementary chaos measures
- **Parameter space exploration**: Systematic 20×20 grid sampling
- **Reproducible results**: Hierarchical random number generation

### Expected Outcomes
This framework enables discovery of:
- **Chaos-order transitions** across heterogeneity parameter space
- **Optimal heterogeneity levels** for different computational goals
- **Interaction effects** between threshold and synaptic heterogeneity
- **Design principles** for robust yet flexible neural networks

## Future Extensions

### Planned Developments
1. **Encoding Capacity Studies**: Information-theoretic analysis of 20-channel dynamic inputs
2. **Task Performance Evaluation**: Classification, regression, and working memory tasks
3. **Extended Network Models**: Different neuron types, synaptic plasticity, network topologies

### Extensibility Features
The modular architecture supports easy extensions:
- New chaos measures in `analysis/spike_analysis.py`
- Alternative neuron models in `src/lif_neuron.py`
- Custom connectivity patterns in `src/synaptic_model.py`
- Additional experiments following the template pattern

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **CPU**: Multi-core system (4+ cores recommended)
- **Memory**: 4GB minimum, 8GB+ recommended
- **Storage**: 1GB free space
- **MPI**: OpenMPI or compatible implementation

### Python Dependencies
See `requirements.txt` for complete list. Key packages:
- numpy>=1.20.0
- scipy>=1.7.0
- mpi4py>=3.1.0
- psutil>=5.8.0
- matplotlib>=3.3.0

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{spiking_rnn_heterogeneity_framework,
  title = {Spiking RNN Heterogeneity Studies Framework},
  author = {[Your Name]},
  year = {2024},
  url = {[Your Repository URL]},
  note = {Framework for studying chaos in heterogeneous spiking neural networks}
}
```

## License

[Your License Choice - e.g., MIT, GPL, etc.]

## Support and Contributing

### Getting Help
- **Installation Issues**: Run `python tests/test_installation.py` first
- **Experimental Design**: Check the parameter scaling table above
- **Performance Problems**: See troubleshooting section
- **Scientific Questions**: Refer to the analysis examples

### Contributing
Contributions are welcome! Areas for improvement:
- Additional chaos measures and analysis methods
- Support for different neuron models
- Enhanced visualization tools
- Performance optimizations
- Documentation improvements

### Contact
[Your Contact Information]

---

**Framework Version**: 2.0  
**Last Updated**: [Current Date]  
**Python Compatibility**: 3.8+  
**MPI Compatibility**: OpenMPI 3.0+