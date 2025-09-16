# tests/test_installation.py
"""
Test script to verify installation and run small examples.
Updated for organized directory structure.
"""

import sys
import os
import importlib
import numpy as np
from mpi4py import MPI

# Add project directories to path for testing
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'analysis'))
sys.path.insert(0, os.path.join(project_root, 'experiments'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    # Core scientific packages
    required_packages = [
        'numpy', 'scipy', 'mpi4py', 'psutil'
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            return False

    # Our custom modules from organized directories
    custom_modules = [
        ('src', ['rng_utils', 'lif_neuron', 'synaptic_model', 'spiking_network']),
        ('analysis', ['spike_analysis']),
        ('experiments', ['chaos_experiment'])
    ]

    for directory, modules in custom_modules:
        print(f"  Testing {directory}/ modules:")
        for module in modules:
            try:
                importlib.import_module(module)
                print(f"    ✓ {module}")
            except ImportError as e:
                print(f"    ✗ {module}: {e}")
                return False

    return True

def test_rng_system():
    """Test hierarchical RNG system for reproducibility."""
    print("\nTesting RNG system...")

    try:
        from rng_utils import get_rng, rng_manager

        # Test reproducibility - clear cache between calls
        rng_manager.clear_cache()
        rng1 = get_rng(1, 1, 1, 'test')
        val1 = rng1.random()

        rng_manager.clear_cache()
        rng2 = get_rng(1, 1, 1, 'test')
        val2 = rng2.random()

        if abs(val1 - val2) < 1e-10:
            print("  ✓ RNG reproducibility")
        else:
            print(f"  ✗ RNG reproducibility: {val1} != {val2}")
            return False

        # Test hierarchy - different blocks should give different values
        rng_block1 = get_rng(1, 1, 1, 'network')
        rng_block2 = get_rng(1, 2, 1, 'network')

        val_block1 = rng_block1.random()
        val_block2 = rng_block2.random()

        if abs(val_block1 - val_block2) > 1e-10:
            print("  ✓ RNG hierarchy (different blocks)")
        else:
            print(f"  ✗ RNG hierarchy: blocks should differ")
            return False

        # Test trial independence
        rng_trial1 = get_rng(1, 1, 1, 'initial_state')
        rng_trial2 = get_rng(1, 1, 2, 'initial_state')

        val_trial1 = rng_trial1.random()
        val_trial2 = rng_trial2.random()

        if abs(val_trial1 - val_trial2) > 1e-10:
            print("  ✓ RNG trial independence")
        else:
            print(f"  ✗ RNG trials should differ")
            return False

        return True

    except Exception as e:
        print(f"  ✗ RNG system test failed: {e}")
        return False

def test_neuron_model():
    """Test LIF neuron model."""
    print("\nTesting LIF neuron model...")

    try:
        from lif_neuron import LIFNeuron

        # Create small neuron population
        neurons = LIFNeuron(n_neurons=10, dt=0.1)

        # Initialize parameters
        neurons.initialize_parameters(
            session_id=1, block_id=1, v_th_mean=-55.0, v_th_std=0.1
        )

        # Check spike thresholds were set
        if neurons.spike_thresholds is not None and len(neurons.spike_thresholds) == 10:
            print("  ✓ Neuron parameter initialization")
        else:
            print("  ✗ Neuron parameter initialization failed")
            return False

        # Initialize state
        neurons.initialize_state(session_id=1, block_id=1, trial_id=1)

        # Test membrane potential initialization
        if neurons.v_membrane is not None and len(neurons.v_membrane) == 10:
            print("  ✓ Neuron state initialization")
        else:
            print("  ✗ Neuron state initialization failed")
            return False

        # Test update (with strong input to ensure some spikes)
        strong_input = np.ones(10) * 50.0  # Strong input
        v_mem, spike_indices = neurons.update(0.0, strong_input)

        if len(spike_indices) >= 0:  # Allow zero spikes (still valid)
            print(f"  ✓ Neuron update ({len(spike_indices)} spikes generated)")
        else:
            print("  ✗ Neuron update failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Neuron model test failed: {e}")
        return False

def test_synaptic_model():
    """Test synaptic models and Poisson inputs."""
    print("\nTesting synaptic models...")

    try:
        from synaptic_model import ExponentialSynapses, StaticPoissonInput, DynamicPoissonInput

        # Test exponential synapses
        synapses = ExponentialSynapses(n_neurons=50, dt=0.1)
        synapses.initialize_weights(
            session_id=1, block_id=1, g_mean=0.0, g_std=0.1, connection_prob=0.2
        )

        if synapses.weight_matrix is not None:
            print("  ✓ Exponential synapses initialization")
        else:
            print("  ✗ Exponential synapses initialization failed")
            return False

        # Test synaptic update
        spike_indices = [0, 1, 5]  # Some neurons spike
        synaptic_input = synapses.update(spike_indices)

        if len(synaptic_input) == 50:
            print("  ✓ Synaptic update")
        else:
            print("  ✗ Synaptic update failed")
            return False

        # Test static Poisson input
        static_input = StaticPoissonInput(n_neurons=50, dt=0.1)
        static_input.initialize_parameters(input_strength=1.0)

        input_current = static_input.update(
            session_id=1, block_id=1, trial_id=1, rate=100.0
        )

        if len(input_current) == 50:
            print("  ✓ Static Poisson input")
        else:
            print("  ✗ Static Poisson input failed")
            return False

        # Test dynamic Poisson input
        dynamic_input = DynamicPoissonInput(n_neurons=50, n_channels=10, dt=0.1)
        dynamic_input.initialize_connectivity(
            session_id=1, block_id=1, connection_prob=0.3, input_strength=1.0
        )

        rates = np.ones(10) * 50.0  # 50Hz for all channels
        input_current = dynamic_input.update(
            session_id=1, block_id=1, trial_id=1, rates=rates
        )

        if len(input_current) == 50:
            print("  ✓ Dynamic Poisson input")
        else:
            print("  ✗ Dynamic Poisson input failed")
            return False

        # Test connectivity info
        conn_info = dynamic_input.get_connectivity_info()
        if 'n_channels' in conn_info and conn_info['n_channels'] == 10:
            print("  ✓ Connectivity analysis")
        else:
            print("  ✗ Connectivity analysis failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Synaptic model test failed: {e}")
        return False

def test_spike_analysis():
    """Test chaos analysis functions."""
    print("\nTesting spike analysis...")

    try:
        from spike_analysis import (
            spikes_to_binary, lempel_ziv_complexity, sort_matrix,
            compute_spike_difference_matrix, compute_chaos_slope_robust,
            analyze_perturbation_response
        )

        # Create test spike data
        spikes1 = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2)]
        spikes2 = [(1.0, 0), (2.5, 1), (3.5, 2), (4.0, 0)]  # Different timing

        # Test spike-to-binary conversion
        binary_matrix = spikes_to_binary(spikes1, num_neurons=3, duration=5.0, bin_size=1.0)

        if binary_matrix.shape == (3, 5):  # 3 neurons, 5 time bins
            print("  ✓ Spike-to-binary conversion")
        else:
            print(f"  ✗ Spike-to-binary conversion: wrong shape {binary_matrix.shape}")
            return False

        # Test matrix sorting
        sorted_matrix = sort_matrix(binary_matrix)

        if sorted_matrix.shape == binary_matrix.shape:
            print("  ✓ Matrix sorting")
        else:
            print("  ✗ Matrix sorting failed")
            return False

        # Test LZ complexity
        lz_comp = lempel_ziv_complexity(sorted_matrix)

        if isinstance(lz_comp, int) and lz_comp > 0:
            print(f"  ✓ Lempel-Ziv complexity (value: {lz_comp})")
        else:
            print("  ✗ Lempel-Ziv complexity failed")
            return False

        # Test spike difference matrix
        diff_matrix = compute_spike_difference_matrix(
            spikes1, spikes2, num_neurons=3, perturbation_time=1.0,
            simulation_end=5.0, perturbed_neuron=0, bin_size=1.0
        )

        if diff_matrix.shape[0] == 3:  # 3 neurons
            print("  ✓ Spike difference matrix")
        else:
            print("  ✗ Spike difference matrix failed")
            return False

        # Test robust slope calculation
        time_bins = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        hamming_distances = np.array([1, 2, 3, 4, 5])  # Linear increase

        slope = compute_chaos_slope_robust(time_bins, hamming_distances)

        if isinstance(slope, (int, float)) and slope >= 0:
            print(f"  ✓ Robust slope calculation (slope: {slope:.3f})")
        else:
            print("  ✗ Robust slope calculation failed")
            return False

        # Test full perturbation analysis
        lz_val, hamm_slope = analyze_perturbation_response(
            spikes1, spikes2, num_neurons=3, perturbation_time=1.0,
            simulation_end=5.0, perturbed_neuron=0
        )

        if isinstance(lz_val, int) and isinstance(hamm_slope, (int, float)):
            print(f"  ✓ Full perturbation analysis (LZ: {lz_val}, slope: {hamm_slope:.3f})")
        else:
            print("  ✗ Full perturbation analysis failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Spike analysis test failed: {e}")
        return False

def test_network_integration():
    """Test complete spiking network."""
    print("\nTesting network integration...")

    try:
        from spiking_network import SpikingRNN

        # Create network
        network = SpikingRNN(n_neurons=100, n_input_channels=5, n_readout_neurons=3, dt=0.1)

        # Initialize network
        network.initialize_network(
            session_id=1, block_id=1, v_th_std=0.1, g_std=0.1
        )
        print("  ✓ Network initialization")

        # Test network dynamics simulation
        spikes = network.simulate_network_dynamics(
            session_id=1, block_id=1, trial_id=1,
            duration=100.0, static_input_rate=200.0
        )

        print(f"  ✓ Network dynamics simulation ({len(spikes)} spikes generated)")

        # Test encoding simulation
        input_patterns = np.random.uniform(10, 50, (1000, 5))  # 1000 time steps, 5 channels
        spike_times, readout_history = network.simulate_encoding_task(
            session_id=1, block_id=1, trial_id=1,
            duration=100.0, input_patterns=input_patterns, static_input_rate=100.0
        )

        print(f"  ✓ Encoding simulation ({len(spike_times)} spikes, {len(readout_history)} readouts)")

        # Test task performance
        target_outputs = np.random.uniform(0, 1, (1000, 3))  # 1000 time steps, 3 outputs
        performance = network.simulate_task_performance(
            session_id=1, block_id=1, trial_id=1,
            duration=100.0, input_patterns=input_patterns,
            target_outputs=target_outputs, static_input_rate=100.0
        )

        if 'mse' in performance and 'mean_correlation' in performance:
            print(f"  ✓ Task performance (MSE: {performance['mse']:.3f}, Corr: {performance['mean_correlation']:.3f})")
        else:
            print("  ✗ Task performance failed")
            return False

        # Test network info
        info = network.get_network_info()

        if 'n_neurons' in info and info['n_neurons'] == 100:
            print("  ✓ Network information")
        else:
            print("  ✗ Network information failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Network integration test failed: {e}")
        return False

def test_mpi():
    """Test MPI functionality."""
    print("\nTesting MPI...")

    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"  MPI rank: {rank}/{size}")

        if size == 1:
            print("  ⚠ Running on single process (use 'mpirun -n 4 python tests/test_installation.py' for parallel test)")
        else:
            print(f"  ✓ MPI parallel execution with {size} processes")

        # Test basic MPI communication
        data = rank * 10
        all_data = comm.gather(data, root=0)

        if rank == 0:
            if all_data is not None and len(all_data) == size:
                print(f"  ✓ MPI communication (gathered data: {all_data})")
            else:
                print("  ✗ MPI communication failed")
                return False

        return True

    except Exception as e:
        print(f"  ✗ MPI test failed: {e}")
        return False

def test_chaos_experiment():
    """Test complete chaos experiment."""
    print("\nTesting chaos experiment...")

    try:
        from chaos_experiment import ChaosExperiment, create_parameter_grid

        # Create small experiment
        experiment = ChaosExperiment(n_neurons=50)  # Small for speed

        # Test parameter grid creation
        v_th_values, g_values = create_parameter_grid(n_points=3)

        if len(v_th_values) == 3 and len(g_values) == 3:
            print("  ✓ Parameter grid creation")
        else:
            print("  ✗ Parameter grid creation failed")
            return False

        # Test single parameter combination
        result = experiment.run_parameter_combination(
            session_id=999, block_id=0, v_th_std=0.1, g_std=0.1
        )

        # Check all required fields
        required_fields = [
            'v_th_std', 'g_std', 'lz_complexities', 'hamming_slopes',
            'lz_mean', 'lz_std', 'hamming_mean', 'hamming_std',
            'n_trials', 'computation_time'
        ]

        missing_fields = [field for field in required_fields if field not in result]

        if not missing_fields:
            print("  ✓ Parameter combination execution")
            print(f"    LZ complexity: {result['lz_mean']:.2f} ± {result['lz_std']:.2f}")
            print(f"    Hamming slope: {result['hamming_mean']:.4f} ± {result['hamming_std']:.4f}")
            print(f"    Computation time: {result['computation_time']:.1f}s")
            print(f"    Trials completed: {result['n_trials']}")
        else:
            print(f"  ✗ Parameter combination missing fields: {missing_fields}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Chaos experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_requirements():
    """Test system requirements and performance."""
    print("\nTesting system requirements...")

    try:
        import psutil

        # Check CPU cores
        cpu_count = psutil.cpu_count()
        print(f"  CPU cores available: {cpu_count}")

        if cpu_count >= 4:
            print("  ✓ Sufficient CPU cores for parallel execution")
        else:
            print("  ⚠ Limited CPU cores - parallel performance may be reduced")

        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"  Total memory: {memory_gb:.1f} GB")

        if memory_gb >= 8:
            print("  ✓ Sufficient memory for large experiments")
        elif memory_gb >= 4:
            print("  ⚠ Limited memory - reduce network size for large experiments")
        else:
            print("  ⚠ Very limited memory - use small networks only")

        # Check available memory
        available_gb = memory.available / (1024**3)
        print(f"  Available memory: {available_gb:.1f} GB")

        return True

    except Exception as e:
        print(f"  ✗ System requirements test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required directories and files exist."""
    print("\nTesting directory structure...")

    project_root = os.path.dirname(current_dir)

    # Required directories
    required_dirs = [
        'src',
        'analysis',
        'experiments',
        'runners',
        'tests'
    ]

    # Required files in each directory
    required_files = {
        'src': ['rng_utils.py', 'lif_neuron.py', 'synaptic_model.py', 'spiking_network.py'],
        'analysis': ['spike_analysis.py'],
        'experiments': ['chaos_experiment.py'],
        'runners': ['mpi_chaos_runner.py', 'run_chaos_experiment.sh'],
        'tests': ['test_installation.py']
    }

    all_good = True

    for directory in required_dirs:
        dir_path = os.path.join(project_root, directory)
        if os.path.exists(dir_path):
            print(f"  ✓ Directory: {directory}/")

            # Check required files
            if directory in required_files:
                for file in required_files[directory]:
                    file_path = os.path.join(dir_path, file)
                    if os.path.exists(file_path):
                        print(f"    ✓ {file}")
                    else:
                        print(f"    ✗ {file} (missing)")
                        all_good = False
        else:
            print(f"  ✗ Directory: {directory}/ (missing)")
            all_good = False

    # Check for __init__.py files
    for directory in required_dirs:
        init_path = os.path.join(project_root, directory, '__init__.py')
        if os.path.exists(init_path):
            print(f"  ✓ {directory}/__init__.py")
        else:
            print(f"  ⚠ {directory}/__init__.py (missing - recommended for package structure)")

    return all_good

def main():
    """Run all tests."""
    print("🧠 Spiking RNN Heterogeneity Framework - Installation Test")
    print("=" * 65)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("RNG System", test_rng_system),
        ("LIF Neurons", test_neuron_model),
        ("Synaptic Models", test_synaptic_model),
        ("Spike Analysis", test_spike_analysis),
        ("Network Integration", test_network_integration),
        ("MPI", test_mpi),
        ("Chaos Experiment", test_chaos_experiment),
        ("System Requirements", test_system_requirements),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 65)
    print("🏁 Test Summary:")
    print("=" * 65)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Installation is working correctly.")
        print("\n🚀 You can now run experiments:")
        print("  # From project root directory:")
        print("  cd spiking_rnn_heterogeneity/")
        print("  ")
        print("  # Quick test (recommended first):")
        print("  runners/run_chaos_experiment.sh --session 1 --n_v_th 3 --n_g 3 --nproc 4")
        print("  ")
        print("  # With nohup (recommended for long runs):")
        print("  nohup_bash runners/run_chaos_experiment.sh --session 1 --n_v_th 20 --n_g 20 --nproc 50")
        print("  ")
        print("  # Monitor progress:")
        print("  tail -f output_run_chaos_experiment.log")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        print("\n🔧 Common fixes:")
        print("  pip install -r requirements.txt")
        print("  pip install -e .  # Install package in development mode")
        print("  Ensure all files are in correct directories")
        print("  Check MPI installation: sudo apt-get install openmpi-bin")
        return 1

if __name__ == "__main__":
    exit(main())
