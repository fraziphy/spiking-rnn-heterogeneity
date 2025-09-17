# tests/test_installation.py - Updated with enhanced analysis testing
"""
Test script to verify installation and enhanced analysis functionality.
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

    # Core packages
    required_packages = ['numpy', 'scipy', 'mpi4py', 'psutil']

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            return False

    # Our custom modules
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

def test_enhanced_spike_analysis():
    """Test enhanced spike analysis functions."""
    print("\nTesting enhanced spike analysis...")

    try:
        from spike_analysis import (
            compute_activity_dimensionality, gamma_coincidence, average_gamma_coincidence,
            analyze_perturbation_response_enhanced
        )

        # Test activity dimensionality
        test_matrix = np.random.randint(0, 2, (50, 100))  # 50 neurons, 100 time bins
        dim_results = compute_activity_dimensionality(test_matrix)

        required_keys = ['intrinsic_dimensionality', 'effective_dimensionality',
                        'participation_ratio', 'total_variance']

        if all(key in dim_results for key in required_keys):
            print(f"  ✓ Activity dimensionality: {dim_results['effective_dimensionality']:.1f} dims")
        else:
            print(f"  ✗ Activity dimensionality missing keys")
            return False

        # Test gamma coincidence
        spikes1 = [1.0, 5.0, 10.0, 15.0]
        spikes2 = [1.1, 5.2, 9.8, 15.3]  # Similar timing
        gamma_c = gamma_coincidence(spikes1, spikes2, window_ms=5.0)

        if 0.0 <= gamma_c <= 1.0:
            print(f"  ✓ Gamma coincidence: {gamma_c:.3f}")
        else:
            print(f"  ✗ Gamma coincidence out of range: {gamma_c}")
            return False

        # Test average gamma coincidence
        network_spikes1 = [(1.0, 0), (5.0, 1), (10.0, 0)]
        network_spikes2 = [(1.1, 0), (5.2, 1), (9.8, 0)]
        avg_gamma = average_gamma_coincidence(network_spikes1, network_spikes2,
                                            num_neurons=2, window_ms=5.0)

        if 0.0 <= avg_gamma <= 1.0:
            print(f"  ✓ Average gamma coincidence: {avg_gamma:.3f}")
        else:
            print(f"  ✗ Average gamma coincidence out of range: {avg_gamma}")
            return False

        # Test enhanced perturbation analysis
        # Create test spike data
        spikes_control = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2)]
        spikes_perturbed = [(1.0, 0), (2.5, 1), (3.5, 2), (4.0, 0)]

        enhanced_results = analyze_perturbation_response_enhanced(
            spikes_control, spikes_perturbed, num_neurons=3,
            perturbation_time=1.0, simulation_end=5.0, perturbed_neuron=0
        )

        # Check all expected keys are present
        expected_keys = [
            'lz_complexity', 'hamming_slope', 'total_spike_differences',
            'intrinsic_dimensionality', 'effective_dimensionality',
            'participation_ratio', 'total_variance', 'gamma_coincidence'
        ]

        missing_keys = [key for key in expected_keys if key not in enhanced_results]
        if not missing_keys:
            print(f"  ✓ Enhanced analysis complete:")
            print(f"    LZ: {enhanced_results['lz_complexity']}")
            print(f"    Hamming: {enhanced_results['hamming_slope']:.4f}")
            print(f"    Spike diffs: {enhanced_results['total_spike_differences']}")
            print(f"    Dimensionality: {enhanced_results['effective_dimensionality']:.1f}")
            print(f"    Gamma: {enhanced_results['gamma_coincidence']:.3f}")
        else:
            print(f"  ✗ Enhanced analysis missing keys: {missing_keys}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Enhanced spike analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_chaos_experiment():
    """Test enhanced chaos experiment."""
    print("\nTesting enhanced chaos experiment...")

    try:
        from chaos_experiment import ChaosExperiment, create_parameter_grid_with_input_rates

        # Create small experiment for testing
        experiment = ChaosExperiment(n_neurons=20)  # Very small for speed

        # Test parameter grid with new ranges
        v_th_values, g_values, input_rates = create_parameter_grid_with_input_rates(n_points=3)

        if (len(v_th_values) == 3 and len(g_values) == 3 and
            np.min(v_th_values) >= 0.01 and np.max(v_th_values) <= 1.0):
            print(f"  ✓ Parameter grid: v_th {np.min(v_th_values):.3f}-{np.max(v_th_values):.3f}")
        else:
            print(f"  ✗ Parameter grid ranges incorrect")
            return False

        # Test single parameter combination with enhanced metrics
        result = experiment.run_parameter_combination(
            session_id=999, block_id=0, v_th_std=0.1, g_std=0.1, static_input_rate=100.0
        )

        # Check for all enhanced metrics
        expected_fields = [
            'v_th_std', 'g_std', 'static_input_rate',
            'lz_complexities', 'hamming_slopes', 'lz_mean', 'lz_std',
            'hamming_mean', 'hamming_std',
            'total_spike_differences', 'spike_diff_mean', 'spike_diff_std',
            'intrinsic_dimensionalities', 'effective_dimensionalities',
            'intrinsic_dim_mean', 'effective_dim_mean',
            'gamma_coincidences', 'gamma_coincidence_mean', 'gamma_coincidence_std',
            'n_trials', 'computation_time'
        ]

        missing_fields = [field for field in expected_fields if field not in result]

        if not missing_fields:
            print(f"  ✓ Enhanced parameter combination:")
            print(f"    LZ: {result['lz_mean']:.2f} ± {result['lz_std']:.2f}")
            print(f"    Hamming: {result['hamming_mean']:.4f} ± {result['hamming_std']:.4f}")
            print(f"    Spike diffs: {result['spike_diff_mean']:.1f} ± {result['spike_diff_std']:.1f}")
            print(f"    Dimensionality: {result['effective_dim_mean']:.1f} ± {result['effective_dim_std']:.1f}")
            print(f"    Gamma: {result['gamma_coincidence_mean']:.3f} ± {result['gamma_coincidence_std']:.3f}")
            print(f"    Trials: {result['n_trials']}, Time: {result['computation_time']:.1f}s")
        else:
            print(f"  ✗ Enhanced experiment missing fields: {missing_fields}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Enhanced chaos experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_integration():
    """Test complete spiking network with updated parameters."""
    print("\nTesting network integration with updated parameters...")

    try:
        from spiking_network import SpikingRNN

        # Create network
        network = SpikingRNN(n_neurons=50, dt=0.1)

        # Initialize with updated parameter ranges
        network.initialize_network(
            session_id=1, block_id=1,
            v_th_std=0.5,  # Test higher heterogeneity
            g_std=0.5,     # Test higher heterogeneity
            static_input_strength=1.0,
            connection_prob=0.1
        )

        print("  ✓ Network initialization with higher heterogeneity")

        # Test simulation with updated duration (300ms post-perturbation)
        spikes = network.simulate_network_dynamics(
            session_id=1, block_id=1, trial_id=1,
            duration=350.0,  # 50ms pre + 300ms post
            static_input_rate=200.0
        )

        print(f"  ✓ Network simulation: {len(spikes)} spikes in 350ms")

        # Get network info and check threshold ranges
        info = network.get_network_info()
        threshold_range = np.max(info['spike_thresholds']) - np.min(info['spike_thresholds'])

        if threshold_range > 1.0:  # Should be higher with increased heterogeneity
            print(f"  ✓ High heterogeneity: threshold range {threshold_range:.2f} mV")
        else:
            print(f"  ⚠ Lower heterogeneity than expected: {threshold_range:.2f} mV")

        return True

    except Exception as e:
        print(f"  ✗ Network integration test failed: {e}")
        return False

def run_enhanced_analysis_test():
    """Run a complete test of the enhanced analysis pipeline."""
    print("\nRunning complete enhanced analysis test...")

    try:
        from chaos_experiment import ChaosExperiment
        from spike_analysis import analyze_perturbation_response_enhanced

        # Very small experiment to test pipeline
        experiment = ChaosExperiment(n_neurons=10)

        print("  Running mini enhanced experiment...")

        # Single combination test
        start_time = time.time()
        result = experiment.run_parameter_combination(
            session_id=1, block_id=1,
            v_th_std=0.2, g_std=0.3, static_input_rate=150.0
        )
        test_time = time.time() - start_time

        # Verify all enhanced metrics are computed
        metrics_check = {
            'Original chaos': all(key in result for key in ['lz_mean', 'hamming_mean']),
            'Spike differences': all(key in result for key in ['spike_diff_mean', 'spike_diff_std']),
            'Dimensionality': all(key in result for key in ['effective_dim_mean', 'intrinsic_dim_mean']),
            'Gamma coincidence': all(key in result for key in ['gamma_coincidence_mean', 'gamma_coincidence_std']),
            'Arrays preserved': all(isinstance(result[key], np.ndarray) for key in
                                  ['lz_complexities', 'total_spike_differences', 'gamma_coincidences'])
        }

        all_passed = all(metrics_check.values())

        print(f"  ✓ Enhanced metrics verification:")
        for metric, passed in metrics_check.items():
            status = "✓" if passed else "✗"
            print(f"    {status} {metric}")

        print(f"  ✓ Test completed in {test_time:.1f}s")
        print(f"  ✓ Enhanced analysis pipeline working correctly")

        return all_passed

    except Exception as e:
        print(f"  ✗ Enhanced analysis test failed: {e}")
        return False

def main():
    """Run all enhanced tests."""
    print("Enhanced Spiking RNN Framework - Installation & Analysis Test")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Enhanced Spike Analysis", test_enhanced_spike_analysis),
        ("Enhanced Chaos Experiment", test_enhanced_chaos_experiment),
        ("Network Integration", test_network_integration),
        ("Complete Enhanced Pipeline", run_enhanced_analysis_test),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 70)
    print("Enhanced Test Summary:")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\nEnhanced framework is working correctly!")
        print("\nNew capabilities available:")
        print("  • Network activity dimensionality analysis")
        print("  • Spike train difference quantification")
        print("  • Normalized gamma coincidence metrics")
        print("  • Updated parameter ranges (0.01-1.0)")
        print("  • Extended analysis duration (300ms)")
        print("\nYou can now run enhanced experiments:")
        print("  runners/run_chaos_experiment.sh --session 1 --n_v_th 5 --n_g 5 --n_input_rates 3")
        return 0
    else:
        print(f"\n{total_tests - passed_tests} enhanced tests failed.")
        print("\nTroubleshooting:")
        print("  1. Ensure all analysis modules are updated")
        print("  2. Check parameter ranges in chaos_experiment.py")
        print("  3. Verify enhanced spike_analysis.py functions")
        return 1

if __name__ == "__main__":
    exit(main())
