# tests/test_installation.py - Fixed for enhanced implementation
"""
Test script to verify installation and functionality of enhanced framework.
"""

import sys
import os
import importlib
import numpy as np
import time
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
            compute_activity_dimensionality, kistler_coincidence_factor,
            gamma_coincidence, average_gamma_coincidence,
            analyze_perturbation_response_enhanced, lempel_ziv_complexity,
            compute_spatial_pattern_complexity, find_stable_period
        )

        # Test new LZ complexity function
        test_sequence = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        lz_result = lempel_ziv_complexity(test_sequence)

        if lz_result > 0:
            print(f"  ✓ New LZ complexity function: {lz_result}")
        else:
            print(f"  ✗ New LZ complexity function failed: {lz_result}")
            return False

        # Test spatial pattern complexity and PCI
        test_matrix = np.array([
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        spatial_results = compute_spatial_pattern_complexity(test_matrix)

        expected_keys = ['lz_spatial_patterns', 'pci_raw', 'pci_normalized',
                        'pci_with_threshold', 'spatial_entropy', 'pattern_fraction']

        if all(key in spatial_results for key in expected_keys):
            print(f"  ✓ Spatial pattern complexity and PCI:")
            print(f"    LZ spatial: {spatial_results['lz_spatial_patterns']}")
            print(f"    PCI raw: {spatial_results['pci_raw']}")
            print(f"    PCI normalized: {spatial_results['pci_normalized']:.3f}")
            print(f"    Pattern fraction: {spatial_results['pattern_fraction']:.3f}")
        else:
            print(f"  ✗ Spatial pattern complexity missing keys")
            return False

        # Test pattern stability
        repeating_seq = [1, 2, 1, 2, 1, 2, 1, 2]
        stability_result = find_stable_period(repeating_seq, min_repeats=3)

        if stability_result is not None and stability_result['period'] == 2:
            print(f"  ✓ Pattern stability detection: period={stability_result['period']}")
        else:
            print(f"  ✗ Pattern stability detection failed")
            return False

        # Test activity dimensionality
        test_matrix_dim = np.random.randint(0, 2, (50, 100))  # 50 neurons, 100 time bins
        dim_results = compute_activity_dimensionality(test_matrix_dim)

        required_keys = ['intrinsic_dimensionality', 'effective_dimensionality',
                        'participation_ratio', 'total_variance']

        if all(key in dim_results for key in required_keys):
            print(f"  ✓ Activity dimensionality: {dim_results['effective_dimensionality']:.1f} dims")
        else:
            print(f"  ✗ Activity dimensionality missing keys")
            return False

        # Test Kistler coincidence
        spikes1 = [1.0, 5.0, 10.0, 15.0]
        spikes2 = [1.1, 5.2, 9.8, 15.3]  # Similar timing
        kistler_c = kistler_coincidence_factor(spikes1, spikes2, delta=2.0, duration=20.0)

        if 0.0 <= kistler_c <= 1.0:
            print(f"  ✓ Kistler coincidence: {kistler_c:.3f}")
        else:
            print(f"  ✗ Kistler coincidence out of range: {kistler_c}")
            return False

        # Test gamma coincidence (backward compatibility)
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
        spikes_control = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2)]
        spikes_perturbed = [(1.0, 0), (2.5, 1), (3.5, 2), (4.0, 0)]

        enhanced_results = analyze_perturbation_response_enhanced(
            spikes_control, spikes_perturbed, num_neurons=3,
            perturbation_time=1.0, simulation_end=5.0, perturbed_neuron=0
        )

        # Check all expected keys are present (updated for new measures)
        expected_keys = [
            'lz_matrix_flattened', 'hamming_slope', 'lz_spatial_patterns',
            'pci_raw', 'pci_normalized', 'pci_with_threshold',
            'total_spike_differences', 'kistler_delta_2ms', 'kistler_delta_5ms',
            'gamma_window_5ms', 'gamma_window_10ms', 'control_firing_stats',
            'perturbed_firing_stats', 'stable_period'
        ]

        missing_keys = [key for key in expected_keys if key not in enhanced_results]
        if not missing_keys:
            print(f"  ✓ Enhanced analysis complete:")
            print(f"    LZ (flattened): {enhanced_results['lz_matrix_flattened']}")
            print(f"    LZ (spatial): {enhanced_results['lz_spatial_patterns']}")
            print(f"    PCI (normalized): {enhanced_results['pci_normalized']:.3f}")
            print(f"    Hamming: {enhanced_results['hamming_slope']:.4f}")
            print(f"    Kistler (2ms): {enhanced_results['kistler_delta_2ms']:.3f}")
        else:
            print(f"  ✗ Enhanced analysis missing keys: {missing_keys}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Enhanced spike analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chaos_experiment():
    """Test enhanced chaos experiment."""
    print("\nTesting enhanced chaos experiment...")

    try:
        from chaos_experiment import ChaosExperiment, create_parameter_grid

        # Test parameter grid creation with extended range
        v_th_stds, g_stds, input_rates = create_parameter_grid(
            n_v_th_points=3, n_g_points=3, input_rate_range=(50.0, 1000.0)
        )

        if (len(v_th_stds) == 3 and len(g_stds) == 3 and
            np.min(input_rates) >= 50.0 and np.max(input_rates) <= 1000.0):
            print(f"  ✓ Enhanced parameter grid: rates {np.min(input_rates):.0f}-{np.max(input_rates):.0f}Hz")
        else:
            print(f"  ✗ Parameter grid ranges incorrect")
            return False

        # Test single parameter combination with enhanced measures
        experiment = ChaosExperiment(n_neurons=20, synaptic_mode="dynamic")  # Very small for speed

        result = experiment.run_parameter_combination(
            session_id=999,
            v_th_std=0.5,
            g_std=0.5,
            v_th_distribution="normal",
            static_input_rate=200.0
        )

        # Check for enhanced fields
        expected_fields = [
            'session_id', 'v_th_std', 'g_std', 'synaptic_mode',
            'lz_matrix_flattened_values', 'lz_spatial_patterns_values',
            'pci_raw_values', 'pci_normalized_values', 'pci_with_threshold_values',
            'hamming_slopes', 'kistler_delta_2ms_values', 'gamma_window_5ms_values',
            'stable_pattern_fraction', 'control_mean_firing_rate_mean',
            'control_percent_silent_mean', 'n_trials', 'computation_time'
        ]

        missing_fields = [field for field in expected_fields if field not in result]

        if not missing_fields:
            print(f"  ✓ Enhanced chaos experiment:")
            print(f"    Session: {result['session_id']}")
            print(f"    Parameters: v_th_std={result['v_th_std']}, g_std={result['g_std']}")
            print(f"    Mode: {result['synaptic_mode']}")
            print(f"    LZ (flattened): {result['lz_matrix_flattened_mean']:.2f}")
            print(f"    LZ (spatial): {result['lz_spatial_patterns_mean']:.2f}")
            print(f"    PCI (normalized): {result['pci_normalized_mean']:.3f}")
            print(f"    Stable patterns: {result['stable_pattern_fraction']:.2f}")
            print(f"    Trials: {result['n_trials']}, Time: {result['computation_time']:.1f}s")
        else:
            print(f"  ✗ Enhanced chaos experiment missing fields: {missing_fields}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Enhanced chaos experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_integration():
    """Test complete spiking network with enhanced features."""
    print("\nTesting network integration...")

    try:
        from spiking_network import SpikingRNN

        # Create network
        network = SpikingRNN(n_neurons=50, dt=0.1, synaptic_mode="dynamic")

        # Initialize with random structure
        network.initialize_network(
            session_id=1,
            v_th_std=0.5,
            g_std=0.5,
            v_th_distribution="normal"
        )

        print("  ✓ Network initialization with random structure")

        # Test simulation
        spikes = network.simulate_network_dynamics(
            session_id=1,
            v_th_std=0.5,
            g_std=0.5,
            trial_id=1,
            duration=100.0,
            static_input_rate=200.0
        )

        print(f"  ✓ Network simulation: {len(spikes)} spikes in 100ms")

        # Get network info
        info = network.get_network_info()
        if 'spike_thresholds' in info and len(info['spike_thresholds']) > 0:
            threshold_range = np.max(info['spike_thresholds']) - np.min(info['spike_thresholds'])
            print(f"  ✓ Heterogeneity: threshold range {threshold_range:.2f} mV")
        else:
            print("  ✓ Network info available")

        return True

    except Exception as e:
        print(f"  ✗ Network integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_synaptic_modes():
    """Test both synaptic modes with enhanced analysis."""
    print("\nTesting synaptic modes...")

    try:
        from spiking_network import SpikingRNN

        # Test dynamic mode
        network_dynamic = SpikingRNN(n_neurons=20, synaptic_mode="dynamic")
        network_dynamic.initialize_network(session_id=1, v_th_std=0.5, g_std=0.5)

        # Test immediate mode
        network_immediate = SpikingRNN(n_neurons=20, synaptic_mode="immediate")
        network_immediate.initialize_network(session_id=1, v_th_std=0.5, g_std=0.5)

        # Check synaptic modes
        dynamic_info = network_dynamic.get_network_info()
        immediate_info = network_immediate.get_network_info()

        if dynamic_info['synaptic_mode'] == 'dynamic':
            print("  ✓ Dynamic synaptic mode")
        else:
            print("  ✗ Dynamic synaptic mode not set")
            return False

        if immediate_info['synaptic_mode'] == 'immediate':
            print("  ✓ Immediate synaptic mode")
        else:
            print("  ✗ Immediate synaptic mode not set")
            return False

        # Check weight statistics for normalization
        dynamic_stats = network_dynamic.synapses.get_weight_statistics()
        immediate_stats = network_immediate.synapses.get_weight_statistics()

        expected_factor = 50.0  # tau_syn/dt = 5/0.1
        actual_factor = immediate_stats.get('normalization_factor', 0)

        if abs(actual_factor - expected_factor) < 1e-6:
            print("  ✓ Impact normalization factor correct")
        else:
            print(f"  ✗ Impact normalization factor incorrect: {actual_factor} vs {expected_factor}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Synaptic modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Enhanced Spiking RNN Framework - Installation Test")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Enhanced Spike Analysis", test_enhanced_spike_analysis),
        ("Enhanced Chaos Experiment", test_chaos_experiment),
        ("Network Integration", test_network_integration),
        ("Synaptic Modes", test_synaptic_modes),
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
    print("Enhanced Installation Test Summary:")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\nEnhanced framework is working correctly!")
        print("\nNew capabilities verified:")
        print("  • 4 LZ-based complexity measures (flattened, spatial, PCI variants)")
        print("  • Kistler coincidence factor with multiple deltas")
        print("  • Enhanced gamma coincidence with multiple windows")
        print("  • Pattern stability detection")
        print("  • Multi-bin dimensionality analysis")
        print("  • Comprehensive firing rate statistics")
        print("  • Extended Poisson rate range (up to 1000 Hz)")
        print("  • Session averaging for all new measures")

        print("\nYou can now run enhanced experiments:")
        print("  ./runners/run_chaos_experiment.sh --session_ids '1 2' --n_v_th 3 --n_g 3 --input_rate_max 1000")
        return 0
    else:
        print(f"\n{total_tests - passed_tests} tests failed.")
        print("\nTroubleshooting:")
        print("  1. Check that all enhanced modules are properly updated")
        print("  2. Verify function names match new implementation")
        print("  3. Ensure MPI installation supports enhanced features")
        return 1

if __name__ == "__main__":
    exit(main())
