# tests/test_installation.py - Complete installation verification with original + new tests
"""
Test script to verify installation and functionality with updated features.
Quick sanity checks for basic functionality.
"""

import sys
import os
import importlib
import numpy as np

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
        ('analysis', ['spontaneous_analysis', 'stability_analysis']),
        ('experiments', ['spontaneous_experiment', 'stability_experiment'])
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

def test_spontaneous_analysis():
    """Test spontaneous activity analysis functions."""
    print("\nTesting spontaneous activity analysis...")

    try:
        from spontaneous_analysis import (
            compute_activity_dimensionality_multi_bin, analyze_firing_rates_and_silence,
            analyze_spontaneous_activity, analyze_population_poisson_properties
        )

        # Test firing rate analysis
        test_spikes = [(10.0, 0), (15.0, 1), (20.0, 0), (25.0, 2)]
        firing_stats = analyze_firing_rates_and_silence(test_spikes, num_neurons=5, duration=1000.0)

        expected_keys = ['mean_firing_rate', 'std_firing_rate', 'percent_silent', 'percent_active']
        if all(key in firing_stats for key in expected_keys):
            print(f"  ✓ Firing rate analysis:")
            print(f"    Mean rate: {firing_stats['mean_firing_rate']:.2f} Hz")
            print(f"    Silent: {firing_stats['percent_silent']:.1f}%")
        else:
            print(f"  ✗ Firing rate analysis missing keys")
            return False

        # Test multi-bin dimensionality
        test_spikes_long = [(i*10.0, i%3) for i in range(50)]
        dim_results = compute_activity_dimensionality_multi_bin(
            test_spikes_long, num_neurons=10, duration=500.0,
            bin_sizes=[0.1, 2.0, 5.0, 20.0, 50.0, 100.0]
        )

        expected_bins = ['bin_0.1ms', 'bin_2.0ms', 'bin_5.0ms', 'bin_20.0ms', 'bin_50.0ms', 'bin_100.0ms']
        if all(bin_key in dim_results for bin_key in expected_bins):
            print(f"  ✓ Multi-bin dimensionality analysis:")
            print(f"    5ms bin effective dim: {dim_results['bin_5.0ms']['effective_dimensionality']:.1f}")
            print(f"    50ms bin effective dim: {dim_results['bin_50.0ms']['effective_dimensionality']:.1f}")
        else:
            print(f"  ✗ Multi-bin dimensionality missing bins")
            return False

        # Test Poisson analysis
        poisson_results = analyze_population_poisson_properties(
            test_spikes_long, num_neurons=10, duration=500.0
        )

        if 'population_statistics' in poisson_results:
            pop_stats = poisson_results['population_statistics']
            print(f"  ✓ Poisson analysis:")
            print(f"    Mean CV ISI: {pop_stats.get('mean_cv_isi', 'N/A')}")
            print(f"    Mean Fano factor: {pop_stats.get('mean_fano_factor', 'N/A')}")
        else:
            print(f"  ✗ Poisson analysis failed")
            return False

        # Test complete spontaneous analysis
        spontaneous_results = analyze_spontaneous_activity(
            test_spikes_long, num_neurons=10, duration=500.0
        )

        required_keys = ['firing_stats', 'dimensionality_metrics', 'poisson_analysis', 'duration_ms', 'total_spikes']
        if all(key in spontaneous_results for key in required_keys):
            print(f"  ✓ Complete spontaneous analysis:")
            print(f"    Duration: {spontaneous_results['duration_ms']:.0f} ms")
            print(f"    Total spikes: {spontaneous_results['total_spikes']}")
        else:
            print(f"  ✗ Complete spontaneous analysis missing keys")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Spontaneous analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stability_analysis():
    """Test network stability analysis functions with NEW measures."""
    print("\nTesting network stability analysis (updated measures)...")

    try:
        from stability_analysis import (
            lempel_ziv_complexity, unified_coincidence_factor,
            analyze_perturbation_response, compute_shannon_entropy, find_settling_time
        )

        # Test LZ complexity
        test_sequence = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        lz_result = lempel_ziv_complexity(test_sequence)

        if lz_result > 0:
            print(f"  ✓ LZ complexity: {lz_result}")
        else:
            print(f"  ✗ LZ complexity failed: {lz_result}")
            return False

        # Test Shannon entropy
        test_seq = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        shannon_ent = compute_shannon_entropy(test_seq)

        if shannon_ent > 0:
            print(f"  ✓ Shannon entropy: {shannon_ent:.3f}")
        else:
            print(f"  ✗ Shannon entropy failed")
            return False

        # Test settling time
        symbol_seq = np.array([1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        pert_bin = 5
        settling = find_settling_time(symbol_seq, pert_bin, bin_size=1.0, min_zero_duration_ms=5.0)

        if not np.isnan(settling):
            print(f"  ✓ Settling time detection: {settling:.1f} ms")
        else:
            print(f"  ✗ Settling time detection failed")
            return False

        # Test unified coincidence calculation
        spikes1 = [1.0, 5.0, 10.0, 15.0]
        spikes2 = [1.1, 5.2, 9.8, 15.3]
        kistler_c, gamma_c = unified_coincidence_factor(spikes1, spikes2, delta=2.0, duration=20.0)

        if not np.isnan(gamma_c):
            print(f"  ✓ Unified coincidence calculation:")
            print(f"    Kistler: {kistler_c:.3f}, Gamma: {gamma_c:.3f}")
        else:
            print(f"  ✗ Unified coincidence out of range")
            return False

        # Test complete perturbation analysis (NEW measures)
        spikes_control = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2)]
        spikes_perturbed = [(1.0, 0), (2.5, 1), (3.5, 2), (4.0, 0)]

        stability_results = analyze_perturbation_response(
            spikes_control, spikes_perturbed, num_neurons=3,
            perturbation_time=1.0, simulation_end=5.0, perturbed_neuron=0,
            dt=0.1
        )

        # Updated expected keys (NEW measures)
        expected_keys = [
            'lz_spatial_patterns', 'lz_column_wise', 'shannon_entropy_symbols', 'shannon_entropy_spikes',
            'unique_patterns_count', 'settling_time_ms', 'total_spike_differences',
            'kistler_delta_0.1ms', 'kistler_delta_2.0ms', 'kistler_delta_5.0ms',
            'gamma_window_0.1ms', 'gamma_window_2.0ms', 'gamma_window_5.0ms'
        ]

        missing_keys = [key for key in expected_keys if key not in stability_results]

        if not missing_keys:
            print(f"  ✓ Complete stability analysis (updated):")
            print(f"    LZ spatial: {stability_results['lz_spatial_patterns']}")
            print(f"    LZ column-wise: {stability_results['lz_column_wise']}")
            print(f"    Shannon (symbols): {stability_results['shannon_entropy_symbols']:.3f}")
            print(f"    Shannon (spikes): {stability_results['shannon_entropy_spikes']:.3f}")
            print(f"    Settling time: {stability_results['settling_time_ms']:.1f} ms")
            print(f"    Kistler (0.1ms): {stability_results['kistler_delta_0.1ms']:.3f}")
            print(f"    Kistler (2ms): {stability_results['kistler_delta_2.0ms']:.3f}")
        else:
            print(f"  ✗ Complete stability analysis missing keys: {missing_keys}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Stability analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spontaneous_experiment():
    """Test spontaneous activity experiment."""
    print("\nTesting spontaneous activity experiment...")

    try:
        from spontaneous_experiment import SpontaneousExperiment, create_parameter_grid

        # Test parameter grid creation
        v_th_stds, g_stds, input_rates = create_parameter_grid(
            n_v_th_points=3, n_g_points=3, n_input_rates=3
        )

        if len(v_th_stds) == 3 and len(g_stds) == 3 and len(input_rates) == 3:
            print(f"  ✓ Parameter grid creation")
        else:
            print(f"  ✗ Parameter grid sizes incorrect")
            return False

        # Test single parameter combination (very small network)
        experiment = SpontaneousExperiment(n_neurons=10, synaptic_mode="filter")

        result = experiment.run_parameter_combination(
            session_id=999,
            v_th_std=0.5,
            g_std=0.5,
            v_th_distribution="normal",
            static_input_rate=200.0,
            duration=100.0
        )

        # Check for spontaneous activity fields
        expected_fields = [
            'session_id', 'v_th_std', 'g_std', 'synaptic_mode', 'static_input_mode', 'duration',
            'mean_firing_rate_values', 'percent_silent_values',
            'effective_dimensionality_bin_5.0ms_values', 'total_spikes_values',
            'mean_cv_isi_values', 'mean_fano_factor_values',
            'n_trials', 'computation_time'
        ]

        missing_fields = [field for field in expected_fields if field not in result]

        if not missing_fields:
            print(f"  ✓ Spontaneous activity experiment:")
            print(f"    Duration: {result['duration']:.0f} ms")
            print(f"    Synaptic mode: {result['synaptic_mode']}")
            print(f"    Static input mode: {result['static_input_mode']}")
            print(f"    Mean firing rate: {result['mean_firing_rate_mean']:.2f} Hz")
            print(f"    Silent neurons: {result['percent_silent_mean']:.1f}%")
            print(f"    Trials: {result['n_trials']}, Time: {result['computation_time']:.1f}s")
        else:
            print(f"  ✗ Spontaneous activity experiment missing fields: {missing_fields}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Spontaneous activity experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stability_experiment():
    """Test network stability experiment with NEW measures."""
    print("\nTesting network stability experiment (updated measures)...")

    try:
        from stability_experiment import StabilityExperiment, create_parameter_grid

        # Test parameter grid creation
        v_th_stds, g_stds, input_rates = create_parameter_grid(
            n_v_th_points=3, n_g_points=3, n_input_rates=3
        )

        if len(v_th_stds) == 3 and len(g_stds) == 3 and len(input_rates) == 3:
            print(f"  ✓ Parameter grid creation")
        else:
            print(f"  ✗ Parameter grid sizes incorrect")
            return False

        # Test single parameter combination (very small network)
        experiment = StabilityExperiment(n_neurons=10, synaptic_mode="filter")

        result = experiment.run_parameter_combination(
            session_id=999,
            v_th_std=0.5,
            g_std=0.5,
            v_th_distribution="normal",
            static_input_rate=200.0
        )

        # Check for NEW stability fields
        expected_fields = [
            'session_id', 'v_th_std', 'g_std', 'synaptic_mode', 'static_input_mode',
            'lz_spatial_patterns_values', 'lz_column_wise_values',
            'shannon_entropy_symbols_values', 'shannon_entropy_spikes_values',
            'settling_time_ms_values', 'settled_fraction',
            'kistler_delta_0.1ms_values', 'kistler_delta_2.0ms_values',
            'gamma_window_0.1ms_values', 'gamma_window_2.0ms_values',
            'n_trials', 'computation_time'
        ]

        missing_fields = [field for field in expected_fields if field not in result]

        if not missing_fields:
            print(f"  ✓ Network stability experiment (updated):")
            print(f"    Synaptic mode: {result['synaptic_mode']}")
            print(f"    Static input mode: {result['static_input_mode']}")
            print(f"    LZ spatial: {result['lz_spatial_patterns_mean']:.2f}")
            print(f"    LZ column-wise: {result['lz_column_wise_mean']:.2f}")
            print(f"    Shannon (symbols): {result['shannon_entropy_symbols_mean']:.3f}")
            print(f"    Settling time: {result.get('settling_time_ms_mean', np.nan):.1f} ms")
            print(f"    Settled fraction: {result['settled_fraction']:.2f}")
            print(f"    Kistler (0.1ms): {result['kistler_delta_0.1ms_mean']:.3f}")
            print(f"    Kistler (2ms): {result['kistler_delta_2.0ms_mean']:.3f}")
            print(f"    Trials: {result['n_trials']}, Time: {result['computation_time']:.1f}s")
        else:
            print(f"  ✗ Network stability experiment missing fields: {missing_fields}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Network stability experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pulse_filter_modes():
    """Test pulse and filter synaptic modes."""
    print("\nTesting pulse and filter synaptic modes...")

    try:
        from spiking_network import SpikingRNN

        # Test pulse mode
        network_pulse = SpikingRNN(n_neurons=50, synaptic_mode="pulse", static_input_mode="independent")
        if network_pulse.synaptic_mode == "pulse":
            print("  ✓ Pulse synapse mode works")
        else:
            print("  ✗ Pulse mode not set correctly")
            return False

        # Test filter mode
        network_filter = SpikingRNN(n_neurons=50, synaptic_mode="filter", static_input_mode="independent")
        if network_filter.synaptic_mode == "filter":
            print("  ✓ Filter synapse mode works")
        else:
            print("  ✗ Filter mode not set correctly")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Pulse/filter modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_static_input_modes():
    """Test all three static input modes."""
    print("\nTesting static input modes...")

    try:
        from spiking_network import SpikingRNN

        modes = ["independent", "common_stochastic", "common_tonic"]

        for mode in modes:
            network = SpikingRNN(n_neurons=50, synaptic_mode="filter", static_input_mode=mode)
            if network.static_input_mode == mode:
                print(f"  ✓ Static input mode '{mode}' works")
            else:
                print(f"  ✗ Static input mode '{mode}' not set correctly")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Static input modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Installation Test - Complete Verification")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Spontaneous Activity Analysis", test_spontaneous_analysis),
        ("Network Stability Analysis (Updated)", test_stability_analysis),
        ("Spontaneous Activity Experiment", test_spontaneous_experiment),
        ("Network Stability Experiment (Updated)", test_stability_experiment),
        ("Pulse/Filter Modes", test_pulse_filter_modes),
        ("Static Input Modes", test_static_input_modes),
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
    print("Installation Test Summary:")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:45s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\nInstallation successful with all features!")
        print("\nVerified features:")
        print("  - Core module imports")
        print("  - Spontaneous activity analysis (6 bin sizes, Poisson tests)")
        print("  - Stability analysis (LZ column-wise, Shannon, settling, 0.1ms coincidence)")
        print("  - Pulse and filter synaptic modes")
        print("  - Three static input modes (independent, common_stochastic, common_tonic)")
        print("  - Both experiment types with all new measures")
        print("\nReady to run experiments:")
        print("  ./runners/run_spontaneous_experiment.sh --synaptic_mode filter --static_input_mode independent")
        print("  ./runners/run_stability_experiment.sh --synaptic_mode pulse --static_input_mode common_tonic")
        return 0
    else:
        print(f"\n{total_tests - passed_tests} tests failed.")
        print("Please check implementation of failed components.")
        return 1

if __name__ == "__main__":
    exit(main())
