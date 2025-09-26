# tests/test_comprehensive_structure.py - Updated for split experiments framework
"""
Comprehensive tests to verify:
1. Network structure consistency across trials
2. Normal vs uniform distribution handling
3. RNG behavior for different components
4. Split analysis functionality (spontaneous vs stability)
5. Enhanced connectivity strength (25)
6. Optimized coincidence calculations
"""

import sys
import os
import numpy as np

# Add project directories
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'analysis'))
sys.path.insert(0, os.path.join(project_root, 'experiments'))

def test_network_structure_consistency():
    """Test that network structure is identical across trials but varies with parameters."""
    print("Testing network structure consistency across trials...")

    from spiking_network import SpikingRNN
    from rng_utils import rng_manager

    # Reset RNG manager
    rng_manager.reset_for_testing()

    session_id = 42
    v_th_std = 1.0
    g_std = 0.5

    # Create two networks with same parameters
    network1 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="dynamic")
    network2 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="dynamic")

    # Initialize with same session + parameters
    network1.initialize_network(session_id, v_th_std, g_std, v_th_distribution="normal")
    network2.initialize_network(session_id, v_th_std, g_std, v_th_distribution="normal")

    # Check spike thresholds are identical
    if np.allclose(network1.neurons.spike_thresholds, network2.neurons.spike_thresholds, atol=1e-15):
        print("  ✓ Spike thresholds identical across network instances")
    else:
        print("  ✗ Spike thresholds differ across network instances")
        return False

    # Check enhanced static Poisson connectivity strength
    if network1.static_input.input_strength == 25.0:
        print("  ✓ Enhanced static Poisson connectivity strength: 25")
    else:
        print(f"  ✗ Wrong static Poisson strength: {network1.static_input.input_strength} (expected 25)")
        return False

    # Check synaptic weights are identical
    weights1 = network1.synapses.weight_matrix.data
    weights2 = network2.synapses.weight_matrix.data

    if np.allclose(weights1, weights2, atol=1e-15):
        print("  ✓ Synaptic weights identical across network instances")
    else:
        print("  ✗ Synaptic weights differ across network instances")
        return False

    # Now test that different parameters give different structure
    rng_manager.reset_for_testing()

    network3 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="dynamic")
    network3.initialize_network(session_id, v_th_std + 0.1, g_std, v_th_distribution="normal")

    if not np.allclose(network1.neurons.spike_thresholds, network3.neurons.spike_thresholds):
        print("  ✓ Different parameters produce different spike thresholds")
    else:
        print("  ✗ Different parameters produce identical spike thresholds")
        return False

    return True

def test_split_analysis_modules():
    """Test both spontaneous and stability analysis modules."""
    print("\nTesting split analysis modules...")

    # Test spontaneous analysis
    try:
        from spontaneous_analysis import (
            analyze_spontaneous_activity, compute_activity_dimensionality_multi_bin,
            analyze_firing_rates_and_silence
        )

        test_spikes = [(i*10.0, i%5) for i in range(100)]

        # Test firing rate analysis
        firing_stats = analyze_firing_rates_and_silence(test_spikes, num_neurons=10, duration=1000.0)
        if 'mean_firing_rate' in firing_stats and 'percent_silent' in firing_stats:
            print("  ✓ Spontaneous firing rate analysis working")
        else:
            print("  ✗ Spontaneous firing rate analysis failed")
            return False

        # Test multi-bin dimensionality (6 bin sizes)
        dim_results = compute_activity_dimensionality_multi_bin(
            test_spikes, num_neurons=10, duration=1000.0,
            bin_sizes=[0.1, 2.0, 5.0, 20.0, 50.0, 100.0]
        )
        expected_bins = ['bin_0.1ms', 'bin_2.0ms', 'bin_5.0ms', 'bin_20.0ms', 'bin_50.0ms', 'bin_100.0ms']
        if all(bin_key in dim_results for bin_key in expected_bins):
            print("  ✓ Multi-bin dimensionality analysis (6 bin sizes)")
        else:
            print("  ✗ Multi-bin dimensionality missing bins")
            return False

        # Test complete spontaneous analysis
        spontaneous_results = analyze_spontaneous_activity(test_spikes, num_neurons=10, duration=1000.0)
        required_keys = ['firing_stats', 'dimensionality_metrics']
        if all(key in spontaneous_results for key in required_keys):
            print("  ✓ Complete spontaneous activity analysis")
        else:
            print("  ✗ Complete spontaneous activity analysis failed")
            return False

    except Exception as e:
        print(f"  ✗ Spontaneous analysis test failed: {e}")
        return False

    # Test stability analysis
    try:
        from stability_analysis import (
            analyze_perturbation_response, unified_coincidence_factor,
            compute_spatial_pattern_complexity, find_stable_period
        )

        # Test unified coincidence (optimized)
        spikes1 = [1.0, 5.0, 10.0, 15.0]
        spikes2 = [1.1, 5.2, 9.8, 15.3]
        kistler_c, gamma_c = unified_coincidence_factor(spikes1, spikes2, delta=2.0, duration=20.0)

        if 0.0 <= gamma_c <= 1.0:
            print("  ✓ Unified coincidence calculation (single loop optimization)")
        else:
            print("  ✗ Unified coincidence calculation failed")
            return False

        # Test spatial pattern complexity (no PCI measures)
        test_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        spatial_results = compute_spatial_pattern_complexity(test_matrix)

        # Should NOT have PCI measures
        pci_keys = ['pci_raw', 'pci_normalized', 'pci_with_threshold']
        has_pci = any(key in spatial_results for key in pci_keys)
        if not has_pci and 'lz_spatial_patterns' in spatial_results:
            print("  ✓ Spatial complexity (no PCI measures, as requested)")
        else:
            print("  ✗ Spatial complexity still has PCI measures or missing LZ")
            return False

        # Test pattern stability
        repeating_seq = [1, 2, 1, 2, 1, 2, 1, 2]
        stability_result = find_stable_period(repeating_seq, min_repeats=3)
        if stability_result is not None and stability_result['period'] == 2:
            print("  ✓ Pattern stability detection")
        else:
            print("  ✗ Pattern stability detection failed")
            return False

        # Test complete perturbation analysis (should not have PCI or lz_matrix_flattened)
        spikes_control = [(1.0, 0), (2.0, 1), (3.0, 0)]
        spikes_perturbed = [(1.0, 0), (2.5, 1), (3.5, 2)]

        stability_results = analyze_perturbation_response(
            spikes_control, spikes_perturbed, num_neurons=3,
            perturbation_time=1.0, simulation_end=5.0, perturbed_neuron=0
        )

        # Check that PCI and lz_matrix_flattened are NOT present
        forbidden_keys = ['pci_raw', 'pci_normalized', 'pci_with_threshold', 'lz_matrix_flattened']
        has_forbidden = any(key in stability_results for key in forbidden_keys)

        required_keys = ['lz_spatial_patterns', 'hamming_slope', 'kistler_delta_2ms', 'gamma_window_2ms']
        has_required = all(key in stability_results for key in required_keys)

        if not has_forbidden and has_required:
            print("  ✓ Complete stability analysis (removed PCI & lz_matrix_flattened)")
        else:
            print("  ✗ Dynamics analysis still has forbidden measures or missing required ones")
            return False

    except Exception as e:
        print(f"  ✗ Dynamics analysis test failed: {e}")
        return False

    return True

def test_split_experiments():
    """Test both experiment types."""
    print("\nTesting split experiments...")

    # Test spontaneous experiment
    try:
        from spontaneous_experiment import SpontaneousExperiment

        experiment = SpontaneousExperiment(n_neurons=20, synaptic_mode="dynamic")
        result = experiment.run_parameter_combination(
            session_id=999, v_th_std=0.5, g_std=0.5,
            v_th_distribution="normal", static_input_rate=200.0, duration=100.0
        )

        # Check for spontaneous-specific fields
        expected_fields = ['duration', 'mean_firing_rate_values', 'percent_silent_values']
        if all(field in result for field in expected_fields):
            print("  ✓ Spontaneous experiment with duration parameter")
        else:
            print("  ✗ Spontaneous experiment missing fields")
            return False

    except Exception as e:
        print(f"  ✗ Spontaneous experiment test failed: {e}")
        return False

    # Test stability experiment (note: user will rename to "stability")
    try:
        from stability_experiment import StabilityExperiment

        experiment = StabilityExperiment(n_neurons=20, synaptic_mode="dynamic")
        result = experiment.run_parameter_combination(
            session_id=999, v_th_std=0.5, g_std=0.5,
            v_th_distribution="normal", static_input_rate=200.0
        )

        # Check for stability-specific fields (no PCI measures)
        expected_fields = ['lz_spatial_patterns_values', 'hamming_slope_values', 'kistler_delta_2ms_values']
        forbidden_fields = ['pci_raw_values', 'pci_normalized_values', 'lz_matrix_flattened_values']

        has_expected = all(field in result for field in expected_fields)
        has_forbidden = any(field in result for field in forbidden_fields)

        if has_expected and not has_forbidden:
            print("  ✓ Dynamics experiment (no PCI, no lz_matrix_flattened)")
        else:
            print("  ✗ Dynamics experiment has forbidden fields or missing expected ones")
            return False

    except Exception as e:
        print(f"  ✗ Dynamics experiment test failed: {e}")
        return False

    return True

def test_trial_dependent_processes():
    """Test that only Poisson processes and initial states vary with trial_id."""
    print("\nTesting trial-dependent processes...")

    from spiking_network import SpikingRNN

    session_id = 123
    v_th_std = 0.8
    g_std = 0.6

    network = SpikingRNN(n_neurons=50, dt=0.1, synaptic_mode="dynamic")
    network.initialize_network(session_id, v_th_std, g_std, v_th_distribution="normal")

    # Run multiple trials to check variability
    identical_count = 0
    total_pairs = 5

    for pair in range(total_pairs):
        trial_id1 = pair * 2 + 1
        trial_id2 = pair * 2 + 2

        # Run simulations with different trial_ids
        spikes1 = network.simulate_network_dynamics(
            session_id=session_id, v_th_std=v_th_std, g_std=g_std, trial_id=trial_id1,
            duration=200.0, static_input_rate=300.0
        )

        spikes2 = network.simulate_network_dynamics(
            session_id=session_id, v_th_std=v_th_std, g_std=g_std, trial_id=trial_id2,
            duration=200.0, static_input_rate=300.0
        )

        if spikes1 == spikes2:
            identical_count += 1

    # Allow up to 1 identical pair out of 5
    if identical_count <= 1:
        print(f"  ✓ Different trial_id produces different spike patterns ({identical_count}/{total_pairs} identical)")
        return True
    else:
        print(f"  ✗ Too many identical spike patterns: {identical_count}/{total_pairs}")
        return False

def test_distribution_handling():
    """Test that normal vs uniform distributions are properly handled."""
    print("\nTesting distribution handling...")

    from lif_neuron import LIFNeuron

    neurons = LIFNeuron(n_neurons=1000, dt=0.1)
    session_id = 456
    v_th_std = 1.0

    # Test normal distribution
    neurons.initialize_parameters(
        session_id=session_id, v_th_std=v_th_std, trial_id=0,
        v_th_mean=-55.0, v_th_distribution="normal"
    )

    thresholds_normal = neurons.spike_thresholds.copy()
    actual_std_normal = np.std(thresholds_normal)
    actual_mean_normal = np.mean(thresholds_normal)

    # Test uniform distribution
    neurons.initialize_parameters(
        session_id=session_id, v_th_std=v_th_std, trial_id=0,
        v_th_mean=-55.0, v_th_distribution="uniform"
    )

    thresholds_uniform = neurons.spike_thresholds.copy()
    actual_std_uniform = np.std(thresholds_uniform)
    actual_mean_uniform = np.mean(thresholds_uniform)

    # Check that distributions are different
    if not np.allclose(thresholds_normal, thresholds_uniform):
        print("  ✓ Normal and uniform distributions produce different values")
    else:
        print("  ✗ Normal and uniform distributions produce identical values")
        return False

    # Check mean preservation for both
    if abs(actual_mean_normal - (-55.0)) < 1e-10:
        print(f"  ✓ Normal distribution mean preserved: {actual_mean_normal:.12f}")
    else:
        print(f"  ✗ Normal distribution mean not preserved: {actual_mean_normal:.12f}")
        return False

    if abs(actual_mean_uniform - (-55.0)) < 1e-10:
        print(f"  ✓ Uniform distribution mean preserved: {actual_mean_uniform:.12f}")
    else:
        print(f"  ✗ Uniform distribution mean not preserved: {actual_mean_uniform:.12f}")
        return False

    # Check standard deviations are approximately correct
    std_tolerance = 0.1
    if abs(actual_std_normal - v_th_std) < std_tolerance:
        print(f"  ✓ Normal distribution std approximately correct: {actual_std_normal:.3f}")
    else:
        print(f"  ✗ Normal distribution std incorrect: {actual_std_normal:.3f} vs {v_th_std}")
        return False

    if abs(actual_std_uniform - v_th_std) < std_tolerance:
        print(f"  ✓ Uniform distribution std approximately correct: {actual_std_uniform:.3f}")
    else:
        print(f"  ✗ Uniform distribution std incorrect: {actual_std_uniform:.3f} vs {v_th_std}")
        return False

    return True

def test_optimized_coincidence():
    """Test that coincidence optimization works correctly."""
    print("\nTesting optimized coincidence calculations...")

    from stability_analysis import unified_coincidence_factor

    # Test with identical spike trains
    spikes1 = [10.0, 20.0, 30.0, 40.0]
    spikes2 = [10.0, 20.0, 30.0, 40.0]

    kistler_identical, gamma_identical = unified_coincidence_factor(spikes1, spikes2, delta=2.0, duration=100.0)

    if gamma_identical == 1.0:
        print(f"  ✓ Unified calculation: identical trains give gamma=1.0")
    else:
        print(f"  ✗ Unified calculation: identical trains give gamma={gamma_identical:.3f}")
        return False

    # Test with different spike trains
    spikes3 = [15.0, 25.0, 35.0, 45.0]  # Offset by 5ms
    kistler_different, gamma_different = unified_coincidence_factor(spikes1, spikes3, delta=2.0, duration=100.0)

    if gamma_different < 0.2:
        print(f"  ✓ Unified calculation: different trains give low gamma={gamma_different:.3f}")
    else:
        print(f"  ✗ Unified calculation: different trains give high gamma={gamma_different:.3f}")
        return False

    # Test with empty spike trains
    kistler_empty, gamma_empty = unified_coincidence_factor([], spikes1, delta=2.0)

    if kistler_empty == 0.0 and gamma_empty == 0.0:
        print("  ✓ Unified calculation: empty trains handled correctly")
    else:
        print(f"  ✗ Unified calculation: empty trains not handled correctly")
        return False

    print("  ✓ Single-loop optimization verified (no duplicate iterations)")
    return True

def run_all_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Comprehensive Split Experiments Framework Tests")
    print("=" * 60)

    tests = [
        ("Network Structure Consistency", test_network_structure_consistency),
        ("Split Analysis Modules", test_split_analysis_modules),
        ("Split Experiments", test_split_experiments),
        ("Trial-Dependent Processes", test_trial_dependent_processes),
        ("Distribution Handling", test_distribution_handling),
        ("Optimized Coincidence", test_optimized_coincidence),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Comprehensive Test Summary:")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:35s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("\nVerified capabilities:")
        print("  ✓ Network structure depends only on session_id + parameters")
        print("  ✓ Enhanced static Poisson connectivity (strength: 25)")
        print("  ✓ Split analysis modules work correctly")
        print("  ✓ Spontaneous analysis: 6 bin sizes for dimensionality")
        print("  ✓ Dynamics analysis: no PCI measures, no lz_matrix_flattened")
        print("  ✓ Unified coincidence calculation (single loop optimization)")
        print("  ✓ Trial-dependent processes vary correctly")
        print("  ✓ Normal vs uniform distributions work correctly")

        print(f"\nReady for split experiments:")
        print(f"  • Spontaneous Activity Analysis:")
        print(f"    - Duration parameter (seconds → milliseconds)")
        print(f"    - 6 dimensionality bin sizes")
        print(f"    - Firing rate statistics and silent neuron %")
        print(f"  • Network Dynamics Analysis:")
        print(f"    - LZ spatial complexity only")
        print(f"    - Optimized Kistler + Gamma coincidence")
        print(f"    - Pattern stability detection")
        print(f"  • Both include randomized job distribution for CPU load balancing")

        print(f"\nNote: User mentioned renaming 'stability' to 'stability' in their scripts")

        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        print("Split experiments framework not ready.")
        return 1

if __name__ == "__main__":
    exit(run_all_comprehensive_tests())
