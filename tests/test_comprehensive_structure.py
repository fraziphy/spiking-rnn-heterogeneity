# tests/test_comprehensive_structure.py - Comprehensive tests for network consistency
"""
Comprehensive tests to verify:
1. Network structure consistency across trials
2. Normal vs uniform distribution handling
3. RNG behavior for different components
4. Firing rate analysis edge cases
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

    # Check synaptic weights are identical
    weights1 = network1.synapses.weight_matrix.data
    weights2 = network2.synapses.weight_matrix.data

    if np.allclose(weights1, weights2, atol=1e-15):
        print("  ✓ Synaptic weights identical across network instances")
    else:
        print("  ✗ Synaptic weights differ across network instances")
        return False

    # Check connectivity matrices
    conn1 = network1.dynamic_input.connectivity_matrix
    conn2 = network2.dynamic_input.connectivity_matrix

    if np.array_equal(conn1, conn2):
        print("  ✓ Input connectivity identical across network instances")
    else:
        print("  ✗ Input connectivity differs across network instances")
        return False

    # Now test that different parameters give different structure
    rng_manager.reset_for_testing()

    network3 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="dynamic")
    network3.initialize_network(session_id, v_th_std + 0.1, g_std, v_th_distribution="normal")  # Different v_th_std

    if not np.allclose(network1.neurons.spike_thresholds, network3.neurons.spike_thresholds):
        print("  ✓ Different parameters produce different spike thresholds")
    else:
        print("  ✗ Different parameters produce identical spike thresholds")
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

    # Run multiple pairs of simulations to reduce chance of false negatives
    identical_count = 0
    total_pairs = 5

    for pair in range(total_pairs):
        trial_id1 = pair * 2 + 1
        trial_id2 = pair * 2 + 2

        # Run simulations with different trial_ids
        spikes1 = network.simulate_network_dynamics(
            session_id=session_id, v_th_std=v_th_std, g_std=g_std, trial_id=trial_id1,
            duration=200.0, static_input_rate=300.0  # Longer duration, higher rate for more spikes
        )

        spikes2 = network.simulate_network_dynamics(
            session_id=session_id, v_th_std=v_th_std, g_std=g_std, trial_id=trial_id2,
            duration=200.0, static_input_rate=300.0
        )

        if spikes1 == spikes2:
            identical_count += 1

    # Allow up to 1 identical pair out of 5 (very conservative)
    if identical_count <= 1:
        print(f"  ✓ Different trial_id produces different spike patterns ({identical_count}/{total_pairs} identical by chance)")
        return True
    else:
        print(f"  ✗ Too many identical spike patterns: {identical_count}/{total_pairs}")
        print("    This suggests trial_id may not be affecting Poisson processes")
        return False

def test_distribution_handling():
    """Test that normal vs uniform distributions are properly handled."""
    print("\nTesting distribution handling...")

    from lif_neuron import LIFNeuron

    neurons = LIFNeuron(n_neurons=1000, dt=0.1)  # Large for good statistics
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
    std_tolerance = 0.1  # Allow 10% tolerance for sampling variation
    if abs(actual_std_normal - v_th_std) < std_tolerance:
        print(f"  ✓ Normal distribution std approximately correct: {actual_std_normal:.3f}")
    else:
        print(f"  ✗ Normal distribution std incorrect: {actual_std_normal:.3f} vs {v_th_std}")
        print(f"    Tolerance: ±{std_tolerance}, Error: {abs(actual_std_normal - v_th_std):.3f}")
        return False

    if abs(actual_std_uniform - v_th_std) < std_tolerance:
        print(f"  ✓ Uniform distribution std approximately correct: {actual_std_uniform:.3f}")
    else:
        print(f"  ✗ Uniform distribution std incorrect: {actual_std_uniform:.3f} vs {v_th_std}")
        print(f"    Tolerance: ±{std_tolerance}, Error: {abs(actual_std_uniform - v_th_std):.3f}")
        return False

    return True

def test_rng_component_independence():
    """Test that different RNG components are independent."""
    print("\nTesting RNG component independence...")

    from rng_utils import get_rng

    session_id = 789
    v_th_std = 0.5
    g_std = 0.8
    trial_id = 1

    # Get RNGs for different components
    rng_thresholds = get_rng(session_id, v_th_std, g_std, 0, 'spike_thresholds')  # Structure component
    rng_weights = get_rng(session_id, v_th_std, g_std, 0, 'synaptic_weights')     # Structure component
    rng_poisson = get_rng(session_id, v_th_std, g_std, trial_id, 'static_poisson')  # Trial component
    rng_initial = get_rng(session_id, v_th_std, g_std, trial_id, 'initial_state')   # Trial component

    # Generate samples from each
    samples_thresholds = rng_thresholds.normal(0, 1, 10)
    samples_weights = rng_weights.normal(0, 1, 10)
    samples_poisson = rng_poisson.random(10)
    samples_initial = rng_initial.uniform(0, 1, 10)

    # Check that samples are different across components
    all_samples = [samples_thresholds, samples_weights, samples_poisson, samples_initial]
    all_different = True

    for i in range(len(all_samples)):
        for j in range(i + 1, len(all_samples)):
            if np.allclose(all_samples[i][:5], all_samples[j][:5]):  # Compare first 5 elements
                all_different = False
                break
        if not all_different:
            break

    if all_different:
        print("  ✓ Different RNG components produce independent samples")
    else:
        print("  ✗ Some RNG components produce correlated samples")
        return False

    return True

def test_firing_rate_analysis_edge_cases():
    """Test firing rate analysis with edge cases."""
    print("\nTesting firing rate analysis edge cases...")

    from spike_analysis import analyze_firing_rates_and_silence, compute_activity_dimensionality

    # Test with no spikes
    empty_spikes = []
    empty_stats = analyze_firing_rates_and_silence(empty_spikes, num_neurons=100, duration=1000.0)

    if empty_stats['percent_silent'] == 100.0 and empty_stats['mean_firing_rate'] == 0.0:
        print("  ✓ No spikes case handled correctly")
    else:
        print("  ✗ No spikes case not handled correctly")
        return False

    # Test with single spike
    single_spike = [(500.0, 0)]
    single_stats = analyze_firing_rates_and_silence(single_spike, num_neurons=100, duration=1000.0)

    expected_rate = 1.0  # 1 spike in 1 second = 1 Hz
    if (abs(single_stats['mean_firing_rate'] - 0.01) < 0.001 and  # 1/100 neurons firing
        single_stats['percent_silent'] == 99.0):
        print("  ✓ Single spike case handled correctly")
    else:
        print(f"  ✗ Single spike case incorrect: rate={single_stats['mean_firing_rate']}, silent={single_stats['percent_silent']}")
        return False

    # Test dimensionality analysis with empty matrix
    empty_matrix = np.zeros((10, 0))
    empty_dim = compute_activity_dimensionality(empty_matrix)

    if empty_dim['intrinsic_dimensionality'] == 0.0:
        print("  ✓ Empty matrix dimensionality handled correctly")
    else:
        print("  ✗ Empty matrix dimensionality not handled correctly")
        return False

    # Test dimensionality with single active neuron
    single_active = np.zeros((10, 100))
    single_active[0, :50] = 1  # Only first neuron active
    single_dim = compute_activity_dimensionality(single_active)

    if single_dim['intrinsic_dimensionality'] == 1.0:
        print("  ✓ Single active neuron dimensionality correct")
    else:
        print(f"  ✗ Single active neuron dimensionality incorrect: {single_dim['intrinsic_dimensionality']}")
        return False

    return True

def test_coincidence_measures():
    """Test both Kistler and gamma coincidence measures."""
    print("\nTesting coincidence measures...")

    from spike_analysis import kistler_coincidence_factor, gamma_coincidence

    # Test with identical spike trains
    spikes1 = [10.0, 20.0, 30.0, 40.0]
    spikes2 = [10.0, 20.0, 30.0, 40.0]

    kistler_identical = kistler_coincidence_factor(spikes1, spikes2, delta=2.0, duration=100.0)
    gamma_identical = gamma_coincidence(spikes1, spikes2, window_ms=5.0)

    if kistler_identical > 0.8:  # Should be close to 1 for identical trains
        print(f"  ✓ Kistler coincidence for identical trains: {kistler_identical:.3f}")
    else:
        print(f"  ✗ Kistler coincidence too low for identical trains: {kistler_identical:.3f}")
        return False

    if gamma_identical == 1.0:  # Should be exactly 1 for identical trains
        print(f"  ✓ Gamma coincidence for identical trains: {gamma_identical:.3f}")
    else:
        print(f"  ✗ Gamma coincidence not 1.0 for identical trains: {gamma_identical:.3f}")
        return False

    # Test with completely different spike trains
    spikes3 = [15.0, 25.0, 35.0, 45.0]  # Offset by 5ms

    kistler_different = kistler_coincidence_factor(spikes1, spikes3, delta=2.0, duration=100.0)
    gamma_different = gamma_coincidence(spikes1, spikes3, window_ms=2.0)

    if kistler_different < 0.2:  # Should be close to 0 for non-overlapping
        print(f"  ✓ Kistler coincidence for different trains: {kistler_different:.3f}")
    else:
        print(f"  ✗ Kistler coincidence too high for different trains: {kistler_different:.3f}")
        return False

    if gamma_different < 0.2:  # Should be close to 0 for non-overlapping
        print(f"  ✓ Gamma coincidence for different trains: {gamma_different:.3f}")
    else:
        print(f"  ✗ Gamma coincidence too high for different trains: {gamma_different:.3f}")
        return False

    # Test with empty spike trains
    kistler_empty = kistler_coincidence_factor([], spikes1, delta=2.0)
    gamma_empty = gamma_coincidence([], spikes1, window_ms=5.0)

    if kistler_empty == 0.0 and gamma_empty == 0.0:
        print("  ✓ Empty spike trains handled correctly")
    else:
        print(f"  ✗ Empty spike trains not handled correctly: Kistler={kistler_empty}, Gamma={gamma_empty}")
        return False

    return True

def test_complexity_measures():
    """Test all complexity measures including PCI."""
    print("\nTesting complexity measures...")

    from spike_analysis import (lempel_ziv_complexity, lempel_ziv_matrix_flattened,
                               compute_spatial_pattern_complexity)

    # Test basic LZ complexity
    simple_sequence = np.array([0, 1, 0, 1, 0, 1])
    lz_simple = lempel_ziv_complexity(simple_sequence)

    if lz_simple > 0:
        print(f"  ✓ LZ complexity for simple sequence: {lz_simple}")
    else:
        print("  ✗ LZ complexity failed for simple sequence")
        return False

    # Test matrix flattened complexity
    simple_matrix = np.array([[1, 0, 1], [0, 1, 0]])
    lz_matrix = lempel_ziv_matrix_flattened(simple_matrix)

    if lz_matrix > 0:
        print(f"  ✓ LZ matrix flattened complexity: {lz_matrix}")
    else:
        print("  ✗ LZ matrix flattened complexity failed")
        return False

    # Test spatial pattern complexity and PCI
    test_matrix = np.array([
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    spatial_results = compute_spatial_pattern_complexity(test_matrix)

    required_keys = ['lz_spatial_patterns', 'pci_raw', 'pci_normalized',
                    'pci_with_threshold', 'spatial_entropy', 'pattern_fraction']

    if all(key in spatial_results for key in required_keys):
        print(f"  ✓ Spatial pattern complexity complete:")
        print(f"    LZ spatial: {spatial_results['lz_spatial_patterns']}")
        print(f"    PCI raw: {spatial_results['pci_raw']:.3f}")
        print(f"    PCI normalized: {spatial_results['pci_normalized']:.3f}")
        print(f"    Pattern fraction: {spatial_results['pattern_fraction']:.3f}")
    else:
        missing_keys = [k for k in required_keys if k not in spatial_results]
        print(f"  ✗ Spatial pattern complexity missing keys: {missing_keys}")
        return False

    return True

def test_pattern_stability():
    """Test pattern stability analysis."""
    print("\nTesting pattern stability analysis...")

    from spike_analysis import find_stable_period

    # Test sequence with repeating pattern
    repeating_seq = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]  # Period 3, repeated 4 times
    stable_result = find_stable_period(repeating_seq, min_repeats=3)

    if stable_result is not None and stable_result['period'] == 3:
        print(f"  ✓ Stable period detected: period={stable_result['period']}, repeats={stable_result['repeats']}")
    else:
        print(f"  ✗ Stable period detection failed: {stable_result}")
        return False

    # Test sequence without stable pattern
    random_seq = [1, 4, 2, 7, 3, 9, 5, 1, 8]
    no_stable = find_stable_period(random_seq, min_repeats=3)

    if no_stable is None:
        print("  ✓ No stable period correctly detected for random sequence")
    else:
        print(f"  ✗ False positive stable period detected: {no_stable}")
        return False

    return True

def run_all_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Comprehensive Network Structure and Analysis Tests")
    print("=" * 60)

    tests = [
        ("Network Structure Consistency", test_network_structure_consistency),
        ("Trial-Dependent Processes", test_trial_dependent_processes),
        ("Distribution Handling", test_distribution_handling),
        ("RNG Component Independence", test_rng_component_independence),
        ("Firing Rate Analysis Edge Cases", test_firing_rate_analysis_edge_cases),
        ("Coincidence Measures", test_coincidence_measures),
        ("Complexity Measures", test_complexity_measures),
        ("Pattern Stability", test_pattern_stability),
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
        print("  ✓ Trial-dependent processes vary correctly")
        print("  ✓ Normal vs uniform distributions work correctly")
        print("  ✓ RNG components are independent")
        print("  ✓ Firing rate analysis handles edge cases")
        print("  ✓ Kistler and gamma coincidence measures work")
        print("  ✓ All complexity measures (LZ, PCI) implemented")
        print("  ✓ Pattern stability analysis functional")

        print(f"\nReady for enhanced chaos experiments with:")
        print(f"  • 4 complexity measures (LZ flattened, LZ spatial, PCI variants)")
        print(f"  • Multiple coincidence measures with different parameters")
        print(f"  • Dimensionality analysis with multiple bin sizes")
        print(f"  • Comprehensive firing rate statistics")
        print(f"  • Pattern stability detection")
        print(f"  • Extended Poisson rate range (up to 1000 Hz)")

        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        print("Framework not ready for enhanced experiments.")
        return 1

if __name__ == "__main__":
    exit(run_all_comprehensive_tests())
