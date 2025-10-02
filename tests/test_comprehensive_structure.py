# tests/test_comprehensive_structure.py - Complete with pulse/filter and input mode tests
"""
Comprehensive tests to verify:
1. Pulse vs filter synapse terminology and behavior
2. Three static input modes: independent, common_stochastic, common_tonic
3. Network structure consistency across trials
4. RNG behavior for different components
5. Split analysis functionality
6. New stability measures (lz_column_wise, delta=0.1ms)
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

def test_pulse_filter_terminology():
    """Test that pulse/filter terminology is correctly implemented."""
    print("Testing pulse/filter terminology...")

    from spiking_network import SpikingRNN

    # Test pulse mode
    try:
        network_pulse = SpikingRNN(n_neurons=50, synaptic_mode="pulse")
        if network_pulse.synaptic_mode == "pulse":
            print("  ✓ Pulse synaptic mode accepted")
        else:
            print("  ✗ Pulse mode not set correctly")
            return False
    except Exception as e:
        print(f"  ✗ Pulse mode failed: {e}")
        return False

    # Test filter mode
    try:
        network_filter = SpikingRNN(n_neurons=50, synaptic_mode="filter")
        if network_filter.synaptic_mode == "filter":
            print("  ✓ Filter synaptic mode accepted")
        else:
            print("  ✗ Filter mode not set correctly")
            return False
    except Exception as e:
        print(f"  ✗ Filter mode failed: {e}")
        return False

    # Test old terminology is rejected
    try:
        network_old = SpikingRNN(n_neurons=50, synaptic_mode="immediate")
        print("  ✗ Old 'immediate' terminology still accepted (should be 'pulse')")
        return False
    except ValueError:
        print("  ✓ Old 'immediate' terminology correctly rejected")

    try:
        network_old = SpikingRNN(n_neurons=50, synaptic_mode="dynamic")
        print("  ✗ Old 'dynamic' terminology still accepted (should be 'filter')")
        return False
    except ValueError:
        print("  ✓ Old 'dynamic' terminology correctly rejected")

    return True


def test_static_input_modes():
    """Test the three static input modes."""
    print("\nTesting static input modes...")

    from spiking_network import SpikingRNN

    modes = ["independent", "common_stochastic", "common_tonic"]

    for mode in modes:
        try:
            network = SpikingRNN(n_neurons=50, synaptic_mode="filter", static_input_mode=mode)
            if network.static_input_mode == mode:
                print(f"  ✓ Static input mode '{mode}' accepted")
            else:
                print(f"  ✗ Static input mode '{mode}' not set correctly")
                return False
        except Exception as e:
            print(f"  ✗ Static input mode '{mode}' failed: {e}")
            return False

    # Test invalid mode is rejected
    try:
        network_invalid = SpikingRNN(n_neurons=50, static_input_mode="invalid_mode")
        print("  ✗ Invalid static input mode accepted")
        return False
    except ValueError:
        print("  ✓ Invalid static input mode correctly rejected")

    return True


def test_common_stochastic_input():
    """Test that common_stochastic gives identical Poisson spikes across neurons."""
    print("\nTesting common_stochastic input...")

    from synaptic_model import StaticPoissonInput, Synapse

    # Create common_stochastic input
    static_input = StaticPoissonInput(n_neurons=100, dt=0.1, static_input_mode="common_stochastic")
    static_input.initialize_parameters(input_strength=1.0)

    # Create synapse for filtering (testing through complete pathway)
    synapse = Synapse(n_neurons=100, dt=0.1, synaptic_mode="filter")

    session_id = 42
    v_th_std = 0.5
    g_std = 0.3
    trial_id = 1
    rate = 500.0

    # Collect input across neurons for multiple timesteps
    neuron_inputs = []
    for time_step in range(50):
        events = static_input.generate_events(session_id, v_th_std, g_std, trial_id, rate, time_step)
        input_current = synapse.apply_to_input(events)
        neuron_inputs.append(input_current.copy())

    neuron_inputs = np.array(neuron_inputs)  # Shape: (timesteps, neurons)

    # Check: At each timestep, all neurons should receive identical filtered input
    timesteps_with_variation = 0
    for t in range(len(neuron_inputs)):
        if not np.allclose(neuron_inputs[t], neuron_inputs[t, 0]):
            timesteps_with_variation += 1

    if timesteps_with_variation == 0:
        print("  ✓ Common stochastic: all neurons receive identical input at each timestep")
    else:
        print(f"  ✗ Common stochastic: {timesteps_with_variation} timesteps had variation across neurons")
        return False

    # Check: Different timesteps should have different patterns (stochastic)
    unique_patterns = len(set(tuple(row) for row in neuron_inputs))
    if unique_patterns > 10:
        print(f"  ✓ Common stochastic: stochastic across time ({unique_patterns} unique patterns)")
    else:
        print(f"  ✗ Common stochastic: not stochastic enough ({unique_patterns} unique patterns)")
        return False

    return True


def test_independent_stochastic_input():
    """Test that independent gives different Poisson spikes across neurons."""
    print("\nTesting independent stochastic input...")

    from synaptic_model import StaticPoissonInput, Synapse

    # Create independent input
    static_input = StaticPoissonInput(n_neurons=100, dt=0.1, static_input_mode="independent")
    static_input.initialize_parameters(input_strength=1.0)

    # Create synapse for filtering
    synapse = Synapse(n_neurons=100, dt=0.1, synaptic_mode="filter")

    session_id = 42
    v_th_std = 0.5
    g_std = 0.3
    trial_id = 1
    rate = 500.0

    # Collect input across neurons for multiple timesteps
    neuron_inputs = []
    for time_step in range(50):
        events = static_input.generate_events(session_id, v_th_std, g_std, trial_id, rate, time_step)
        input_current = synapse.apply_to_input(events)
        neuron_inputs.append(input_current.copy())

    neuron_inputs = np.array(neuron_inputs)  # Shape: (timesteps, neurons)

    # Check: Neurons should have different patterns
    neuron_patterns_different = 0
    for n1 in range(0, 100, 10):  # Sample every 10th neuron
        for n2 in range(n1 + 1, min(n1 + 10, 100)):
            if not np.array_equal(neuron_inputs[:, n1], neuron_inputs[:, n2]):
                neuron_patterns_different += 1

    if neuron_patterns_different > 0:
        print(f"  ✓ Independent: neurons have different input patterns ({neuron_patterns_different} pairs differ)")
    else:
        print("  ✗ Independent: all sampled neurons have identical patterns")
        return False

    return True


def test_common_tonic_input():
    """Test that common_tonic gives deterministic constant input."""
    print("\nTesting common_tonic input...")

    from synaptic_model import StaticPoissonInput, Synapse

    # Create common_tonic input
    static_input = StaticPoissonInput(n_neurons=100, dt=0.1, static_input_mode="common_tonic")
    static_input.initialize_parameters(input_strength=1.0)

    # Create synapse for filtering
    synapse = Synapse(n_neurons=100, dt=0.1, synaptic_mode="filter")

    session_id = 42
    v_th_std = 0.5
    g_std = 0.3
    trial_id = 1
    rate = 500.0

    # Collect input for multiple timesteps
    inputs_over_time = []
    for time_step in range(50):
        events = static_input.generate_events(session_id, v_th_std, g_std, trial_id, rate, time_step)
        input_current = synapse.apply_to_input(events)
        inputs_over_time.append(input_current.copy())

    inputs_over_time = np.array(inputs_over_time)

    # Check 1: All neurons should receive identical input at each timestep
    for t in range(len(inputs_over_time)):
        if not np.allclose(inputs_over_time[t], inputs_over_time[t, 0]):
            print(f"  ✗ Common tonic: neurons differ at timestep {t}")
            return False

    print("  ✓ Common tonic: all neurons receive identical input")

    # Check 2: With filter synapses, input should accumulate
    mean_input_last = np.mean(inputs_over_time[-10:, 0])
    if mean_input_last > 0:
        print(f"  ✓ Common tonic: non-zero accumulated input ({mean_input_last:.3f})")
    else:
        print("  ✗ Common tonic: input is zero")
        return False

    return True


def test_input_mode_trial_dependence():
    """Test that input modes are trial-dependent."""
    print("\nTesting input mode trial-dependence...")

    from synaptic_model import StaticPoissonInput, Synapse

    # Test with independent mode
    static_input = StaticPoissonInput(n_neurons=50, dt=0.1, static_input_mode="independent")
    static_input.initialize_parameters(input_strength=1.0)

    session_id = 42
    v_th_std = 0.5
    g_std = 0.3
    rate = 500.0

    # Run two different trials (without synapse for simplicity - just check events)
    events_trial1 = []
    for time_step in range(20):
        events = static_input.generate_events(session_id, v_th_std, g_std, trial_id=1,
                                             rate=rate, time_step=time_step)
        events_trial1.append(events.copy())

    events_trial2 = []
    for time_step in range(20):
        events = static_input.generate_events(session_id, v_th_std, g_std, trial_id=2,
                                             rate=rate, time_step=time_step)
        events_trial2.append(events.copy())

    events_trial1 = np.array(events_trial1)
    events_trial2 = np.array(events_trial2)

    # Trials should differ
    if not np.array_equal(events_trial1, events_trial2):
        print("  ✓ Different trials produce different input patterns")
        return True
    else:
        print("  ✗ Different trials produce identical input patterns")
        return False


def test_lz_column_wise():
    """Test the new lz_column_wise measure."""
    print("\nTesting lz_column_wise computation...")

    from stability_analysis import analyze_perturbation_response

    # Create simple spike patterns
    spikes_control = [(t*0.5, t%3) for t in range(20)]
    spikes_perturbed = [(t*0.5 + 0.1, t%3) for t in range(20)]

    result = analyze_perturbation_response(
        spikes_control, spikes_perturbed,
        num_neurons=3, perturbation_time=1.0,
        simulation_end=10.0, perturbed_neuron=0, dt=0.1
    )

    # Check lz_column_wise exists
    if 'lz_column_wise' in result:
        lz_col = result['lz_column_wise']
        print(f"  ✓ lz_column_wise computed: {lz_col}")

        # Should be a positive integer
        if lz_col > 0 and isinstance(lz_col, (int, np.integer)):
            print("  ✓ lz_column_wise has valid value")
            return True
        else:
            print(f"  ✗ lz_column_wise has invalid value: {lz_col}")
            return False
    else:
        print("  ✗ lz_column_wise not found in results")
        return False


def test_coincidence_delta_01ms():
    """Test that delta=0.1ms coincidence measure is computed."""
    print("\nTesting coincidence with delta=0.1ms...")

    from stability_analysis import average_coincidence_multi_window

    # Create spike trains
    spikes1 = [(1.0, 0), (2.0, 0), (3.0, 1)]
    spikes2 = [(1.05, 0), (2.03, 0), (3.01, 1)]

    result = average_coincidence_multi_window(
        spikes1, spikes2, num_neurons=2,
        delta_values=[0.1, 2.0, 5.0], duration=5.0
    )

    # Check all three deltas are present
    expected_keys = ['kistler_delta_0.1ms', 'kistler_delta_2.0ms', 'kistler_delta_5.0ms',
                     'gamma_window_0.1ms', 'gamma_window_2.0ms', 'gamma_window_5.0ms']

    missing_keys = [k for k in expected_keys if k not in result]

    if not missing_keys:
        print("  ✓ All coincidence measures computed (0.1ms, 2ms, 5ms)")
        print(f"    Kistler 0.1ms: {result['kistler_delta_0.1ms']:.3f}")
        print(f"    Kistler 2.0ms: {result['kistler_delta_2.0ms']:.3f}")
        return True
    else:
        print(f"  ✗ Missing coincidence measures: {missing_keys}")
        return False


def test_pulse_vs_filter_behavior():
    """Test that pulse and filter synapses behave differently."""
    print("\nTesting pulse vs filter synapse behavior...")

    from synaptic_model import Synapse

    # Create pulse synapse
    synapse_pulse = Synapse(n_neurons=10, dt=0.1, synaptic_mode="pulse")
    synapse_pulse.initialize_weights(session_id=1, v_th_std=0.5, g_std=0.5, g_mean=1.0, connection_prob=0.5)

    # Create filter synapse
    synapse_filter = Synapse(n_neurons=10, dt=0.1, synaptic_mode="filter")
    synapse_filter.initialize_weights(session_id=1, v_th_std=0.5, g_std=0.5, g_mean=1.0, connection_prob=0.5)

    # Apply spike to both
    spike_indices = [0, 1, 2]

    current_pulse1 = synapse_pulse.update(spike_indices)
    current_filter1 = synapse_filter.update(spike_indices)

    # Step 2: No spikes
    current_pulse2 = synapse_pulse.update([])
    current_filter2 = synapse_filter.update([])

    # Pulse should be zero on step 2, filter should decay
    pulse_went_to_zero = np.allclose(current_pulse2, 0.0)
    filter_decayed = np.any(current_filter2 > 0) and np.all(current_filter2 < current_filter1 + 1e-10)

    if pulse_went_to_zero:
        print("  ✓ Pulse synapses reset to zero without spikes")
    else:
        print("  ✗ Pulse synapses did not reset to zero")
        return False

    if filter_decayed:
        print("  ✓ Filter synapses decay exponentially")
    else:
        print("  ✗ Filter synapses did not decay properly")
        return False

    return True


def test_network_structure_consistency():
    """Test that network structure is identical across trials but varies with parameters."""
    print("\nTesting network structure consistency across trials...")

    from spiking_network import SpikingRNN
    from rng_utils import rng_manager

    # Reset RNG manager
    rng_manager.reset_for_testing()

    session_id = 42
    v_th_std = 1.0
    g_std = 0.5

    # Create two networks with same parameters
    network1 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="filter")
    network2 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="filter")

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
    weights1 = network1.recurrent_synapses.weight_matrix.data
    weights2 = network2.recurrent_synapses.weight_matrix.data

    if np.allclose(weights1, weights2, atol=1e-15):
        print("  ✓ Synaptic weights identical across network instances")
    else:
        print("  ✗ Synaptic weights differ across network instances")
        return False

    # Test that different parameters give different structure
    rng_manager.reset_for_testing()

    network3 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode="filter")
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

    # Test stability analysis with NEW measures
    try:
        from stability_analysis import (
            analyze_perturbation_response, unified_coincidence_factor,
            compute_shannon_entropy, find_settling_time, lempel_ziv_complexity
        )

        # Test unified coincidence
        spikes1 = [1.0, 5.0, 10.0, 15.0]
        spikes2 = [1.1, 5.2, 9.8, 15.3]
        kistler_c, gamma_c = unified_coincidence_factor(spikes1, spikes2, delta=2.0, duration=20.0)

        if not np.isnan(gamma_c):
            print("  ✓ Unified coincidence calculation works")
        else:
            print("  ✗ Unified coincidence calculation failed")
            return False

        # Test Shannon entropy
        test_seq = np.array([0, 1, 2, 0, 1, 2, 0])
        shannon_ent = compute_shannon_entropy(test_seq)
        if shannon_ent > 0:
            print(f"  ✓ Shannon entropy calculation: {shannon_ent:.3f}")
        else:
            print("  ✗ Shannon entropy calculation failed")
            return False

        # Test settling time
        symbol_seq = np.array([1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        pert_bin = 5
        settling = find_settling_time(symbol_seq, pert_bin, bin_size=1.0, min_zero_duration_ms=5.0)
        if not np.isnan(settling):
            print(f"  ✓ Settling time detection: {settling:.1f} ms")
        else:
            print("  ✗ Settling time detection failed")
            return False

    except Exception as e:
        print(f"  ✗ Stability analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def run_all_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Comprehensive Tests - Pulse/Filter and Input Modes")
    print("=" * 70)

    tests = [
        ("Pulse/Filter Terminology", test_pulse_filter_terminology),
        ("Static Input Modes", test_static_input_modes),
        ("Common Stochastic Input", test_common_stochastic_input),
        ("Independent Stochastic Input", test_independent_stochastic_input),
        ("Common Tonic Input", test_common_tonic_input),
        ("Input Mode Trial Dependence", test_input_mode_trial_dependence),
        ("LZ Column-Wise", test_lz_column_wise),
        ("Coincidence Delta 0.1ms", test_coincidence_delta_01ms),
        ("Pulse vs Filter Behavior", test_pulse_vs_filter_behavior),
        ("Network Structure Consistency", test_network_structure_consistency),
        ("Split Analysis Modules", test_split_analysis_modules),
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

    print("\n" + "=" * 70)
    print("Comprehensive Test Summary:")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:40s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nVerified capabilities:")
        print("  ✓ Pulse/filter synapse terminology")
        print("  ✓ Three static input modes (independent, common_stochastic, common_tonic)")
        print("  ✓ Input modes are trial-dependent")
        print("  ✓ LZ column-wise complexity")
        print("  ✓ Coincidence at 0.1ms, 2ms, 5ms")
        print("  ✓ Network structure consistency")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        return 1


if __name__ == "__main__":
    exit(run_all_comprehensive_tests())
