# tests/test_comprehensive_structure.py
"""
Comprehensive tests to verify:
1. Pulse vs filter synapse terminology and behavior
2. Three static input modes: independent, common_stochastic, common_tonic
3. Three HD input modes: independent, common_stochastic, common_tonic
4. Network structure consistency across trials
5. RNG behavior for different components
6. Split analysis functionality
7. New stability measures (lz_column_wise, delta=0.1ms)
8. Refactored structure with common_utils
9. Base experiment class functionality
"""

import sys
import os
import numpy as np

# Add project directories
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def test_dimensionality_svd():
    """Test the new SVD-based dimensionality computation."""
    print("\nTesting dimensionality computation with SVD...")

    from analysis.common_utils import compute_dimensionality_svd

    # Create test data
    data = np.random.randn(100, 50)  # 100 timebins, 50 neurons

    dim_metrics = compute_dimensionality_svd(data, variance_threshold=0.95)

    # Check required fields
    required_keys = ['participation_ratio', 'effective_dimensionality',
                     'intrinsic_dimensionality', 'total_variance', 'n_components']

    missing_keys = [k for k in required_keys if k not in dim_metrics]
    if not missing_keys:
        print(f"  ✓ All required fields present")
        print(f"    Participation ratio: {dim_metrics['participation_ratio']:.2f}")
        print(f"    Effective dim: {dim_metrics['effective_dimensionality']:.2f}")
        print(f"    Intrinsic dim: {dim_metrics['intrinsic_dimensionality']}")
        print(f"    Total variance: {dim_metrics['total_variance']:.2f}")
    else:
        print(f"  ✗ Missing fields: {missing_keys}")
        return False

    # Test with empty data
    empty_data = np.zeros((1, 50))
    dim_empty = compute_dimensionality_svd(empty_data)
    if dim_empty['participation_ratio'] == 0.0:
        print(f"  ✓ Handles empty data correctly")
    else:
        print(f"  ✗ Empty data handling failed")
        return False

    return True

# Update the tests list in run_all_comprehensive_tests() (around line 425):
# Add after ("Coincidence Delta 0.1ms", test_coincidence_delta_01ms),
("Dimensionality SVD", test_dimensionality_svd),


def test_pulse_filter_terminology():
    """Test that pulse/filter terminology is correctly implemented."""
    print("Testing pulse/filter terminology...")

    from src.spiking_network import SpikingRNN

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

    from src.spiking_network import SpikingRNN

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


def test_hd_input_modes():
    """Test the three HD input modes."""
    print("\nTesting HD input modes...")

    from src.spiking_network import SpikingRNN

    modes = ["independent", "common_stochastic", "common_tonic"]

    for mode in modes:
        try:
            network = SpikingRNN(n_neurons=50, synaptic_mode="filter",
                                hd_input_mode=mode, n_hd_channels=10)
            if network.hd_input_mode == mode:
                print(f"  ✓ HD input mode '{mode}' accepted")
            else:
                print(f"  ✗ HD input mode '{mode}' not set correctly")
                return False
        except Exception as e:
            print(f"  ✗ HD input mode '{mode}' failed: {e}")
            return False

    # Test invalid mode is rejected
    try:
        network_invalid = SpikingRNN(n_neurons=50, hd_input_mode="invalid_mode", n_hd_channels=10)
        print("  ✗ Invalid HD input mode accepted")
        return False
    except ValueError:
        print("  ✓ Invalid HD input mode correctly rejected")

    return True


def test_common_stochastic_input():
    """Test that common_stochastic gives identical Poisson spikes across neurons."""
    print("\nTesting common_stochastic input...")

    from src.synaptic_model import StaticPoissonInput

    # Create common_stochastic input
    static_input = StaticPoissonInput(n_neurons=100, dt=0.1, static_input_mode="common_stochastic")
    static_input.initialize_parameters(input_strength=1.0)

    session_id = 42
    v_th_std = 0.5
    g_std = 0.3
    trial_id = 1
    rate = 5000.0  # INCREASED: Use higher rate for more spikes

    # Collect RAW EVENTS to test stochasticity
    raw_events = []
    for time_step in range(100):  # INCREASED: More timesteps
        events = static_input.generate_events(session_id, v_th_std, g_std, trial_id, rate, time_step)
        raw_events.append(events.copy())

    raw_events = np.array(raw_events)  # Shape: (timesteps, neurons)

    # Check 1: At each timestep, all neurons should have identical values
    all_identical = True
    for t in range(len(raw_events)):
        unique_values = np.unique(raw_events[t])
        if len(unique_values) > 1:
            all_identical = False
            break

    if all_identical:
        print("  ✓ Common stochastic: all neurons receive identical input at each timestep")
    else:
        print(f"  ✗ Common stochastic: neurons have different values at some timesteps")
        return False

    # Check 2: Different timesteps should have different patterns (stochastic)
    timesteps_with_spikes = np.sum([np.any(raw_events[t] > 0) for t in range(len(raw_events))])
    timesteps_without_spikes = np.sum([np.all(raw_events[t] == 0) for t in range(len(raw_events))])

    # With high rate, should have many timesteps with spikes
    if timesteps_with_spikes > 10:
        print(f"  ✓ Common stochastic: stochastic across time ({timesteps_with_spikes} with spikes, {timesteps_without_spikes} without)")
    else:
        print(f"  ✗ Common stochastic: not enough spikes ({timesteps_with_spikes} with spikes)")
        return False

    return True


def test_independent_stochastic_input():
    """Test that independent gives different Poisson spikes across neurons."""
    print("\nTesting independent stochastic input...")

    from src.synaptic_model import StaticPoissonInput, Synapse

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

    from src.synaptic_model import StaticPoissonInput, Synapse

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

    from src.synaptic_model import StaticPoissonInput

    # Test with independent mode
    static_input = StaticPoissonInput(n_neurons=50, dt=0.1, static_input_mode="independent")
    static_input.initialize_parameters(input_strength=1.0)

    session_id = 42
    v_th_std = 0.5
    g_std = 0.3
    rate = 500.0

    # Run two different trials
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

    from analysis.stability_analysis import analyze_perturbation_response

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

    from analysis.stability_analysis import average_coincidence_multi_window

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

    from src.synaptic_model import Synapse

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

    from src.spiking_network import SpikingRNN
    from src.rng_utils import rng_manager

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


def test_common_utils_integration():
    """Test that analysis modules correctly use common_utils."""
    print("\nTesting common_utils integration...")

    try:
        from analysis.common_utils import spikes_to_binary, compute_participation_ratio
        from analysis.spontaneous_analysis import analyze_spontaneous_activity
        from analysis.stability_analysis import analyze_perturbation_response

        # Test that spontaneous_analysis uses common_utils
        spikes = [(i*10.0, i%3) for i in range(20)]
        result = analyze_spontaneous_activity(
            spikes,
            num_neurons=5,
            duration=400.0,  # CHANGE: 200ms transient + 200ms data
            transient_time=200.0
        )
        assert 'dimensionality_metrics' in result
        print("  ✓ spontaneous_analysis integrates with common_utils")

        # Test that stability_analysis uses common_utils
        spikes_ctrl = [(1.0, 0), (2.0, 1)]
        spikes_pert = [(1.1, 0), (2.1, 1)]
        result = analyze_perturbation_response(
            spikes_ctrl, spikes_pert, num_neurons=2,
            perturbation_time=1.0, simulation_end=5.0, perturbed_neuron=0
        )
        assert 'lz_spatial_patterns' in result
        print("  ✓ stability_analysis integrates with common_utils")

        return True

    except Exception as e:
        print(f"  ✗ common_utils integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_experiment_functionality():
    """Test BaseExperiment class functionality."""
    print("\nTesting BaseExperiment functionality...")

    try:
        from experiments.base_experiment import BaseExperiment

        # Test 1: Simple parameter grid (spontaneous/stability)
        v_th, g, rates = BaseExperiment.create_parameter_grid(
            n_v_th_points=3, n_g_points=3, n_input_rates=2,
            v_th_std_range=(0.0, 2.0), g_std_range=(0.0, 2.0),
            input_rate_range=(50.0, 100.0)
        )
        assert len(v_th) == 3 and len(g) == 3 and len(rates) == 2
        assert v_th[0] == 0.0 and v_th[-1] == 2.0  # Check actual values
        assert rates[0] == 50.0 and rates[-1] == 100.0
        print("  ✓ Parameter grid creation (simple) - values correct")

        # Test 2: With HD input dimensions (encoding/categorical)
        v_th, g, hd_dims, rates = BaseExperiment.create_parameter_grid(
            n_v_th_points=3, n_g_points=3, n_input_rates=2,
            n_hd_input_points=4, hd_input_dim_range=(1, 10)
        )
        assert len(hd_dims) == 4
        assert hd_dims[0] == 1 and hd_dims[-1] == 10  # Check actual values
        print("  ✓ Parameter grid with HD input dimensions - values correct")

        # Test 3: With HD input AND output dimensions (temporal task)
        v_th, g, hd_in, hd_out, rates = BaseExperiment.create_parameter_grid(
            n_v_th_points=2, n_g_points=2, n_input_rates=2,
            n_hd_input_points=3, hd_input_dim_range=(1, 5),
            n_hd_output_points=3, hd_output_dim_range=(2, 8)
        )
        assert len(hd_in) == 3 and len(hd_out) == 3
        assert hd_in[0] == 1 and hd_out[-1] == 8
        print("  ✓ Parameter grid with HD input AND output dimensions - values correct")

        # Test 4: Error handling - missing hd_input_dim_range
        try:
            BaseExperiment.create_parameter_grid(
                n_v_th_points=3, n_g_points=3, n_input_rates=2,
                n_hd_input_points=4  # Missing hd_input_dim_range!
            )
            print("  ✗ Should have raised ValueError for missing hd_input_dim_range")
            return False
        except ValueError:
            print("  ✓ Correctly raises error when hd_input_dim_range missing")

        # Test 5: Error handling - hd_output without hd_input
        try:
            BaseExperiment.create_parameter_grid(
                n_v_th_points=3, n_g_points=3, n_input_rates=2,
                n_hd_output_points=3, hd_output_dim_range=(1, 5)  # No input!
            )
            print("  ✗ Should have raised ValueError for output without input")
            return False
        except ValueError:
            print("  ✓ Correctly raises error when hd_output without hd_input")

        return True

    except Exception as e:
        print(f"  ✗ BaseExperiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_inheritance():
    """Test that all experiments inherit from BaseExperiment."""
    print("\nTesting experiment inheritance...")

    try:
        from experiments import (
            StabilityExperiment, TaskPerformanceExperiment
        )
        from experiments.base_experiment import BaseExperiment

        # Check inheritance
        assert issubclass(StabilityExperiment, BaseExperiment)
        assert issubclass(TaskPerformanceExperiment, BaseExperiment)
        print("  ✓ All experiments inherit from BaseExperiment")

        # Test that they all have required methods
        stab = StabilityExperiment(n_neurons=10, use_cached_transients=False)
        task = TaskPerformanceExperiment(
            task_type='categorical',
            n_neurons=10,
            n_input_patterns=2,
            input_dim_intrinsic=2,
            input_dim_embedding=5
        )

        for exp in [stab, task]:
            assert hasattr(exp, 'extract_trial_arrays')
            assert hasattr(exp, 'compute_all_statistics')
            assert hasattr(exp, 'create_parameter_combinations')
        print("  ✓ All experiments have required methods")

        return True

    except Exception as e:
        print(f"  ✗ Inheritance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 70)
    print("COMPREHENSIVE STRUCTURE TESTS")
    print("Pulse/Filter, Input Modes, Structure, Refactored Code")
    print("=" * 70)

    tests = [
        ("Pulse/Filter Terminology", test_pulse_filter_terminology),
        ("Static Input Modes", test_static_input_modes),
        ("HD Input Modes", test_hd_input_modes),
        ("Common Stochastic Input", test_common_stochastic_input),
        ("Independent Stochastic Input", test_independent_stochastic_input),
        ("Common Tonic Input", test_common_tonic_input),
        ("Input Mode Trial Dependence", test_input_mode_trial_dependence),
        ("LZ Column-Wise", test_lz_column_wise),
        ("Coincidence Delta 0.1ms", test_coincidence_delta_01ms),
        ("Pulse vs Filter Behavior", test_pulse_vs_filter_behavior),
        ("Network Structure Consistency", test_network_structure_consistency),
        ("Common Utils Integration", test_common_utils_integration),
        ("BaseExperiment Functionality", test_base_experiment_functionality),
        ("Experiment Inheritance", test_experiment_inheritance),
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
    print("COMPLETE COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:40s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("\nVerified capabilities:")
        print("  ✓ Pulse/filter synapse terminology")
        print("  ✓ Three static input modes (independent, common_stochastic, common_tonic)")
        print("  ✓ Three HD input modes (independent, common_stochastic, common_tonic)")
        print("  ✓ Input modes are trial-dependent")
        print("  ✓ LZ column-wise complexity")
        print("  ✓ Coincidence at 0.1ms, 2ms, 5ms")
        print("  ✓ Network structure consistency")
        print("  ✓ Common utils integration")
        print("  ✓ BaseExperiment class functionality")
        print("  ✓ Proper experiment inheritance")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        return 1


if __name__ == "__main__":
    exit(run_all_comprehensive_tests())
