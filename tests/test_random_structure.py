# tests/test_random_structure.py - Tests for random structure with mean centering
"""
Tests to verify that network structure depends on session_id AND parameters,
and that mean centering works correctly.
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

def test_parameter_dependent_structure():
    """Test that network structure depends on both session_id AND parameters."""
    print("Testing parameter-dependent structure...")

    from rng_utils import get_rng, rng_manager

    # Reset RNG manager to ensure test independence
    rng_manager.reset_for_testing()

    session_id = 42
    trial_id = 0

    # Same session, different parameters should give different RNGs
    rng1 = get_rng(session_id, 1.0, 0.5, trial_id, 'spike_thresholds')
    rng2 = get_rng(session_id, 1.0, 1.0, trial_id, 'spike_thresholds')  # Different g_std

    # Generate samples
    samples1 = rng1.normal(0, 1, 10)
    samples2 = rng2.normal(0, 1, 10)

    if not np.allclose(samples1, samples2):
        print("  ✓ Different parameters produce different structure")
    else:
        print("  ✗ Different parameters produce identical structure")
        return False

    # Reset and test same parameters
    rng_manager.reset_for_testing()

    # Same parameters should give identical RNGs
    rng3 = get_rng(session_id, 1.0, 0.5, trial_id, 'spike_thresholds')
    rng4 = get_rng(session_id, 1.0, 0.5, trial_id, 'spike_thresholds')

    samples3 = rng3.normal(0, 1, 10)
    samples4 = rng4.normal(0, 1, 10)

    if np.allclose(samples3, samples4):
        print("  ✓ Same parameters produce identical structure")
        return True
    else:
        print("  ✗ Same parameters produce different structure")
        print(f"    samples3: {samples3[:3]}")
        print(f"    samples4: {samples4[:3]}")
        return False

def test_mean_centering_lif():
    """Test LIF neuron mean centering with proper tolerance."""
    print("\nTesting LIF neuron mean centering...")

    from lif_neuron import LIFNeuron
    from rng_utils import rng_manager

    neurons = LIFNeuron(n_neurons=1000, dt=0.1)

    # Test normal distribution
    for v_th_std in [0.5, 1.0, 2.0]:
        # Use a different session_id each time to avoid any possible interference
        session_id = int(v_th_std * 1000)

        neurons.initialize_parameters(
            session_id=session_id,
            v_th_std=v_th_std,
            trial_id=0,
            v_th_mean=-55.0,
            v_th_distribution="normal"
        )

        actual_mean = np.mean(neurons.spike_thresholds)
        actual_std = np.std(neurons.spike_thresholds)

        # Use realistic tolerance for floating-point arithmetic
        # With 1000 numbers, expect ~1e-13 precision, use 1e-12 for safety
        mean_error = abs(actual_mean - (-55.0))
        if mean_error < 1e-12:  # More realistic tolerance
            print(f"  ✓ Normal dist, std={v_th_std}: mean preserved ({actual_mean:.12f})")
        else:
            print(f"  ✗ Normal dist, std={v_th_std}: mean error too large ({mean_error:.2e})")
            print(f"    Expected: -55.0, Got: {actual_mean:.15f}")
            return False

        # Allow larger tolerance for higher std values due to clipping effects
        std_tolerance = 0.05 if v_th_std <= 1.0 else 0.1  # More lenient for std=2.0
        if abs(actual_std - v_th_std) < std_tolerance:
            print(f"  ✓ Normal dist, std={v_th_std}: std correct ({actual_std:.3f})")
        else:
            print(f"  ✗ Normal dist, std={v_th_std}: std incorrect ({actual_std:.3f})")
            print(f"    Expected: {v_th_std}, tolerance: ±{std_tolerance}")
            return False

    # Test uniform distribution
    for v_th_std in [0.5, 1.0, 2.0]:
        session_id = int(v_th_std * 1000) + 1000  # Different from normal tests

        neurons.initialize_parameters(
            session_id=session_id,
            v_th_std=v_th_std,
            trial_id=0,
            v_th_mean=-55.0,
            v_th_distribution="uniform"
        )

        actual_mean = np.mean(neurons.spike_thresholds)
        actual_std = np.std(neurons.spike_thresholds)

        mean_error = abs(actual_mean - (-55.0))
        if mean_error < 1e-12:
            print(f"  ✓ Uniform dist, std={v_th_std}: mean preserved ({actual_mean:.12f})")
        else:
            print(f"  ✗ Uniform dist, std={v_th_std}: mean error too large ({mean_error:.2e})")
            return False

        # Allow larger tolerance for higher std values due to clipping effects
        std_tolerance = 0.05 if v_th_std <= 1.0 else 0.1  # More lenient for std=2.0
        if abs(actual_std - v_th_std) < std_tolerance:
            print(f"  ✓ Uniform dist, std={v_th_std}: std correct ({actual_std:.3f})")
        else:
            print(f"  ✗ Uniform dist, std={v_th_std}: std incorrect ({actual_std:.3f})")
            return False

    return True

def test_mean_centering_synapses():
    """Test synaptic weight mean centering."""
    print("\nTesting synaptic weight mean centering...")

    from synaptic_model import ExponentialSynapses

    synapses = ExponentialSynapses(n_neurons=500, dt=0.1, synaptic_mode="dynamic")

    for g_std in [0.5, 1.0, 2.0]:
        synapses.initialize_weights(
            session_id=456,
            v_th_std=1.0,
            g_std=g_std,
            g_mean=0.0,
            connection_prob=0.1
        )

        weight_stats = synapses.get_weight_statistics()

        if abs(weight_stats['effective_mean'] - 0.0) < 1e-8:
            print(f"  ✓ g_std={g_std}: mean preserved ({weight_stats['effective_mean']:.10f})")
        else:
            print(f"  ✗ g_std={g_std}: mean not preserved ({weight_stats['effective_mean']:.10f})")
            return False

        if abs(weight_stats['effective_std'] - g_std) < 0.1:  # Allow 10% tolerance for sparse case
            print(f"  ✓ g_std={g_std}: std correct ({weight_stats['effective_std']:.3f})")
        else:
            print(f"  ✗ g_std={g_std}: std incorrect ({weight_stats['effective_std']:.3f})")
            return False

    return True

def test_synaptic_mode_normalization():
    """Test synaptic mode impact normalization."""
    print("\nTesting synaptic mode normalization...")

    from synaptic_model import ExponentialSynapses

    session_id = 789
    v_th_std = 1.0
    g_std = 1.0

    # Test dynamic mode
    synapses_dynamic = ExponentialSynapses(n_neurons=100, dt=0.1, synaptic_mode="dynamic")
    synapses_dynamic.initialize_weights(session_id, v_th_std, g_std)

    # Test immediate mode
    synapses_immediate = ExponentialSynapses(n_neurons=100, dt=0.1, synaptic_mode="immediate")
    synapses_immediate.initialize_weights(session_id, v_th_std, g_std)

    # Get statistics
    dynamic_stats = synapses_dynamic.get_weight_statistics()
    immediate_stats = synapses_immediate.get_weight_statistics()

    # Check normalization factor
    expected_factor = 5.0 / 0.1  # tau_syn / dt = 50
    actual_factor = immediate_stats['normalization_factor']

    if abs(actual_factor - expected_factor) < 1e-6:
        print(f"  ✓ Normalization factor correct: {actual_factor}")
    else:
        print(f"  ✗ Normalization factor incorrect: {actual_factor} vs {expected_factor}")
        return False

    # Check that effective means are both zero
    if abs(dynamic_stats['effective_mean']) < 1e-8 and abs(immediate_stats['effective_mean']) < 1e-8:
        print(f"  ✓ Both modes have zero effective mean")
    else:
        print(f"  ✗ Effective means not zero: dynamic={dynamic_stats['effective_mean']:.8f}, immediate={immediate_stats['effective_mean']:.8f}")
        return False

    # Check that effective stds are similar
    std_ratio = immediate_stats['effective_std'] / dynamic_stats['effective_std']
    if 0.9 < std_ratio < 1.1:  # Allow 10% difference
        print(f"  ✓ Effective standard deviations similar (ratio: {std_ratio:.3f})")
    else:
        print(f"  ✗ Effective standard deviations differ (ratio: {std_ratio:.3f})")
        return False

    return True

def test_session_averaging_function():
    """Test session averaging functionality."""
    print("\nTesting session averaging function...")

    from chaos_experiment import average_across_sessions
    import tempfile

    # Create dummy results for two sessions
    dummy_results_1 = [
        {
            'session_id': 1,
            'v_th_std': 1.0,
            'g_std': 0.5,
            'v_th_distribution': 'normal',
            'static_input_rate': 200.0,
            'synaptic_mode': 'dynamic',
            'lz_complexities': np.array([10, 12, 11]),
            'hamming_slopes': np.array([0.1, 0.12, 0.11]),
            'total_spike_differences': np.array([100, 120, 110]),
            'intrinsic_dimensionalities': np.array([5, 6, 5]),
            'effective_dimensionalities': np.array([3, 4, 3]),
            'participation_ratios': np.array([2.5, 3.0, 2.5]),
            'total_variances': np.array([100, 120, 100]),
            'gamma_coincidences': np.array([0.5, 0.6, 0.5]),
            'n_trials': 3,
            'computation_time': 10.0,
            'combination_index': 1
        }
    ]

    dummy_results_2 = [
        {
            'session_id': 2,
            'v_th_std': 1.0,
            'g_std': 0.5,
            'v_th_distribution': 'normal',
            'static_input_rate': 200.0,
            'synaptic_mode': 'dynamic',
            'lz_complexities': np.array([9, 11, 10]),
            'hamming_slopes': np.array([0.09, 0.11, 0.10]),
            'total_spike_differences': np.array([90, 110, 100]),
            'intrinsic_dimensionalities': np.array([4, 5, 4]),
            'effective_dimensionalities': np.array([2, 3, 2]),
            'participation_ratios': np.array([2.0, 2.5, 2.0]),
            'total_variances': np.array([90, 110, 90]),
            'gamma_coincidences': np.array([0.4, 0.5, 0.4]),
            'n_trials': 3,
            'computation_time': 12.0,
            'combination_index': 1
        }
    ]

    # Save to temporary files
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f1:
        import pickle
        pickle.dump(dummy_results_1, f1)
        file1 = f1.name

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f2:
        import pickle
        pickle.dump(dummy_results_2, f2)
        file2 = f2.name

    try:
        # Test averaging
        averaged = average_across_sessions([file1, file2])

        if len(averaged) == 1:
            print("  ✓ Correct number of averaged combinations")
        else:
            print(f"  ✗ Expected 1 combination, got {len(averaged)}")
            return False

        result = averaged[0]

        # Check averaged values
        expected_lz_mean = np.mean([10, 12, 11, 9, 11, 10])  # Combined arrays
        if abs(result['lz_mean'] - expected_lz_mean) < 1e-10:
            print(f"  ✓ LZ mean correctly averaged: {result['lz_mean']}")
        else:
            print(f"  ✗ LZ mean incorrectly averaged: {result['lz_mean']} vs {expected_lz_mean}")
            return False

        if result['n_sessions'] == 2:
            print("  ✓ Session count correct")
        else:
            print(f"  ✗ Session count incorrect: {result['n_sessions']}")
            return False

        if result['total_trials'] == 6:  # 3 trials × 2 sessions
            print("  ✓ Total trials count correct")
        else:
            print(f"  ✗ Total trials count incorrect: {result['total_trials']}")
            return False

        print("  ✓ Session averaging function works correctly")
        return True

    finally:
        # Cleanup temporary files
        os.unlink(file1)
        os.unlink(file2)

def run_all_tests():
    """Run all tests for random structure implementation."""
    print("Random Structure Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Parameter-Dependent Structure", test_parameter_dependent_structure),
        ("LIF Neuron Mean Centering", test_mean_centering_lif),
        ("Synaptic Weight Mean Centering", test_mean_centering_synapses),
        ("Synaptic Mode Normalization", test_synaptic_mode_normalization),
        ("Session Averaging Function", test_session_averaging_function),
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
    print("Random Structure Test Summary:")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:30s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL RANDOM STRUCTURE TESTS PASSED!")
        print("\nVerified properties:")
        print("  ✓ Network structure depends on session_id AND parameters")
        print("  ✓ Mean centering works for both normal and uniform distributions")
        print("  ✓ Synaptic weights have exact zero mean")
        print("  ✓ Synaptic modes have fair impact normalization")
        print("  ✓ Session averaging combines results correctly")

        print(f"\nImplementation ready for:")
        print(f"  • Direct heterogeneity studies (v_th_std, g_std)")
        print(f"  • Synaptic mode comparisons (immediate vs dynamic)")
        print(f"  • Session averaging for robust statistics")
        print(f"  • Mean-centered parameter distributions")

        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        print("Random structure requirements not satisfied.")
        return 1

if __name__ == "__main__":
    exit(run_all_tests())
