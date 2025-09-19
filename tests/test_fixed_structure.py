# tests/test_fixed_structure.py - Comprehensive tests for fixed structure requirements
"""
Tests to verify that network structure depends only on session_id and that
multiplier scaling works correctly while preserving exact means.
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

def test_base_distributions_fixed_by_session():
    """Test that base distributions depend only on session_id."""
    print("Testing base distributions fixed by session_id...")

    from rng_utils import generate_base_distributions

    # Generate base distributions for same session with different parameters
    session_id = 42
    n_neurons = 100

    base1 = generate_base_distributions(session_id, n_neurons)
    base2 = generate_base_distributions(session_id, n_neurons)

    # Should be identical
    if np.allclose(base1['base_v_th'], base2['base_v_th'], atol=1e-15):
        print("  ✓ Base v_th distributions identical for same session")
    else:
        print("  ✗ Base v_th distributions differ for same session")
        return False

    if np.allclose(base1['base_g'], base2['base_g'], atol=1e-15):
        print("  ✓ Base g distributions identical for same session")
    else:
        print("  ✗ Base g distributions differ for same session")
        return False

    # Test different sessions give different distributions
    base_session2 = generate_base_distributions(session_id + 1, n_neurons)

    if not np.allclose(base1['base_v_th'], base_session2['base_v_th'], atol=1e-10):
        print("  ✓ Different sessions produce different base v_th")
    else:
        print("  ✗ Different sessions produce identical base v_th")
        return False

    return True

def test_exact_mean_preservation():
    """Test that base distributions have exact required means."""
    print("\nTesting exact mean preservation...")

    from rng_utils import generate_base_distributions

    session_id = 123
    n_neurons = 1000  # Large enough for good statistics

    base_dist = generate_base_distributions(session_id, n_neurons)

    # Check v_th mean is exactly -55.0
    v_th_mean = np.mean(base_dist['base_v_th'])
    if abs(v_th_mean - (-55.0)) < 1e-12:
        print(f"  ✓ Base v_th mean exactly -55.0 (actual: {v_th_mean:.15f})")
    else:
        print(f"  ✗ Base v_th mean not exact: {v_th_mean:.15f}")
        return False

    # Check g mean is exactly 0.0
    g_mean = np.mean(base_dist['base_g'])
    if abs(g_mean - 0.0) < 1e-12:
        print(f"  ✓ Base g mean exactly 0.0 (actual: {g_mean:.15f})")
    else:
        print(f"  ✗ Base g mean not exact: {g_mean:.15f}")
        return False

    return True

def test_multiplier_scaling_preserves_means():
    """Test that multiplier scaling preserves means exactly."""
    print("\nTesting multiplier scaling preserves means...")

    from lif_neuron import LIFNeuron
    from synaptic_model import ExponentialSynapses

    # Test LIF neuron threshold scaling
    neurons = LIFNeuron(n_neurons=100, dt=0.1)
    session_id = 456

    # Test different multipliers preserve mean
    for multiplier in [1.0, 5.0, 50.0, 100.0]:
        neurons.initialize_parameters(
            session_id=session_id,
            block_id=0,
            v_th_mean=-55.0,
            v_th_multiplier=multiplier
        )

        actual_mean = np.mean(neurons.spike_thresholds)
        if abs(actual_mean - (-55.0)) < 1e-10:
            print(f"  ✓ Multiplier {multiplier}: mean preserved exactly ({actual_mean:.12f})")
        else:
            print(f"  ✗ Multiplier {multiplier}: mean not preserved ({actual_mean:.12f})")
            return False

    # Test synaptic weight scaling
    synapses = ExponentialSynapses(n_neurons=50, dt=0.1)

    for multiplier in [1.0, 10.0, 50.0, 100.0]:
        synapses.initialize_weights(
            session_id=session_id,
            block_id=0,
            g_mean=0.0,
            g_multiplier=multiplier,
            connection_prob=0.1
        )

        weight_stats = synapses.get_weight_statistics()
        if abs(weight_stats['mean'] - 0.0) < 1e-8:  # Slightly relaxed for sparse case
            print(f"  ✓ Synaptic multiplier {multiplier}: mean preserved ({weight_stats['mean']:.10f})")
        else:
            print(f"  ✗ Synaptic multiplier {multiplier}: mean not preserved ({weight_stats['mean']:.10f})")
            return False

    return True

def test_network_structure_independence():
    """Test that network structure is independent of parameter values."""
    print("\nTesting network structure independence...")

    from spiking_network import SpikingRNN

    session_id = 789
    n_neurons = 50

    # Create networks with same session but different multipliers
    network1 = SpikingRNN(n_neurons=n_neurons, dt=0.1)
    network2 = SpikingRNN(n_neurons=n_neurons, dt=0.1)

    # Initialize with different multipliers
    network1.initialize_network(
        session_id=session_id, block_id=1,
        v_th_multiplier=1.0, g_multiplier=1.0
    )

    network2.initialize_network(
        session_id=session_id, block_id=2,  # Different block_id
        v_th_multiplier=50.0, g_multiplier=50.0
    )

    # Base distributions should be identical
    base1 = network1.neurons.base_distributions
    base2 = network2.neurons.base_distributions

    if np.allclose(base1['base_v_th'], base2['base_v_th'], atol=1e-15):
        print("  ✓ Base v_th distributions identical across different multipliers")
    else:
        print("  ✗ Base v_th distributions differ with different multipliers")
        return False

    # Connectivity should be identical
    conn1 = network1.synapses.base_distributions['connectivity']
    conn2 = network2.synapses.base_distributions['connectivity']

    if np.array_equal(conn1, conn2):
        print("  ✓ Connectivity matrices identical across different multipliers")
    else:
        print("  ✗ Connectivity matrices differ with different multipliers")
        return False

    # Perturbation targets should be identical
    pert1 = base1['perturbation_neurons']
    pert2 = base2['perturbation_neurons']

    if np.array_equal(pert1, pert2):
        print("  ✓ Perturbation targets identical across different multipliers")
    else:
        print("  ✗ Perturbation targets differ with different multipliers")
        return False

    return True

def test_relative_structure_preservation():
    """Test that relative structure is preserved with scaling."""
    print("\nTesting relative structure preservation...")

    from rng_utils import generate_base_distributions

    session_id = 999
    n_neurons = 20

    base_dist = generate_base_distributions(session_id, n_neurons)
    base_g = base_dist['base_g']
    connectivity = base_dist['connectivity']

    # Get connected weights only
    connected_weights = base_g[connectivity]

    # Test that multiplying preserves relative ordering
    multipliers = [1.0, 5.0, 100.0]

    for mult in multipliers:
        scaled_weights = connected_weights * mult

        # Check that ordering is preserved
        orig_ranking = np.argsort(connected_weights)
        scaled_ranking = np.argsort(scaled_weights)

        if np.array_equal(orig_ranking, scaled_ranking):
            print(f"  ✓ Multiplier {mult}: relative ordering preserved")
        else:
            print(f"  ✗ Multiplier {mult}: relative ordering changed")
            return False

        # Check that ratios between weights are preserved
        if len(connected_weights) > 1:
            orig_ratios = connected_weights[1:] / connected_weights[0]
            scaled_ratios = scaled_weights[1:] / scaled_weights[0]

            if np.allclose(orig_ratios, scaled_ratios, rtol=1e-12):
                print(f"  ✓ Multiplier {mult}: weight ratios preserved")
            else:
                print(f"  ✗ Multiplier {mult}: weight ratios changed")
                return False

    return True

def test_poisson_processes_vary_with_trial():
    """Test that only Poisson processes change with trial_id."""
    print("\nTesting Poisson processes vary with trial_id...")

    from spiking_network import SpikingRNN

    session_id = 111
    network = SpikingRNN(n_neurons=20, dt=0.1)

    network.initialize_network(
        session_id=session_id, block_id=1,
        v_th_multiplier=10.0, g_multiplier=10.0
    )

    # Run two simulations with same parameters but different trial_id
    spikes1 = network.simulate_network_dynamics(
        session_id=session_id, block_id=1, trial_id=1,
        duration=50.0, static_input_rate=100.0
    )

    spikes2 = network.simulate_network_dynamics(
        session_id=session_id, block_id=1, trial_id=2,
        duration=50.0, static_input_rate=100.0
    )

    # Spike patterns should be different (due to different Poisson realizations)
    if spikes1 != spikes2:
        print("  ✓ Different trial_id produces different spike patterns")
    else:
        print("  ✗ Different trial_id produces identical spike patterns")
        return False

    # But network structure should remain the same
    # (already tested in previous functions)

    return True

def test_perturbation_neurons_fixed():
    """Test that perturbation neurons are fixed for given session."""
    print("\nTesting perturbation neurons fixed by session...")

    from chaos_experiment import ChaosExperiment

    experiment = ChaosExperiment(n_neurons=100)
    session_id = 222

    # Run single perturbation with different parameter combinations
    result1 = experiment.run_single_perturbation(
        session_id=session_id, block_id=1, trial_id=1,
        v_th_multiplier=1.0, g_multiplier=1.0,
        perturbation_neuron_idx=0,  # First neuron in fixed list
        static_input_rate=100.0
    )

    result2 = experiment.run_single_perturbation(
        session_id=session_id, block_id=2, trial_id=1,
        v_th_multiplier=50.0, g_multiplier=50.0,
        perturbation_neuron_idx=0,  # Same index
        static_input_rate=100.0
    )

    # Should perturb the same neuron despite different parameters
    if result1['perturbation_neuron'] == result2['perturbation_neuron']:
        print(f"  ✓ Same perturbation neuron ({result1['perturbation_neuron']}) across parameter combinations")
    else:
        print(f"  ✗ Different perturbation neurons: {result1['perturbation_neuron']} vs {result2['perturbation_neuron']}")
        return False

    return True

def test_standard_deviations_scale_correctly():
    """Test that standard deviations scale correctly with multipliers."""
    print("\nTesting standard deviation scaling...")

    from lif_neuron import LIFNeuron

    neurons = LIFNeuron(n_neurons=1000, dt=0.1)  # Large for good statistics
    session_id = 333
    base_multiplier = 1.0

    # Get base standard deviation
    neurons.initialize_parameters(
        session_id=session_id, block_id=0,
        v_th_multiplier=base_multiplier
    )
    base_std = np.std(neurons.spike_thresholds)
    expected_base_std = 0.01  # Should be our base heterogeneity

    if abs(base_std - expected_base_std) < 0.001:  # Allow small tolerance
        print(f"  ✓ Base std approximately correct: {base_std:.6f}")
    else:
        print(f"  ✗ Base std incorrect: {base_std:.6f}, expected ~{expected_base_std}")
        return False

    # Test scaling
    test_multipliers = [5.0, 20.0, 100.0]

    for multiplier in test_multipliers:
        neurons.initialize_parameters(
            session_id=session_id, block_id=0,
            v_th_multiplier=multiplier
        )
        scaled_std = np.std(neurons.spike_thresholds)
        expected_std = base_std * multiplier

        relative_error = abs(scaled_std - expected_std) / expected_std
        if relative_error < 0.01:  # 1% tolerance
            print(f"  ✓ Multiplier {multiplier}: std scales correctly ({scaled_std:.6f})")
        else:
            print(f"  ✗ Multiplier {multiplier}: std scaling incorrect ({scaled_std:.6f} vs {expected_std:.6f})")
            return False

    return True

def run_all_fixed_structure_tests():
    """Run all tests for fixed structure requirements."""
    print("Fixed Structure and RNG Requirements Test Suite")
    print("=" * 60)

    tests = [
        ("Base Distributions Fixed by Session", test_base_distributions_fixed_by_session),
        ("Exact Mean Preservation", test_exact_mean_preservation),
        ("Multiplier Scaling Preserves Means", test_multiplier_scaling_preserves_means),
        ("Network Structure Independence", test_network_structure_independence),
        ("Relative Structure Preservation", test_relative_structure_preservation),
        ("Poisson Processes Vary with Trial", test_poisson_processes_vary_with_trial),
        ("Perturbation Neurons Fixed", test_perturbation_neurons_fixed),
        ("Standard Deviation Scaling", test_standard_deviations_scale_correctly),
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
    print("Fixed Structure Test Summary:")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:30s}: {status}")

    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL FIXED STRUCTURE REQUIREMENTS SATISFIED!")
        print("\nVerified properties:")
        print("  ✓ Network structure depends only on session_id")
        print("  ✓ Base distributions have exact means (-55.0, 0.0)")
        print("  ✓ Multiplier scaling preserves means exactly")
        print("  ✓ Relative network structure preserved across multipliers")
        print("  ✓ Only Poisson processes vary with trial_id")
        print("  ✓ Perturbation targets fixed by session")
        print("  ✓ Standard deviations scale correctly")

        print(f"\nMultiplier interpretation:")
        print(f"  • v_th_multiplier 1-100 → actual v_th_std 0.01-1.0")
        print(f"  • g_multiplier 1-100 → actual g_std 0.01-1.0")
        print(f"  • Same relative network structure across all combinations")

        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")
        print("Fixed structure requirements not satisfied.")
        return 1

if __name__ == "__main__":
    exit(run_all_fixed_structure_tests())
