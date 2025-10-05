#!/usr/bin/env python3
# test_encoding_implementation.py - Test HD input encoding implementation
"""
Test script to verify the HD input encoding system is working correctly.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments')

def test_rng_extension():
    """Test that RNG system supports HD parameters."""
    print("Testing RNG extension...")
    from rng_utils import get_rng

    # Test without HD parameters (backward compatibility)
    rng1 = get_rng(1, 0.5, 0.5, 1, 'test_component')
    assert rng1 is not None
    print("  ✓ RNG works without HD parameters")

    # Test with HD parameters
    rng2 = get_rng(1, 0.5, 0.5, 1, 'test_component', hd_dim=5, embed_dim=10)
    assert rng2 is not None
    print("  ✓ RNG works with HD parameters")

    # Test that different HD values give different seeds
    rng3 = get_rng(1, 0.5, 0.5, 1, 'test_component', hd_dim=3, embed_dim=10)
    val2 = rng2.random()
    val3 = rng3.random()
    assert val2 != val3, "Different HD dims should give different random values"
    print("  ✓ Different HD dims produce different RNG streams")

    print("RNG extension test PASSED\n")


def test_hd_input_generator():
    """Test HD input generator."""
    print("Testing HD input generator...")
    from hd_input_generator import HDInputGenerator, run_rate_rnn, make_embedding

    # Test rate RNN
    rates, time = run_rate_rnn(
        n_neurons=100,
        T=350.0,
        dt=0.1,
        g=1.2,
        session_id=1,
        hd_dim=5,
        embed_dim=10
    )
    assert rates.shape[0] == 3000, f"Expected 3000 timesteps, got {rates.shape[0]}"
    assert rates.shape[1] == 100, f"Expected 100 neurons, got {rates.shape[1]}"
    print("  ✓ Rate RNN produces correct output shape")

    # Test embedding
    Y, components = make_embedding(rates, k=10, d=5, session_id=1)
    assert Y.shape[0] == 3000, f"Expected 3000 timesteps, got {Y.shape[0]}"
    assert Y.shape[1] == 10, f"Expected 10 channels, got {Y.shape[1]}"
    assert len(components) == 5, f"Expected 5 components, got {len(components)}"
    print("  ✓ Embedding produces correct output shape")

    # Test HD input generator
    generator = HDInputGenerator(embed_dim=10, dt=0.1)
    generator.initialize_base_input(session_id=1, hd_dim=5)
    assert generator.Y_base is not None
    assert generator.Y_base.shape == (3000, 10)
    print("  ✓ HDInputGenerator initializes correctly")

    # Test trial input generation
    Y_trial = generator.generate_trial_input(
        session_id=1,
        v_th_std=0.5,
        g_std=0.5,
        trial_id=1,
        hd_dim=5,
        noise_std=0.5,
        rate_scale=1.0
    )
    assert Y_trial.shape == (3000, 10)
    assert np.all(Y_trial >= 0), "All rates should be non-negative"
    print("  ✓ Trial input generation works correctly")

    # Test that different trials give different noise
    Y_trial2 = generator.generate_trial_input(
        session_id=1,
        v_th_std=0.5,
        g_std=0.5,
        trial_id=2,
        hd_dim=5,
        noise_std=0.5,
        rate_scale=1.0
    )
    assert not np.array_equal(Y_trial, Y_trial2), "Different trials should have different noise"
    print("  ✓ Different trials produce different noise")

    print("HD input generator test PASSED\n")


def test_hd_dynamic_input():
    """Test HDDynamicInput class."""
    print("Testing HDDynamicInput class...")
    from synaptic_model import HDDynamicInput
    from rng_utils import get_rng

    # Test initialization
    hd_input = HDDynamicInput(n_neurons=100, n_channels=10, dt=0.1, hd_input_mode="independent")
    assert hd_input.n_neurons == 100
    assert hd_input.n_channels == 10
    print("  ✓ HDDynamicInput initializes correctly")

    # Test connectivity
    hd_input.initialize_connectivity(
        session_id=1,
        hd_dim=5,
        embed_dim=10,
        connection_prob=0.3,
        input_strength=1.0
    )
    assert hd_input.connectivity_matrix is not None
    assert hd_input.connectivity_matrix.shape == (100, 10)

    # Check connection probability
    conn_prob = np.mean(hd_input.connectivity_matrix)
    assert 0.2 < conn_prob < 0.4, f"Connection probability {conn_prob} out of expected range"
    print("  ✓ Connectivity matrix correct shape and density")

    # Test event generation - independent mode
    rates = np.ones(10) * 100.0  # 100 Hz
    events = hd_input.generate_events(
        session_id=1,
        v_th_std=0.5,
        g_std=0.5,
        trial_id=1,
        hd_dim=5,
        embed_dim=10,
        rates=rates,
        time_step=0
    )
    assert events.shape == (100,)
    assert np.sum(events) > 0, "Should have some events"
    print("  ✓ Independent mode event generation works")

    # Test common_tonic mode
    hd_input_tonic = HDDynamicInput(n_neurons=100, n_channels=10, dt=0.1, hd_input_mode="common_tonic")
    hd_input_tonic.initialize_connectivity(1, 5, 10, 0.3, 1.0)
    events_tonic = hd_input_tonic.generate_events(1, 0.5, 0.5, 1, 5, 10, rates, 0)
    assert events_tonic.shape == (100,)
    print("  ✓ Common tonic mode event generation works")

    # Test common_stochastic mode
    hd_input_stoch = HDDynamicInput(n_neurons=100, n_channels=10, dt=0.1, hd_input_mode="common_stochastic")
    hd_input_stoch.initialize_connectivity(1, 5, 10, 0.3, 1.0)
    events_stoch = hd_input_stoch.generate_events(1, 0.5, 0.5, 1, 5, 10, rates, 0)
    assert events_stoch.shape == (100,)
    print("  ✓ Common stochastic mode event generation works")

    # Test connectivity info
    info = hd_input.get_connectivity_info()
    assert 'neurons_per_channel_mean' in info
    assert 'pairwise_overlap_mean' in info
    print("  ✓ Connectivity info retrieval works")

    print("HDDynamicInput test PASSED\n")


def test_encoding_experiment():
    """Test encoding experiment."""
    print("Testing EncodingExperiment...")
    from encoding_experiment import EncodingExperiment

    # Create experiment
    experiment = EncodingExperiment(
        n_neurons=100,  # Small for testing
        dt=0.1,
        synaptic_mode="filter",
        static_input_mode="independent",
        hd_input_mode="independent",
        embed_dim=5
    )
    assert experiment.n_neurons == 100
    assert experiment.embed_dim == 5
    print("  ✓ EncodingExperiment initializes correctly")

    # Test single trial (this will take a moment)
    print("  Running single trial test (may take ~10 seconds)...")

    # Initialize the HD generator first
    experiment.hd_generator.initialize_base_input(
        session_id=1,
        hd_dim=3,
        rate_rnn_params={'n_neurons': 100, 'T': 350.0, 'g': 1.2}  # Small for testing
    )

    result = experiment.run_single_trial(
        session_id=1,
        v_th_std=0.5,
        g_std=0.5,
        trial_id=1,
        hd_dim=3,
        v_th_distribution="normal",
        static_input_rate=200.0
    )

    assert 'spike_times' in result
    assert 'readout_history' in result
    assert 'hd_input_patterns' in result
    assert result['hd_input_patterns'].shape == (3000, 5)
    print("  ✓ Single trial completes successfully")
    print(f"    Spikes recorded: {result['n_spikes']}")

    print("EncodingExperiment test PASSED\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing HD Input Encoding Implementation")
    print("=" * 70 + "\n")

    # Track test results
    test_results = []

    try:
        # Test 1: RNG Extension
        test_rng_extension()
        test_results.append(("RNG Extension (HD Parameters)", "PASS"))

        # Test 2: HD Input Generator
        test_hd_input_generator()
        test_results.append(("HD Input Generator (Rate RNN + Embedding)", "PASS"))

        # Test 3: HD Dynamic Input
        test_hd_dynamic_input()
        test_results.append(("HD Dynamic Input (Connectivity + Events)", "PASS"))

        # Test 4: Encoding Experiment
        test_encoding_experiment()
        test_results.append(("Encoding Experiment (Full Pipeline)", "PASS"))

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("HD Input Encoding Test Summary:")
    print("=" * 70)

    for test_name, status in test_results:
        status_symbol = "✅" if status == "PASS" else "❌"
        print(f"  {test_name:<50} : {status_symbol} {status}")

    print("\n" + f"Results: {len(test_results)}/{len(test_results)} tests passed")
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 70)

    print("\n✓ HD Input Encoding System Ready")
    print("\nCapabilities:")
    print("  • Generate HD inputs with controlled intrinsic dimensionality")
    print("  • Embed d-dimensional signals in k-dimensional space")
    print("  • Three input modes: independent, common_stochastic, common_tonic")
    print("  • Pulse vs filter synaptic transmission")
    print("  • 20 trials per parameter combination")
    print("  • Session averaging support")

    print("\nNext Steps:")
    print("  1. Implement decoding analysis in analysis/encoding_analysis.py")
    print("  2. Create MPI runner: runners/mpi_encoding_runner.py")
    print("  3. Create shell script: runners/run_encoding_experiment.sh")
    print("  4. Run experiments across HD dimensionalities and heterogeneity")

    print("\nExample Usage:")
    print("  from experiments.encoding_experiment import EncodingExperiment")
    print("  exp = EncodingExperiment(n_neurons=1000, embed_dim=10)")
    print("  results = exp.run_full_experiment(session_id=1, ...)")
    print("")


if __name__ == "__main__":
    main()
