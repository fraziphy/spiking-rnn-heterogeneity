# tests/test_encoding_implementation.py
"""
Complete HD input encoding implementation tests.
Tests ALL components: RNG, HD input, connectivity, experiments, decoding.
"""

import numpy as np
import sys
import os

# Add src to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def test_rng_extension():
    """Test that RNG system supports HD parameters."""
    print("Testing RNG extension for HD inputs...")
    from src.rng_utils import get_rng

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

    # Test that different embed_dim gives different seeds
    rng4 = get_rng(1, 0.5, 0.5, 1, 'test_component', hd_dim=5, embed_dim=8)
    val4 = rng4.random()
    assert val2 != val4, "Different embed dims should give different random values"
    print("  ✓ Different embed dims produce different RNG streams")

    print("RNG extension test PASSED\n")
    return True


def test_rate_rnn():
    """Test rate RNN for HD signal generation."""
    print("Testing rate RNN...")
    from src.hd_input import run_rate_rnn

    # Test rate RNN
    rates, time = run_rate_rnn(
        n_neurons=100,
        T=500.0,
        dt=0.1,
        g=1.2,
        session_id=1,
        hd_dim=5,
        embed_dim=10
    )

    # Should remove 200ms transient
    expected_steps = int((500.0 - 200.0) / 0.1)
    assert rates.shape[0] == expected_steps, f"Expected {expected_steps} timesteps, got {rates.shape[0]}"
    assert rates.shape[1] == 100, f"Expected 100 neurons, got {rates.shape[1]}"
    print(f"  ✓ Rate RNN produces correct output shape: {rates.shape}")

    # Check that rates are bounded
    assert np.all(rates >= -1.0) and np.all(rates <= 1.0), "Rates should be in tanh range"
    print("  ✓ Rates are properly bounded")

    # Test that different sessions give different outputs
    rates2, _ = run_rate_rnn(100, 500.0, 0.1, 1.2, session_id=2, hd_dim=5, embed_dim=10)
    assert not np.array_equal(rates, rates2), "Different sessions should give different rates"
    print("  ✓ Different sessions produce different rates")

    return True


def test_embedding():
    """Test HD embedding generation."""
    print("\nTesting HD embedding...")
    from src.hd_input import run_rate_rnn, make_embedding

    # Generate rate RNN output
    rates, _ = run_rate_rnn(100, 500.0, 0.1, 1.2, session_id=1, hd_dim=5, embed_dim=10)

    # Test embedding
    Y, components = make_embedding(rates, k=10, d=5, session_id=1)

    assert Y.shape[0] == rates.shape[0], f"Expected {rates.shape[0]} timesteps, got {Y.shape[0]}"
    assert Y.shape[1] == 10, f"Expected 10 channels, got {Y.shape[1]}"
    assert len(components) == 5, f"Expected 5 components, got {len(components)}"
    print(f"  ✓ Embedding produces correct shape: {Y.shape}")
    print(f"  ✓ Correct number of components: {len(components)}")

    # Check that components are unique
    assert len(set(components)) == 5, "Components should be unique"
    print("  ✓ Components are unique")

    # Check that embedding is normalized
    channel_stds = np.std(Y, axis=0)
    assert np.allclose(channel_stds, 1.0, rtol=0.1), "Channels should be normalized"
    print("  ✓ Embedding is normalized")

    # Test that different intrinsic dims give different embeddings
    Y2, _ = make_embedding(rates, k=10, d=3, session_id=1)
    assert not np.array_equal(Y, Y2), "Different intrinsic dims should give different embeddings"
    print("  ✓ Different intrinsic dims produce different embeddings")

    return True


def test_hd_input_generator():
    """Test HDInputGenerator class."""
    print("\nTesting HDInputGenerator...")
    from src.hd_input import HDInputGenerator
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with inputs subdirectory (like task experiments)
        inputs_dir = os.path.join(tmpdir, 'inputs')

        # Test initialization
        generator = HDInputGenerator(embed_dim=10, dt=0.1, signal_cache_dir=inputs_dir)
        assert generator.embed_dim == 10
        assert generator.dt == 0.1
        print("  ✓ HDInputGenerator initializes correctly")

        # Test base input initialization
        generator.initialize_base_input(session_id=1, hd_dim=5, pattern_id=0)  # ADD pattern_id
        assert generator.Y_base is not None
        assert generator.Y_base.shape == (3000, 10)  # 300ms encoding period
        print(f"  ✓ Base input initialized: shape {generator.Y_base.shape}")

        # Test caching
        cache_file = generator._get_signal_filename(1, 5, 0)  # ADD pattern_id
        assert os.path.exists(cache_file), "Cache file not created"
        print("  ✓ Signal caching works")

        # Test loading from cache
        generator2 = HDInputGenerator(embed_dim=10, dt=0.1, signal_cache_dir=inputs_dir)
        generator2.initialize_base_input(session_id=1, hd_dim=5, pattern_id=0)
        assert np.array_equal(generator.Y_base, generator2.Y_base)
        print("  ✓ Cache loading works")

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
        print("  ✓ Trial input generation works")

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

        # Test statistics
        stats = generator.get_base_statistics()
        assert 'n_timesteps' in stats
        assert 'embed_dim' in stats
        assert stats['embed_dim'] == 10
        print("  ✓ Statistics retrieval works")

    return True


def test_hd_dynamic_input():
    """Test HDDynamicInput class."""
    print("\nTesting HDDynamicInput class...")
    from src.synaptic_model import HDDynamicInput

    # Test initialization
    hd_input = HDDynamicInput(n_neurons=100, n_channels=10, dt=0.1, hd_input_mode="independent")
    assert hd_input.n_neurons == 100
    assert hd_input.n_channels == 10
    assert hd_input.hd_input_mode == "independent"
    print("  ✓ HDDynamicInput initializes correctly")

    # Test connectivity initialization
    hd_input.initialize_connectivity(
        session_id=1,
        hd_dim=5,
        embed_dim=10,
        connection_prob=0.3,
        input_strength=1.0
    )
    assert hd_input.connectivity_matrix is not None
    assert hd_input.connectivity_matrix.shape == (100, 10)
    print("  ✓ Connectivity matrix initialized: shape (100, 10)")

    # Check connection probability
    conn_prob = np.mean(hd_input.connectivity_matrix)
    assert 0.2 < conn_prob < 0.4, f"Connection probability {conn_prob} out of expected range"
    print(f"  ✓ Connection probability: {conn_prob:.3f}")

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
    assert np.sum(events_tonic) > 0
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
    assert 'hd_input_mode' in info
    print("  ✓ Connectivity info retrieval works")

    return True


def test_encoding_experiment_single_trial():
    """Test encoding experiment single trial."""
    print("\nTesting EncodingExperiment single trial...")
    from experiments.encoding_experiment import EncodingExperiment

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
    assert experiment.transient_time == 200.0
    assert experiment.encoding_time == 300.0
    print("  ✓ EncodingExperiment initializes correctly")
    print(f"    Transient: {experiment.transient_time} ms, Encoding: {experiment.encoding_time} ms")

    # Initialize the HD generator
    experiment.hd_generator.initialize_base_input(
        session_id=1,
        hd_dim=3,
        pattern_id=0,
        rate_rnn_params={'n_neurons': 100, 'T': 500.0, 'g': 1.2}
    )
    print("  ✓ HD generator initialized")

    # Run single trial
    print("  Running single trial (may take ~10 seconds)...")
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
    assert 'n_spikes' in result
    assert 'trial_id' in result
    print(f"  ✓ Single trial completes successfully")
    print(f"    Spikes recorded: {result['n_spikes']}")

    return True


def test_encoding_experiment_smart_storage():
    """Test encoding experiment smart storage logic."""
    print("\nTesting encoding smart storage logic...")
    from experiments.encoding_experiment import EncodingExperiment
    from analysis.statistics_utils import get_extreme_combinations

    # Setup extreme combinations
    v_th_stds = np.array([0.0, 2.0, 4.0])
    g_stds = np.array([0.0, 2.0, 4.0])
    extremes = get_extreme_combinations(v_th_stds, g_stds)
    print(f"  Extreme combinations: {extremes}")

    # Create experiment with hd_dim=1, embed_dim=1
    experiment = EncodingExperiment(n_neurons=50, embed_dim=1)
    experiment.hd_generator.initialize_base_input(session_id=1, hd_dim=1, pattern_id=0)
    print("  ✓ Experiment initialized for low-dim test")

    # Run extreme combination (should save neuron data)
    print("  Running extreme combination (0.0, 0.0)...")
    result_extreme = experiment.run_parameter_combination(
        session_id=1, v_th_std=0.0, g_std=0.0, hd_dim=1,
        extreme_combos=extremes
    )

    assert result_extreme['saved_neuron_data'] == True
    assert 'decoder_weights' in result_extreme['decoding']
    assert 'spike_jitter_per_fold' in result_extreme['decoding']
    assert 'spike_thresholds' in result_extreme['decoding']
    print("  ✓ Neuron data saved for extreme combo")

    # Run non-extreme combination (should NOT save neuron data)
    print("  Running non-extreme combination (2.0, 2.0)...")
    result_normal = experiment.run_parameter_combination(
        session_id=1, v_th_std=2.0, g_std=2.0, hd_dim=1,
        extreme_combos=extremes
    )

    assert result_normal['saved_neuron_data'] == False
    assert 'weight_participation_ratio_mean' in result_normal['decoding']
    assert 'decoder_weights' not in result_normal['decoding']
    print("  ✓ Only summary stats saved for non-extreme combo")

    # Test high-dim (should NOT save neuron data even at extremes)
    experiment_hd = EncodingExperiment(n_neurons=50, embed_dim=10)
    experiment_hd.hd_generator.initialize_base_input(session_id=1, hd_dim=5, pattern_id=0)
    print("  Running high-dim combination...")

    result_hd = experiment_hd.run_parameter_combination(
        session_id=1, v_th_std=0.0, g_std=0.0, hd_dim=5,
        extreme_combos=extremes
    )

    assert result_hd['saved_neuron_data'] == False
    print("  ✓ High-dim doesn't save neuron data")

    return True


def test_decoding_analysis():
    """Test decoding analysis functions."""
    print("\nTesting decoding analysis...")
    from analysis.encoding_analysis import decode_hd_input

    # Create mock trial results
    n_trials = 5
    n_neurons = 50
    duration = 300.0

    trial_results = []
    for trial_id in range(n_trials):
        # Generate random spikes
        n_spikes = np.random.randint(50, 150)
        spike_times = [(np.random.rand() * duration, np.random.randint(0, n_neurons))
                      for _ in range(n_spikes)]
        trial_results.append({
            'spike_times': spike_times,
            'n_spikes': n_spikes,
            'trial_id': trial_id
        })

    print(f"  Created {len(trial_results)} mock trials")

    # Create mock HD input
    n_timesteps = int(duration / 0.1)
    hd_input = np.random.randn(n_timesteps, 5)  # 5 channels

    # Run decoding
    print("  Running decoding analysis (may take ~30 seconds)...")
    decoding_results = decode_hd_input(
        trial_results=trial_results,
        hd_input_ground_truth=hd_input,
        n_neurons=n_neurons,
        session_id=1,
        v_th_std=0.5,
        g_std=0.5,
        hd_dim=3,
        embed_dim=5,
        encoding_duration=duration,
        n_splits=n_trials
    )

    # Check required fields
    required_fields = [
        'test_rmse_mean', 'test_rmse_std', 'test_r2_mean', 'test_correlation_mean',
        'decoder_weights', 'weight_svd_analysis', 'decoded_pca_analysis',
        'spike_jitter_per_fold', 'n_folds', 'n_neurons', 'n_channels'
    ]

    for field in required_fields:
        assert field in decoding_results, f"Missing field: {field}"

    print("  ✓ All required fields present")
    print(f"    Test RMSE: {decoding_results['test_rmse_mean']:.4f}")
    print(f"    Test R²: {decoding_results['test_r2_mean']:.4f}")
    print(f"    Test Correlation: {decoding_results['test_correlation_mean']:.4f}")

    # Check dimensionality analysis
    assert len(decoding_results['weight_svd_analysis']) == n_trials
    assert len(decoding_results['decoded_pca_analysis']) == n_trials
    print("  ✓ Dimensionality analysis completed")

    return True


def main():
    """Run all encoding implementation tests."""
    print("=" * 70)
    print("COMPLETE HD INPUT ENCODING IMPLEMENTATION TESTS")
    print("=" * 70 + "\n")

    tests = [
        ("RNG Extension (HD Parameters)", test_rng_extension),
        ("Rate RNN", test_rate_rnn),
        ("HD Embedding", test_embedding),
        ("HD Input Generator", test_hd_input_generator),
        ("HD Dynamic Input", test_hd_dynamic_input),
        ("Encoding Experiment Single Trial", test_encoding_experiment_single_trial),
        ("Encoding Smart Storage", test_encoding_experiment_smart_storage),
        ("Decoding Analysis", test_decoding_analysis),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("HD INPUT ENCODING TEST SUMMARY")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:40s}: {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL HD ENCODING TESTS PASSED!")
        print("\nVerified capabilities:")
        print("  ✓ RNG extended for HD parameters")
        print("  ✓ Rate RNN with 200ms transient removal")
        print("  ✓ HD embedding (d-dimensional → k-dimensional)")
        print("  ✓ HD input generation and caching")
        print("  ✓ HD dynamic input (3 modes)")
        print("  ✓ Encoding experiment (200ms transient + 300ms encoding)")
        print("  ✓ Smart storage (neuron data only for low-dim extremes)")
        print("  ✓ Decoding analysis (SVD, PCA, spike jitter)")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
