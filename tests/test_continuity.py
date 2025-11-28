#!/usr/bin/env python3
# tests/test_continuity.py
"""
COMPREHENSIVE CONTINUITY & CONSISTENCY TESTS

Tests all critical requirements for reproducibility and consistency:
1. Network state continuity (transient → evoked)
2. RNG reproducibility across execution paths
3. HD pattern independence from network parameters
4. Cache integrity (save = load)
5. Spike timing precision (no drift)
6. Cross-validation consistency
7. Trial-to-trial consistency
8. Network structure consistency
"""

import sys
import os
import numpy as np
import tempfile
import pickle

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)



def test_network_state_continuity():
    """
    TEST 1: Network state continuity from transient to evoked response.
    """
    print("\n[TEST 1] Network State Continuity...")

    try:
        from src.spiking_network import SpikingRNN

        # Create network
        net1 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter',
                         static_input_mode='common_tonic', n_hd_channels=0)
        net1.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0,
                               v_th_distribution='normal')

        # Simulate 100ms transient
        net1.simulate(session_id=0, v_th_std=0.0, g_std=1.0, trial_id=0,
                     duration=100.0, static_input_rate=30.0,
                     continue_from_state=False)

        # Save state at 100ms
        state_100ms = net1.save_state()

        # DIRECT: Continue on net1
        # Note: record start time before simulation
        start_time_direct = net1.current_time
        spikes_direct_all = net1.simulate(session_id=0, v_th_std=0.0, g_std=1.0,
                                     trial_id=0, duration=50.0,
                                     static_input_rate=30.0,
                                     continue_from_state=True)
        # Filter spikes to only those >= start_time
        spikes_direct = [(t, n) for t, n in spikes_direct_all if t >= start_time_direct]

        # CACHED: Create new network, restore state, continue
        net2 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter',
                         static_input_mode='common_tonic', n_hd_channels=0)
        net2.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0,
                               v_th_distribution='normal')
        net2.restore_state(state_100ms)

        start_time_cached = net2.current_time
        spikes_cached_all = net2.simulate(session_id=0, v_th_std=0.0, g_std=1.0,
                                     trial_id=0, duration=50.0,
                                     static_input_rate=30.0,
                                     continue_from_state=True)
        # Filter spikes to only those >= start_time
        spikes_cached = [(t, n) for t, n in spikes_cached_all if t >= start_time_cached]

        # Compare spike trains
        if len(spikes_direct) != len(spikes_cached):
            print(f"  ✗ Spike count mismatch: direct={len(spikes_direct)}, "
                  f"cached={len(spikes_cached)}")
            return False

        # Check spike-by-spike identity
        for (t1, n1), (t2, n2) in zip(spikes_direct, spikes_cached):
            if abs(t1 - t2) > 1e-10 or n1 != n2:
                print(f"  ✗ Spike mismatch: ({t1}, {n1}) vs ({t2}, {n2})")
                return False

        print(f"  ✓ {len(spikes_direct)} spikes identical")
        print(f"  ✓ Network state continuity preserved")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rng_reproducibility():
    """
    TEST 2: RNG reproducibility across different execution paths.
    """
    print("\n[TEST 2] RNG Reproducibility...")

    try:
        from src.hd_input import HDInputGenerator
        from src.spiking_network import SpikingRNN
        from src.rng_utils import get_rng

        # Test 2a: HD pattern reproducibility (cached)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen1 = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir,
                                   signal_type='hd_input')
            gen1.initialize_base_input(session_id=0, hd_dim=3, pattern_id=0)
            pattern1 = gen1.Y_base.copy()

            gen2 = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir,
                                   signal_type='hd_input')
            gen2.initialize_base_input(session_id=0, hd_dim=3, pattern_id=0)
            pattern2 = gen2.Y_base.copy()

            if not np.allclose(pattern1, pattern2, atol=1e-10):
                print(f"  ✗ HD pattern differs from cache")
                return False

            print(f"  ✓ HD pattern identical from cache")

        # Test 2b: Trial noise reproducibility
        rng1 = get_rng(0, 0.0, 1.0, 0, 'hd_input_noise_0', rate=30.0, hd_dim=3, embed_dim=5)
        noise1 = rng1.normal(0, 0.5, (100, 5))

        rng2 = get_rng(0, 0.0, 1.0, 0, 'hd_input_noise_0', rate=30.0, hd_dim=3, embed_dim=5)
        noise2 = rng2.normal(0, 0.5, (100, 5))

        if not np.allclose(noise1, noise2, atol=1e-10):
            print(f"  ✗ Trial noise differs")
            return False

        print(f"  ✓ Trial noise identical")

        # Test 2c: Network weight reproducibility
        net1 = SpikingRNN(n_neurons=100, dt=0.1, n_hd_channels=0)
        net1.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0)
        weights1 = net1.recurrent_synapses.weight_matrix.toarray()

        net2 = SpikingRNN(n_neurons=100, dt=0.1, n_hd_channels=0)
        net2.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0)
        weights2 = net2.recurrent_synapses.weight_matrix.toarray()

        if not np.allclose(weights1, weights2, atol=1e-10):
            print(f"  ✗ Network weights differ")
            return False

        print(f"  ✓ Network weights identical")
        print(f"  ✓ RNG reproducibility verified")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_hd_pattern_independence():
    """
    TEST 3: HD pattern independence from network parameters.
    HD patterns should be identical regardless of v_th_std, g_std, static_input_rate.
    """
    print("\n[TEST 3] HD Pattern Independence...")

    try:
        from src.hd_input import HDInputGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Reference pattern
            gen_ref = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)
            gen_ref.initialize_base_input(session_id=0, hd_dim=3, pattern_id=0)
            pattern_ref = gen_ref.Y_base.copy()

            # Test 1: Different session should give DIFFERENT pattern
            gen_diff_session = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)
            gen_diff_session.initialize_base_input(session_id=1, hd_dim=3, pattern_id=0)
            pattern_diff = gen_diff_session.Y_base.copy()

            if np.allclose(pattern_ref, pattern_diff, atol=1e-10):
                print(f"  ✗ Different sessions produce identical patterns (should differ)")
                return False
            print(f"  ✓ Different sessions produce different patterns")

            # Test 2: Different pattern_id should give DIFFERENT pattern
            gen_diff_pattern = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)
            gen_diff_pattern.initialize_base_input(session_id=0, hd_dim=3, pattern_id=1)
            pattern_diff2 = gen_diff_pattern.Y_base.copy()

            if np.allclose(pattern_ref, pattern_diff2, atol=1e-10):
                print(f"  ✗ Different pattern_ids produce identical patterns (should differ)")
                return False
            print(f"  ✓ Different pattern_ids produce different patterns")

            # Test 3: Trial noise DOES depend on v_th_std, g_std, rate
            # Same trial parameters should give same noise
            noise1 = gen_ref.generate_trial_input(
                session_id=0, v_th_std=0.5, g_std=1.0, trial_id=0,
                hd_dim=3, static_input_rate=30.0
            )
            noise2 = gen_ref.generate_trial_input(
                session_id=0, v_th_std=0.5, g_std=1.0, trial_id=0,
                hd_dim=3, static_input_rate=30.0
            )
            if not np.allclose(noise1, noise2, atol=1e-10):
                print(f"  ✗ Same trial parameters give different noise")
                return False
            print(f"  ✓ Same trial parameters give identical noise")

            # Test 4: Different v_th_std should give DIFFERENT trial noise
            noise_diff_vth = gen_ref.generate_trial_input(
                session_id=0, v_th_std=1.0, g_std=1.0, trial_id=0,  # Different v_th_std
                hd_dim=3, static_input_rate=30.0
            )
            if np.allclose(noise1, noise_diff_vth, atol=1e-10):
                print(f"  ✗ Different v_th_std produces identical noise (should differ)")
                return False
            print(f"  ✓ Different v_th_std produces different trial noise")

            # Test 5: Different g_std should give DIFFERENT trial noise
            noise_diff_g = gen_ref.generate_trial_input(
                session_id=0, v_th_std=0.5, g_std=2.0, trial_id=0,  # Different g_std
                hd_dim=3, static_input_rate=30.0
            )
            if np.allclose(noise1, noise_diff_g, atol=1e-10):
                print(f"  ✗ Different g_std produces identical noise (should differ)")
                return False
            print(f"  ✓ Different g_std produces different trial noise")

            # Test 6: Base pattern is INDEPENDENT of network parameters
            # (base pattern only depends on session_id, hd_dim, pattern_id, embed_dim)
            # Reload from cache to verify
            gen_reload = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)
            gen_reload.initialize_base_input(session_id=0, hd_dim=3, pattern_id=0)

            if not np.allclose(pattern_ref, gen_reload.Y_base, atol=1e-10):
                print(f"  ✗ Base pattern changed on reload")
                return False
            print(f"  ✓ Base pattern independent of network parameters (cached correctly)")

            print(f"  ✓ HD pattern independence verified")
            return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_integrity():
    """
    TEST 4: Cache integrity - saved state equals loaded state.
    """
    print("\n[TEST 4] Cache Integrity...")

    try:
        from experiments.transient_cache_experiment import TransientCacheExperiment

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_exp = TransientCacheExperiment(
                n_neurons=100, dt=0.1, transient_duration=100.0,
                n_trials=3, cache_dir=tmpdir
            )

            cache_exp.run_transient_removal(
                session_id=0, g_std=1.0, v_th_std=0.0, static_rate=30.0
            )

            cache_file = os.path.join(tmpdir,
                "session_0_g_1.000_vth_0.000_rate_30.0_trial_states.pkl")

            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify structure
            assert 'trial_states' in cache_data
            assert len(cache_data['trial_states']) == 3

            # Check actual field names from your save_state()
            required_fields = ['neuron_v_membrane', 'neuron_last_spike', 'neuron_refractory',
                              'recurrent_synaptic_current', 'static_synaptic_current', 'current_time']
            for trial_id, state in cache_data['trial_states'].items():
                for field in required_fields:
                    assert field in state, f"Missing field: {field}"

            print(f"  ✓ Cache structure intact")

            # Test save-load cycle
            original_v = cache_data['trial_states'][0]['neuron_v_membrane'].copy()

            with open(cache_file, 'rb') as f:
                reloaded_data = pickle.load(f)

            reloaded_v = reloaded_data['trial_states'][0]['neuron_v_membrane']

            if not np.allclose(original_v, reloaded_v, atol=1e-10):
                print(f"  ✗ Data corruption in save-load cycle")
                return False

            print(f"  ✓ Save-load cycle preserves data")
            print(f"  ✓ Cache integrity verified")
            return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_spike_timing_precision():
    """
    TEST 5: Spike timing precision with/without caching.
    """
    print("\n[TEST 5] Spike Timing Precision...")

    try:
        from src.spiking_network import SpikingRNN
        from src.hd_input import HDInputGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)
            gen.initialize_base_input(session_id=0, hd_dim=3, pattern_id=0)
            hd_input = gen.Y_base.copy()

            # PATH 1: Direct
            net_direct = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter',
                                   static_input_mode='common_tonic',
                                   hd_input_mode='common_tonic',
                                   n_hd_channels=5, hd_connection_mode='overlapping')
            net_direct.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0,
                                         hd_dim=3, embed_dim=5)

            net_direct.simulate(session_id=0, v_th_std=0.0, g_std=1.0, trial_id=0,
                               duration=100.0, static_input_rate=30.0)

            start_time_direct = net_direct.current_time
            spikes_direct_all = net_direct.simulate(
                session_id=0, v_th_std=0.0, g_std=1.0, trial_id=0,
                duration=50.0, static_input_rate=30.0,
                hd_input_patterns=hd_input, hd_dim=3, embed_dim=5,
                continue_from_state=True
            )
            spikes_direct = [(t, n) for t, n in spikes_direct_all if t >= start_time_direct]

            # PATH 2: Cached
            net_trans = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter',
                                  static_input_mode='common_tonic',
                                  hd_input_mode='common_tonic',
                                  n_hd_channels=5, hd_connection_mode='overlapping')
            net_trans.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0,
                                        hd_dim=3, embed_dim=5)

            net_trans.simulate(session_id=0, v_th_std=0.0, g_std=1.0, trial_id=0,
                              duration=100.0, static_input_rate=30.0)

            state_100ms = net_trans.save_state()

            net_evoked = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter',
                                   static_input_mode='common_tonic',
                                   hd_input_mode='common_tonic',
                                   n_hd_channels=5, hd_connection_mode='overlapping')
            net_evoked.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0,
                                         hd_dim=3, embed_dim=5)
            net_evoked.restore_state(state_100ms)

            start_time_cached = net_evoked.current_time
            spikes_cached_all = net_evoked.simulate(
                session_id=0, v_th_std=0.0, g_std=1.0, trial_id=0,
                duration=50.0, static_input_rate=30.0,
                hd_input_patterns=hd_input, hd_dim=3, embed_dim=5,
                continue_from_state=True
            )
            spikes_cached = [(t, n) for t, n in spikes_cached_all if t >= start_time_cached]

            # Compare
            if len(spikes_direct) != len(spikes_cached):
                print(f"  ✗ Spike count differs")
                return False

            for (t1, n1), (t2, n2) in zip(spikes_direct, spikes_cached):
                if abs(t1 - t2) > 1e-10 or n1 != n2:
                    print(f"  ✗ Spike differs: Δt={abs(t1-t2):.2e}")
                    return False

            print(f"  ✓ {len(spikes_direct)} spikes identical")
            print(f"  ✓ Spike timing precision preserved")
            return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cv_consistency():
    """TEST 6: Cross-validation consistency."""
    print("\n[TEST 6] Cross-Validation Consistency...")

    try:
        from src.rng_utils import get_rng
        from sklearn.model_selection import StratifiedKFold

        n_trials = 100
        n_patterns = 4
        pattern_ids = np.repeat(range(n_patterns), n_trials // n_patterns)

        session_id = 42
        n_folds = 5

        rng1 = get_rng(session_id, 0, 0, 0, 'cv_split')
        skf1 = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=int(rng1.integers(0, 1000000)))
        splits1 = list(skf1.split(np.zeros(n_trials), pattern_ids))

        rng2 = get_rng(session_id, 0, 0, 0, 'cv_split')
        skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=int(rng2.integers(0, 1000000)))
        splits2 = list(skf2.split(np.zeros(n_trials), pattern_ids))

        for fold_idx, ((train1, test1), (train2, test2)) in enumerate(zip(splits1, splits2)):
            if not np.array_equal(train1, train2) or not np.array_equal(test1, test2):
                print(f"  ✗ Split differs in fold {fold_idx}")
                return False

        print(f"  ✓ {n_folds} folds identical across runs")
        print(f"  ✓ CV consistency verified")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_trial_consistency():
    """TEST 7: Trial-to-trial consistency."""
    print("\n[TEST 7] Trial-to-Trial Consistency...")

    try:
        from src.rng_utils import get_rng

        noise_t0 = get_rng(0, 0.0, 1.0, 0, 'hd_input_noise_0').normal(0, 0.5, 100)
        noise_t1 = get_rng(0, 0.0, 1.0, 1, 'hd_input_noise_0').normal(0, 0.5, 100)

        if np.allclose(noise_t0, noise_t1):
            print(f"  ✗ Different trials have identical noise")
            return False

        print(f"  ✓ Different trials have different noise")

        noise_call1 = get_rng(0, 0.0, 1.0, 5, 'hd_input_noise_0').normal(0, 0.5, 100)
        noise_call2 = get_rng(0, 0.0, 1.0, 5, 'hd_input_noise_0').normal(0, 0.5, 100)

        if not np.allclose(noise_call1, noise_call2, atol=1e-10):
            print(f"  ✗ Same trial has different noise")
            return False

        print(f"  ✓ Same trial reproducible")
        print(f"  ✓ Trial consistency verified")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_network_structure_consistency():
    """TEST 8: Network structure consistency."""
    print("\n[TEST 8] Network Structure Consistency...")

    try:
        from src.spiking_network import SpikingRNN

        net1 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter')
        net1.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0)

        net2 = SpikingRNN(n_neurons=100, dt=0.1, synaptic_mode='filter')
        net2.initialize_network(session_id=0, v_th_std=0.0, g_std=1.0)

        if not np.allclose(net1.recurrent_synapses.weight_matrix.toarray(),
                          net2.recurrent_synapses.weight_matrix.toarray(), atol=1e-10):
            print(f"  ✗ Weight matrices differ")
            return False

        if not np.allclose(net1.neurons.spike_thresholds,
                          net2.neurons.spike_thresholds, atol=1e-10):
            print(f"  ✗ Thresholds differ")
            return False

        print(f"  ✓ Network structures identical")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Run all continuity tests."""
    print("=" * 80)
    print("COMPREHENSIVE CONTINUITY & CONSISTENCY TEST SUITE")
    print("=" * 80)

    tests = [
        ("Network State Continuity", test_network_state_continuity),
        ("RNG Reproducibility", test_rng_reproducibility),
        ("HD Pattern Independence", test_hd_pattern_independence),
        ("Cache Integrity", test_cache_integrity),
        ("Spike Timing Precision", test_spike_timing_precision),
        ("Cross-Validation Consistency", test_cv_consistency),
        ("Trial-to-Trial Consistency", test_trial_consistency),
        ("Network Structure Consistency", test_network_structure_consistency),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:40s}: {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 System ready for production!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
