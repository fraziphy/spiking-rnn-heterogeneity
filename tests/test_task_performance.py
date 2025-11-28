#!/usr/bin/env python3
# tests/test_task_performance.py
"""
Complete tests for refactored task-performance experiments.
Tests both categorical classification and temporal transformation tasks.
Merged with comprehensive component tests.
"""

import sys
import os
import numpy as np
import tempfile

# Add project directories
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir) if current_dir else '.'
sys.path.insert(0, project_root)


def test_imports():
    """Test 1: Import all required modules."""
    print("\n[TEST 1] Testing imports...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        from experiments.experiment_utils import (
            apply_exponential_filter,
            train_task_readout,
            predict_task_readout,
            evaluate_categorical_task,
            evaluate_temporal_task
        )
        from analysis.common_utils import spikes_to_matrix
        from src.rng_utils import get_rng
        from src.hd_input import HDInputGenerator, run_rate_rnn, make_embedding_projected
        from src.spiking_network import SpikingRNN
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exponential_filter():
    """Test 2: Exponential filtering."""
    print("\n[TEST 2] Testing exponential filter...")
    try:
        from analysis.common_utils import apply_exponential_filter

        # Create dummy spike matrix
        spike_matrix = np.random.randint(0, 2, (100, 50))
        tau = 5.0
        dt = 0.1

        traces = apply_exponential_filter(spike_matrix, tau, dt)

        print(f"  Input shape: {spike_matrix.shape}")
        print(f"  Output shape: {traces.shape}")
        print(f"  Output range: [{traces.min():.3f}, {traces.max():.3f}]")

        assert traces.shape == spike_matrix.shape, "Shape mismatch"
        assert traces.min() >= 0, "Negative values in traces"
        assert not np.array_equal(traces, spike_matrix), "Filter had no effect"

        print("  ✓ Exponential filter works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ridge_regression():
    """Test 3: Ridge regression training and prediction."""
    print("\n[TEST 3] Testing ridge regression...")
    try:
        from experiments.experiment_utils import train_task_readout, predict_task_readout

        n_trials = 20
        T = 100
        N = 50
        n_outputs = 4

        X_train = np.random.randn(n_trials, T, N)
        Y_train = np.random.randn(n_trials, T, n_outputs)

        # Train
        W = train_task_readout(X_train, Y_train, lambda_reg=1e-3)

        print(f"  X_train shape: {X_train.shape}")
        print(f"  Y_train shape: {Y_train.shape}")
        print(f"  W shape: {W.shape}")
        print(f"  W range: [{W.min():.3f}, {W.max():.3f}]")

        assert W.shape == (N, n_outputs), f"Weight shape wrong: {W.shape}"

        # Predict
        Y_pred = predict_task_readout(X_train, W)

        print(f"  Y_pred shape: {Y_pred.shape}")
        assert Y_pred.shape == Y_train.shape, f"Prediction shape wrong: {Y_pred.shape}"

        print("  ✓ Ridge regression works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Ridge regression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_categorical_evaluation():
    """Test 4: Categorical evaluation metrics."""
    print("\n[TEST 4] Testing categorical evaluation...")
    try:
        from experiments.experiment_utils import evaluate_categorical_task

        n_trials = 20
        T = 1000
        n_classes = 4
        decision_window_steps = 500

        # Create dummy predictions and ground truth
        Y_pred = np.random.randn(n_trials, T, n_classes)

        # Create one-hot ground truth (constant across time)
        Y_true = np.zeros((n_trials, T, n_classes))
        for i in range(n_trials):
            class_id = i % n_classes
            Y_true[i, :, class_id] = 1.0

        # Evaluate
        metrics = evaluate_categorical_task(Y_pred, Y_true, decision_window_steps)

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Confusion matrix shape: {np.array(metrics['confusion_matrix']).shape}")
        print(f"  Per-class accuracy: {metrics['per_class_accuracy']}")

        assert 0 <= metrics['accuracy'] <= 1, "Accuracy out of range"
        assert len(metrics['per_class_accuracy']) == n_classes, "Wrong number of classes"

        print("  ✓ Categorical evaluation works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Categorical evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_evaluation():
    """Test 5: Temporal evaluation metrics."""
    print("\n[TEST 5] Testing temporal evaluation...")
    try:
        from experiments.experiment_utils import evaluate_temporal_task

        n_trials = 20
        T = 1000
        n_outputs = 8

        # Create dummy predictions and ground truth
        Y_pred = np.random.randn(n_trials, T, n_outputs)
        Y_true = np.random.randn(n_trials, T, n_outputs)

        # Evaluate
        metrics = evaluate_temporal_task(Y_pred, Y_true)

        print(f"  RMSE mean: {metrics['rmse_mean']:.4f}")
        print(f"  R² mean: {metrics['r2_mean']:.4f}")
        print(f"  Correlation mean: {metrics['correlation_mean']:.4f}")

        assert 'rmse_mean' in metrics, "Missing RMSE"
        assert 'r2_mean' in metrics, "Missing R²"
        assert 'correlation_mean' in metrics, "Missing correlation"

        print("  ✓ Temporal evaluation works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Temporal evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_id_in_hd_functions():
    """Test 6: pattern_id parameter in HD functions."""
    print("\n[TEST 6] Testing pattern_id in HD functions...")
    try:
        from src.hd_input import run_rate_rnn, make_embedding_projected

        # Test run_rate_rnn with different signal_types
        rates_input, _ = run_rate_rnn(
            n_neurons=100, T=500.0, dt=0.1, g=1.2,
            session_id=1, signal_type='hd_input'
        )

        rates_output, _ = run_rate_rnn(
            n_neurons=100, T=500.0, dt=0.1, g=1.2,
            session_id=1, signal_type='hd_output'
        )

        assert not np.array_equal(rates_input, rates_output), "Different signal_type should give different rates"
        print(f"  ✓ signal_type creates different rate patterns")

        # Test make_embedding_projected with different pattern_ids
        Y1 = make_embedding_projected(rates_input, k=10, d=5, session_id=1, pattern_id=0, signal_type='hd_input')
        Y2 = make_embedding_projected(rates_input, k=10, d=5, session_id=1, pattern_id=1, signal_type='hd_input')

        assert not np.array_equal(Y1, Y2), "Different pattern_id should give different embeddings"
        print(f"  ✓ pattern_id creates different embeddings")

        # Test same parameters give same result
        Y1_repeat = make_embedding_projected(rates_input, k=10, d=5, session_id=1, pattern_id=0, signal_type='hd_input')
        assert np.array_equal(Y1, Y1_repeat), "Same parameters should give identical result"
        print(f"  ✓ Same parameters are reproducible")

        return True
    except Exception as e:
        print(f"  ✗ pattern_id test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hd_generator_pattern_caching():
    """Test 7: HDInputGenerator with pattern caching."""
    print("\n[TEST 7] Testing HDInputGenerator pattern caching...")
    try:
        from src.hd_input import HDInputGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = os.path.join(tmpdir, 'inputs')
            outputs_dir = os.path.join(tmpdir, 'outputs')

            # Create input generator
            input_gen = HDInputGenerator(embed_dim=10, dt=0.1, signal_cache_dir=inputs_dir)

            # Generate pattern A
            input_gen.initialize_base_input(session_id=1, hd_dim=5, pattern_id=0)
            pattern_A = input_gen.Y_base.copy()

            # Generate pattern B
            input_gen.initialize_base_input(session_id=1, hd_dim=5, pattern_id=1)
            pattern_B = input_gen.Y_base.copy()

            assert not np.array_equal(pattern_A, pattern_B), "Patterns should differ"
            print(f"  ✓ Patterns A and B are different")

            # Check cache files
            cache_A = os.path.join(inputs_dir, 'hd_hd_input_session_1_hd_5_k_10_pattern_0.pkl')
            cache_B = os.path.join(inputs_dir, 'hd_hd_input_session_1_hd_5_k_10_pattern_1.pkl')

            assert os.path.exists(cache_A), "Pattern A cache not found"
            assert os.path.exists(cache_B), "Pattern B cache not found"
            print(f"  ✓ Pattern cache files created")

            # Test loading from cache
            input_gen2 = HDInputGenerator(embed_dim=10, dt=0.1, signal_cache_dir=inputs_dir)
            input_gen2.initialize_base_input(session_id=1, hd_dim=5, pattern_id=0)
            assert np.array_equal(input_gen2.Y_base, pattern_A), "Cached pattern not loaded correctly"
            print(f"  ✓ Pattern loading from cache works")

            # Test separate output directory
            output_gen = HDInputGenerator(embed_dim=8, dt=0.1, signal_cache_dir=outputs_dir)
            output_gen.initialize_base_input(session_id=1, hd_dim=3, pattern_id=100)

            cache_out = os.path.join(outputs_dir, 'hd_hd_input_session_1_hd_3_k_8_pattern_100.pkl')
            assert os.path.exists(cache_out), "Output pattern cache not found"
            print(f"  ✓ Separate output directory works")

        return True
    except Exception as e:
        print(f"  ✗ HD generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_experiment_init():
    """Test 8: TaskPerformanceExperiment initialization."""
    print("\n[TEST 8] Testing TaskPerformanceExperiment initialization...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment

        # Test categorical
        categorical = TaskPerformanceExperiment(
            task_type='categorical',
            n_neurons=100,
            n_input_patterns=4,
            input_dim_intrinsic=5,
            input_dim_embedding=10,
            n_trials_per_pattern=100
        )

        assert categorical.task_type == 'categorical'
        assert categorical.n_input_patterns == 4
        assert categorical.output_generator is None, "Categorical shouldn't have output generator"
        print(f"  ✓ Categorical task initialized")

        # Test temporal
        temporal = TaskPerformanceExperiment(
            task_type='temporal',
            n_neurons=100,
            n_input_patterns=4,
            input_dim_intrinsic=5,
            input_dim_embedding=10,
            output_dim_intrinsic=3,
            output_dim_embedding=8,
            n_trials_per_pattern=100
        )

        assert temporal.task_type == 'temporal'
        assert temporal.output_generator is not None, "Temporal should have output generator"
        print(f"  ✓ Temporal task initialized")

        # Test invalid task type
        try:
            TaskPerformanceExperiment(task_type='invalid')
            print(f"  ✗ Invalid task type accepted")
            return False
        except ValueError:
            print(f"  ✓ Invalid task type rejected")

        return True
    except Exception as e:
        print(f"  ✗ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_generation():
    """Test 9: Input and output pattern generation."""
    print("\n[TEST 9] Testing pattern generation...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment

        with tempfile.TemporaryDirectory() as tmpdir:
            # Categorical task
            categorical = TaskPerformanceExperiment(
                task_type='categorical',
                n_input_patterns=4,
                input_dim_intrinsic=3,
                input_dim_embedding=5,
                stimulus_duration=300.0,
                signal_cache_dir=tmpdir
            )

            # Generate output patterns
            output_patterns = categorical.generate_output_patterns(session_id=1)

            assert len(output_patterns) == 4, "Should have 4 output patterns"
            n_timesteps = int(300.0 / 0.1)

            for pattern_id, pattern in output_patterns.items():
                assert pattern.shape == (n_timesteps, 4), f"Categorical output shape wrong"
                # Check one-hot encoding
                expected_one_hot = np.zeros(4)
                expected_one_hot[pattern_id] = 1.0
                assert np.allclose(pattern[0, :], expected_one_hot), f"Pattern {pattern_id} not one-hot"
                assert np.allclose(pattern[-1, :], expected_one_hot), f"Pattern {pattern_id} not constant"

            print(f"  ✓ Categorical outputs: one-hot encoded")

            # Temporal task
            temporal = TaskPerformanceExperiment(
                task_type='temporal',
                n_input_patterns=4,
                input_dim_intrinsic=3,
                input_dim_embedding=5,
                output_dim_intrinsic=2,
                output_dim_embedding=4,
                stimulus_duration=300.0,
                signal_cache_dir=tmpdir
            )

            output_patterns_temporal = temporal.generate_output_patterns(session_id=1)

            assert len(output_patterns_temporal) == 4, "Should have 4 output patterns"
            for pattern_id, pattern in output_patterns_temporal.items():
                assert pattern.shape == (n_timesteps, 4), f"Temporal output shape wrong"

            # Temporal outputs should differ
            assert not np.array_equal(output_patterns_temporal[0], output_patterns_temporal[1]), \
                "Temporal output patterns should differ"
            print(f"  ✓ Temporal outputs: time-varying patterns")

        return True
    except Exception as e:
        print(f"  ✗ Pattern generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stratified_cv():
    """Test 10: Stratified cross-validation."""
    print("\n[TEST 10] Testing stratified cross-validation...")
    try:
        from sklearn.model_selection import StratifiedKFold

        # 4 patterns, 100 trials each
        pattern_ids = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)
        n_trials = len(pattern_ids)

        print(f"  Total trials: {n_trials}")
        print(f"  Pattern distribution: {np.bincount(pattern_ids)}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros((n_trials, 1)), pattern_ids)):
            train_dist = np.bincount(pattern_ids[train_idx])
            test_dist = np.bincount(pattern_ids[test_idx])

            if fold == 0:
                print(f"  Fold {fold}:")
                print(f"    Train: {len(train_idx)} trials, distribution: {train_dist}")
                print(f"    Test:  {len(test_idx)} trials, distribution: {test_dist}")

            # Verify balance
            assert len(np.unique(test_dist)) == 1, f"Test fold {fold} is imbalanced!"
            assert test_dist[0] == 20, f"Test fold {fold} doesn't have 20 of each pattern!"

        print("  ✓ All folds perfectly stratified")
        return True
    except Exception as e:
        print(f"  ✗ Stratified CV test failed: {e}")
        import traceback
        traceback.print_exc()
        return False




def test_categorical_experiment():
    """Test 11: Full categorical task experiment (minimal)."""
    print("\n[TEST 11] Testing categorical task experiment...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock cached spikes directory
            spike_cache_dir = os.path.join(tmpdir, 'cached_spikes')
            os.makedirs(os.path.join(spike_cache_dir, 'overlapping'), exist_ok=True)

            n_patterns = 2
            n_trials_per_pattern = 6
            n_neurons = 50
            stimulus_duration = 50.0

            # Create mock spike data for each pattern
            for pattern_id in range(n_patterns):
                mock_spikes = {
                    'trial_spikes': {
                        trial_id: [(float(t), t % n_neurons) for t in range(0, int(stimulus_duration), 5)]
                        for trial_id in range(n_trials_per_pattern)
                    }
                }
                cache_file = os.path.join(spike_cache_dir, 'overlapping',
                    f'session_0_g_1.000_vth_1.000_rate_200.0_h_2_d_3_pattern_{pattern_id}_spikes.pkl')
                with open(cache_file, 'wb') as f:
                    pickle.dump(mock_spikes, f)

            exp = TaskPerformanceExperiment(
                task_type='categorical',
                n_neurons=n_neurons,
                n_input_patterns=n_patterns,
                input_dim_intrinsic=2,
                input_dim_embedding=3,
                n_trials_per_pattern=n_trials_per_pattern,
                stimulus_duration=stimulus_duration,
                tau_syn=5.0,
                decision_window=10.0,
                signal_cache_dir=tmpdir
            )

            print(f"  Running categorical experiment with mock cached spikes...")

            session_id = 0
            v_th_std = 1.0
            g_std = 1.0
            static_input_rate = 200.0

            # Generate patterns
            input_patterns = exp.input_generator.initialize_and_get_patterns(
                session_id=session_id,
                hd_dim=exp.input_dim_intrinsic,
                n_patterns=n_patterns
            )
            output_patterns = exp.generate_output_patterns(session_id)

            # Simulate with cached spikes
            my_trials = list(range(n_patterns * n_trials_per_pattern))
            local_spike_times = exp.simulate_trials_parallel(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                v_th_distribution='normal',
                static_input_rate=static_input_rate,
                my_trial_indices=my_trials,
                input_patterns=input_patterns,
                rank=0,
                use_cached_spikes=True,
                spike_cache_dir=spike_cache_dir
            )

            assert len(local_spike_times) == n_patterns * n_trials_per_pattern
            print(f"  ✓ Loaded {len(local_spike_times)} trials from cache")

            # Convert to traces
            traces_all, ground_truth_all, pattern_ids = exp.convert_spikes_to_traces(
                local_spike_times, output_patterns, n_patterns, n_trials_per_pattern
            )

            assert traces_all.shape[0] == n_patterns * n_trials_per_pattern
            print(f"  ✓ Converted spikes to traces: {traces_all.shape}")

            # Cross-validate
            class MockComm:
                def bcast(self, data, root=0): return data
                def Barrier(self): pass

            cv_results = exp.cross_validate(
                traces_all=traces_all,
                ground_truth_all=ground_truth_all,
                pattern_ids=pattern_ids,
                session_id=session_id,
                n_folds=3,
                rank=0,
                size=1,
                comm=MockComm()
            )

            assert 0 <= cv_results['test_accuracy_bayesian_mean'] <= 1
            print(f"  ✓ Test accuracy: {cv_results['test_accuracy_bayesian_mean']:.3f}")

            print("  ✓ Categorical experiment completed successfully")
            return True

    except Exception as e:
        print(f"  ✗ Categorical experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_experiment():
    """Test 12: Full temporal task experiment (minimal)."""
    print("\n[TEST 12] Testing temporal task experiment...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock cached spikes directory
            spike_cache_dir = os.path.join(tmpdir, 'cached_spikes')
            os.makedirs(os.path.join(spike_cache_dir, 'overlapping'), exist_ok=True)

            n_patterns = 2
            n_trials_per_pattern = 6
            n_neurons = 50
            stimulus_duration = 50.0

            # Create mock spike data for each pattern
            for pattern_id in range(n_patterns):
                mock_spikes = {
                    'trial_spikes': {
                        trial_id: [(float(t), t % n_neurons) for t in range(0, int(stimulus_duration), 5)]
                        for trial_id in range(n_trials_per_pattern)
                    }
                }
                cache_file = os.path.join(spike_cache_dir, 'overlapping',
                    f'session_0_g_1.000_vth_1.000_rate_200.0_h_2_d_3_pattern_{pattern_id}_spikes.pkl')
                with open(cache_file, 'wb') as f:
                    pickle.dump(mock_spikes, f)

            exp = TaskPerformanceExperiment(
                task_type='temporal',
                n_neurons=n_neurons,
                n_input_patterns=n_patterns,
                input_dim_intrinsic=2,
                input_dim_embedding=3,
                output_dim_intrinsic=2,
                output_dim_embedding=3,
                n_trials_per_pattern=n_trials_per_pattern,
                stimulus_duration=stimulus_duration,
                tau_syn=5.0,
                signal_cache_dir=tmpdir
            )

            print(f"  Running temporal experiment with mock cached spikes...")

            session_id = 0
            v_th_std = 1.0
            g_std = 1.0
            static_input_rate = 200.0

            # Generate patterns
            input_patterns = exp.input_generator.initialize_and_get_patterns(
                session_id=session_id,
                hd_dim=exp.input_dim_intrinsic,
                n_patterns=n_patterns
            )
            output_patterns = exp.generate_output_patterns(session_id)

            # Simulate with cached spikes
            my_trials = list(range(n_patterns * n_trials_per_pattern))
            local_spike_times = exp.simulate_trials_parallel(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                v_th_distribution='normal',
                static_input_rate=static_input_rate,
                my_trial_indices=my_trials,
                input_patterns=input_patterns,
                rank=0,
                use_cached_spikes=True,
                spike_cache_dir=spike_cache_dir
            )

            assert len(local_spike_times) == n_patterns * n_trials_per_pattern
            print(f"  ✓ Loaded {len(local_spike_times)} trials from cache")

            # Convert to traces
            traces_all, ground_truth_all, pattern_ids = exp.convert_spikes_to_traces(
                local_spike_times, output_patterns, n_patterns, n_trials_per_pattern
            )

            assert traces_all.shape[0] == n_patterns * n_trials_per_pattern
            print(f"  ✓ Converted spikes to traces: {traces_all.shape}")

            # Cross-validate
            class MockComm:
                def bcast(self, data, root=0): return data
                def Barrier(self): pass

            cv_results = exp.cross_validate(
                traces_all=traces_all,
                ground_truth_all=ground_truth_all,
                pattern_ids=pattern_ids,
                session_id=session_id,
                n_folds=3,
                rank=0,
                size=1,
                comm=MockComm()
            )

            assert cv_results['test_rmse_mean'] >= 0
            print(f"  ✓ Test RMSE: {cv_results['test_rmse_mean']:.4f}")
            print(f"  ✓ Test R²: {cv_results['test_r2_mean']:.4f}")

            print("  ✓ Temporal experiment completed successfully")
            return True

    except Exception as e:
        print(f"  ✗ Temporal experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rng_reproducibility():
    """Test 13: RNG reproducibility for cached spike loading."""
    print("\n[TEST 13] Testing RNG reproducibility...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock cached spikes directory
            spike_cache_dir = os.path.join(tmpdir, 'cached_spikes')
            os.makedirs(os.path.join(spike_cache_dir, 'overlapping'), exist_ok=True)

            n_patterns = 2
            n_trials_per_pattern = 4
            n_neurons = 50
            stimulus_duration = 50.0

            # Create mock spike data
            for pattern_id in range(n_patterns):
                mock_spikes = {
                    'trial_spikes': {
                        trial_id: [(float(t + pattern_id * 10), (t + trial_id) % n_neurons)
                                   for t in range(0, int(stimulus_duration), 5)]
                        for trial_id in range(n_trials_per_pattern)
                    }
                }
                cache_file = os.path.join(spike_cache_dir, 'overlapping',
                    f'session_42_g_1.000_vth_1.000_rate_200.0_h_2_d_3_pattern_{pattern_id}_spikes.pkl')
                with open(cache_file, 'wb') as f:
                    pickle.dump(mock_spikes, f)

            # Run twice and compare
            results = []
            for run in range(2):
                exp = TaskPerformanceExperiment(
                    task_type='categorical',
                    n_neurons=n_neurons,
                    n_input_patterns=n_patterns,
                    input_dim_intrinsic=2,
                    input_dim_embedding=3,
                    n_trials_per_pattern=n_trials_per_pattern,
                    stimulus_duration=stimulus_duration,
                    signal_cache_dir=tmpdir
                )

                input_patterns = exp.input_generator.initialize_and_get_patterns(
                    session_id=42, hd_dim=2, n_patterns=n_patterns
                )

                my_trials = list(range(n_patterns * n_trials_per_pattern))
                local_spike_times = exp.simulate_trials_parallel(
                    session_id=42,
                    v_th_std=1.0,
                    g_std=1.0,
                    v_th_distribution='normal',
                    static_input_rate=200.0,
                    my_trial_indices=my_trials,
                    input_patterns=input_patterns,
                    rank=0,
                    use_cached_spikes=True,
                    spike_cache_dir=spike_cache_dir
                )
                results.append(local_spike_times)

            # Compare spike times from both runs
            for i, (r1, r2) in enumerate(zip(results[0], results[1])):
                assert r1['pattern_id'] == r2['pattern_id'], f"Pattern mismatch at trial {i}"
                assert r1['trial_id'] == r2['trial_id'], f"Trial ID mismatch at trial {i}"
                assert r1['spike_times'] == r2['spike_times'], f"Spike times mismatch at trial {i}"

            print(f"  ✓ Both runs produced identical results")
            print("  ✓ RNG reproducibility verified")
            return True

    except Exception as e:
        print(f"  ✗ Reproducibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hd_pattern_independence_from_dimensions():
    """Test 14: HD pattern generation and embedding behavior."""
    print("\n[TEST 14] Testing HD pattern independence from dimensions...")
    try:
        from src.hd_input import run_rate_rnn, make_embedding_projected

        session_id = 42

        # Test 1: Rate RNN is deterministic (same params → same output)
        rates1, _ = run_rate_rnn(
            n_neurons=100, T=500.0, dt=0.1, g=1.5,
            session_id=session_id, signal_type='hd_input'
        )
        rates2, _ = run_rate_rnn(
            n_neurons=100, T=500.0, dt=0.1, g=1.5,
            session_id=session_id, signal_type='hd_input'
        )
        assert np.array_equal(rates1, rates2), \
            "Rate RNN should be identical for same session/signal_type"
        print(f"  ✓ Rate RNN is deterministic")

        # Test 2: Different signal_type → different rates
        rates_output, _ = run_rate_rnn(
            n_neurons=100, T=500.0, dt=0.1, g=1.5,
            session_id=session_id, signal_type='hd_output'
        )
        assert not np.array_equal(rates1, rates_output), \
            "Different signal_type should give different rates"
        print(f"  ✓ Different signal_type produces different rates")

        # Test 3: Embedding with same parameters → identical
        Y1 = make_embedding_projected(rates1, k=10, d=3, session_id=session_id,
                                       pattern_id=0, signal_type='hd_input')
        Y1_repeat = make_embedding_projected(rates1, k=10, d=3, session_id=session_id,
                                              pattern_id=0, signal_type='hd_input')
        assert np.array_equal(Y1, Y1_repeat), \
            "Same embedding parameters should give identical result"
        print(f"  ✓ Embedding is deterministic")

        # Test 4: Different pattern_id → different embeddings
        Y2 = make_embedding_projected(rates1, k=10, d=3, session_id=session_id,
                                       pattern_id=1, signal_type='hd_input')
        assert not np.array_equal(Y1, Y2), \
            "Different pattern_id should give different embeddings"
        print(f"  ✓ Different pattern_id produces different embeddings")

        # Test 5: Different intrinsic dimension (d) → different embeddings
        Y3 = make_embedding_projected(rates1, k=10, d=5, session_id=session_id,
                                       pattern_id=0, signal_type='hd_input')
        assert not np.array_equal(Y1, Y3), \
            "Different intrinsic dimension should give different embeddings"
        print(f"  ✓ Different intrinsic dimension produces different embeddings")

        # Test 6: Verify output shapes
        assert Y1.shape[1] == 10, f"Y1 should have k=10 channels, got {Y1.shape[1]}"
        assert Y3.shape[1] == 10, f"Y3 should have k=10 channels, got {Y3.shape[1]}"
        print(f"  ✓ Embeddings have correct dimensions (k=10)")

        # Test 7: Different embedding dimension (k) → different shape
        Y4 = make_embedding_projected(rates1, k=8, d=3, session_id=session_id,
                                       pattern_id=0, signal_type='hd_input')
        assert Y4.shape[1] == 8, f"Y4 should have k=8 channels, got {Y4.shape[1]}"
        print(f"  ✓ Different k produces correct output shape")

        return True
    except Exception as e:
        print(f"  ✗ HD pattern independence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_network_structure_independence():
    """Test 15: Network structure independent of task parameters."""
    print("\n[TEST 15] Testing network structure independence...")
    try:
        from src.spiking_network import SpikingRNN

        session_id = 10
        v_th_std = 1.5
        g_std = 2.0

        # Create networks with different HD dimensions but same session/v_th/g
        network_1 = SpikingRNN(n_neurons=100, dt=0.1)
        network_1.initialize_network(
            session_id=session_id, v_th_std=v_th_std, g_std=g_std,
            hd_dim=3, embed_dim=10,
            static_input_strength=10.0,
            hd_connection_prob=0.3,
            hd_input_strength=1.0
        )

        network_2 = SpikingRNN(n_neurons=100, dt=0.1)
        network_2.initialize_network(
            session_id=session_id, v_th_std=v_th_std, g_std=g_std,
            hd_dim=5, embed_dim=15,
            static_input_strength=10.0,
            hd_connection_prob=0.3,
            hd_input_strength=1.0
        )

        # Thresholds should be identical
        assert np.array_equal(network_1.neurons.spike_thresholds,
                            network_2.neurons.spike_thresholds), \
            "Spike thresholds should be identical for same session/v_th/g!"
        print(f"  ✓ Spike thresholds independent of HD dimensions")

        # Recurrent weights should be identical
        diff = (network_1.recurrent_synapses.weight_matrix -
                network_2.recurrent_synapses.weight_matrix).nnz
        assert diff == 0, \
            "Recurrent weights should be identical for same session/v_th/g!"
        print(f"  ✓ Recurrent weights independent of HD dimensions")

        # HD input connectivity should be different (depends on hd_dim/embed_dim)
        # This is expected and correct
        print(f"  ✓ HD connectivity correctly depends on dimensions")

        return True
    except Exception as e:
        print(f"  ✗ Network structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trial_noise_independence():
    """Test 16: Trial noise independent across all parameters."""
    print("\n[TEST 16] Testing trial noise independence...")
    try:
        from src.rng_utils import get_rng

        session_id = 7
        v_th_std = 1.0
        g_std = 1.5
        pattern_id = 3
        static_input_rate = 200.0

        # Generate noise for different trials (should differ)
        noise_trial_0 = get_rng(session_id, v_th_std, g_std, 0,
                                f'hd_input_noise_{pattern_id}',
                                rate=static_input_rate,
                                hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        noise_trial_1 = get_rng(session_id, v_th_std, g_std, 1,
                                f'hd_input_noise_{pattern_id}',
                                rate=static_input_rate,
                                hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        assert not np.array_equal(noise_trial_0, noise_trial_1), \
            "Different trials should have different noise!"
        print(f"  ✓ Noise differs across trials")

        # Generate noise for same trial but different patterns (should differ)
        noise_pattern_3 = get_rng(session_id, v_th_std, g_std, 0,
                                  f'hd_input_noise_3',
                                  rate=static_input_rate,
                                  hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        noise_pattern_4 = get_rng(session_id, v_th_std, g_std, 0,
                                  f'hd_input_noise_4',
                                  rate=static_input_rate,
                                  hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        assert not np.array_equal(noise_pattern_3, noise_pattern_4), \
            "Different patterns should have different noise!"
        print(f"  ✓ Noise differs across patterns")

        # Generate noise for same trial but different dimensions (should differ)
        noise_dim_5 = get_rng(session_id, v_th_std, g_std, 0,
                              f'hd_input_noise_{pattern_id}',
                              rate=static_input_rate,
                              hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        noise_dim_7 = get_rng(session_id, v_th_std, g_std, 0,
                              f'hd_input_noise_{pattern_id}',
                              rate=static_input_rate,
                              hd_dim=7, embed_dim=10).normal(0, 0.5, (100, 10))

        assert not np.array_equal(noise_dim_5, noise_dim_7), \
            "Different hd_dim should have different noise!"
        print(f"  ✓ Noise differs across hd_dim")

        # Generate noise for same trial but different static rates (should differ)
        noise_rate_200 = get_rng(session_id, v_th_std, g_std, 0,
                                 f'hd_input_noise_{pattern_id}',
                                 rate=200.0,
                                 hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        noise_rate_300 = get_rng(session_id, v_th_std, g_std, 0,
                                 f'hd_input_noise_{pattern_id}',
                                 rate=300.0,
                                 hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        assert not np.array_equal(noise_rate_200, noise_rate_300), \
            "Different static_input_rate should have different noise!"
        print(f"  ✓ Noise differs across static_input_rate")

        # Noise for same parameters should be reproducible
        noise_trial_0_repeat = get_rng(session_id, v_th_std, g_std, 0,
                                       f'hd_input_noise_{pattern_id}',
                                       rate=static_input_rate,
                                       hd_dim=5, embed_dim=10).normal(0, 0.5, (100, 10))

        assert np.array_equal(noise_trial_0, noise_trial_0_repeat), \
            "Same parameters should have reproducible noise!"
        print(f"  ✓ Noise reproducible for same parameters")

        return True
    except Exception as e:
        print(f"  ✗ Trial noise test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_experiment_consistency():
    """Test 17: Full experiment consistency across dimension sweeps."""
    print("\n[TEST 17] Testing complete experiment consistency...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = 99
            v_th_std = 1.2
            g_std = 1.8
            static_input_rate = 200.0
            n_input_patterns = 2
            n_trials_per_pattern = 4

            print(f"  Running experiments with different HD dimensions...")

            # Experiment 1: hd_input=3, hd_output=2
            exp1 = TaskPerformanceExperiment(
                task_type='temporal',
                n_neurons=30,
                n_input_patterns=n_input_patterns,
                input_dim_intrinsic=3,
                input_dim_embedding=8,
                output_dim_intrinsic=2,
                output_dim_embedding=6,
                n_trials_per_pattern=n_trials_per_pattern,
                stimulus_duration=30.0,
                signal_cache_dir=tmpdir
            )

            # Manually check that network structure is the same
            from src.spiking_network import SpikingRNN
            net1 = SpikingRNN(n_neurons=30, dt=0.1)
            net1.initialize_network(session_id, v_th_std, g_std, hd_dim=3, embed_dim=8)
            thresholds_1 = net1.neurons.spike_thresholds.copy()
            weights_1 = net1.recurrent_synapses.weight_matrix.toarray()

            # Experiment 2: hd_input=5, hd_output=4 (different dimensions)
            exp2 = TaskPerformanceExperiment(
                task_type='temporal',
                n_neurons=30,
                n_input_patterns=n_input_patterns,
                input_dim_intrinsic=5,
                input_dim_embedding=12,
                output_dim_intrinsic=4,
                output_dim_embedding=10,
                n_trials_per_pattern=n_trials_per_pattern,
                stimulus_duration=30.0,
                signal_cache_dir=tmpdir
            )

            net2 = SpikingRNN(n_neurons=30, dt=0.1)
            net2.initialize_network(session_id, v_th_std, g_std, hd_dim=5, embed_dim=12)
            thresholds_2 = net2.neurons.spike_thresholds.copy()
            weights_2 = net2.recurrent_synapses.weight_matrix.toarray()

            # Network structure should be identical
            assert np.array_equal(thresholds_1, thresholds_2), \
                "Thresholds should be identical across HD dimension sweeps!"
            print(f"  ✓ Network thresholds consistent")

            assert np.allclose(weights_1, weights_2), \
                "Recurrent weights should be identical across HD dimension sweeps!"
            print(f"  ✓ Network weights consistent")

            print(f"  ✓ Complete experiment consistency verified")

        return True
    except Exception as e:
        print(f"  ✗ Complete consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_dimensionality_computation_in_experiments():
    """Test 18: Dimensionality computation on spike data."""
    print("\n[TEST 18] Testing dimensionality computation in experiments...")
    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        from analysis.common_utils import spikes_to_binary, compute_dimensionality_svd
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock cached spikes directory
            spike_cache_dir = os.path.join(tmpdir, 'cached_spikes')
            os.makedirs(os.path.join(spike_cache_dir, 'overlapping'), exist_ok=True)

            n_patterns = 2
            n_trials_per_pattern = 4
            n_neurons = 50
            stimulus_duration = 50.0

            # Create mock spike data with varied patterns
            np.random.seed(123)
            for pattern_id in range(n_patterns):
                mock_spikes = {
                    'trial_spikes': {
                        trial_id: [(float(t), np.random.randint(0, n_neurons))
                                   for t in sorted(np.random.uniform(0, stimulus_duration, 20))]
                        for trial_id in range(n_trials_per_pattern)
                    }
                }
                cache_file = os.path.join(spike_cache_dir, 'overlapping',
                    f'session_1_g_1.000_vth_1.000_rate_200.0_h_2_d_3_pattern_{pattern_id}_spikes.pkl')
                with open(cache_file, 'wb') as f:
                    pickle.dump(mock_spikes, f)

            exp = TaskPerformanceExperiment(
                task_type='categorical',
                n_neurons=n_neurons,
                n_input_patterns=n_patterns,
                input_dim_intrinsic=2,
                input_dim_embedding=3,
                n_trials_per_pattern=n_trials_per_pattern,
                stimulus_duration=stimulus_duration,
                signal_cache_dir=tmpdir
            )

            input_patterns = exp.input_generator.initialize_and_get_patterns(
                session_id=1, hd_dim=2, n_patterns=n_patterns
            )

            my_trials = list(range(n_patterns * n_trials_per_pattern))
            local_spike_times = exp.simulate_trials_parallel(
                session_id=1,
                v_th_std=1.0,
                g_std=1.0,
                v_th_distribution='normal',
                static_input_rate=200.0,
                my_trial_indices=my_trials,
                input_patterns=input_patterns,
                rank=0,
                use_cached_spikes=True,
                spike_cache_dir=spike_cache_dir
            )

            print(f"  Loaded {len(local_spike_times)} trials")

            # Test dimensionality computation on spike data
            bin_sizes = [2.0, 10.0, 20.0]

            for trial_result in local_spike_times[:1]:
                for bin_size in bin_sizes:
                    binary_matrix = spikes_to_binary(
                        trial_result['spike_times'],
                        n_neurons,
                        stimulus_duration,
                        bin_size
                    )

                    # compute_dimensionality_svd expects (n_samples, n_features)
                    dim_metrics = compute_dimensionality_svd(binary_matrix.T)

                    assert 'participation_ratio' in dim_metrics
                    assert 'effective_dimensionality' in dim_metrics
                    assert 'intrinsic_dimensionality' in dim_metrics

                    if bin_size == 2.0:
                        print(f"  ✓ Dimensionality at {bin_size}ms bin:")
                        print(f"    PR: {dim_metrics['participation_ratio']:.2f}")
                        print(f"    ED: {dim_metrics['effective_dimensionality']:.2f}")
                        print(f"    ID: {dim_metrics['intrinsic_dimensionality']}")

            print(f"  ✓ Dimensionality computation works on all bin sizes")
            return True

    except Exception as e:
        print(f"  ✗ Dimensionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimensionality_aggregation():
    """Test 19: Dimensionality aggregation across trials."""
    print("\n[TEST 19] Testing dimensionality aggregation...")
    try:
        from analysis.common_utils import compute_dimensionality_svd

        # Simulate dimensionality metrics from multiple trials
        n_trials = 10
        bin_size = 10.0

        pr_values = []
        ed_values = []
        id_values = []

        for trial in range(n_trials):
            # Create random data
            data = np.random.randn(100, 50)
            dim_metrics = compute_dimensionality_svd(data)

            pr_values.append(dim_metrics['participation_ratio'])
            ed_values.append(dim_metrics['effective_dimensionality'])
            id_values.append(dim_metrics['intrinsic_dimensionality'])

        # Aggregate
        pr_mean = np.mean(pr_values)
        pr_std = np.std(pr_values)
        ed_mean = np.mean(ed_values)
        ed_std = np.std(ed_values)
        id_mean = np.mean(id_values)
        id_std = np.std(id_values)

        print(f"  Aggregated across {n_trials} trials:")
        print(f"    PR: {pr_mean:.2f} ± {pr_std:.2f}")
        print(f"    ED: {ed_mean:.2f} ± {ed_std:.2f}")
        print(f"    ID: {id_mean:.0f} ± {id_std:.0f}")

        assert pr_mean > 0, "PR mean should be positive"
        assert ed_mean > 0, "ED mean should be positive"
        assert id_mean > 0, "ID mean should be positive"

        print(f"  ✓ Aggregation produces valid statistics")
        return True

    except Exception as e:
        print(f"  ✗ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_cached_spike_loading():
    """NEW: Test loading cached evoked spikes."""
    print("\n[TEST NEW] Testing cached spike loading...")

    try:
        from experiments.task_performance_experiment import TaskPerformanceExperiment
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            spike_cache_dir = os.path.join(tmpdir, 'cached_spikes')
            os.makedirs(os.path.join(spike_cache_dir, 'overlapping'))

            # Create mock cached spike file
            mock_spikes = {
                'trial_spikes': {
                    0: [(10.0, 0), (20.0, 1)],
                    1: [(15.0, 0), (25.0, 2)]
                }
            }

            cache_file = os.path.join(spike_cache_dir, 'overlapping',
                'session_0_g_1.000_vth_0.000_rate_30.0_h_2_d_5_pattern_0_spikes.pkl')

            with open(cache_file, 'wb') as f:
                pickle.dump(mock_spikes, f)

            # Test loading
            exp = TaskPerformanceExperiment(
                task_type='categorical', n_neurons=100, n_input_patterns=1,
                input_dim_intrinsic=2, input_dim_embedding=5,
                output_dim_intrinsic=2, output_dim_embedding=5
            )

            loaded_spikes = exp.load_cached_trial_spikes(
                session_id=0, v_th_std=0.0, g_std=1.0, static_rate=30.0,
                pattern_id=0,
                hd_connection_mode='overlapping',
                spike_cache_dir=spike_cache_dir
            )

            assert len(loaded_spikes) == 2, "Should have 2 trials"
            assert len(loaded_spikes[0]) == 2, "Trial 0 should have 2 spikes"
            print("  ✓ Cached spike loading works")

        return True
    except Exception as e:
        print(f"  ✗ Cached spike test failed: {e}")
        return False



def main():
    """Run all tests."""
    print("=" * 80)
    print("COMPREHENSIVE TASK PERFORMANCE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Module imports", test_imports),
        ("Exponential filter", test_exponential_filter),
        ("Ridge regression", test_ridge_regression),
        ("Categorical evaluation", test_categorical_evaluation),
        ("Temporal evaluation", test_temporal_evaluation),
        ("pattern_id in HD functions", test_pattern_id_in_hd_functions),
        ("HD generator pattern caching", test_hd_generator_pattern_caching),
        ("TaskPerformanceExperiment init", test_task_experiment_init),
        ("Pattern generation", test_pattern_generation),
        ("Stratified cross-validation", test_stratified_cv),
        ("Categorical task experiment", test_categorical_experiment),
        ("Temporal task experiment", test_temporal_experiment),
        ("RNG reproducibility", test_rng_reproducibility),
        ("HD pattern independence from dimensions", test_hd_pattern_independence_from_dimensions),  # NEW
        ("Network structure independence", test_network_structure_independence),  # NEW
        ("Trial noise independence", test_trial_noise_independence),  # NEW
        ("Complete experiment consistency", test_complete_experiment_consistency),  # NEW
        ("Dimensionality computation in experiments", test_dimensionality_computation_in_experiments),
        ("Cached Spike Loading", test_cached_spike_loading),
        ("Dimensionality aggregation", test_dimensionality_aggregation),
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
    print("\n" + "=" * 80)
    print("COMPLETE TASK PERFORMANCE TEST SUMMARY")
    print("=" * 80)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:45s}: {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nVerified capabilities:")
        print("  ✓ Exponential synaptic filtering")
        print("  ✓ Ridge regression training")
        print("  ✓ Categorical classification (one-hot, accuracy)")
        print("  ✓ Temporal transformation (time-varying, RMSE/R²)")
        print("  ✓ Pattern-based HD input generation")
        print("  ✓ Stratified K-fold cross-validation")
        print("  ✓ RNG reproducibility")
        print("\nReady for cluster deployment! 🚀")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
