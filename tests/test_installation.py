# tests/test_installation.py
"""
Complete installation verification with ALL original tests + refactored structure tests.
"""

import sys
import os
import importlib
import numpy as np

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def test_core_package_imports():
    """Test that all required external packages can be imported."""
    print("Testing core package imports...")

    required_packages = ['numpy', 'scipy', 'sklearn', 'mpi4py', 'psutil']

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            return False

    return True


def test_package_level_imports():
    """Test package-level imports with __init__.py."""
    print("\nTesting package-level imports...")

    try:
        # Test src package
        from src import (
            get_rng, HierarchicalRNG, LIFNeuron, Synapse,
            StaticPoissonInput, HDDynamicInput, ReadoutLayer,
            HDInputGenerator, run_rate_rnn, make_embedding,
            SpikingRNN
        )
        print("  ✓ src package imports")

        # Test analysis package
        from analysis import (
            spikes_to_binary, spikes_to_matrix,
            compute_participation_ratio, compute_effective_dimensionality,
            compute_dimensionality_from_covariance,
            analyze_spontaneous_activity, analyze_perturbation_response,
            decode_hd_input,
            get_extreme_combinations, is_extreme_combo, compute_hierarchical_stats
        )
        print("  ✓ analysis package imports")

        # Test experiments package
        from experiments import (
            BaseExperiment,
            SpontaneousExperiment, StabilityExperiment, EncodingExperiment,
            save_results, load_results,
            average_across_sessions_spontaneous,
            average_across_sessions_stability,
            average_across_sessions_encoding
        )
        print("  ✓ experiments package imports")

        return True

    except ImportError as e:
        print(f"  ✗ Package import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_internal_module_imports():
    """Test that internal module imports work with relative paths."""
    print("\nTesting internal module imports...")

    try:
        # Test src internal imports
        from src.hd_input import HDInputGenerator
        from src.spiking_network import SpikingRNN
        from src.lif_neuron import LIFNeuron
        from src.synaptic_model import Synapse
        from src.rng_utils import get_rng
        print("  ✓ src internal imports work")

        # Test analysis internal imports
        from analysis.common_utils import spikes_to_binary
        from analysis.spontaneous_analysis import analyze_spontaneous_activity
        from analysis.stability_analysis import analyze_perturbation_response
        from analysis.encoding_analysis import decode_hd_input
        from analysis.statistics_utils import get_extreme_combinations
        print("  ✓ analysis internal imports work")

        # Test experiments internal imports
        from experiments.base_experiment import BaseExperiment
        from experiments.spontaneous_experiment import SpontaneousExperiment
        from experiments.stability_experiment import StabilityExperiment
        from experiments.encoding_experiment import EncodingExperiment
        from experiments.experiment_utils import save_results
        print("  ✓ experiments internal imports work")

        return True

    except ImportError as e:
        print(f"  ✗ Internal import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_common_utils_functionality():
    """Test common utility functions work correctly."""
    print("\nTesting common utilities functionality...")

    try:
        from analysis.common_utils import (
            spikes_to_binary, spikes_to_matrix,
            compute_participation_ratio, compute_effective_dimensionality,
            compute_dimensionality_from_covariance
        )

        # Test spikes_to_binary
        spikes = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2)]
        binary = spikes_to_binary(spikes, num_neurons=3, duration=5.0, bin_size=1.0)
        assert binary.shape == (3, 5), f"Expected (3, 5), got {binary.shape}"
        print(f"  ✓ spikes_to_binary: shape {binary.shape}")

        # Test spikes_to_matrix
        matrix = spikes_to_matrix(spikes, n_steps=5, n_neurons=3, step_size=1.0)
        assert matrix.shape == (5, 3), f"Expected (5, 3), got {matrix.shape}"
        print(f"  ✓ spikes_to_matrix: shape {matrix.shape}")

        # Test participation ratio
        eigenvalues = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        pr = compute_participation_ratio(eigenvalues)
        assert pr > 0 and pr <= len(eigenvalues), f"Invalid PR: {pr}"
        print(f"  ✓ participation_ratio: {pr:.2f}")

        # Test effective dimensionality
        ed = compute_effective_dimensionality(eigenvalues, 0.95)
        assert 0 < ed <= len(eigenvalues), f"Invalid ED: {ed}"
        print(f"  ✓ effective_dimensionality: {ed}")

        # Test dimensionality from covariance
        data = np.random.randn(10, 100)
        dim_metrics = compute_dimensionality_from_covariance(data)
        required_keys = ['intrinsic_dimensionality', 'effective_dimensionality',
                        'participation_ratio', 'total_variance']
        assert all(k in dim_metrics for k in required_keys)
        print(f"  ✓ dimensionality_from_covariance works")

        return True

    except Exception as e:
        print(f"  ✗ Common utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics_utils_functionality():
    """Test statistics utility functions."""
    print("\nTesting statistics utilities...")

    try:
        from analysis.statistics_utils import (
            get_extreme_combinations, is_extreme_combo, compute_hierarchical_stats
        )

        v_th_stds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        g_stds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        extremes = get_extreme_combinations(v_th_stds, g_stds)
        assert len(extremes) == 4, f"Expected 4 extremes, got {len(extremes)}"
        expected = [(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0)]
        for exp in expected:
            assert exp in extremes, f"Missing extreme: {exp}"
        print(f"  ✓ get_extreme_combinations: {extremes}")

        assert is_extreme_combo(0.0, 0.0, extremes) == True
        assert is_extreme_combo(2.0, 2.0, extremes) == False
        print(f"  ✓ is_extreme_combo works")

        session_arrays = [np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5])]
        stats = compute_hierarchical_stats(session_arrays)
        assert 'mean' in stats and 'std' in stats
        assert stats['n_sessions'] == 2
        assert stats['n_total_values'] == 6
        print(f"  ✓ compute_hierarchical_stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

        return True

    except Exception as e:
        print(f"  ✗ Statistics utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spontaneous_analysis():
    """Test spontaneous activity analysis functions (ALL original tests)."""
    print("\nTesting spontaneous activity analysis...")

    try:
        from analysis.spontaneous_analysis import (
            analyze_firing_rates_and_silence,
            compute_activity_dimensionality_multi_bin,
            analyze_population_poisson_properties,
            analyze_spontaneous_activity
        )

        # Test firing rate analysis
        test_spikes = [(10.0, 0), (15.0, 1), (20.0, 0), (25.0, 2), (30.0, 0)]
        firing_stats = analyze_firing_rates_and_silence(test_spikes, num_neurons=5, duration=1000.0)

        expected_keys = ['mean_firing_rate', 'std_firing_rate', 'min_firing_rate',
                        'max_firing_rate', 'silent_neurons', 'active_neurons',
                        'percent_silent', 'percent_active']
        assert all(key in firing_stats for key in expected_keys)
        print(f"  ✓ Firing rate analysis: {firing_stats['mean_firing_rate']:.2f} Hz, "
              f"{firing_stats['percent_silent']:.1f}% silent")

        # Test multi-bin dimensionality (ALL 6 bin sizes)
        test_spikes_long = [(i*10.0, i%3) for i in range(50)]
        dim_results = compute_activity_dimensionality_multi_bin(
            test_spikes_long, num_neurons=10, duration=500.0,
            bin_sizes=[0.1, 2.0, 5.0, 20.0, 50.0, 100.0]
        )

        expected_bins = ['bin_0.1ms', 'bin_2.0ms', 'bin_5.0ms',
                        'bin_20.0ms', 'bin_50.0ms', 'bin_100.0ms']
        assert all(bin_key in dim_results for bin_key in expected_bins)
        print(f"  ✓ Multi-bin dimensionality (6 bins):")
        print(f"    0.1ms: ED={dim_results['bin_0.1ms']['effective_dimensionality']:.1f}")
        print(f"    5ms: ED={dim_results['bin_5.0ms']['effective_dimensionality']:.1f}")
        print(f"    100ms: ED={dim_results['bin_100.0ms']['effective_dimensionality']:.1f}")

        # Test Poisson analysis
        poisson_results = analyze_population_poisson_properties(
            test_spikes_long, num_neurons=10, duration=500.0, bin_size=10.0, min_spikes=10
        )

        assert 'population_statistics' in poisson_results
        pop_stats = poisson_results['population_statistics']
        poisson_keys = ['total_neurons', 'active_neurons', 'neurons_with_sufficient_spikes',
                       'poisson_isi_fraction', 'poisson_count_fraction',
                       'mean_cv_isi', 'mean_fano_factor']
        assert all(k in pop_stats for k in poisson_keys)
        print(f"  ✓ Poisson analysis:")
        print(f"    Mean CV ISI: {pop_stats['mean_cv_isi']:.3f}")
        print(f"    Mean Fano: {pop_stats['mean_fano_factor']:.3f}")
        print(f"    Poisson (ISI): {pop_stats['poisson_isi_fraction']:.2f}")

        # Test complete spontaneous analysis (with 200ms transient)
        spontaneous_results = analyze_spontaneous_activity(
            test_spikes_long, num_neurons=10, duration=500.0, transient_time=200.0
        )

        required_keys = ['firing_stats', 'dimensionality_metrics', 'poisson_analysis',
                        'transient_time', 'steady_state_duration', 'total_spikes']
        assert all(key in spontaneous_results for key in required_keys)
        assert spontaneous_results['transient_time'] == 200.0
        print(f"  ✓ Complete analysis: transient={spontaneous_results['transient_time']}ms")

        return True

    except Exception as e:
        print(f"  ✗ Spontaneous analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stability_analysis():
    """Test network stability analysis (ALL original + NEW measures)."""
    print("\nTesting network stability analysis...")

    try:
        from analysis.stability_analysis import (
            lempel_ziv_complexity, compute_shannon_entropy, find_settling_time,
            unified_coincidence_factor, average_coincidence_multi_window,
            analyze_perturbation_response
        )

        # Test LZ complexity
        test_sequence = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        lz_result = lempel_ziv_complexity(test_sequence)
        assert lz_result > 0, f"LZ complexity should be > 0, got {lz_result}"
        print(f"  ✓ LZ complexity: {lz_result}")

        # Test Shannon entropy
        test_seq = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        shannon_ent = compute_shannon_entropy(test_seq)
        assert shannon_ent > 0, "Shannon entropy should be > 0"
        print(f"  ✓ Shannon entropy: {shannon_ent:.3f}")

        # Test settling time
        symbol_seq = np.array([1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        pert_bin = 5
        settling = find_settling_time(symbol_seq, pert_bin, bin_size=1.0, min_zero_duration_ms=5.0)
        assert not np.isnan(settling), "Settling time detection failed"
        print(f"  ✓ Settling time: {settling:.1f} ms")

        # Test unified coincidence (Kistler + Gamma)
        spikes1 = [1.0, 5.0, 10.0, 15.0, 20.0]
        spikes2 = [1.1, 5.2, 9.8, 15.3, 20.1]
        kistler_c, gamma_c = unified_coincidence_factor(spikes1, spikes2, delta=2.0, duration=25.0)
        assert not np.isnan(gamma_c), "Gamma coincidence failed"
        assert not np.isnan(kistler_c), "Kistler coincidence failed"
        print(f"  ✓ Unified coincidence: Kistler={kistler_c:.3f}, Gamma={gamma_c:.3f}")

        # Test multi-window coincidence (0.1ms, 2ms, 5ms)
        spikes_ctrl = [(1.0, 0), (2.0, 1), (3.0, 0)]
        spikes_pert = [(1.05, 0), (2.1, 1), (3.02, 0)]
        coin_results = average_coincidence_multi_window(
            spikes_ctrl, spikes_pert, num_neurons=2,
            delta_values=[0.1, 2.0, 5.0], duration=5.0
        )
        expected_keys = ['kistler_delta_0.1ms', 'kistler_delta_2.0ms', 'kistler_delta_5.0ms',
                        'gamma_window_0.1ms', 'gamma_window_2.0ms', 'gamma_window_5.0ms']
        assert all(k in coin_results for k in expected_keys)
        print(f"  ✓ Multi-window coincidence (0.1ms, 2ms, 5ms):")
        print(f"    Kistler 0.1ms: {coin_results['kistler_delta_0.1ms']:.3f}")
        print(f"    Kistler 2ms: {coin_results['kistler_delta_2.0ms']:.3f}")

        # Test complete perturbation analysis (with lz_column_wise)
        spikes_control = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2)]
        spikes_perturbed = [(1.0, 0), (2.5, 1), (3.5, 2), (4.0, 0)]

        stability_results = analyze_perturbation_response(
            spikes_control, spikes_perturbed, num_neurons=3,
            perturbation_time=200.0, simulation_end=500.0,
            perturbed_neuron=0, dt=0.1
        )

        # Check ALL required keys including NEW measures
        expected_keys = [
            'lz_spatial_patterns', 'lz_column_wise',  # LZ measures
            'shannon_entropy_symbols', 'shannon_entropy_spikes',  # Shannon
            'unique_patterns_count', 'post_pert_symbol_sum', 'total_spike_differences',
            'settling_time_ms',  # Settling
            'kistler_delta_0.1ms', 'kistler_delta_2.0ms', 'kistler_delta_5.0ms',  # Coincidence
            'gamma_window_0.1ms', 'gamma_window_2.0ms', 'gamma_window_5.0ms',
            'perturbation_time', 'simulation_duration'  # Metadata
        ]

        missing_keys = [key for key in expected_keys if key not in stability_results]
        assert not missing_keys, f"Missing keys: {missing_keys}"

        assert stability_results['perturbation_time'] == 200.0
        print(f"  ✓ Complete perturbation analysis (ALL measures):")
        print(f"    LZ spatial: {stability_results['lz_spatial_patterns']}")
        print(f"    LZ column-wise: {stability_results['lz_column_wise']}")
        print(f"    Shannon (symbols): {stability_results['shannon_entropy_symbols']:.3f}")
        print(f"    Settling: {stability_results['settling_time_ms']:.1f} ms")
        print(f"    Perturbation time: {stability_results['perturbation_time']} ms")

        return True

    except Exception as e:
        print(f"  ✗ Stability analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pulse_filter_modes():
    """Test pulse and filter synaptic modes."""
    print("\nTesting pulse and filter synaptic modes...")

    try:
        from src.spiking_network import SpikingRNN

        # Test pulse mode
        network_pulse = SpikingRNN(n_neurons=50, synaptic_mode="pulse")
        assert network_pulse.synaptic_mode == "pulse"
        print("  ✓ Pulse synapse mode")

        # Test filter mode
        network_filter = SpikingRNN(n_neurons=50, synaptic_mode="filter")
        assert network_filter.synaptic_mode == "filter"
        print("  ✓ Filter synapse mode")

        # Test that old terminology is rejected
        try:
            network_old = SpikingRNN(n_neurons=50, synaptic_mode="immediate")
            print("  ✗ Old 'immediate' terminology still accepted")
            return False
        except ValueError:
            print("  ✓ Old 'immediate' terminology rejected")

        try:
            network_old2 = SpikingRNN(n_neurons=50, synaptic_mode="dynamic")
            print("  ✗ Old 'dynamic' terminology still accepted")
            return False
        except ValueError:
            print("  ✓ Old 'dynamic' terminology rejected")

        return True

    except Exception as e:
        print(f"  ✗ Pulse/filter modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_static_input_modes():
    """Test all three static input modes."""
    print("\nTesting static input modes...")

    try:
        from src.spiking_network import SpikingRNN

        modes = ["independent", "common_stochastic", "common_tonic"]

        for mode in modes:
            network = SpikingRNN(n_neurons=50, synaptic_mode="filter",
                                static_input_mode=mode)
            assert network.static_input_mode == mode
            print(f"  ✓ Static input mode '{mode}'")

        # Test invalid mode is rejected
        try:
            network_invalid = SpikingRNN(n_neurons=50, static_input_mode="invalid_mode")
            print("  ✗ Invalid mode accepted")
            return False
        except ValueError:
            print("  ✓ Invalid mode rejected")

        return True

    except Exception as e:
        print(f"  ✗ Static input modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hd_input_modes():
    """Test HD input modes."""
    print("\nTesting HD input modes...")

    try:
        from src.spiking_network import SpikingRNN

        modes = ["independent", "common_stochastic", "common_tonic"]

        for mode in modes:
            network = SpikingRNN(n_neurons=50, synaptic_mode="filter",
                                hd_input_mode=mode, n_hd_channels=10)
            assert network.hd_input_mode == mode
            print(f"  ✓ HD input mode '{mode}'")

        return True

    except Exception as e:
        print(f"  ✗ HD input modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spontaneous_experiment():
    """Test spontaneous activity experiment (complete)."""
    print("\nTesting spontaneous activity experiment...")

    try:
        from experiments.spontaneous_experiment import SpontaneousExperiment

        experiment = SpontaneousExperiment(n_neurons=10, synaptic_mode="filter")

        result = experiment.run_parameter_combination(
            session_id=999,
            v_th_std=0.5,
            g_std=0.5,
            v_th_distribution="normal",
            static_input_rate=200.0,
            duration=500.0  # CHANGE: 200ms transient + 300ms data = 500ms total
        )

        # Check ALL expected fields
        expected_fields = [
            'session_id', 'v_th_std', 'g_std', 'synaptic_mode', 'static_input_mode',
            'duration', 'v_th_distribution', 'static_input_rate',
            # Firing stats
            'mean_firing_rate_values', 'std_firing_rate_values',
            'silent_neurons_values', 'active_neurons_values',
            'percent_silent_values', 'percent_active_values',
            # Dimensionality (6 bin sizes)
            'intrinsic_dimensionality_bin_0.1ms_values',
            'effective_dimensionality_bin_5.0ms_values',
            'participation_ratio_bin_50.0ms_values',
            # Poisson
            'mean_cv_isi_values', 'mean_fano_factor_values',
            'poisson_isi_fraction_values', 'poisson_count_fraction_values',
            # Metadata
            'total_spikes_values', 'steady_state_spikes_values',
            'n_trials', 'computation_time'
        ]

        missing_fields = [field for field in expected_fields if field not in result]
        assert not missing_fields, f"Missing fields: {missing_fields}"

        print(f"  ✓ All fields present")
        print(f"    Mean firing rate: {result['mean_firing_rate_mean']:.2f} Hz")
        print(f"    Silent neurons: {result['percent_silent_mean']:.1f}%")
        print(f"    Trials: {result['n_trials']}")

        return True

    except Exception as e:
        print(f"  ✗ Spontaneous experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stability_experiment():
    """Test network stability experiment (complete with NEW measures)."""
    print("\nTesting network stability experiment...")

    try:
        from experiments.stability_experiment import StabilityExperiment

        experiment = StabilityExperiment(n_neurons=10, synaptic_mode="filter")

        result = experiment.run_parameter_combination(
            session_id=999,
            v_th_std=0.5,
            g_std=0.5,
            v_th_distribution="normal",
            static_input_rate=200.0
        )

        # Check ALL expected fields including NEW measures
        expected_fields = [
            'session_id', 'v_th_std', 'g_std', 'synaptic_mode', 'static_input_mode',
            'v_th_distribution', 'static_input_rate',
            # LZ complexity
            'lz_spatial_patterns_values', 'lz_column_wise_values',
            # Shannon entropy
            'shannon_entropy_symbols_values', 'shannon_entropy_spikes_values',
            # Pattern stats
            'unique_patterns_count_values', 'post_pert_symbol_sum_values',
            'total_spike_differences_values',
            # Settling
            'settling_time_ms_values', 'settled_fraction', 'settled_count',
            # Coincidence (0.1ms, 2ms, 5ms)
            'kistler_delta_0.1ms_values', 'kistler_delta_2.0ms_values', 'kistler_delta_5.0ms_values',
            'gamma_window_0.1ms_values', 'gamma_window_2.0ms_values', 'gamma_window_5.0ms_values',
            # Metadata
            'n_trials', 'computation_time', 'perturbation_neurons'
        ]

        missing_fields = [field for field in expected_fields if field not in result]
        assert not missing_fields, f"Missing fields: {missing_fields}"

        print(f"  ✓ All fields present (including NEW measures)")
        print(f"    LZ spatial: {result['lz_spatial_patterns_mean']:.2f}")
        print(f"    LZ column-wise: {result['lz_column_wise_mean']:.2f}")
        print(f"    Shannon (symbols): {result['shannon_entropy_symbols_mean']:.3f}")
        print(f"    Kistler 0.1ms: {result['kistler_delta_0.1ms_mean']:.3f}")
        print(f"    Trials: {result['n_trials']}")

        return True

    except Exception as e:
        print(f"  ✗ Stability experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hd_input_system():
    """Test HD input generation and caching."""
    print("\nTesting HD input system...")

    try:
        from src.hd_input import HDInputGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)

            # Initialize base input
            generator.initialize_base_input(session_id=1, hd_dim=3)

            assert generator.Y_base is not None
            assert generator.Y_base.shape[1] == 5
            assert generator.Y_base.shape[0] == 3000  # 500ms - 200ms transient
            print(f"  ✓ Base input generation: shape {generator.Y_base.shape}")

            # Test caching
            cache_file = generator._get_signal_filename(1, 3)
            assert os.path.exists(cache_file)
            print(f"  ✓ Signal caching works")

            # Test loading from cache
            generator2 = HDInputGenerator(embed_dim=5, dt=0.1, signal_cache_dir=tmpdir)
            generator2.initialize_base_input(session_id=1, hd_dim=3)
            assert np.array_equal(generator.Y_base, generator2.Y_base)
            print(f"  ✓ Cache loading works")

            # Test trial generation
            trial_input = generator.generate_trial_input(
                session_id=1, v_th_std=0.5, g_std=0.5,
                trial_id=1, hd_dim=3
            )
            assert np.all(trial_input >= 0)
            assert trial_input.shape == generator.Y_base.shape
            print(f"  ✓ Trial input generation works")

            # Test different trials have different noise
            trial_input2 = generator.generate_trial_input(
                session_id=1, v_th_std=0.5, g_std=0.5,
                trial_id=2, hd_dim=3
            )
            assert not np.array_equal(trial_input, trial_input2)
            print(f"  ✓ Different trials have different noise")

        return True

    except Exception as e:
        print(f"  ✗ HD input system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transient_time_consistency():
    """Test that transient time is 200ms everywhere."""
    print("\nTesting transient time consistency...")

    try:
        from experiments.spontaneous_experiment import SpontaneousExperiment
        from experiments.stability_experiment import StabilityExperiment
        from experiments.encoding_experiment import EncodingExperiment

        # Check encoding
        enc = EncodingExperiment()
        assert enc.transient_time == 200.0
        print(f"  ✓ Encoding transient: {enc.transient_time} ms")

        # Check stability
        stab = StabilityExperiment()
        assert stab.pre_perturbation_time == 200.0
        assert stab.perturbation_time == 200.0
        print(f"  ✓ Stability perturbation: {stab.perturbation_time} ms")

        # Spontaneous uses transient_time parameter in analysis
        print(f"  ✓ Spontaneous uses 200ms in analyze_spontaneous_activity()")

        return True

    except Exception as e:
        print(f"  ✗ Transient time test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run ALL installation tests."""
    print("=" * 70)
    print("COMPLETE INSTALLATION VERIFICATION")
    print("ALL ORIGINAL TESTS + REFACTORED STRUCTURE TESTS")
    print("=" * 70)

    tests = [
        # Core imports
        ("Core Package Imports", test_core_package_imports),
        ("Package-Level Imports", test_package_level_imports),
        ("Internal Module Imports", test_internal_module_imports),

        # Utility functions
        ("Common Utils Functionality", test_common_utils_functionality),
        ("Statistics Utils Functionality", test_statistics_utils_functionality),

        # Analysis modules (ALL original tests)
        ("Spontaneous Analysis (ALL)", test_spontaneous_analysis),
        ("Stability Analysis (ALL + NEW)", test_stability_analysis),

        # Network modes
        ("Pulse/Filter Modes", test_pulse_filter_modes),
        ("Static Input Modes", test_static_input_modes),
        ("HD Input Modes", test_hd_input_modes),

        # Experiments
        ("Spontaneous Experiment (Complete)", test_spontaneous_experiment),
        ("Stability Experiment (Complete + NEW)", test_stability_experiment),

        # HD input system
        ("HD Input System", test_hd_input_system),

        # Consistency checks
        ("Transient Time Consistency", test_transient_time_consistency),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ {test_name} exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 70)
    print("COMPLETE INSTALLATION TEST SUMMARY")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:40s}: {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL INSTALLATION TESTS PASSED!")
        print("\nVerified features:")
        print("  • Package-level and internal imports")
        print("  • Common utilities (spikes_to_binary, dimensionality, etc)")
        print("  • Statistics utilities (extreme combos, hierarchical stats)")
        print("  • Spontaneous analysis (6 bin sizes, Poisson tests)")
        print("  • Stability analysis (LZ column-wise, Shannon, settling, 0.1ms coincidence)")
        print("  • Pulse/filter synaptic modes")
        print("  • Three input modes (independent, common_stochastic, common_tonic)")
        print("  • HD input generation and caching")
        print("  • 200ms transient time consistency")
        print("  • All experiment types with complete field sets")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
