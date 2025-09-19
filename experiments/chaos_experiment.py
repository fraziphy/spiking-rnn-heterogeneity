# experiments/chaos_experiment.py - Updated with enhanced analysis
"""
Chaos analysis experiment with enhanced metrics: dimensionality, matrix differences, and gamma coincidence.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Tuple, Any

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.spike_analysis import analyze_perturbation_response_enhanced
except ImportError:
    # Fallback imports...
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    src_dir = os.path.join(project_root, 'src')
    analysis_dir = os.path.join(project_root, 'analysis')
    sys.path.insert(0, src_dir)
    sys.path.insert(0, analysis_dir)
    from spiking_network import SpikingRNN
    from rng_utils import get_rng
    from spike_analysis import analyze_perturbation_response_enhanced

class ChaosExperiment:
    """Enhanced chaos experiment with dimensionality, matrix differences, and gamma coincidence."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt
        self.pre_perturbation_time = 50.0
        self.post_perturbation_time = 300.0  # Updated to 300ms as requested
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time
        self.perturbation_time = self.pre_perturbation_time
        self.n_perturbation_trials = 100

    def run_single_perturbation(self, session_id: int, block_id: int, trial_id: int,
                              v_th_std: float, g_std: float, perturbation_neuron: int,
                              static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single perturbation with enhanced analysis."""
        network_control = SpikingRNN(self.n_neurons, dt=self.dt)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt)

        network_params = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'static_input_strength': 1.0,
            'dynamic_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        for network in [network_control, network_perturbed]:
            network.initialize_network(session_id, block_id, **network_params)

        spikes_control = network_control.simulate_network_dynamics(
            session_id=session_id,
            block_id=block_id,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate
        )

        spikes_perturbed = network_perturbed.simulate_network_dynamics(
            session_id=session_id,
            block_id=block_id,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate,
            perturbation_time=self.perturbation_time,
            perturbation_neuron=perturbation_neuron
        )

        # Use enhanced analysis function
        analysis_results = analyze_perturbation_response_enhanced(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=self.perturbation_time,
            simulation_end=self.total_duration,
            perturbed_neuron=perturbation_neuron
        )

        return analysis_results

    def run_parameter_combination(self, session_id: int, block_id: int,
                                v_th_std: float, g_std: float,
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run parameter combination with enhanced metrics."""
        start_time = time.time()

        # Store all trial results for enhanced statistics
        trial_results = []

        rng = get_rng(session_id, block_id, 0, 'perturbation')
        perturbation_neurons = rng.choice(
            self.n_neurons, size=self.n_perturbation_trials, replace=False
        )

        for trial_idx, perturb_neuron in enumerate(perturbation_neurons):
            trial_id = trial_idx + 1

            trial_result = self.run_single_perturbation(
                session_id=session_id,
                block_id=block_id,
                trial_id=trial_id,
                v_th_std=v_th_std,
                g_std=g_std,
                perturbation_neuron=perturb_neuron,
                static_input_rate=static_input_rate
            )

            trial_results.append(trial_result)

        # Compile enhanced statistics
        lz_complexities = [r['lz_complexity'] for r in trial_results]
        hamming_slopes = [r['hamming_slope'] for r in trial_results]
        total_spike_diffs = [r['total_spike_differences'] for r in trial_results]
        intrinsic_dims = [r['intrinsic_dimensionality'] for r in trial_results]
        effective_dims = [r['effective_dimensionality'] for r in trial_results]
        participation_ratios = [r['participation_ratio'] for r in trial_results]
        total_variances = [r['total_variance'] for r in trial_results]
        gamma_coincidences = [r['gamma_coincidence'] for r in trial_results]

        results = {
            # Parameter information
            'v_th_std': v_th_std,
            'g_std': g_std,
            'static_input_rate': static_input_rate,

            # Original chaos measures - arrays and statistics
            'lz_complexities': np.array(lz_complexities),
            'hamming_slopes': np.array(hamming_slopes),
            'lz_mean': np.mean(lz_complexities),
            'lz_std': np.std(lz_complexities),
            'hamming_mean': np.mean(hamming_slopes),
            'hamming_std': np.std(hamming_slopes),

            # NEW MEASURE 1: Matrix differences - arrays and statistics
            'total_spike_differences': np.array(total_spike_diffs),
            'spike_diff_mean': np.mean(total_spike_diffs),
            'spike_diff_std': np.std(total_spike_diffs),

            # NEW MEASURE 2: Network dimensionality - arrays and statistics
            'intrinsic_dimensionalities': np.array(intrinsic_dims),
            'effective_dimensionalities': np.array(effective_dims),
            'participation_ratios': np.array(participation_ratios),
            'total_variances': np.array(total_variances),
            'intrinsic_dim_mean': np.mean(intrinsic_dims),
            'intrinsic_dim_std': np.std(intrinsic_dims),
            'effective_dim_mean': np.mean(effective_dims),
            'effective_dim_std': np.std(effective_dims),
            'participation_ratio_mean': np.mean(participation_ratios),
            'participation_ratio_std': np.std(participation_ratios),
            'total_variance_mean': np.mean(total_variances),
            'total_variance_std': np.std(total_variances),

            # NEW MEASURE 3: Gamma coincidence - arrays and statistics
            'gamma_coincidences': np.array(gamma_coincidences),
            'gamma_coincidence_mean': np.mean(gamma_coincidences),
            'gamma_coincidence_std': np.std(gamma_coincidences),

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'perturbation_neurons': perturbation_neurons.tolist()
        }

        return results

    def run_full_experiment(self, session_id: int, v_th_std_values: np.ndarray,
                          g_std_values: np.ndarray,
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full experiment with enhanced analysis."""
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

        results = []
        block_id = 0

        total_combinations = len(v_th_std_values) * len(g_std_values) * len(static_input_rates)
        print(f"Starting enhanced experiment: {total_combinations} parameter combinations")
        print(f"  v_th_std values: {len(v_th_std_values)} (range: {np.min(v_th_std_values):.3f}-{np.max(v_th_std_values):.3f})")
        print(f"  g_std values: {len(g_std_values)} (range: {np.min(g_std_values):.3f}-{np.max(g_std_values):.3f})")
        print(f"  static_input_rates: {len(static_input_rates)}")
        print(f"  Enhanced metrics: LZ, Hamming, Spike Diffs, Dimensionality, Gamma Coincidence")

        combo_idx = 0
        for input_rate in static_input_rates:
            for i, v_th_std in enumerate(v_th_std_values):
                for j, g_std in enumerate(g_std_values):
                    combo_idx += 1

                    print(f"[{combo_idx}/{total_combinations}] Enhanced analysis: "
                          f"input_rate={input_rate:.1f}Hz, v_th_std={v_th_std:.3f}, g_std={g_std:.3f}")

                    result = self.run_parameter_combination(
                        session_id=session_id,
                        block_id=block_id,
                        v_th_std=v_th_std,
                        g_std=g_std,
                        static_input_rate=input_rate
                    )

                    result['block_id'] = block_id
                    result['combination_index'] = combo_idx
                    results.append(result)

                    print(f"  LZ: {result['lz_mean']:.2f}±{result['lz_std']:.2f}, "
                          f"Hamming: {result['hamming_mean']:.4f}±{result['hamming_std']:.4f}")
                    print(f"  Spike Diffs: {result['spike_diff_mean']:.1f}±{result['spike_diff_std']:.1f}, "
                          f"Dim: {result['effective_dim_mean']:.1f}±{result['effective_dim_std']:.1f}, "
                          f"Gamma: {result['gamma_coincidence_mean']:.3f}±{result['gamma_coincidence_std']:.3f}")
                    print(f"  Time: {result['computation_time']:.1f}s")

                    block_id += 1

        print(f"Enhanced experiment completed: {len(results)} combinations processed")
        return results

def create_parameter_grid_with_input_rates(n_points: int = 10,
                                         input_rate_range: Tuple[float, float] = (50.0, 500.0),
                                         n_input_rates: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create parameter grids with updated ranges."""
    # Updated ranges as requested: 0.01 to 1.0
    v_th_std_values = np.linspace(0.01, 1.0, n_points)
    g_std_values = np.linspace(0.01, 1.0, n_points)
    static_input_rates = np.linspace(input_rate_range[0], input_rate_range[1], n_input_rates)

    return v_th_std_values, g_std_values, static_input_rates

def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    """Save enhanced experimental results."""
    if not os.path.isabs(filename):
        if use_data_subdir:
            results_dir = os.path.join(os.getcwd(), "results", "data")
        else:
            results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        full_path = os.path.join(results_dir, filename)
    else:
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        full_path = filename

    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

    file_size = os.path.getsize(full_path) / (1024 * 1024)
    print(f"Enhanced results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load enhanced experimental results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Enhanced results loaded: {len(results)} combinations from {filename}")
    return results
