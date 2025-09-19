# experiments/chaos_experiment.py - Updated with multiplier scaling and fixed structure
"""
Chaos analysis experiment with fixed network structure and heterogeneity multipliers.
Network topology depends only on session_id, heterogeneity scales with multipliers.
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
    from src.rng_utils import get_rng, generate_base_distributions
    from analysis.spike_analysis import analyze_perturbation_response_enhanced
except ImportError:
    try:
        from ..src.spiking_network import SpikingRNN
        from ..src.rng_utils import get_rng, generate_base_distributions
        from ..analysis.spike_analysis import analyze_perturbation_response_enhanced
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        src_dir = os.path.join(project_root, 'src')
        analysis_dir = os.path.join(project_root, 'analysis')
        sys.path.insert(0, src_dir)
        sys.path.insert(0, analysis_dir)
        from spiking_network import SpikingRNN
        from rng_utils import get_rng, generate_base_distributions
        from spike_analysis import analyze_perturbation_response_enhanced

class ChaosExperiment:
    """Enhanced chaos experiment with fixed network structure and multiplier scaling."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt
        self.pre_perturbation_time = 50.0
        self.post_perturbation_time = 300.0
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time
        self.perturbation_time = self.pre_perturbation_time
        self.n_perturbation_trials = 20  # Updated to 100 as requested

        # Base distribution parameters
        self.base_v_th_std = 0.01  # Base heterogeneity
        self.base_g_std = 0.01     # Base heterogeneity

    def run_single_perturbation(self, session_id: int, block_id: int, trial_id: int,
                              v_th_multiplier: float, g_multiplier: float,
                              perturbation_neuron_idx: int,
                              static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run single perturbation with fixed structure and multiplier scaling."""

        # Create identical networks with fixed structure
        network_control = SpikingRNN(self.n_neurons, dt=self.dt)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt)

        # Network parameters with multiplier scaling
        network_params = {
            'v_th_multiplier': v_th_multiplier,  # Changed from v_th_std
            'g_multiplier': g_multiplier,        # Changed from g_std
            'static_input_strength': 1.0,
            'dynamic_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        # Initialize both networks with identical fixed structure
        for network in [network_control, network_perturbed]:
            network.initialize_network(session_id, block_id, **network_params)

        # Get the actual perturbation neuron from fixed list
        base_distributions = generate_base_distributions(session_id, self.n_neurons)

        available_neurons = len(base_distributions['perturbation_neurons'])
        safe_idx = perturbation_neuron_idx % available_neurons
        perturbation_neuron = int(base_distributions['perturbation_neurons'][safe_idx])

        # Run control simulation (no perturbation)
        spikes_control = network_control.simulate_network_dynamics(
            session_id=session_id,
            block_id=block_id,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate
        )

        # Run perturbed simulation (auxiliary spike at perturbation_time)
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

        # Add perturbation info
        analysis_results['perturbation_neuron'] = perturbation_neuron
        analysis_results['perturbation_neuron_idx'] = perturbation_neuron_idx

        return analysis_results

    def run_parameter_combination(self, session_id: int, block_id: int,
                                v_th_multiplier: float, g_multiplier: float,
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """Run parameter combination with multiplier scaling."""
        start_time = time.time()

        # Store all trial results for enhanced statistics
        trial_results = []

        # Use fixed perturbation neuron indices (first 100 from fixed list)
        perturbation_neuron_indices = list(range(self.n_perturbation_trials))

        for trial_idx in range(self.n_perturbation_trials):
            trial_id = trial_idx + 1

            trial_result = self.run_single_perturbation(
                session_id=session_id,
                block_id=block_id,
                trial_id=trial_id,
                v_th_multiplier=v_th_multiplier,
                g_multiplier=g_multiplier,
                perturbation_neuron_idx=perturbation_neuron_indices[trial_idx],
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

        # Get actual heterogeneity values for reporting
        actual_v_th_std = self.base_v_th_std * v_th_multiplier
        actual_g_std = self.base_g_std * g_multiplier

        results = {
            # Parameter information (both multipliers and actual values)
            'v_th_multiplier': v_th_multiplier,
            'g_multiplier': g_multiplier,
            'v_th_std': actual_v_th_std,  # Computed actual std
            'g_std': actual_g_std,        # Computed actual std
            'base_v_th_std': self.base_v_th_std,
            'base_g_std': self.base_g_std,
            'static_input_rate': static_input_rate,

            # Original chaos measures - arrays and statistics
            'lz_complexities': np.array(lz_complexities),
            'hamming_slopes': np.array(hamming_slopes),
            'lz_mean': np.mean(lz_complexities),
            'lz_std': np.std(lz_complexities),
            'hamming_mean': np.mean(hamming_slopes),
            'hamming_std': np.std(hamming_slopes),

            # Enhanced measures
            'total_spike_differences': np.array(total_spike_diffs),
            'spike_diff_mean': np.mean(total_spike_diffs),
            'spike_diff_std': np.std(total_spike_diffs),

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

            'gamma_coincidences': np.array(gamma_coincidences),
            'gamma_coincidence_mean': np.mean(gamma_coincidences),
            'gamma_coincidence_std': np.std(gamma_coincidences),

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'perturbation_neurons': [r['perturbation_neuron'] for r in trial_results]
        }

        return results

    def run_full_experiment(self, session_id: int, v_th_multipliers: np.ndarray,
                          g_multipliers: np.ndarray,
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run full experiment with multiplier scaling."""
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

        results = []
        block_id = 0

        total_combinations = len(v_th_multipliers) * len(g_multipliers) * len(static_input_rates)
        print(f"Starting fixed-structure experiment: {total_combinations} parameter combinations")
        print(f"  v_th_multipliers: {len(v_th_multipliers)} (range: {np.min(v_th_multipliers):.1f}-{np.max(v_th_multipliers):.1f})")
        print(f"  g_multipliers: {len(g_multipliers)} (range: {np.min(g_multipliers):.1f}-{np.max(g_multipliers):.1f})")
        print(f"  Base heterogeneities: v_th_std=0.01, g_std=0.01")
        print(f"  Actual ranges: v_th_std={0.01*np.min(v_th_multipliers):.3f}-{0.01*np.max(v_th_multipliers):.2f}, g_std={0.01*np.min(g_multipliers):.3f}-{0.01*np.max(g_multipliers):.2f}")
        print(f"  Fixed structure: Only session_id={session_id} determines network topology")
        print(f"  Trials per combination: {self.n_perturbation_trials}")

        combo_idx = 0
        for input_rate in static_input_rates:
            for i, v_th_mult in enumerate(v_th_multipliers):
                for j, g_mult in enumerate(g_multipliers):
                    combo_idx += 1

                    actual_v_th_std = self.base_v_th_std * v_th_mult
                    actual_g_std = self.base_g_std * g_mult

                    print(f"[{combo_idx}/{total_combinations}] Fixed-structure analysis: "
                          f"input_rate={input_rate:.1f}Hz, v_th_mult={v_th_mult:.1f}→{actual_v_th_std:.3f}, "
                          f"g_mult={g_mult:.1f}→{actual_g_std:.3f}")

                    result = self.run_parameter_combination(
                        session_id=session_id,
                        block_id=block_id,
                        v_th_multiplier=v_th_mult,
                        g_multiplier=g_mult,
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

        print(f"Fixed-structure experiment completed: {len(results)} combinations processed")
        return results


def create_parameter_grid_with_multipliers(n_points: int = 10,
                                         multiplier_range: Tuple[float, float] = (1.0, 100.0),
                                         input_rate_range: Tuple[float, float] = (50.0, 500.0),
                                         n_input_rates: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create parameter grids with multiplier scaling.

    Args:
        n_points: Number of points per multiplier dimension
        multiplier_range: (min_multiplier, max_multiplier)
        input_rate_range: (min_rate, max_rate) in Hz
        n_input_rates: Number of input rate values

    Returns:
        Tuple of (v_th_multipliers, g_multipliers, static_input_rates)
    """
    v_th_multipliers = np.linspace(multiplier_range[0], multiplier_range[1], n_points)
    g_multipliers = np.linspace(multiplier_range[0], multiplier_range[1], n_points)
    static_input_rates = np.linspace(input_rate_range[0], input_rate_range[1], n_input_rates)

    return v_th_multipliers, g_multipliers, static_input_rates



def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    """Save enhanced experimental results with multiplier information."""
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
    print(f"Fixed-structure results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load enhanced experimental results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Fixed-structure results loaded: {len(results)} combinations from {filename}")
    return results
