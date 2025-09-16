# experiments/chaos_experiment.py - Modified version
"""
Chaos analysis experiment with configurable static input rate.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Tuple, Any

# Handle imports for different directory structures
try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.spike_analysis import analyze_perturbation_response
except ImportError:
    try:
        from ..src.spiking_network import SpikingRNN
        from ..src.rng_utils import get_rng
        from ..analysis.spike_analysis import analyze_perturbation_response
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        src_dir = os.path.join(project_root, 'src')
        analysis_dir = os.path.join(project_root, 'analysis')
        sys.path.insert(0, src_dir)
        sys.path.insert(0, analysis_dir)
        from spiking_network import SpikingRNN
        from rng_utils import get_rng
        from spike_analysis import analyze_perturbation_response

class ChaosExperiment:
    """Experiment class for studying chaos in spiking RNNs with configurable input rates."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt
        self.pre_perturbation_time = 50.0
        self.post_perturbation_time = 500.0
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time
        self.perturbation_time = self.pre_perturbation_time
        self.n_perturbation_trials = 20

    def run_single_perturbation(self, session_id: int, block_id: int, trial_id: int,
                              v_th_std: float, g_std: float, perturbation_neuron: int,
                              static_input_rate: float = 200.0) -> Tuple[float, float]:
        """
        Run single perturbation experiment with configurable input rate.
        """
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
            static_input_rate=static_input_rate  # Now configurable
        )

        spikes_perturbed = network_perturbed.simulate_network_dynamics(
            session_id=session_id,
            block_id=block_id,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=static_input_rate,  # Now configurable
            perturbation_time=self.perturbation_time,
            perturbation_neuron=perturbation_neuron
        )

        lz_complexity, hamming_slope = analyze_perturbation_response(
            spikes_control=spikes_control,
            spikes_perturbed=spikes_perturbed,
            num_neurons=self.n_neurons,
            perturbation_time=self.perturbation_time,
            simulation_end=self.total_duration,
            perturbed_neuron=perturbation_neuron
        )

        return lz_complexity, hamming_slope

    def run_parameter_combination(self, session_id: int, block_id: int,
                                v_th_std: float, g_std: float,
                                static_input_rate: float = 200.0) -> Dict[str, Any]:
        """
        Run complete experiment for single parameter combination with configurable input rate.
        """
        start_time = time.time()
        lz_complexities = []
        hamming_slopes = []

        rng = get_rng(session_id, block_id, 0, 'perturbation')
        perturbation_neurons = rng.choice(
            self.n_neurons, size=self.n_perturbation_trials, replace=False
        )

        for trial_idx, perturb_neuron in enumerate(perturbation_neurons):
            trial_id = trial_idx + 1

            lz_comp, hamm_slope = self.run_single_perturbation(
                session_id=session_id,
                block_id=block_id,
                trial_id=trial_id,
                v_th_std=v_th_std,
                g_std=g_std,
                perturbation_neuron=perturb_neuron,
                static_input_rate=static_input_rate  # Pass input rate
            )

            lz_complexities.append(lz_comp)
            hamming_slopes.append(hamm_slope)

        results = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'static_input_rate': static_input_rate,  # Store input rate
            'lz_complexities': np.array(lz_complexities),
            'hamming_slopes': np.array(hamming_slopes),
            'lz_mean': np.mean(lz_complexities),
            'lz_std': np.std(lz_complexities),
            'hamming_mean': np.mean(hamming_slopes),
            'hamming_std': np.std(hamming_slopes),
            'n_trials': len(lz_complexities),
            'computation_time': time.time() - start_time,
            'perturbation_neurons': perturbation_neurons.tolist()
        }

        return results

    def run_full_experiment(self, session_id: int, v_th_std_values: np.ndarray,
                          g_std_values: np.ndarray,
                          static_input_rates: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Run complete experiment across full parameter space including input rates.
        """
        if static_input_rates is None:
            static_input_rates = np.array([200.0])  # Default single value

        results = []
        block_id = 0

        total_combinations = len(v_th_std_values) * len(g_std_values) * len(static_input_rates)
        print(f"Starting full experiment: {total_combinations} parameter combinations")
        print(f"  v_th_std values: {len(v_th_std_values)}")
        print(f"  g_std values: {len(g_std_values)}")
        print(f"  static_input_rates: {len(static_input_rates)}")

        combo_idx = 0
        for input_rate in static_input_rates:
            for i, v_th_std in enumerate(v_th_std_values):
                for j, g_std in enumerate(g_std_values):
                    combo_idx += 1

                    print(f"[{combo_idx}/{total_combinations}] Running "
                          f"input_rate={input_rate:.1f}Hz, v_th_std={v_th_std:.3f}, g_std={g_std:.3f}")

                    start_time = time.time()
                    result = self.run_parameter_combination(
                        session_id=session_id,
                        block_id=block_id,
                        v_th_std=v_th_std,
                        g_std=g_std,
                        static_input_rate=input_rate
                    )
                    end_time = time.time()

                    result['block_id'] = block_id
                    result['combination_index'] = combo_idx
                    results.append(result)

                    print(f"  LZ complexity: {result['lz_mean']:.2f} ± {result['lz_std']:.2f}")
                    print(f"  Hamming slope: {result['hamming_mean']:.4f} ± {result['hamming_std']:.4f}")
                    print(f"  Computation time: {result['computation_time']:.1f}s")

                    block_id += 1

        print(f"Full experiment completed: {len(results)} combinations processed")
        return results

def create_parameter_grid_with_input_rates(n_points: int = 10,
                                         input_rate_range: Tuple[float, float] = (50.0, 500.0),
                                         n_input_rates: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create parameter grids including static input rates.

    Args:
        n_points: Number of points per v_th/g_std dimension
        input_rate_range: (min_rate, max_rate) in Hz
        n_input_rates: Number of input rate values to test

    Returns:
        Tuple of (v_th_std_values, g_std_values, static_input_rates)
    """
    v_th_std_values = np.linspace(0.05, 0.5, n_points)
    g_std_values = np.linspace(0.05, 0.5, n_points)
    static_input_rates = np.linspace(input_rate_range[0], input_rate_range[1], n_input_rates)

    return v_th_std_values, g_std_values, static_input_rates

# Updated save/load functions remain the same
def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    """Save experimental results with proper directory structure."""
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
    print(f"Results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load experimental results from file."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded: {len(results)} combinations from {filename}")
    return results
