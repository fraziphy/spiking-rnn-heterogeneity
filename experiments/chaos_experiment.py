# experiments/chaos_experiment.py
"""
Chaos analysis experiment for studying network dynamics in spiking RNNs.
Updated with proper module imports for organized directory structure.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Tuple, Any

# Handle imports for different directory structures
try:
    # Try package-style imports (if installed with pip install -e .)
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
    from analysis.spike_analysis import analyze_perturbation_response
except ImportError:
    try:
        # Try relative imports within package
        from ..src.spiking_network import SpikingRNN
        from ..src.rng_utils import get_rng
        from ..analysis.spike_analysis import analyze_perturbation_response
    except ImportError:
        # Fallback: add directories to path
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
    """Experiment class for studying chaos in spiking RNNs through perturbation analysis."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1):
        """
        Initialize chaos experiment.

        Args:
            n_neurons: Number of neurons in RNN
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.dt = dt

        # Experiment parameters
        self.pre_perturbation_time = 50.0   # ms
        self.post_perturbation_time = 500.0 # ms
        self.total_duration = self.pre_perturbation_time + self.post_perturbation_time
        self.perturbation_time = self.pre_perturbation_time

        self.n_perturbation_trials = 20  # Number of different neurons to perturb
        self.static_input_rate = 100.0   # Hz - background Poisson process

    def run_single_perturbation(self, session_id: int, block_id: int,
                              trial_id: int, v_th_std: float, g_std: float,
                              perturbation_neuron: int) -> Tuple[float, float]:
        """
        Run single perturbation experiment comparing control vs perturbed network.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            trial_id: Trial ID for RNG
            v_th_std: Spike threshold heterogeneity (std dev)
            g_std: Synaptic weight heterogeneity (std dev)
            perturbation_neuron: Which neuron to perturb

        Returns:
            Tuple of (lz_complexity, hamming_slope)
        """
        # Create two identical networks
        network_control = SpikingRNN(self.n_neurons, dt=self.dt)
        network_perturbed = SpikingRNN(self.n_neurons, dt=self.dt)

        # Network initialization parameters
        network_params = {
            'v_th_std': v_th_std,
            'g_std': g_std,
            'static_input_strength': 1.0,
            'dynamic_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        # Initialize both networks with identical parameters
        for network in [network_control, network_perturbed]:
            network.initialize_network(session_id, block_id, **network_params)

        # Run control simulation (no perturbation)
        spikes_control = network_control.simulate_network_dynamics(
            session_id=session_id,
            block_id=block_id,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=self.static_input_rate
        )

        # Run perturbed simulation (auxiliary spike at perturbation_time)
        spikes_perturbed = network_perturbed.simulate_network_dynamics(
            session_id=session_id,
            block_id=block_id,
            trial_id=trial_id,
            duration=self.total_duration,
            static_input_rate=self.static_input_rate,
            perturbation_time=self.perturbation_time,
            perturbation_neuron=perturbation_neuron
        )

        # Analyze perturbation response using both chaos measures
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
                                v_th_std: float, g_std: float) -> Dict[str, Any]:
        """
        Run complete experiment for single parameter combination.
        Executes 20 perturbation trials with different neurons.

        Args:
            session_id: Session ID for RNG
            block_id: Block ID for RNG
            v_th_std: Spike threshold heterogeneity
            g_std: Synaptic weight heterogeneity

        Returns:
            Dictionary with complete results including timing
        """
        start_time = time.time()

        lz_complexities = []
        hamming_slopes = []

        # Generate random neurons to perturb (one per trial)
        rng = get_rng(session_id, block_id, 0, 'perturbation')
        perturbation_neurons = rng.choice(
            self.n_neurons, size=self.n_perturbation_trials, replace=False
        )

        # Execute perturbation trials
        for trial_idx, perturb_neuron in enumerate(perturbation_neurons):
            trial_id = trial_idx + 1  # Start trial IDs from 1

            lz_comp, hamm_slope = self.run_single_perturbation(
                session_id=session_id,
                block_id=block_id,
                trial_id=trial_id,
                v_th_std=v_th_std,
                g_std=g_std,
                perturbation_neuron=perturb_neuron
            )

            lz_complexities.append(lz_comp)
            hamming_slopes.append(hamm_slope)

        # Compute summary statistics
        results = {
            'v_th_std': v_th_std,
            'g_std': g_std,
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

    def run_full_experiment(self, session_id: int,
                          v_th_std_values: np.ndarray,
                          g_std_values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run complete experiment across full parameter space.

        Args:
            session_id: Session ID for RNG
            v_th_std_values: Array of spike threshold heterogeneity values
            g_std_values: Array of synaptic weight heterogeneity values

        Returns:
            List of result dictionaries for all parameter combinations
        """
        results = []

        block_id = 0
        total_combinations = len(v_th_std_values) * len(g_std_values)

        print(f"Starting full experiment: {total_combinations} parameter combinations")

        for i, v_th_std in enumerate(v_th_std_values):
            for j, g_std in enumerate(g_std_values):
                combo_idx = i * len(g_std_values) + j + 1

                print(f"[{combo_idx}/{total_combinations}] Running v_th_std={v_th_std:.3f}, g_std={g_std:.3f}")

                start_time = time.time()
                result = self.run_parameter_combination(
                    session_id=session_id,
                    block_id=block_id,
                    v_th_std=v_th_std,
                    g_std=g_std
                )
                end_time = time.time()

                # Add metadata
                result['block_id'] = block_id
                result['combination_index'] = combo_idx

                results.append(result)

                # Progress reporting
                print(f"  LZ complexity: {result['lz_mean']:.2f} ± {result['lz_std']:.2f}")
                print(f"  Hamming slope: {result['hamming_mean']:.4f} ± {result['hamming_std']:.4f}")
                print(f"  Computation time: {result['computation_time']:.1f}s")

                block_id += 1

        print(f"Full experiment completed: {len(results)} combinations processed")
        return results

def create_parameter_grid(n_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create parameter grids for heterogeneity study.

    Args:
        n_points: Number of points per parameter dimension

    Returns:
        Tuple of (v_th_std_values, g_std_values)
    """
    # Spike threshold heterogeneity: 0.05 to 0.5 mV
    v_th_std_values = np.linspace(0.05, 0.5, n_points)

    # Synaptic weight heterogeneity: 0.05 to 0.5 (normalized)
    g_std_values = np.linspace(0.05, 0.5, n_points)

    return v_th_std_values, g_std_values

def save_results(results: List[Dict[str, Any]], filename: str,
                use_data_subdir: bool = True):
    """
    Save experimental results with proper directory structure.

    Args:
        results: List of result dictionaries
        filename: Output filename
        use_data_subdir: Whether to save in 'data' subdirectory
    """
    # Handle path creation
    if not os.path.isabs(filename):
        if use_data_subdir:
            # Create results/data directory structure
            results_dir = os.path.join(os.getcwd(), "results", "data")
        else:
            results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        full_path = os.path.join(results_dir, filename)
    else:
        # Absolute path provided
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        full_path = filename

    # Save results as pickle file
    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

    # Report save status
    file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
    print(f"Results saved successfully!")
    print(f"  File: {full_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Combinations: {len(results)}")

def load_results(filename: str) -> List[Dict[str, Any]]:
    """
    Load experimental results from file.

    Args:
        filename: Input filename

    Returns:
        List of result dictionaries
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    print(f"Results loaded: {len(results)} combinations from {filename}")
    return results

if __name__ == "__main__":
    # Example usage and testing
    print("Chaos Experiment - Standalone Test")
    print("=" * 40)

    # Create small experiment for testing
    experiment = ChaosExperiment(n_neurons=100)  # Small network for speed

    # Create small parameter grid
    v_th_std_values, g_std_values = create_parameter_grid(n_points=3)

    print(f"Testing with {len(v_th_std_values)}x{len(g_std_values)} parameter grid")

    # Run experiment
    session_id = 999  # Test session
    results = experiment.run_full_experiment(
        session_id=session_id,
        v_th_std_values=v_th_std_values,
        g_std_values=g_std_values
    )

    # Save test results
    test_filename = f"test_chaos_results_session_{session_id}.pkl"
    save_results(results, test_filename)

    print("\nStandalone test completed successfully!")
    print(f"Results saved as: {test_filename}")
