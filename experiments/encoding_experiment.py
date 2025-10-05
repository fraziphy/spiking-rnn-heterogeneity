# experiments/encoding_experiment.py - HD input encoding capacity study
"""
Encoding capacity experiment: study how networks encode high-dimensional inputs
with varying intrinsic dimensionality.
"""

import numpy as np
import os
import sys
import time
import pickle
import random
from typing import Dict, List, Tuple, Any

import warnings

def compute_safe_mean(array):
    """Compute mean suppressing empty slice warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='Mean of empty slice')
        return float(np.nanmean(array))

def compute_safe_std(array):
    """Compute std suppressing degrees of freedom warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='Degrees of freedom')
        return float(np.nanstd(array))

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.hd_input_generator import HDInputGenerator
    from src.rng_utils import get_rng
    from analysis.encoding_analysis import decode_hd_input
except ImportError:
    try:
        from ..src.spiking_network import SpikingRNN
        from ..src.hd_input_generator import HDInputGenerator
        from ..src.rng_utils import get_rng
        from ..analysis.encoding_analysis import decode_hd_input
    except ImportError:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        src_dir = os.path.join(project_root, 'src')
        analysis_dir = os.path.join(project_root, 'analysis')
        sys.path.insert(0, src_dir)
        sys.path.insert(0, analysis_dir)
        from spiking_network import SpikingRNN
        from hd_input_generator import HDInputGenerator
        from rng_utils import get_rng
        from encoding_analysis import decode_hd_input


class EncodingExperiment:
    """Encoding capacity experiment with HD inputs."""

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 synaptic_mode: str = "filter",
                 static_input_mode: str = "independent",
                 hd_input_mode: str = "independent",
                 embed_dim: int = 10):
        """
        Initialize encoding experiment.

        Args:
            n_neurons: Number of RNN neurons
            dt: Time step (ms)
            synaptic_mode: "pulse" or "filter"
            static_input_mode: Static background mode
            hd_input_mode: HD input mode ("independent", "common_stochastic", "common_tonic")
            embed_dim: HD embedding dimensionality (number of input channels)
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.hd_input_mode = hd_input_mode
        self.embed_dim = embed_dim

        # Timing parameters
        self.transient_time = 50.0  # ms
        self.encoding_time = 300.0  # ms
        self.total_duration = self.transient_time + self.encoding_time

        # Number of trials
        self.n_trials = 20

        # HD input generator
        self.hd_generator = HDInputGenerator(embed_dim=embed_dim, dt=dt)

    def run_single_trial(self, session_id: int, v_th_std: float, g_std: float,
                        trial_id: int, hd_dim: int,
                        v_th_distribution: str = "normal",
                        static_input_rate: float = 200.0,
                        hd_noise_std: float = 0.5,
                        hd_rate_scale: float = 1.0) -> Dict[str, Any]:
        """
        Run single encoding trial.

        Args:
            session_id: Session ID
            v_th_std: Threshold std
            g_std: Weight std
            trial_id: Trial ID
            hd_dim: HD intrinsic dimensionality
            v_th_distribution: Threshold distribution
            static_input_rate: Background input rate
            hd_noise_std: Noise std for HD input
            hd_rate_scale: Rate scaling for HD input

        Returns:
            Trial results with spike times and readout
        """
        # Create network
        network = SpikingRNN(
            self.n_neurons,
            dt=self.dt,
            synaptic_mode=self.synaptic_mode,
            static_input_mode=self.static_input_mode,
            hd_input_mode=self.hd_input_mode,
            n_hd_channels=self.embed_dim
        )

        # Network parameters
        network_params = {
            'v_th_distribution': v_th_distribution,
            'static_input_strength': 10.0,
            'hd_connection_prob': 0.3,
            'hd_input_strength': 1.0,
            'readout_weight_scale': 1.0
        }

        # Initialize network
        network.initialize_network(
            session_id, v_th_std, g_std,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim,
            **network_params
        )

        # Generate HD input for this trial
        hd_input_patterns = self.hd_generator.generate_trial_input(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            hd_dim=hd_dim,
            noise_std=hd_noise_std,
            rate_scale=hd_rate_scale
        )

        # Run encoding simulation
        spike_times, readout_history = network.simulate_encoding_task(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.total_duration,
            hd_input_patterns=hd_input_patterns,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim,
            static_input_rate=static_input_rate,
            transient_time=self.transient_time
        )

        # Extract encoding period spikes (after transient)
        encoding_spikes = [(t - self.transient_time, nid)
                          for t, nid in spike_times
                          if t >= self.transient_time]

        # Extract encoding period readout
        encoding_readout = [(t - self.transient_time, activity)
                           for t, activity in readout_history
                           if t >= self.transient_time]

        return {
            'spike_times': encoding_spikes,
            'readout_history': encoding_readout,
            'hd_input_patterns': hd_input_patterns,
            'n_spikes': len(encoding_spikes),
            'trial_id': trial_id
        }

    def run_parameter_combination(self, session_id: int, v_th_std: float, g_std: float,
                                 hd_dim: int,
                                 v_th_distribution: str = "normal",
                                 static_input_rate: float = 200.0,
                                 hd_noise_std: float = 0.5,
                                 hd_rate_scale: float = 1.0,
                                 signal_manager = None) -> Dict[str, Any]:
        """
        Run full parameter combination with multiple trials.

        Args:
            session_id: Session ID
            v_th_std: Threshold std
            g_std: Weight std
            hd_dim: HD intrinsic dimensionality
            v_th_distribution: Threshold distribution
            static_input_rate: Background input rate
            hd_noise_std: Noise std for HD input
            hd_rate_scale: Rate scaling for HD input
            signal_manager: HDSignalManager instance (optional, for caching)

        Returns:
            Results dictionary with all trials and decoding analysis
        """
        start_time = time.time()

        # Load or generate HD base signal
        if signal_manager is not None:
            signal_data = signal_manager.get_or_generate_signal(
                session_id=session_id,
                hd_dim=hd_dim,
                embed_dim=self.embed_dim
            )
            # Set the generator's base signal
            self.hd_generator.Y_base = signal_data['Y_base']
            self.hd_generator.chosen_components = signal_data['chosen_components']
            self.hd_generator.n_timesteps = signal_data['n_timesteps']
        else:
            # Initialize HD generator base input (not cached)
            rate_rnn_params = {
                'n_neurons': 1000,
                'T': 350.0,  # 50ms transient + 300ms encoding
                'g': 1.2
            }

            self.hd_generator.initialize_base_input(
                session_id=session_id,
                hd_dim=hd_dim,
                rate_rnn_params=rate_rnn_params
            )

        # Run all trials
        trial_results = []

        for trial_idx in range(self.n_trials):
            trial_id = trial_idx + 1

            trial_result = self.run_single_trial(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                hd_dim=hd_dim,
                v_th_distribution=v_th_distribution,
                static_input_rate=static_input_rate,
                hd_noise_std=hd_noise_std,
                hd_rate_scale=hd_rate_scale
            )

            trial_results.append(trial_result)

        # Extract arrays for basic statistics
        n_spikes_array = np.array([r['n_spikes'] for r in trial_results])

        # Perform decoding analysis
        print(f"    Running decoding analysis...")

        decoding_results = decode_hd_input(
            trial_results=trial_results,
            hd_input_ground_truth=self.hd_generator.Y_base,
            n_neurons=self.n_neurons,
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim,
            encoding_duration=self.encoding_time,
            dt=self.dt,
            tau=10.0,
            lambda_reg=1e-3,
            n_splits=self.n_trials  # LOOCV
        )

        # Compile results
        results = {
            # Parameter information
            'session_id': session_id,
            'v_th_std': v_th_std,
            'g_std': g_std,
            'hd_dim': hd_dim,
            'embed_dim': self.embed_dim,
            'v_th_distribution': v_th_distribution,
            'static_input_rate': static_input_rate,
            'hd_noise_std': hd_noise_std,
            'hd_rate_scale': hd_rate_scale,
            'synaptic_mode': self.synaptic_mode,
            'static_input_mode': self.static_input_mode,
            'hd_input_mode': self.hd_input_mode,

            # Trial data
            'trial_results': trial_results,

            # Basic statistics
            'n_spikes_mean': float(np.mean(n_spikes_array)),
            'n_spikes_std': float(np.std(n_spikes_array)),

            # HD input statistics
            'hd_base_stats': self.hd_generator.get_base_statistics(),

            # Decoding analysis
            'decoding': decoding_results,

            # Metadata
            'n_trials': len(trial_results),
            'computation_time': time.time() - start_time,
            'transient_time': self.transient_time,
            'encoding_time': self.encoding_time
        }

        return results

    def run_full_experiment(self, session_id: int, v_th_stds: np.ndarray,
                          g_stds: np.ndarray, hd_dims: np.ndarray,
                          v_th_distribution: str = "normal",
                          static_input_rates: np.ndarray = None,
                          hd_noise_std: float = 0.5,
                          hd_rate_scale: float = 1.0,
                          use_signal_cache: bool = True,
                          signal_cache_dir: str = "hd_signals") -> List[Dict[str, Any]]:
        """
        Run full encoding experiment with randomized job distribution.

        Args:
            session_id: Session ID
            v_th_stds: Array of threshold stds
            g_stds: Array of weight stds
            hd_dims: Array of HD intrinsic dimensionalities
            v_th_distribution: Threshold distribution
            static_input_rates: Array of background rates
            hd_noise_std: Noise std for HD input
            hd_rate_scale: Rate scaling for HD input
            use_signal_cache: Whether to use HD signal caching
            signal_cache_dir: Directory for signal cache

        Returns:
            List of results for all parameter combinations
        """
        if static_input_rates is None:
            static_input_rates = np.array([200.0])

        # Initialize signal manager if caching is enabled
        if use_signal_cache:
            from src.hd_signal_manager import HDSignalManager
            signal_manager = HDSignalManager(signal_cache_dir)

            # Pre-generate all HD signals for this session
            print(f"Pre-generating HD signals for session {session_id}...")
            for hd_dim in hd_dims:
                signal_manager.get_or_generate_signal(session_id, int(hd_dim), self.embed_dim)
            print("Signal pre-generation complete.\n")
        else:
            signal_manager = None

        # Create all parameter combinations
        all_combinations = []
        combo_idx = 0
        for input_rate in static_input_rates:
            for hd_dim in hd_dims:
                for v_th_std in v_th_stds:
                    for g_std in g_stds:
                        all_combinations.append({
                            'combo_idx': combo_idx,
                            'session_id': session_id,
                            'v_th_std': v_th_std,
                            'g_std': g_std,
                            'hd_dim': hd_dim,
                            'v_th_distribution': v_th_distribution,
                            'static_input_rate': input_rate,
                            'hd_noise_std': hd_noise_std,
                            'hd_rate_scale': hd_rate_scale
                        })
                        combo_idx += 1

        # RANDOMIZE job order for better CPU load balancing
        random.shuffle(all_combinations)

        total_combinations = len(all_combinations)

        print(f"Starting encoding experiment with randomized job distribution: {total_combinations} combinations")
        print(f"  Session ID: {session_id}")
        print(f"  v_th_stds: {len(v_th_stds)} (range: {np.min(v_th_stds):.3f}-{np.max(v_th_stds):.3f})")
        print(f"  g_stds: {len(g_stds)} (range: {np.min(g_stds):.3f}-{np.max(g_stds):.3f})")
        print(f"  hd_dims: {len(hd_dims)} (range: {np.min(hd_dims)}-{np.max(hd_dims)})")
        print(f"  embed_dim: {self.embed_dim}")
        print(f"  v_th_distribution: {v_th_distribution}")
        print(f"  Static rates: {static_input_rates}")
        print(f"  Synaptic mode: {self.synaptic_mode}")
        print(f"  Static input mode: {self.static_input_mode}")
        print(f"  HD input mode: {self.hd_input_mode}")
        print(f"  Trials per combination: {self.n_trials}")
        print(f"  Job order: RANDOMIZED for load balancing")

        results = []
        for i, combo in enumerate(all_combinations):
            print(f"[{i+1}/{total_combinations}] Processing randomized job:")
            print(f"    v_th_std={combo['v_th_std']:.3f}, g_std={combo['g_std']:.3f}")
            print(f"    hd_dim={combo['hd_dim']}, rate={combo['static_input_rate']:.0f}Hz")

            result = self.run_parameter_combination(
                session_id=combo['session_id'],
                v_th_std=combo['v_th_std'],
                g_std=combo['g_std'],
                hd_dim=combo['hd_dim'],
                v_th_distribution=combo['v_th_distribution'],
                static_input_rate=combo['static_input_rate'],
                hd_noise_std=combo['hd_noise_std'],
                hd_rate_scale=combo['hd_rate_scale']
            )

            result['original_combination_index'] = combo['combo_idx']
            results.append(result)

            # Progress reporting
            print(f"  Mean spikes: {result['n_spikes_mean']:.0f}±{result['n_spikes_std']:.0f}")
            print(f"  Time: {result['computation_time']:.1f}s")

        # Sort results back by original combination index
        results.sort(key=lambda x: x['original_combination_index'])

        print(f"Encoding experiment completed: {len(results)} combinations processed")
        return results


def create_parameter_grid(n_v_th_points: int = 5, n_g_points: int = 5,
                         n_hd_points: int = 5,
                         v_th_std_range: Tuple[float, float] = (0.0, 4.0),
                         g_std_range: Tuple[float, float] = (0.0, 4.0),
                         hd_dim_range: Tuple[int, int] = (1, 10),
                         input_rate_range: Tuple[float, float] = (100.0, 500.0),
                         n_input_rates: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create parameter grids for encoding experiment."""
    v_th_stds = np.linspace(v_th_std_range[0], v_th_std_range[1], n_v_th_points)
    g_stds = np.linspace(g_std_range[0], g_std_range[1], n_g_points)
    hd_dims = np.linspace(hd_dim_range[0], hd_dim_range[1], n_hd_points, dtype=int)
    static_input_rates = np.linspace(input_rate_range[0], input_rate_range[1], n_input_rates)

    return v_th_stds, g_stds, hd_dims, static_input_rates


def save_results(results: List[Dict[str, Any]], filename: str, use_data_subdir: bool = True):
    """Save encoding experiment results."""
    if not os.path.isabs(filename):
        if use_data_subdir:
            results_dir = os.path.join(os.getcwd(), "results", "data")
            full_path = os.path.join(results_dir, filename)
        else:
            full_path = os.path.join(os.getcwd(), filename)
    else:
        full_path = filename

    directory = os.path.dirname(full_path)
    os.makedirs(directory, exist_ok=True)

    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Encoding results saved: {full_path}")


def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load encoding experiment results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Encoding results loaded: {len(results)} combinations from {filename}")
    return results


def average_across_sessions(results_files: List[str]) -> List[Dict[str, Any]]:
    """Average encoding results across sessions."""
    print(f"Averaging encoding results across {len(results_files)} sessions...")

    # Load all session results
    all_session_results = []
    for file_path in results_files:
        session_results = load_results(file_path)
        all_session_results.append(session_results)
        print(f"  Loaded session with {len(session_results)} combinations")

    # Verify consistency
    n_combinations = len(all_session_results[0])
    for i, session_results in enumerate(all_session_results[1:], 1):
        if len(session_results) != n_combinations:
            raise ValueError(f"Session {i+1} has {len(session_results)} combinations, expected {n_combinations}")

    # Average across sessions
    averaged_results = []

    for combo_idx in range(n_combinations):
        combo_results = [session_results[combo_idx] for session_results in all_session_results]
        first_result = combo_results[0]

        # Concatenate trial results across sessions
        all_trials = []
        for session_result in combo_results:
            all_trials.extend(session_result['trial_results'])

        # Compute statistics across all trials
        n_spikes_array = np.array([t['n_spikes'] for t in all_trials])

        # Create averaged result
        averaged_result = {
            # Parameter information
            'v_th_std': first_result['v_th_std'],
            'g_std': first_result['g_std'],
            'hd_dim': first_result['hd_dim'],
            'embed_dim': first_result['embed_dim'],
            'v_th_distribution': first_result['v_th_distribution'],
            'static_input_rate': first_result['static_input_rate'],
            'hd_noise_std': first_result['hd_noise_std'],
            'hd_rate_scale': first_result['hd_rate_scale'],
            'synaptic_mode': first_result['synaptic_mode'],
            'static_input_mode': first_result['static_input_mode'],
            'hd_input_mode': first_result['hd_input_mode'],
            'original_combination_index': first_result.get('original_combination_index', combo_idx),

            # All trial data
            'trial_results': all_trials,

            # Session-averaged statistics
            'n_spikes_mean': compute_safe_mean(n_spikes_array),
            'n_spikes_std': compute_safe_std(n_spikes_array),

            # HD input statistics
            'hd_base_stats': first_result['hd_base_stats'],

            # Metadata
            'n_sessions': len(combo_results),
            'n_trials_per_session': first_result['n_trials'],
            'total_trials': len(all_trials),
            'total_computation_time': sum(r['computation_time'] for r in combo_results),
            'session_ids_used': [r.get('session_id', 'unknown') for r in combo_results],
            'transient_time': first_result['transient_time'],
            'encoding_time': first_result['encoding_time']
        }

        averaged_results.append(averaged_result)

        if (combo_idx + 1) % 10 == 0:
            print(f"  Averaged {combo_idx + 1}/{n_combinations} combinations")

    print(f"Session averaging completed: {len(averaged_results)} combinations averaged")
    return averaged_results
