# experiments/task_performance_experiment.py
"""
Unified task-performance experiment: categorical classification and temporal transformation.
MODIFIED: Can load cached evoked spikes instead of simulating.
"""

import numpy as np
import os
import sys
import time
import pickle
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
import gc

# Import base class
from .base_experiment import BaseExperiment
from .experiment_utils import (
    apply_exponential_filter,
    train_task_readout,
    predict_task_readout,
    evaluate_categorical_task,
    evaluate_temporal_task
)

try:
    from src.spiking_network import SpikingRNN
    from src.hd_input import HDInputGenerator
    from src.rng_utils import get_rng
    from analysis.common_utils import spikes_to_matrix, compute_empirical_dimensionality
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    for subdir in ['src', 'analysis']:
        sys.path.insert(0, os.path.join(project_root, subdir))
    from spiking_network import SpikingRNN
    from hd_input import HDInputGenerator
    from rng_utils import get_rng
    from common_utils import spikes_to_matrix, compute_empirical_dimensionality


def compute_pattern_dimensionalities(patterns: Dict[int, np.ndarray]) -> List[float]:
    """
    Compute empirical dimensionality for each pattern.

    Args:
        patterns: Dictionary mapping pattern_id to pattern array (n_timesteps, n_features)

    Returns:
        List of dimensionalities, one per pattern
    """
    dims = []
    for pattern_id in sorted(patterns.keys()):
        pattern = patterns[pattern_id]
        dim = compute_empirical_dimensionality(pattern)
        dims.append(dim)
    return dims


class TaskPerformanceExperiment(BaseExperiment):
    """
    Unified task experiment: categorical classification or temporal transformation.
    MODIFIED: Supports loading cached evoked spikes instead of simulating.
    """

    def __init__(self, task_type: str, n_neurons: int = 1000,
                 n_input_patterns: int = 10,
                 input_dim_intrinsic: int = 3, input_dim_embedding: int = 10,
                 output_dim_intrinsic: int = 3, output_dim_embedding: int = 10,
                 dt: float = 0.1, tau_syn: float = 5.0,
                 synaptic_mode: str = "filter",
                 static_input_mode: str = "independent",
                 hd_input_mode: str = "independent",
                 hd_connection_mode: str = "overlapping",
                 signal_cache_dir: str = "hd_signals",
                 decision_window: float = 50.0,
                 stimulus_duration: float = 300.0,
                 n_trials_per_pattern: int = 100,
                 lambda_reg: float = 1e-3,
                 use_distributed_cv: bool = False):
        """
        Initialize task performance experiment.

        Args:
            task_type: 'categorical' or 'temporal'
            n_neurons: Number of neurons
            n_input_patterns: Number of input patterns
            input_dim_intrinsic: Intrinsic dimensionality of input HD space
            input_dim_embedding: Embedding dimension for input
            output_dim_intrinsic: Intrinsic dimensionality of output HD space
            output_dim_embedding: Embedding dimension for output
            dt: Time step (ms)
            tau_syn: Synaptic filter time constant (ms)
            synaptic_mode: Synaptic dynamics mode
            static_input_mode: Static input mode
            hd_input_mode: HD input mode
            hd_connection_mode: 'overlapping' or 'partitioned'
            signal_cache_dir: Directory with pre-generated HD signals
            decision_window: Window for decision in categorical task (ms)
            stimulus_duration: Duration of stimulus (ms)
            n_trials_per_pattern: Number of trials per pattern
            lambda_reg: Ridge regression regularization
            use_distributed_cv: Use distributed cross-validation
        """
        super().__init__(n_neurons, dt)

        if task_type not in ['categorical', 'temporal', 'autoencoding']:  # ADD autoencoding
            raise ValueError("task_type must be 'categorical', 'temporal', or 'autoencoding'")


        self.task_type = task_type
        self.n_input_patterns = n_input_patterns
        self.input_dim_intrinsic = input_dim_intrinsic
        self.input_dim_embedding = input_dim_embedding
        self.output_dim_intrinsic = output_dim_intrinsic
        self.output_dim_embedding = output_dim_embedding
        self.tau_syn = tau_syn
        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.hd_input_mode = hd_input_mode
        self.hd_connection_mode = hd_connection_mode
        self.signal_cache_dir = signal_cache_dir
        self.decision_window = decision_window
        self.stimulus_duration = stimulus_duration
        self.transient_time = 0.0  # No transient when using cached spikes
        self.total_duration = self.stimulus_duration
        self.n_trials_per_pattern = n_trials_per_pattern
        self.lambda_reg = lambda_reg
        self.use_distributed_cv = use_distributed_cv

        # Initialize HD input generator
        self.input_generator = HDInputGenerator(
            embed_dim=input_dim_embedding,
            dt=dt,
            signal_cache_dir=signal_cache_dir,
            signal_type='hd_input'
        )

        # Output generator
        if task_type == 'temporal':
            # Transformation: use separate hd_output signals
            self.output_generator = HDInputGenerator(
                embed_dim=output_dim_embedding,
                dt=dt,
                signal_cache_dir=signal_cache_dir,
                signal_type='hd_output'
            )
        else:
            self.output_generator = None  # No generator for categorical/autoencoding

    def load_cached_trial_spikes(self, session_id: int, v_th_std: float,
                                g_std: float, static_rate: float,
                                pattern_id: int,
                                hd_connection_mode: str,
                                spike_cache_dir: str = "results/cached_spikes") -> List[List[Tuple[float, int]]]:
        """
        Load pre-cached evoked spikes for ALL trials of a pattern.
        Args:
            session_id: Session identifier
            v_th_std: Threshold heterogeneity std
            g_std: Weight heterogeneity std
            static_rate: Static input rate (Hz)
            pattern_id: Pattern identifier
            hd_connection_mode: 'overlapping' or 'partitioned'
            spike_cache_dir: Directory with cached spike files
        Returns:
            List of 100 trials, each trial is a list of (time, neuron_id) spike tuples
        """
        filename = os.path.join(spike_cache_dir, hd_connection_mode,
            f"session_{session_id}_g_{g_std:.3f}_vth_{v_th_std:.3f}_"
            f"rate_{static_rate:.1f}_h_{self.input_dim_intrinsic}_"
            f"d_{self.input_dim_embedding}_pattern_{pattern_id}_spikes.pkl")

        # ADD DEBUG
        if not hasattr(self, '_debug_printed'):
            print(f"   DEBUG: Cache lookup: {filename}")
            print(f"   DEBUG: File exists: {os.path.exists(filename)}")
            self._debug_printed = True

        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)

        # Return ALL 100 trials for this pattern
        return cache_data['trial_spikes']

    def run_single_trial(self, session_id: int, v_th_std: float, g_std: float,
                        trial_id: int, pattern_id: int,
                        noisy_input_pattern: np.ndarray,
                        static_input_rate: float,
                        v_th_distribution: str) -> Dict[str, Any]:
        """
        Run single trial by simulating network response to HD input.
        NOTE: This method is only used when use_cached_spikes=False.

        Args:
            session_id: Session identifier
            v_th_std: Threshold heterogeneity std
            g_std: Weight heterogeneity std
            trial_id: Trial identifier
            pattern_id: Pattern identifier
            noisy_input_pattern: Noisy HD input pattern
            static_input_rate: Static input rate (Hz)
            v_th_distribution: Threshold distribution type

        Returns:
            Dictionary with pattern_id, trial_id, and spike_times
        """
        # Create network
        network = SpikingRNN(
            n_neurons=self.n_neurons,
            dt=self.dt,
            synaptic_mode=self.synaptic_mode,
            static_input_mode=self.static_input_mode,
            hd_input_mode=self.hd_input_mode,
            hd_connection_mode=self.hd_connection_mode,
            n_hd_channels=self.input_dim_embedding
        )

        # Initialize network
        network.initialize_network(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            v_th_distribution=v_th_distribution,
            hd_dim=self.input_dim_intrinsic,
            embed_dim=self.input_dim_embedding
        )

        # Run simulation (no transient when simulating fresh)
        spike_times = network.simulate(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.stimulus_duration,
            hd_input_patterns=noisy_input_pattern,
            hd_dim=self.input_dim_intrinsic,
            embed_dim=self.input_dim_embedding,
            static_input_rate=static_input_rate,
            transient_time=0.0,
            continue_from_state=False
        )

        return {
            'pattern_id': pattern_id,
            'trial_id': trial_id % self.n_trials_per_pattern,
            'spike_times': spike_times
        }

    def simulate_trials_parallel(self, session_id: int, v_th_std: float,
                                g_std: float, v_th_distribution: str,
                                static_input_rate: float,
                                my_trial_indices: List[int],
                                input_patterns: Dict[int, np.ndarray],
                                rank: int,
                                use_cached_spikes: bool = True,
                                hd_connection_mode: str = "overlapping",
                                spike_cache_dir: str = "results/cached_spikes") -> List[Dict[str, Any]]:
        """
        Simulate or load cached spikes for assigned trials.
        Args:
            session_id: Session identifier
            v_th_std: Threshold heterogeneity std
            g_std: Weight heterogeneity std
            v_th_distribution: Threshold distribution type
            static_input_rate: Static input rate (Hz)
            my_trial_indices: List of global trial indices for this rank
            input_patterns: Dictionary of input patterns
            rank: MPI rank
            use_cached_spikes: If True, load from cache; if False, simulate
            hd_connection_mode: 'overlapping' or 'partitioned'
            spike_cache_dir: Directory with cached spike files
        Returns:
            List of trial result dictionaries
        """
        local_results = []

        # Memory-efficient caching: track current pattern
        pattern_spike_cache = {}
        current_pattern_id = None

        for i, trial_idx in enumerate(my_trial_indices):
            pattern_id = trial_idx // self.n_trials_per_pattern
            trial_within_pattern = trial_idx % self.n_trials_per_pattern

            if use_cached_spikes:
                # LOAD CACHED SPIKES
                if rank == 0 and i == 0:  # Print once
                    print(f"   DEBUG: Attempting to load cached spikes from {spike_cache_dir}/{hd_connection_mode}/")

                # Check if we moved to a new pattern
                if pattern_id != current_pattern_id:
                    # Delete old pattern data to free memory
                    if current_pattern_id is not None and current_pattern_id in pattern_spike_cache:
                        del pattern_spike_cache[current_pattern_id]
                        gc.collect()

                    # Load new pattern if not already loaded
                    if pattern_id not in pattern_spike_cache:
                        all_trials_for_pattern = self.load_cached_trial_spikes(
                            session_id, v_th_std, g_std, static_input_rate,
                            pattern_id, hd_connection_mode, spike_cache_dir
                        )
                        pattern_spike_cache[pattern_id] = all_trials_for_pattern

                    current_pattern_id = pattern_id

                # Extract the specific trial
                spike_times = pattern_spike_cache[pattern_id][trial_within_pattern]

                trial_result = {
                    'pattern_id': pattern_id,
                    'trial_id': trial_within_pattern,
                    'global_trial_idx': trial_idx,
                    'spike_times': spike_times
                }
            else:
                # SIMULATE FRESH
                base_pattern = input_patterns[pattern_id]
                rng = get_rng(session_id, v_th_std, g_std, trial_idx,
                            f'hd_input_noise_{pattern_id}',
                            rate=static_input_rate,
                            hd_dim=self.input_dim_intrinsic,
                            embed_dim=self.input_dim_embedding)
                noise = rng.normal(0, 0.5, base_pattern.shape)
                noisy_input = base_pattern + noise
                noisy_input = noisy_input - np.min(noisy_input)
                noisy_input = noisy_input * 1.0  # rate_scale
                trial_result = self.run_single_trial(
                    session_id, v_th_std, g_std, trial_idx, pattern_id,
                    noisy_input, static_input_rate, v_th_distribution)
                trial_result['global_trial_idx'] = trial_idx

            local_results.append(trial_result)

            if (i + 1) % 20 == 0:
                print(f"   Rank {rank}: completed {i+1}/{len(my_trial_indices)} trials")

        # Clean up cached data after loop
        if use_cached_spikes:
            pattern_spike_cache.clear()
            del pattern_spike_cache
            gc.collect()

        return local_results

    def generate_output_patterns(self, session_id: int) -> Dict[int, np.ndarray]:
        """Generate output patterns for each task type."""
        output_patterns = {}
        n_timesteps = int(self.stimulus_duration / self.dt)

        if self.task_type == 'categorical':
            # Create one-hot vectors
            for pattern_id in range(self.n_input_patterns):
                one_hot = np.zeros(self.n_input_patterns)
                one_hot[pattern_id] = 1.0
                output_patterns[pattern_id] = np.tile(one_hot, (n_timesteps, 1))
        elif self.task_type == 'temporal':
            # Load hd_output signals
            for pattern_id in range(self.n_input_patterns):
                self.output_generator.initialize_base_input(
                    session_id=session_id,
                    hd_dim=self.output_dim_intrinsic,
                    pattern_id=pattern_id
                )
                output_patterns[pattern_id] = self.output_generator.Y_base.copy()
        elif self.task_type == 'autoencoding':
            raise ValueError("Autoencoding uses input_patterns in runner")

        return output_patterns

    def convert_spikes_to_traces(self, all_trial_results: List[Dict[str, Any]],
                                 output_patterns: Dict[int, np.ndarray],
                                 n_patterns: int,
                                 n_trials_per_pattern: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spike times to filtered traces for training.

        Args:
            all_trial_results: List of all trial results
            output_patterns: Output patterns (for temporal) or input patterns (for categorical)
            n_patterns: Number of patterns
            n_trials_per_pattern: Number of trials per pattern

        Returns:
            Tuple of (traces_all, ground_truth_all, pattern_ids)
        """
        n_trials = len(all_trial_results)
        n_timesteps = int(self.stimulus_duration / self.dt)


        traces_all = np.zeros((n_trials, n_timesteps, self.n_neurons))

        if self.task_type == 'categorical':
            ground_truth_all = np.zeros((n_trials, n_timesteps, n_patterns))
        else:  # temporal
            ground_truth_all = np.zeros((n_trials, n_timesteps, self.output_dim_embedding))

        pattern_ids = np.zeros(n_trials, dtype=int)

        for idx, trial_result in enumerate(all_trial_results):
            pattern_id = trial_result['pattern_id']
            spike_times = trial_result['spike_times']
            pattern_ids[idx] = pattern_id

            # Convert spikes to binary matrix
            spike_matrix = spikes_to_matrix(
                spike_times,      # spike_list
                n_timesteps,      # n_steps (number of time bins)
                self.n_neurons,   # n_neurons
                self.dt           # step_size
            )

            # Apply exponential filter
            filtered_trace = apply_exponential_filter(spike_matrix, self.tau_syn, self.dt)
            traces_all[idx] = filtered_trace

            # Ground truth
            if self.task_type == 'categorical':
                ground_truth_all[idx, :, pattern_id] = 1.0
            else:  # temporal
                output_pattern = output_patterns[pattern_id]
                n_steps = min(n_timesteps, len(output_pattern))
                ground_truth_all[idx, :n_steps, :] = output_pattern[:n_steps]

        return traces_all, ground_truth_all, pattern_ids



    def cross_validate(self, traces_all: np.ndarray, ground_truth_all: np.ndarray,
                    pattern_ids: np.ndarray, session_id: int, n_folds: int,
                    rank: int, size: int, comm: Any) -> Dict[str, Any]:
        """
        Perform cross-validation (centralized on rank 0).

        Args:
            traces_all: Filtered traces (n_trials, n_timesteps, n_neurons)
            ground_truth_all: Ground truth (n_trials, n_timesteps, n_outputs)
            pattern_ids: Pattern labels for each trial
            session_id: Session identifier for RNG
            n_folds: Number of CV folds
            rank: MPI rank
            size: MPI size
            comm: MPI communicator

        Returns:
            Dictionary with CV results
        """
        if rank != 0:
            return {}

        n_trials = len(traces_all)

        # K-fold split
        rng = get_rng(session_id, 0, 0, 0, 'cv_split')
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 1000000)))

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_trials), pattern_ids)):
            X_train = traces_all[train_idx]
            Y_train = ground_truth_all[train_idx]
            X_test = traces_all[test_idx]
            Y_test = ground_truth_all[test_idx]

            # Train readout
            W = train_task_readout(X_train, Y_train, self.lambda_reg)

            # Predict
            Y_train_pred = predict_task_readout(X_train, W)
            Y_test_pred = predict_task_readout(X_test, W)

            # Evaluate
            if self.task_type == 'categorical':
                decision_window_steps = int(self.decision_window / self.dt)
                train_metrics = evaluate_categorical_task(Y_train_pred, Y_train, decision_window_steps)
                test_metrics = evaluate_categorical_task(Y_test_pred, Y_test, decision_window_steps)

                fold_results.append({
                    'train_accuracy': train_metrics['accuracy'],
                    'train_accuracy_timeaveraged': train_metrics['accuracy_timeaveraged'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_accuracy_timeaveraged': test_metrics['accuracy_timeaveraged'],
                    'test_confidence': test_metrics['mean_confidence'],
                    'test_confusion_matrix': test_metrics['confusion_matrix'],
                    'test_confusion_matrix_timeaveraged': test_metrics['confusion_matrix_timeaveraged'],
                    'test_methods_agreement_rate': test_metrics['methods_agreement_rate']
                })

            else:  # temporal or autoencoding
                train_metrics = evaluate_temporal_task(Y_train_pred, Y_train)
                test_metrics = evaluate_temporal_task(Y_test_pred, Y_test)

                fold_results.append({
                    'train_rmse': train_metrics['rmse_mean'],
                    'train_r2': train_metrics['r2_mean'],
                    'train_correlation': train_metrics['correlation_mean'],
                    'test_rmse': test_metrics['rmse_mean'],
                    'test_r2': test_metrics['r2_mean'],
                    'test_correlation': test_metrics['correlation_mean']
                })

            if (fold_idx + 1) % 5 == 0:
                print(f"    Fold {fold_idx + 1}/{n_folds}")

        # Aggregate results
        if self.task_type == 'categorical':
            test_acc = [f['test_accuracy'] for f in fold_results]
            test_acc_tavg = [f['test_accuracy_timeaveraged'] for f in fold_results]
            test_conf = [f['test_confidence'] for f in fold_results]
            agreement_rates = [f['test_methods_agreement_rate'] for f in fold_results]

            results = {
                'test_accuracy_bayesian_mean': float(np.mean(test_acc)),
                'test_accuracy_bayesian_std': float(np.std(test_acc)),
                'test_accuracy_timeaveraged_mean': float(np.mean(test_acc_tavg)),
                'test_accuracy_timeaveraged_std': float(np.std(test_acc_tavg)),
                'test_confidence_bayesian_mean': float(np.mean(test_conf)),
                'test_confidence_bayesian_std': float(np.std(test_conf)),
                'test_methods_agreement_rate_mean': float(np.mean(agreement_rates)),
                'test_methods_agreement_rate_std': float(np.std(agreement_rates)),
                'cv_accuracy_bayesian_per_fold': test_acc,
                'cv_accuracy_timeaveraged_per_fold': test_acc_tavg,
                'cv_confusion_matrices_bayesian': [f['test_confusion_matrix'] for f in fold_results],
                'cv_confusion_matrices_timeaveraged': [f['test_confusion_matrix_timeaveraged'] for f in fold_results]
            }
        else:  # temporal or autoencoding
            test_rmse = [f['test_rmse'] for f in fold_results]
            test_r2 = [f['test_r2'] for f in fold_results]
            train_rmse = [f['train_rmse'] for f in fold_results]
            train_r2 = [f['train_r2'] for f in fold_results]

            results = {
                'test_rmse_mean': float(np.mean(test_rmse)),
                'test_rmse_std': float(np.std(test_rmse)),
                'test_r2_mean': float(np.mean(test_r2)),
                'test_r2_std': float(np.std(test_r2)),
                'train_rmse_mean': float(np.mean(train_rmse)),
                'train_r2_mean': float(np.mean(train_r2)),
                'cv_rmse_per_fold': test_rmse,
                'cv_r2_per_fold': test_r2
            }

        return results



    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract arrays from trial results (required by base class).

        For task experiments, this is not used because we handle data
        conversion in convert_spikes_to_traces() and cross-validation separately.
        Just return empty dict to satisfy the abstract method requirement.

        Args:
            trial_results: List of trial result dictionaries

        Returns:
            Empty dictionary (method not used in task experiments)
        """
        return {}
