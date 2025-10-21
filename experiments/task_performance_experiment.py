# experiments/task_performance_experiment.py
"""
Unified task-performance experiment: categorical classification and temporal transformation.
NEW: Methods for distributed trial simulation and CV training.
"""

import numpy as np
import os
import sys
import time
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold  # <-- ADD THIS HERE

# Import base class
from .base_experiment import BaseExperiment
from .experiment_utils import (
    apply_exponential_filter,
    train_task_readout,
    predict_task_readout,
    evaluate_categorical_task,
    evaluate_temporal_task
)

# Import with flexible handling
try:
    from src.spiking_network import SpikingRNN
    from src.hd_input import HDInputGenerator
    from src.rng_utils import get_rng
    from analysis.common_utils import spikes_to_matrix
except ImportError:
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    for subdir in ['src', 'analysis']:
        sys.path.insert(0, os.path.join(project_root, subdir))
    from spiking_network import SpikingRNN
    from hd_input import HDInputGenerator
    from rng_utils import get_rng
    from common_utils import spikes_to_matrix

class TaskPerformanceExperiment(BaseExperiment):
    """
    Unified task experiment: categorical classification or temporal transformation.
    NEW: Supports distributed trial simulation and CV training.
    """

    def __init__(self,
                task_type: str = 'categorical',
                n_neurons: int = 1000,
                n_input_patterns: int = 4,
                input_dim_intrinsic: int = 5,
                input_dim_embedding: int = 10,
                output_dim_intrinsic: int = 1,
                output_dim_embedding: int = 1,
                dt: float = 0.1,
                tau_syn: float = 5.0,
                decision_window: float = 50.0,
                synaptic_mode: str = "filter",
                static_input_mode: str = "independent",
                hd_input_mode: str = "independent",
                signal_cache_dir: str = "hd_signals",
                stimulus_duration: float = 300.0,
                n_trials_per_pattern: int = 100,
                lambda_reg: float = 1e-3,
                use_distributed_cv: bool = False):
        """Initialize task-performance experiment."""
        super().__init__(n_neurons, dt)

        if task_type not in ['categorical', 'temporal']:
            raise ValueError(f"task_type must be 'categorical' or 'temporal', got '{task_type}'")

        self.task_type = task_type
        self.n_input_patterns = n_input_patterns

        # Input configuration
        self.input_dim_intrinsic = input_dim_intrinsic
        self.input_dim_embedding = input_dim_embedding

        # Output configuration
        self.output_dim_intrinsic = output_dim_intrinsic
        self.output_dim_embedding = output_dim_embedding

        # Network modes
        self.synaptic_mode = synaptic_mode
        self.static_input_mode = static_input_mode
        self.hd_input_mode = hd_input_mode

        # Timing
        self.transient_time = 200.0
        self.stimulus_duration = stimulus_duration
        self.total_duration = self.transient_time + self.stimulus_duration
        self.decision_window = decision_window

        # Synaptic filtering
        self.tau_syn = tau_syn

        # Training configuration
        self.n_trials_per_pattern = n_trials_per_pattern
        self.lambda_reg = lambda_reg

        self.use_distributed_cv = use_distributed_cv

        # Input generator
        self.input_generator = HDInputGenerator(
            embed_dim=input_dim_embedding,
            dt=dt,
            signal_cache_dir=os.path.join(signal_cache_dir, 'inputs')
        )

        # Output generator (temporal only)
        if task_type == 'temporal':
            self.output_generator = HDInputGenerator(
                embed_dim=output_dim_embedding,
                dt=dt,
                signal_cache_dir=os.path.join(signal_cache_dir, 'outputs')
            )
        else:
            self.output_generator = None

    def generate_output_patterns(self, session_id: int) -> Dict[int, np.ndarray]:
        """Generate output patterns for each class."""
        output_patterns = {}

        # Calculate timesteps based on stimulus_duration
        n_timesteps = int(self.stimulus_duration / self.dt)

        if self.task_type == 'categorical':
            # Categorical: constant one-hot vectors
            for pattern_id in range(self.n_input_patterns):
                one_hot = np.zeros(self.n_input_patterns)
                one_hot[pattern_id] = 1.0
                output_patterns[pattern_id] = np.tile(one_hot, (n_timesteps, 1))

        elif self.task_type == 'temporal':
            # Temporal: time-varying outputs from separate HD generator
            rate_rnn_duration = 200.0 + self.stimulus_duration

            for pattern_id in range(self.n_input_patterns):
                output_pattern_id = pattern_id + 100

                self.output_generator.initialize_base_input(
                    session_id=session_id,
                    hd_dim=self.output_dim_intrinsic,
                    pattern_id=output_pattern_id,
                    rate_rnn_params={
                        'n_neurons': 1000,
                        'T': rate_rnn_duration,
                        'g': 2.0
                    }
                )

                output_patterns[pattern_id] = self.output_generator.Y_base.copy()

        return output_patterns

    # NEW METHOD: Initialize and get all patterns at once
    def initialize_and_get_patterns(self, session_id: int, hd_dim: int,
                                   n_patterns: int) -> Dict[int, np.ndarray]:
        """
        Initialize and return all input patterns.
        Used for distributed execution where all ranks need same patterns.
        """
        patterns = {}

        for pattern_id in range(n_patterns):
            self.input_generator.initialize_base_input(
                session_id=session_id,
                hd_dim=hd_dim,
                pattern_id=pattern_id,
                rate_rnn_params={
                    'n_neurons': 1000,
                    'T': 200.0 + self.stimulus_duration,
                    'g': 2.0
                }
            )
            patterns[pattern_id] = self.input_generator.Y_base.copy()

        return patterns

    def run_single_trial(self, session_id: int, v_th_std: float, g_std: float,
                        trial_id: int, pattern_id: int,
                        noisy_input_pattern: np.ndarray,
                        static_input_rate: float = 200.0,
                        v_th_distribution: str = "normal") -> Dict[str, Any]:
        """Run single trial: network processes noisy input, collect RNN spikes."""

        # Create network WITHOUT readout synapses (offline training)
        network = SpikingRNN(
            n_neurons=self.n_neurons,
            dt=self.dt,
            synaptic_mode=self.synaptic_mode,
            static_input_mode=self.static_input_mode,
            hd_input_mode=self.hd_input_mode,
            n_hd_channels=self.input_dim_embedding,
            use_readout_synapses=False
        )

        # Initialize network
        network.initialize_network(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            v_th_distribution=v_th_distribution,
            hd_dim=self.input_dim_intrinsic,
            embed_dim=self.input_dim_embedding,
            static_input_strength=10.0,
            hd_connection_prob=0.3,
            hd_input_strength=50.0
        )

        # Run simulation (returns spike times only)
        spike_times, _ = network.simulate_encoding_task(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            trial_id=trial_id,
            duration=self.total_duration,
            hd_input_patterns=noisy_input_pattern,
            hd_dim=self.input_dim_intrinsic,
            embed_dim=self.input_dim_embedding,
            static_input_rate=static_input_rate,
            transient_time=self.transient_time
        )

        # Extract spikes during stimulus period only
        stimulus_spikes = [(t - self.transient_time, nid)
                          for t, nid in spike_times
                          if t >= self.transient_time]

        return {
            'pattern_id': pattern_id,
            'trial_id': trial_id,
            'spike_times': stimulus_spikes
        }

    # NEW METHOD: Simulate trials in parallel (distributed across ranks)
    def simulate_trials_parallel(self, session_id: int, v_th_std: float, g_std: float,
                                v_th_distribution: str, static_input_rate: float,
                                my_trial_indices: List[int],
                                input_patterns: Dict[int, np.ndarray],
                                rank: int) -> List[Dict[str, Any]]:
        """Simulate assigned trials on this rank."""
        local_results = []

        for i, trial_idx in enumerate(my_trial_indices):
            # Decode trial index to pattern_id and trial_within_pattern
            pattern_id = trial_idx // self.n_trials_per_pattern
            trial_within_pattern = trial_idx % self.n_trials_per_pattern

            # Generate noisy input for this trial
            base_pattern = input_patterns[pattern_id]

            # Add noise (trial-specific)
            rng = get_rng(session_id, v_th_std, g_std, trial_idx,
                        f'hd_input_noise_{pattern_id}',
                        rate=static_input_rate,  # ADD THIS
                        hd_dim=self.input_dim_intrinsic,
                        embed_dim=self.input_dim_embedding)

            noise = rng.normal(0, 0.5, base_pattern.shape)
            noisy_input = base_pattern + noise
            noisy_input = noisy_input - np.min(noisy_input)
            noisy_input = noisy_input * 1.0  # rate_scale

            # Run trial
            trial_result = self.run_single_trial(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_idx,
                pattern_id=pattern_id,
                noisy_input_pattern=noisy_input,
                static_input_rate=static_input_rate,
                v_th_distribution=v_th_distribution
            )

            trial_result['global_trial_idx'] = trial_idx
            local_results.append(trial_result)

            if (i + 1) % 20 == 0:
                print(f"   Rank {rank}: completed {i+1}/{len(my_trial_indices)} trials")

        return local_results

    # NEW METHOD: Convert spike times to traces (done locally on each rank)
    def convert_spikes_to_traces(self, all_trial_results: List[Dict[str, Any]],
                                 output_patterns: Dict[int, np.ndarray],
                                 n_patterns: int, n_trials_per_pattern: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spike times to filtered traces.
        Each rank does this independently after receiving all spike times.

        Returns:
            traces_all: (n_trials, n_timesteps, n_neurons)
            ground_truth_all: (n_trials, n_timesteps, n_outputs)
            pattern_ids: (n_trials,)
        """
        # Sort by global trial index
        all_trial_results_sorted = sorted(all_trial_results, key=lambda x: x['global_trial_idx'])

        n_timesteps = int(self.stimulus_duration / self.dt)
        n_trials = len(all_trial_results_sorted)

        traces_list = []
        ground_truth_list = []
        pattern_ids_list = []

        for trial_result in all_trial_results_sorted:
            # Convert spikes to matrix
            spike_matrix = spikes_to_matrix(
                trial_result['spike_times'],
                n_timesteps,
                self.n_neurons,
                self.dt
            )

            # Apply exponential filtering
            traces = apply_exponential_filter(spike_matrix, self.tau_syn, self.dt)
            traces_list.append(traces)

            # Get ground truth
            pattern_id = trial_result['pattern_id']
            ground_truth = output_patterns[pattern_id]
            ground_truth_list.append(ground_truth)
            pattern_ids_list.append(pattern_id)

        traces_all = np.array(traces_list)
        ground_truth_all = np.array(ground_truth_list)
        pattern_ids = np.array(pattern_ids_list)

        return traces_all, ground_truth_all, pattern_ids

    # NEW METHOD: Distributed CV training
    def cross_validate_distributed(self, traces_all: np.ndarray,
                                ground_truth_all: np.ndarray,
                                pattern_ids: np.ndarray,
                                session_id: int,
                                n_folds: int,
                                rank: int,
                                size: int,
                                comm) -> Dict[str, Any]:
        """
        Perform stratified K-fold CV with distributed training.

        Each rank does n_folds/size CV iterations.
        """
        # Check that n_folds divides evenly by size
        if n_folds % size != 0:
            if rank == 0:
                print(f"WARNING: n_folds ({n_folds}) not divisible by size ({size})")
                print(f"         Some ranks will do more work than others")

        # Determine which CV iterations this rank handles
        cv_per_rank = n_folds // size
        remainder = n_folds % size

        if rank < remainder:
            my_cv_start = rank * (cv_per_rank + 1)
            my_cv_count = cv_per_rank + 1
        else:
            my_cv_start = rank * cv_per_rank + remainder
            my_cv_count = cv_per_rank

        my_cv_iterations = list(range(my_cv_start, my_cv_start + my_cv_count))

        print(f"   Rank {rank}: CV iterations {my_cv_iterations}")

        # Get RNG for CV splits (same across all ranks for consistency)
        rng_cv = get_rng(session_id, 0.0, 0.0, 0,
                        f'task_{self.task_type}_cv_splits',
                        hd_dim=self.input_dim_intrinsic,
                        embed_dim=self.input_dim_embedding)

        cv_seed = int(rng_cv.integers(0, 2**31 - 1))

        # Create stratified folds (same across all ranks)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
        fold_splits = list(skf.split(traces_all, pattern_ids))

        # Each rank processes its assigned CV iterations
        local_fold_results = []
        local_weights = []

        for cv_iter in my_cv_iterations:
            train_idx, test_idx = fold_splits[cv_iter]

            # Split data
            X_train = traces_all[train_idx]
            X_test = traces_all[test_idx]
            y_train = ground_truth_all[train_idx]
            y_test = ground_truth_all[test_idx]

            # Train readout
            W_readout = train_task_readout(X_train, y_train, self.lambda_reg)
            local_weights.append(W_readout.copy())

            # Predict
            y_pred_train = predict_task_readout(X_train, W_readout)
            y_pred_test = predict_task_readout(X_test, W_readout)

            # Evaluate based on task type
            if self.task_type == 'categorical':
                decision_window_steps = int(self.decision_window / self.dt)

                train_metrics = evaluate_categorical_task(
                    y_pred_train, y_train, decision_window_steps
                )
                test_metrics = evaluate_categorical_task(
                    y_pred_test, y_test, decision_window_steps
                )

                local_fold_results.append({
                    'fold': cv_iter,
                    'train_accuracy': train_metrics['accuracy'],
                    'test_accuracy': test_metrics['accuracy'],
                    'confusion_matrix': test_metrics['confusion_matrix'],
                    'per_class_accuracy': test_metrics['per_class_accuracy']
                })

            else:  # temporal
                train_metrics = evaluate_temporal_task(y_pred_train, y_train)
                test_metrics = evaluate_temporal_task(y_pred_test, y_test)

                local_fold_results.append({
                    'fold': cv_iter,
                    'train_rmse': train_metrics['rmse_mean'],
                    'train_r2': train_metrics['r2_mean'],
                    'train_correlation': train_metrics['correlation_mean'],
                    'test_rmse': test_metrics['rmse_mean'],
                    'test_r2': test_metrics['r2_mean'],
                    'test_correlation': test_metrics['correlation_mean']
                })

        # Gather all results to rank 0
        all_fold_results = comm.gather(local_fold_results, root=0)
        all_weights = comm.gather(local_weights, root=0)

        # Only rank 0 aggregates
        if rank == 0:
            # Flatten gathered results
            fold_results = []
            all_weights_flat = []
            for rank_results in all_fold_results:
                fold_results.extend(rank_results)
            for rank_weights in all_weights:
                all_weights_flat.extend(rank_weights)

            # Sort by fold number
            fold_results = sorted(fold_results, key=lambda x: x['fold'])

            # Aggregate
            if self.task_type == 'categorical':
                train_acc = [f['train_accuracy'] for f in fold_results]
                test_acc = [f['test_accuracy'] for f in fold_results]

                return {
                    'train_accuracy_mean': float(np.mean(train_acc)),
                    'train_accuracy_std': float(np.std(train_acc)),
                    'test_accuracy_mean': float(np.mean(test_acc)),
                    'test_accuracy_std': float(np.std(test_acc)),
                    'cv_accuracy_per_fold': test_acc,
                    'cv_confusion_matrices': [f['confusion_matrix'] for f in fold_results],
                    'cv_per_class_accuracy': [f['per_class_accuracy'] for f in fold_results],
                    'readout_weights': all_weights_flat
                }
            else:  # temporal
                train_rmse = [f['train_rmse'] for f in fold_results]
                test_rmse = [f['test_rmse'] for f in fold_results]
                train_r2 = [f['train_r2'] for f in fold_results]
                test_r2 = [f['test_r2'] for f in fold_results]
                train_corr = [f['train_correlation'] for f in fold_results]
                test_corr = [f['test_correlation'] for f in fold_results]

                return {
                    'train_rmse_mean': float(np.mean(train_rmse)),
                    'train_rmse_std': float(np.std(train_rmse)),
                    'test_rmse_mean': float(np.mean(test_rmse)),
                    'test_rmse_std': float(np.std(test_rmse)),
                    'train_r2_mean': float(np.mean(train_r2)),
                    'train_r2_std': float(np.std(train_r2)),
                    'test_r2_mean': float(np.mean(test_r2)),
                    'test_r2_std': float(np.std(test_r2)),
                    'train_correlation_mean': float(np.mean(train_corr)),
                    'train_correlation_std': float(np.std(train_corr)),
                    'test_correlation_mean': float(np.mean(test_corr)),
                    'test_correlation_std': float(np.std(test_corr)),
                    'cv_rmse_per_fold': test_rmse,
                    'cv_r2_per_fold': test_r2,
                    'cv_correlation_per_fold': test_corr,
                    'readout_weights': all_weights_flat
                }
        else:
            return {}




    # NEW METHOD: Centralized CV training
    def cross_validate_centralized(self, traces_all: np.ndarray,
                                ground_truth_all: np.ndarray,
                                pattern_ids: np.ndarray,
                                session_id: int,
                                n_folds: int,
                                rank: int,
                                comm) -> Dict[str, Any]:
        """
        Perform stratified K-fold CV with centralized training on rank 0.

        Only rank 0 has the data and does all CV training.
        Other ranks just return empty dict.

        Args:
            traces_all: Full traces (only rank 0 has this, others have None)
            ground_truth_all: Full ground truth (only rank 0 has this, others have None)
            pattern_ids: Pattern IDs for stratification (only rank 0 has this, others have None)
            n_folds: Number of CV folds (fixed at 20)
            rank: MPI rank
            comm: MPI communicator

        Returns:
            Aggregated CV results (only on rank 0, empty dict on other ranks)
        """

        # Only rank 0 does the work
        if rank == 0:
            print(f"   Rank 0: Performing all {n_folds} CV folds...")

            # Get RNG for CV splits
            rng_cv = get_rng(session_id, 0.0, 0.0, 0,
                            f'task_{self.task_type}_cv_splits',
                            hd_dim=self.input_dim_intrinsic,
                            embed_dim=self.input_dim_embedding)

            cv_seed = int(rng_cv.integers(0, 2**31 - 1))

            # Create stratified folds
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
            fold_splits = list(skf.split(traces_all, pattern_ids))

            # Process all CV folds on rank 0
            fold_results = []
            all_weights = []

            for cv_iter in range(n_folds):
                if (cv_iter + 1) % 5 == 0:
                    print(f"      Completed {cv_iter + 1}/{n_folds} folds")

                train_idx, test_idx = fold_splits[cv_iter]

                # Split data
                X_train = traces_all[train_idx]
                X_test = traces_all[test_idx]
                y_train = ground_truth_all[train_idx]
                y_test = ground_truth_all[test_idx]

                # Train readout
                W_readout = train_task_readout(X_train, y_train, self.lambda_reg)
                all_weights.append(W_readout.copy())

                # Predict
                y_pred_train = predict_task_readout(X_train, W_readout)
                y_pred_test = predict_task_readout(X_test, W_readout)

                # Evaluate based on task type
                if self.task_type == 'categorical':
                    decision_window_steps = int(self.decision_window / self.dt)

                    train_metrics = evaluate_categorical_task(
                        y_pred_train, y_train, decision_window_steps
                    )
                    test_metrics = evaluate_categorical_task(
                        y_pred_test, y_test, decision_window_steps
                    )

                    fold_results.append({
                        'fold': cv_iter,
                        'train_accuracy': train_metrics['accuracy'],
                        'test_accuracy': test_metrics['accuracy'],
                        'confusion_matrix': test_metrics['confusion_matrix'],
                        'per_class_accuracy': test_metrics['per_class_accuracy']
                    })

                else:  # temporal or auto_encoding
                    train_metrics = evaluate_temporal_task(y_pred_train, y_train)
                    test_metrics = evaluate_temporal_task(y_pred_test, y_test)

                    fold_results.append({
                        'fold': cv_iter,
                        'train_rmse': train_metrics['rmse_mean'],
                        'train_r2': train_metrics['r2_mean'],
                        'train_correlation': train_metrics['correlation_mean'],
                        'test_rmse': test_metrics['rmse_mean'],
                        'test_r2': test_metrics['r2_mean'],
                        'test_correlation': test_metrics['correlation_mean']
                    })

            # Aggregate results
            if self.task_type == 'categorical':
                train_acc = [f['train_accuracy'] for f in fold_results]
                test_acc = [f['test_accuracy'] for f in fold_results]

                return {
                    'train_accuracy_mean': float(np.mean(train_acc)),
                    'train_accuracy_std': float(np.std(train_acc)),
                    'test_accuracy_mean': float(np.mean(test_acc)),
                    'test_accuracy_std': float(np.std(test_acc)),
                    'cv_accuracy_per_fold': test_acc,
                    'cv_confusion_matrices': [f['confusion_matrix'] for f in fold_results],
                    'cv_per_class_accuracy': [f['per_class_accuracy'] for f in fold_results],
                    'readout_weights': all_weights
                }
            else:  # temporal or auto_encoding
                train_rmse = [f['train_rmse'] for f in fold_results]
                test_rmse = [f['test_rmse'] for f in fold_results]
                train_r2 = [f['train_r2'] for f in fold_results]
                test_r2 = [f['test_r2'] for f in fold_results]
                train_corr = [f['train_correlation'] for f in fold_results]
                test_corr = [f['test_correlation'] for f in fold_results]

                return {
                    'train_rmse_mean': float(np.mean(train_rmse)),
                    'train_rmse_std': float(np.std(train_rmse)),
                    'test_rmse_mean': float(np.mean(test_rmse)),
                    'test_rmse_std': float(np.std(test_rmse)),
                    'train_r2_mean': float(np.mean(train_r2)),
                    'train_r2_std': float(np.std(train_r2)),
                    'test_r2_mean': float(np.mean(test_r2)),
                    'test_r2_std': float(np.std(test_r2)),
                    'train_correlation_mean': float(np.mean(train_corr)),
                    'train_correlation_std': float(np.std(train_corr)),
                    'test_correlation_mean': float(np.mean(test_corr)),
                    'test_correlation_std': float(np.std(test_corr)),
                    'cv_rmse_per_fold': test_rmse,
                    'cv_r2_per_fold': test_r2,
                    'cv_correlation_per_fold': test_corr,
                    'readout_weights': all_weights
                }
        else:
            # Other ranks just wait
            return {}

    def cross_validate(self, traces_all: np.ndarray,
                    ground_truth_all: np.ndarray,
                    pattern_ids: np.ndarray,
                    session_id: int,
                    n_folds: int,
                    rank: int,
                    size: int,
                    comm) -> Dict[str, Any]:
        """
        Perform cross-validation using either distributed or centralized approach.

        Dispatches to the appropriate method based on self.use_distributed_cv flag.
        """
        if self.use_distributed_cv:
            if rank == 0:
                print(f"   Using DISTRIBUTED CV ({n_folds} folds across {size} ranks)")
            return self.cross_validate_distributed(
                traces_all=traces_all,
                ground_truth_all=ground_truth_all,
                pattern_ids=pattern_ids,
                session_id=session_id,
                n_folds=n_folds,
                rank=rank,
                size=size,
                comm=comm
            )
        else:
            if rank == 0:
                print(f"   Using CENTRALIZED CV ({n_folds} folds on rank 0 only)")
            return self.cross_validate_centralized(
                traces_all=traces_all,
                ground_truth_all=ground_truth_all,
                pattern_ids=pattern_ids,
                session_id=session_id,
                n_folds=n_folds,
                rank=rank,
                comm=comm
            )

    def extract_trial_arrays(self, trial_results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract arrays from trial results (required by base class)."""
        return {}
