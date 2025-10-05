# analysis/encoding_analysis.py - Decoding and encoding capacity analysis
"""
Complete decoding analysis for HD input encoding experiments.
Includes linear decoder, dimensionality analysis, and spike jitter computation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from sklearn.model_selection import KFold
from scipy.linalg import svd
from sklearn.decomposition import PCA


def spikes_to_matrix(spike_list: List[Tuple[float, int]], n_steps: int,
                    n_neurons: int, step_size: float) -> np.ndarray:
    """
    Convert spike data into a spike matrix.

    Args:
        spike_list: List of spikes [(time, neuron_id), ...]
        n_steps: Number of time steps
        n_neurons: Number of neurons
        step_size: Time step size in ms

    Returns:
        Spike matrix of shape (n_steps, n_neurons)
    """
    spike_matrix = np.zeros((n_steps, n_neurons))
    for spike_time, neuron_id in spike_list:
        time_bin = int(round(spike_time / step_size))
        if 0 <= time_bin < n_steps and 0 <= neuron_id < n_neurons:
            spike_matrix[time_bin, neuron_id] += 1
    return spike_matrix


def fft_convolution_with_padding(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform linear convolution using FFT with proper zero-padding.

    Args:
        signal: Input signal
        kernel: Impulse response or kernel

    Returns:
        Linearly convolved signal
    """
    padded_length = len(signal) + len(kernel) - 1
    padded_signal = np.pad(signal, (0, padded_length - len(signal)))
    padded_kernel = np.pad(kernel, (0, padded_length - len(kernel)))

    convolved = np.fft.ifft(np.fft.fft(padded_signal) * np.fft.fft(padded_kernel))

    return np.real(convolved[:len(signal)])


def filter_spikes_exp_kernel(spike_matrices: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Filter spike matrices using an exponential kernel through convolution.

    Args:
        spike_matrices: Array of spike matrices with shape (n_trials, n_steps, n_neurons)
        kernel: 1D array representing the exponential kernel

    Returns:
        Filtered spike matrices with the same shape as input
    """
    n_trials, n_steps, n_neurons = spike_matrices.shape

    # Apply convolution to each neuron in each trial
    filtered_spikes = np.array([
        np.array([fft_convolution_with_padding(spike_matrices[i][:, j], kernel)
                 for j in range(n_neurons)]).T
        for i in range(n_trials)
    ])

    return filtered_spikes


def compute_spike_time_jitter(spikes: List[List[Tuple[float, int]]],
                              n_neurons: int, window: float = 5.0) -> Dict[int, float]:
    """
    Compute spike time jitter for each neuron across trials.

    Args:
        spikes: List of trials, each is a list of (spike_time, neuron_id) tuples
        n_neurons: Total number of neurons
        window: Maximum time difference (ms) to consider spikes as 'the same event' across trials

    Returns:
        Dictionary of neuron_id -> mean jitter (ms)
    """
    # Organize spike times by neuron and trial
    neuron_trial_spikes = {i: [] for i in range(n_neurons)}
    for trial in spikes:
        spikes_by_neuron = defaultdict(list)
        for t, n in trial:
            spikes_by_neuron[n].append(t)
        for n in range(n_neurons):
            neuron_trial_spikes[n].append(sorted(spikes_by_neuron[n]))

    # For each neuron, align spikes across trials and compute jitter
    neuron_jitter = {}
    for n in range(n_neurons):
        # Collect all spike times across all trials
        all_spikes = [np.array(trial) for trial in neuron_trial_spikes[n]]

        # Skip neurons with no spikes
        if all(len(trial) == 0 for trial in all_spikes):
            neuron_jitter[n] = np.nan
            continue

        # Build spike events: align spikes across trials within 'window' ms
        events = []
        used = [set() for _ in all_spikes]

        for trial_idx, trial_spikes in enumerate(all_spikes):
            for spike_idx, spike_time in enumerate(trial_spikes):
                if spike_idx in used[trial_idx]:
                    continue

                # Start a new event
                event = [(trial_idx, spike_time)]
                used[trial_idx].add(spike_idx)

                # Look for matching spikes in other trials
                for other_idx, other_spikes in enumerate(all_spikes):
                    if other_idx == trial_idx:
                        continue

                    # Find the closest unused spike
                    candidates = [(i, abs(st - spike_time))
                                 for i, st in enumerate(other_spikes)
                                 if i not in used[other_idx]]

                    if candidates:
                        min_i, min_dt = min(candidates, key=lambda x: x[1])
                        if min_dt <= window:
                            event.append((other_idx, other_spikes[min_i]))
                            used[other_idx].add(min_i)

                if len(event) > 1:  # Only consider events seen in >1 trial
                    events.append([st for _, st in event])

        # Compute SD for each event, then mean across events
        if events:
            jitters = [np.std(times) for times in events]
            neuron_jitter[n] = np.mean(jitters)
        else:
            neuron_jitter[n] = np.nan

    return neuron_jitter


def analyze_weight_matrix_svd(weights: np.ndarray) -> Dict[str, Any]:
    """
    Analyze weight matrix using SVD to determine effective dimensionality.

    Args:
        weights: Weight matrix of shape (n_neurons, k)

    Returns:
        Dictionary with SVD analysis results
    """
    # Compute SVD
    U, singular_values, Vt = svd(weights, full_matrices=False)

    # Compute eigenvalues (squared singular values)
    eigenvalues = singular_values ** 2

    # Explained variance ratios
    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative_var = np.cumsum(explained_var_ratio)

    # Effective dimensionality (95% threshold)
    effective_dim_95 = np.searchsorted(cumulative_var, 0.95) + 1
    effective_dim_95 = min(effective_dim_95, len(eigenvalues))

    # Participation ratio
    if total_var > 0:
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    else:
        participation_ratio = 0.0

    return {
        'singular_values': singular_values,
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_var_ratio,
        'cumulative_variance': cumulative_var,
        'effective_dim_95': int(effective_dim_95),
        'participation_ratio': float(participation_ratio),
        'total_variance': float(total_var)
    }


def analyze_decoded_output_pca(decoded_output: np.ndarray) -> Dict[str, Any]:
    """
    Analyze decoded output using PCA to determine dimensionality.

    Args:
        decoded_output: Decoded signal for single trial, shape (n_timesteps, k)

    Returns:
        Dictionary with PCA analysis results
    """
    # Transpose so features are rows, samples are columns
    # PCA on (k, n_timesteps) - treating time as samples
    n_timesteps, k = decoded_output.shape

    if k < 2 or n_timesteps < 2:
        return {
            'explained_variance_ratio': np.array([]),
            'cumulative_variance': np.array([]),
            'effective_dim_95': 0,
            'participation_ratio': 0.0
        }

    # Do PCA
    pca = PCA(n_components=min(k, n_timesteps))
    pca.fit(decoded_output)  # Fit on (n_timesteps, k) - sklearn handles transpose

    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    # Effective dimensionality (95% threshold)
    effective_dim_95 = np.searchsorted(cumulative_var, 0.95) + 1
    effective_dim_95 = min(effective_dim_95, len(explained_var_ratio))

    # Participation ratio from eigenvalues
    eigenvalues = pca.explained_variance_
    if np.sum(eigenvalues) > 0:
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    else:
        participation_ratio = 0.0

    return {
        'explained_variance_ratio': explained_var_ratio,
        'cumulative_variance': cumulative_var,
        'effective_dim_95': int(effective_dim_95),
        'participation_ratio': float(participation_ratio)
    }


class LinearDecoder:
    """Linear decoder with ridge regularization for HD input decoding."""

    def __init__(self, dt: float = 0.1, tau: float = 10.0,
                lambda_reg: float = 1e-3, random_state=None):
        """
        Initialize the LinearDecoder.

        Args:
            dt: Time step (ms)
            tau: Time constant for exponential kernel (ms)
            lambda_reg: Regularization strength
            random_state: numpy.random.Generator, int seed, or None
        """
        self.dt = dt
        self.tau = tau
        self.lambda_reg = lambda_reg

        # Convert Generator to integer seed for sklearn compatibility
        if hasattr(random_state, 'integers'):  # It's a np.random.Generator
            self.random_state = int(random_state.integers(0, 2**31 - 1))
        else:
            self.random_state = random_state  # Already int or None

        self.w = None

        # Create exponential kernel
        self.kernel = np.exp(-np.arange(0, 5 * tau, dt) / tau)

    def preprocess_data(self, spikes_trials_all: List[List[Tuple[float, int]]],
                       n_neurons: int, duration: float) -> np.ndarray:
        """
        Preprocess spike data: convert to matrices and apply exponential kernel.

        Args:
            spikes_trials_all: List of spike time tuples for each trial
            n_neurons: Number of neurons
            duration: Duration of the recording in ms

        Returns:
            Filtered spike matrices of shape (n_trials, n_steps, n_neurons)
        """
        n_steps = int(duration / self.dt)
        spike_matrices = np.array([
            spikes_to_matrix(trial_spikes, n_steps, n_neurons, self.dt)
            for trial_spikes in spikes_trials_all
        ])
        return filter_spikes_exp_kernel(spike_matrices, self.kernel)

    def _ensure_2d_signal(self, signal: np.ndarray) -> np.ndarray:
        """Ensure the signal is a 2D array."""
        if signal.ndim == 1:
            return signal.reshape(1, -1)
        elif signal.ndim == 2:
            return signal
        else:
            raise ValueError("Signal must be either 1D or 2D array.")

    def fit(self, filtered_spikes: np.ndarray, signal: np.ndarray):
        """
        Fit the linear decoder to the data.

        Args:
            filtered_spikes: Filtered spike data (n_trials, n_steps, n_neurons)
            signal: Signal to decode (n_signals, n_steps)
        """
        signal = self._ensure_2d_signal(signal)

        if filtered_spikes.ndim != 3:
            raise ValueError(f"Expected filtered_spikes to be 3D, got shape {filtered_spikes.shape}")

        n_trials, n_steps, n_neurons = filtered_spikes.shape
        X = filtered_spikes.reshape(n_trials * n_steps, n_neurons)
        y = np.tile(signal.T, (n_trials, 1))

        # Solve regularized least squares: (X^T X + λI) w = X^T y
        I = np.eye(n_neurons)
        reg_term = X.T @ X + self.lambda_reg * I
        self.w = np.linalg.solve(reg_term, X.T @ y)


    def predict(self, filtered_spikes: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained decoder.

        Args:
            filtered_spikes: Filtered spike data (n_trials, n_steps, n_neurons)

        Returns:
            Predicted signal (n_trials, n_steps, n_signals)
        """
        if filtered_spikes.ndim == 3:
            # Handle 3D input: (n_trials, n_steps, n_neurons)
            n_trials, n_steps, n_neurons = filtered_spikes.shape
            # Reshape to 2D, predict, then reshape back
            X = filtered_spikes.reshape(n_trials * n_steps, n_neurons)
            predictions = X.dot(self.w)  # (n_trials*n_steps, n_signals)
            return predictions.reshape(n_trials, n_steps, -1)  # (n_trials, n_steps, n_signals)
        else:
            # Handle 2D input: (n_steps, n_neurons)
            return filtered_spikes.dot(self.w)

    def compute_rmse(self, prediction: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Compute Root Mean Square Error per trial, then average.

        Args:
            prediction: Predicted signal (n_trials, n_steps, n_signals)
            signal: Actual signal (n_signals, n_steps)

        Returns:
            RMSE for each signal dimension, averaged over trials
        """
        signal_2d = self._ensure_2d_signal(signal)  # (n_signals, n_steps)

        if prediction.ndim == 3:
            n_trials, n_steps, n_signals = prediction.shape
            # Compute RMSE per trial
            rmse_per_trial = np.sqrt(((prediction - signal_2d.T) ** 2).mean(axis=1))  # (n_trials, n_signals)
            return rmse_per_trial.mean(axis=0)  # Average over trials: (n_signals,)
        else:
            # Fallback for 2D
            return np.sqrt(((prediction - signal_2d.T) ** 2).mean(axis=0))


    def compute_r2(self, prediction: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Compute R² score per trial, then average.

        Args:
            prediction: Predicted signal (n_trials, n_steps, n_signals)
            signal: Actual signal (n_signals, n_steps)

        Returns:
            R² for each signal dimension, averaged over trials
        """
        signal_2d = self._ensure_2d_signal(signal)  # (n_signals, n_steps)

        if prediction.ndim == 3:
            n_trials, n_steps, n_signals = prediction.shape
            # Broadcast signal to match prediction shape
            signal_broadcast = np.broadcast_to(signal_2d.T, (n_trials, n_steps, n_signals))

            # Compute SS_res per trial
            ss_res = ((prediction - signal_broadcast) ** 2).sum(axis=1)  # (n_trials, n_signals)

            # Compute SS_tot (same for all trials)
            signal_mean = signal_2d.mean(axis=1, keepdims=True)  # (n_signals, 1)
            ss_tot = ((signal_2d - signal_mean) ** 2).sum(axis=1)  # (n_signals,)

            # R² per trial
            r2_per_trial = 1 - (ss_res / ss_tot)  # (n_trials, n_signals)
            return r2_per_trial.mean(axis=0)  # Average over trials: (n_signals,)
        else:
            # Fallback for 2D
            ss_res = ((prediction - signal_2d.T) ** 2).sum(axis=0)
            ss_tot = ((signal_2d.T - signal_2d.mean(axis=1, keepdims=True)) ** 2).sum(axis=0)
            return 1 - (ss_res / ss_tot)


    def compute_correlation(self, prediction: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation per trial, then average.

        Args:
            prediction: Predicted signal (n_trials, n_steps, n_signals)
            signal: Actual signal (n_signals, n_steps)

        Returns:
            Correlation for each signal dimension, averaged over trials
        """
        signal_2d = self._ensure_2d_signal(signal)  # (n_signals, n_steps)

        if prediction.ndim == 3:
            n_trials, n_steps, n_signals = prediction.shape
            correlations_per_trial = np.zeros((n_trials, n_signals))

            # Compute correlation per trial, per channel
            for trial in range(n_trials):
                for channel in range(n_signals):
                    pred = prediction[trial, :, channel]
                    true = signal_2d[channel, :]

                    if np.std(pred) > 0 and np.std(true) > 0:
                        correlations_per_trial[trial, channel] = np.corrcoef(pred, true)[0, 1]
                    else:
                        correlations_per_trial[trial, channel] = np.nan

            # Average over trials, ignoring NaN
            return np.nanmean(correlations_per_trial, axis=0)  # (n_signals,)
        else:
            # Fallback for 2D
            n_steps, n_signals = prediction.shape
            correlations = np.zeros(n_signals)
            for i in range(n_signals):
                if np.std(prediction[:, i]) > 0 and np.std(signal_2d[i, :]) > 0:
                    correlations[i] = np.corrcoef(prediction[:, i], signal_2d[i, :])[0, 1]
                else:
                    correlations[i] = np.nan
            return correlations

    def stratified_cv(self, filtered_spikes: np.ndarray, signal: np.ndarray,
                     n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray],
                                                  List[np.ndarray], List[np.ndarray],
                                                  List[Dict], List[Dict], List[Dict]]:
        """
        Perform stratified cross-validation with comprehensive analysis.

        Args:
            filtered_spikes: Filtered spike data (n_trials, n_steps, n_neurons)
            signal: Signal to decode (n_signals, n_steps)
            n_splits: Number of splits for cross-validation

        Returns:
            Tuple of:
            - train_errors: RMSE for training data
            - test_errors: RMSE for test data
            - all_weights: Weight matrices from each fold
            - train_r2_all: R² scores for training
            - test_r2_all: R² scores for test
            - train_corr_all: Correlations for training
            - test_corr_all: Correlations for test
            - weight_svd_all: SVD analysis for each fold
            - decoded_pca_all: PCA analysis of decoded outputs per fold
            - spike_jitter_all: Spike jitter for each fold
        """
        signal = self._ensure_2d_signal(signal)

        n_trials = filtered_spikes.shape[0]
        if n_splits == n_trials:  # LOOCV
            kf = KFold(n_splits=n_trials, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Initialize storage
        train_errors = []
        test_errors = []
        all_weights = []
        train_r2_all = []
        test_r2_all = []
        train_corr_all = []
        test_corr_all = []
        weight_svd_all = []
        decoded_pca_all = []
        train_idx_all = []
        test_idx_all = []

        # Perform cross-validation
        for train_idx, test_idx in kf.split(filtered_spikes):
            # Prepare training data
            X_train = filtered_spikes[train_idx]

            # Fit the model
            self.fit(X_train, signal)

            # Store weights
            all_weights.append(self.w.copy())

            # Analyze weight matrix with SVD
            weight_svd = analyze_weight_matrix_svd(self.w)
            weight_svd_all.append(weight_svd)

            # Make predictions
            train_prediction = self.predict(filtered_spikes[train_idx])
            test_prediction = self.predict(filtered_spikes[test_idx])

            # Analyze decoded output PCA for each test trial
            test_pca_results = []
            for test_trial_idx in range(test_prediction.shape[0]):
                trial_decoded = test_prediction[test_trial_idx]  # shape (n_steps, k)
                pca_result = analyze_decoded_output_pca(trial_decoded)
                test_pca_results.append(pca_result)

            # Average PCA metrics across test trials
            avg_pca_result = {
                'effective_dim_95': np.mean([r['effective_dim_95'] for r in test_pca_results]),
                'participation_ratio': np.mean([r['participation_ratio'] for r in test_pca_results]),
                'per_trial_results': test_pca_results
            }
            decoded_pca_all.append(avg_pca_result)

            # Compute errors
            train_errors.append(self.compute_rmse(train_prediction, signal))
            test_errors.append(self.compute_rmse(test_prediction, signal))

            # Compute R²
            train_r2_all.append(self.compute_r2(train_prediction, signal))
            test_r2_all.append(self.compute_r2(test_prediction, signal))

            # Compute correlations
            train_corr_all.append(self.compute_correlation(train_prediction, signal))
            test_corr_all.append(self.compute_correlation(test_prediction, signal))

            train_idx_all.append(train_idx)
            test_idx_all.append(test_idx)

        # Convert to arrays
        train_errors = np.array(train_errors)
        test_errors = np.array(test_errors)
        train_r2_all = np.array(train_r2_all)
        test_r2_all = np.array(test_r2_all)
        train_corr_all = np.array(train_corr_all)
        test_corr_all = np.array(test_corr_all)

        return (train_errors, test_errors, all_weights, train_r2_all, test_r2_all,
                train_corr_all, test_corr_all, weight_svd_all, decoded_pca_all,
                train_idx_all, test_idx_all)


def decode_hd_input(trial_results: List[Dict[str, Any]],
                    hd_input_ground_truth: np.ndarray,
                    n_neurons: int,
                    session_id: int,
                    v_th_std: float,
                    g_std: float,
                    hd_dim: int,
                    embed_dim: int,
                    encoding_duration: float = 300.0,
                    dt: float = 0.1,
                    tau: float = 10.0,
                    lambda_reg: float = 1e-3,
                    n_splits: int = 20) -> Dict[str, Any]:
    """
    Decode HD input from spike times across trials with full analysis.

    Args:
        trial_results: List of trial result dictionaries with 'spike_times' key
        hd_input_ground_truth: Ground truth HD signal, shape (n_timesteps, k)
        n_neurons: Number of neurons
        session_id: Session ID for RNG
        v_th_std: Threshold std for RNG
        g_std: Weight std for RNG
        hd_dim: HD intrinsic dimensionality for RNG
        embed_dim: HD embedding dimensionality for RNG
        encoding_duration: Encoding period duration (ms)
        dt: Time step (ms)
        tau: Decoder kernel time constant (ms)
        lambda_reg: Ridge regularization strength
        n_splits: Number of cross-validation splits (20 for LOOCV)

    Returns:
        Complete decoding analysis results
    """
    from rng_utils import get_rng

    # Get RNG for decoder
    rng_decoder = get_rng(session_id, v_th_std, g_std, 0, 'decoder_cv_splits',
                         hd_dim=hd_dim, embed_dim=embed_dim)

    # Initialize decoder
    decoder = LinearDecoder(dt=dt, tau=tau, lambda_reg=lambda_reg,
                           random_state=rng_decoder)

    # Extract spike times from trials
    spikes_trials_all = [trial['spike_times'] for trial in trial_results]

    # Preprocess spikes
    filtered_spikes = decoder.preprocess_data(spikes_trials_all, n_neurons, encoding_duration)

    # Perform cross-validation with full analysis
    (train_errors, test_errors, all_weights, train_r2, test_r2,
     train_corr, test_corr, weight_svd, decoded_pca,
     train_idx_all, test_idx_all) = decoder.stratified_cv(
        filtered_spikes, hd_input_ground_truth.T, n_splits=n_splits
    )

    # Compute spike jitter for each fold
    spike_jitter_all = []
    for train_idx in train_idx_all:
        train_spikes = [spikes_trials_all[i] for i in train_idx]
        jitter = compute_spike_time_jitter(train_spikes, n_neurons)
        spike_jitter_all.append(jitter)

    # Compile results
    results = {
        # Performance metrics
        'test_rmse_mean': float(np.mean(test_errors)),
        'test_rmse_std': float(np.std(test_errors)),
        'test_rmse_per_fold': test_errors,  # shape (n_folds, k)
        'test_r2_mean': float(np.mean(test_r2)),
        'test_r2_std': float(np.std(test_r2)),
        'test_r2_per_fold': test_r2,  # shape (n_folds, k)
        'test_correlation_mean': float(np.mean(test_corr)),
        'test_correlation_std': float(np.std(test_corr)),
        'test_correlation_per_fold': test_corr,  # shape (n_folds, k)

        # Training metrics (for comparison)
        'train_rmse_mean': float(np.mean(train_errors)),
        'train_r2_mean': float(np.mean(train_r2)),
        'train_correlation_mean': float(np.mean(train_corr)),

        # Weight analysis
        'decoder_weights': all_weights,  # List of (n_neurons, k) arrays
        'weight_svd_analysis': weight_svd,  # List of SVD dicts per fold

        # Decoded output dimensionality
        'decoded_pca_analysis': decoded_pca,  # List of PCA dicts per fold

        # Spike jitter
        'spike_jitter_per_fold': spike_jitter_all,  # List of dicts per fold

        # Cross-validation indices
        'train_indices': train_idx_all,
        'test_indices': test_idx_all,

        # Metadata
        'n_folds': len(train_errors),
        'n_neurons': n_neurons,
        'n_channels': hd_input_ground_truth.shape[1],
        'encoding_duration': encoding_duration,
        'decoder_params': {
            'dt': dt,
            'tau': tau,
            'lambda_reg': lambda_reg
        }
    }

    return results
