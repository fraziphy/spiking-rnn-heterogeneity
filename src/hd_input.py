# src/hd_input.py - Unified HD input generation and management
"""
High-dimensional input generator with caching for encoding experiments.
Combines functionality from hd_input_generator.py and hd_signal_manager.py.
"""

import numpy as np
import os
import pickle
from typing import Tuple, Dict, Optional
from sklearn.decomposition import PCA
from scipy.linalg import qr
from .rng_utils import get_rng


def run_rate_rnn(n_neurons: int, T: float, dt: float, g: float,
                 session_id: int, hd_dim: int, embed_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run rate RNN to generate temporal patterns for HD inputs.

    Args:
        n_neurons: Number of neurons in rate RNN
        T: Total duration (ms)
        dt: Time step (ms)
        g: Coupling strength
        session_id: Session ID for reproducibility
        hd_dim: HD intrinsic dimensionality (affects seed)
        embed_dim: HD embedding dimensionality (affects seed)

    Returns:
        rates: Neural activity after transient removal, shape (n_steps, n_neurons)
        time: Time array
    """
    # Get RNG for this session/hd_dim/embed_dim combination
    rng = get_rng(session_id, 0.0, 0.0, 0, 'rate_rnn_for_hd',
                  hd_dim=hd_dim, embed_dim=embed_dim)

    n_steps = int(T / dt)
    time = np.arange(0, T, dt)

    # Initialize weight matrix
    W = rng.normal(0, 1/np.sqrt(n_neurons), (n_neurons, n_neurons))
    np.fill_diagonal(W, 0)

    # Initialize state
    x = rng.normal(0, 0.1, n_neurons)

    # Simulate
    rates = np.zeros((n_steps, n_neurons))
    for t_idx in range(n_steps):
        phi_x = np.tanh(x)
        dx = -x + g * W @ phi_x
        x += dx * dt / 2
        rates[t_idx] = phi_x

    # Remove transient (first 200ms as per refactoring plan)
    transient_steps = int(200.0 / dt)
    rates_clean = rates[transient_steps:]

    return rates_clean, time[transient_steps:]


def make_embedding(Rates: np.ndarray, k: int, d: int,
                   session_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate k-dimensional embedding with intrinsic dimensionality d.

    Args:
        Rates: RNN activity, shape (T, n_neurons)
        k: Embedding dimensionality (ambient space)
        d: Intrinsic dimensionality (d <= k)
        session_id: Session ID for reproducibility

    Returns:
        Y_embedded: Embedded trajectory, shape (T, k)
        chosen_components: Indices of PCA components chosen
    """
    # Get RNG for embedding (fixed per session/d/k)
    rng = get_rng(session_id, 0.0, 0.0, 0, 'hd_embedding',
                  hd_dim=d, embed_dim=k)

    # Step 1: PCA decomposition
    pca = PCA(n_components=k)
    S = pca.fit_transform(Rates)  # shape (T, k)

    # Step 2: Normalize each component
    S_norm = S / (S.std(axis=0, keepdims=True) + 1e-12)

    # Step 3: Random rotation in k-dimensional space
    A = rng.normal(size=(k, k))
    Q, _ = qr(A)
    Y = S_norm @ Q

    # Step 4: Pick d random columns (intrinsic dimensionality)
    chosen_components = rng.choice(k, size=d, replace=False)
    Y_d = Y[:, chosen_components]  # (T, d)

    # Step 5: Generate fresh random orthogonal basis for embedding
    A = rng.normal(size=(k, k))
    Q, _ = qr(A)
    U = Q[:, :d]  # (k, d) orthonormal embedding directions

    # Step 6: Embed d-dimensional signal into k-dimensional space
    Y_embedded = Y_d @ U.T  # (T, k)

    # Step 7: Normalize across channels
    Y_embedded = Y_embedded / (Y_embedded.std(axis=0, keepdims=True) + 1e-12)

    return Y_embedded, chosen_components


class HDInputGenerator:
    """High-dimensional input generator with signal caching."""

    def __init__(self, embed_dim: int = 10, dt: float = 0.1,
                 signal_cache_dir: Optional[str] = None):
        """
        Initialize HD input generator.

        Args:
            embed_dim: Embedding dimensionality k (number of input channels)
            dt: Time step (ms)
            signal_cache_dir: Directory for signal caching (None = no caching)
        """
        self.embed_dim = embed_dim
        self.dt = dt
        self.signal_cache_dir = signal_cache_dir

        if signal_cache_dir:
            os.makedirs(signal_cache_dir, exist_ok=True)

        # Will be initialized per session/hd_dim/embed_dim
        self.Y_base = None
        self.chosen_components = None
        self.n_timesteps = None

    def _get_signal_filename(self, session_id: int, hd_dim: int) -> str:
        """Generate filename for HD signal cache."""
        if not self.signal_cache_dir:
            return None
        return os.path.join(self.signal_cache_dir,
                          f"hd_signal_session_{session_id}_hd_{hd_dim}_k_{self.embed_dim}.pkl")

    def _signal_exists(self, session_id: int, hd_dim: int) -> bool:
        """Check if HD signal already exists in cache."""
        filename = self._get_signal_filename(session_id, hd_dim)
        return filename and os.path.exists(filename)

    def _save_signal(self, session_id: int, hd_dim: int, rate_rnn_params: dict):
        """Save generated signal to cache."""
        if not self.signal_cache_dir:
            return

        signal_data = {
            'Y_base': self.Y_base,
            'chosen_components': self.chosen_components,
            'n_timesteps': self.n_timesteps,
            'session_id': session_id,
            'hd_dim': hd_dim,
            'embed_dim': self.embed_dim,
            'rate_rnn_params': rate_rnn_params,
            'statistics': self.get_base_statistics()
        }

        filename = self._get_signal_filename(session_id, hd_dim)
        with open(filename, 'wb') as f:
            pickle.dump(signal_data, f)

    def _load_signal(self, session_id: int, hd_dim: int) -> bool:
        """Load signal from cache. Returns True if successful."""
        filename = self._get_signal_filename(session_id, hd_dim)
        if not filename or not os.path.exists(filename):
            return False

        with open(filename, 'rb') as f:
            signal_data = pickle.load(f)

        self.Y_base = signal_data['Y_base']
        self.chosen_components = signal_data['chosen_components']
        self.n_timesteps = signal_data['n_timesteps']

        return True

    def initialize_base_input(self, session_id: int, hd_dim: int,
                             rate_rnn_params: dict = None):
        """
        Generate or load base HD input (fixed per session/hd_dim/embed_dim).

        Args:
            session_id: Session ID
            hd_dim: Intrinsic dimensionality (d)
            rate_rnn_params: Parameters for rate RNN
                - n_neurons: default 1000
                - T: default 500ms (200ms transient + 300ms encoding)
                - g: default 1.2
        """
        # Try to load from cache first
        if self._load_signal(session_id, hd_dim):
            return

        # Generate new signal
        if rate_rnn_params is None:
            rate_rnn_params = {'n_neurons': 1000, 'T': 500.0, 'g': 2.0}

        # Run rate RNN
        rates, _ = run_rate_rnn(
            n_neurons=rate_rnn_params.get('n_neurons', 1000),
            T=rate_rnn_params.get('T', 500.0),
            dt=self.dt,
            g=rate_rnn_params.get('g', 2.0),
            session_id=session_id,
            hd_dim=hd_dim,
            embed_dim=self.embed_dim
        )

        # Generate embedding
        self.Y_base, self.chosen_components = make_embedding(
            Rates=rates,
            k=self.embed_dim,
            d=hd_dim,
            session_id=session_id
        )

        self.n_timesteps = self.Y_base.shape[0]

        # Save to cache
        self._save_signal(session_id, hd_dim, rate_rnn_params)

    def generate_trial_input(self, session_id: int, v_th_std: float, g_std: float,
                            trial_id: int, hd_dim: int,
                            noise_std: float = 0.5,
                            rate_scale: float = 1.0) -> np.ndarray:
        """
        Generate HD input for a single trial with added noise.

        Args:
            session_id: Session ID
            v_th_std: Threshold std (for RNG)
            g_std: Weight std (for RNG)
            trial_id: Trial ID (for trial-specific noise)
            hd_dim: Intrinsic dimensionality
            noise_std: Standard deviation of additive Gaussian noise
            rate_scale: Multiplicative scaling factor for rates

        Returns:
            Y_trial: HD input with noise, shape (n_timesteps, embed_dim)
        """
        if self.Y_base is None:
            raise ValueError("Must call initialize_base_input() first")

        # Get RNG for trial-specific noise
        rng = get_rng(session_id, v_th_std, g_std, trial_id, 'hd_input_noise',
                     hd_dim=hd_dim, embed_dim=self.embed_dim)

        # Add Gaussian noise
        noise = rng.normal(0, noise_std, self.Y_base.shape)
        Y_noisy = self.Y_base + noise

        # Shift to positive values
        Y_positive = Y_noisy - np.min(Y_noisy)

        # Apply rate scaling
        Y_trial = Y_positive * rate_scale

        return Y_trial

    def get_base_statistics(self) -> dict:
        """Get statistics about the base HD input."""
        if self.Y_base is None:
            return {'error': 'Base input not initialized'}

        return {
            'n_timesteps': self.n_timesteps,
            'embed_dim': self.embed_dim,
            'mean': float(np.mean(self.Y_base)),
            'std': float(np.std(self.Y_base)),
            'min': float(np.min(self.Y_base)),
            'max': float(np.max(self.Y_base)),
            'chosen_components': self.chosen_components.tolist() if self.chosen_components is not None else []
        }
