# src/hd_input.py - Updated with signal_type and projection-based embedding
"""
High-dimensional input generator with caching and projection-based embedding.
Supports separate HD input and HD output generation via signal_type parameter.
"""

import numpy as np
import os
import pickle
from typing import Tuple, Dict, Optional, Literal
from sklearn.decomposition import PCA
from scipy.linalg import qr
from .rng_utils import get_rng


def run_rate_rnn(n_neurons: int, T: float, dt: float, g: float,
                 session_id: int, signal_type: Literal['hd_input', 'hd_output']) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run rate RNN to generate temporal patterns for HD signals.

    Args:
        n_neurons: Number of neurons in rate RNN
        T: Total duration (ms)
        dt: Time step (ms)
        g: Coupling strength
        session_id: Session ID for reproducibility
        signal_type: 'hd_input' or 'hd_output' (ensures different signals)

    Returns:
        rates: Neural activity after transient removal, shape (n_steps, n_neurons)
        time: Time array
    """
    # Get RNG for this session/signal_type combination
    rng = get_rng(session_id, 0.0, 0.0, 0, f'rate_rnn_{signal_type}')

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

    # Remove transient (first 200ms)
    transient_steps = int(200.0 / dt)
    rates_clean = rates[transient_steps:]

    return rates_clean, time[transient_steps:]


def make_embedding_projected(Rates: np.ndarray, k: int, d: int,
                              session_id: int, pattern_id: int,
                              signal_type: Literal['hd_input', 'hd_output'],
                              kk: int = 7) -> np.ndarray:
    """
    PROJECT kk -> d -> k (not random selection!)

    Flow: Rates (T, n_neurons) -> PCA -> (T, kk) -> project -> (T, d) -> embed -> (T, k)

    Args:
        Rates: Neural rates (T, n_neurons)
        k: Embedding dimensionality
        d: Intrinsic dimensionality
        session_id: Random seed component
        pattern_id: Pattern identifier
        signal_type: 'hd_input' or 'hd_output'
        kk: Intermediate PCA dimensions

    Returns:
        Y_embedded: Embedded signal (T, k)
    """
    # Get 3 RNGs for different stages
    rng_rot = get_rng(session_id, 0.0, 0.0, 0,
                      f'{signal_type}_rotation_{pattern_id}_{k}_{d}')
    rng_proj = get_rng(session_id, 0.0, 0.0, 0,
                       f'{signal_type}_projection_{pattern_id}_{k}_{d}')
    rng_embed = get_rng(session_id, 0.0, 0.0, 0,
                        f'{signal_type}_embedding_{pattern_id}_{k}_{d}')

    # Step 1: PCA decomposition to kk dimensions
    pca = PCA(n_components=kk)
    S = pca.fit_transform(Rates)  # shape (T, kk)

    # Step 2: Normalize each component
    S_norm = S / (S.std(axis=0, keepdims=True) + 1e-12)

    # Step 3: Random rotation in kk-dimensional space
    A = rng_rot.normal(size=(kk, kk))
    Q, _ = qr(A)
    Y = S_norm @ Q

    # Step 4: Project to d-dimensional subspace
    A = rng_proj.normal(size=(kk, kk))
    Q, _ = qr(A)
    U_proj = Q[:, :d]  # (kk, d) orthonormal projection directions
    Y_d = Y @ U_proj  # (T, d)

    # Step 5: Embed d-dimensional signal into k-dimensional space
    A = rng_embed.normal(size=(k, k))
    Q, _ = qr(A)
    U_embed = Q[:, :d]  # (k, d) orthonormal embedding directions
    Y_embedded = Y_d @ U_embed.T  # (T, k)

    # Step 6: Normalize across channels
    Y_embedded = Y_embedded / (Y_embedded.std(axis=0, keepdims=True) + 1e-12)

    return Y_embedded


class HDInputGenerator:
    """High-dimensional input generator with signal caching and pattern support."""

    def __init__(self, embed_dim: int = 10, dt: float = 0.1,
                 signal_cache_dir: Optional[str] = None,
                 signal_type: Literal['hd_input', 'hd_output'] = 'hd_input'):
        """
        Initialize HD input generator.

        Args:
            embed_dim: Embedding dimensionality k (number of input channels)
            dt: Time step (ms)
            signal_cache_dir: Directory for signal caching (None = no caching)
            signal_type: 'hd_input' or 'hd_output'
        """
        self.embed_dim = embed_dim
        self.dt = dt
        self.signal_cache_dir = signal_cache_dir
        self.signal_type = signal_type

        if signal_cache_dir:
            os.makedirs(signal_cache_dir, exist_ok=True)

        # Will be initialized per session/hd_dim/embed_dim/pattern_id/signal_type
        self.Y_base = None
        self.n_timesteps = None

    def _get_signal_filename(self, session_id: int, hd_dim: int, pattern_id: int) -> str:
        """Generate filename for HD signal cache."""
        if not self.signal_cache_dir:
            return None
        return os.path.join(self.signal_cache_dir,
                          f"hd_{self.signal_type}_session_{session_id}_hd_{hd_dim}_k_{self.embed_dim}_pattern_{pattern_id}.pkl")

    def _signal_exists(self, session_id: int, hd_dim: int, pattern_id: int) -> bool:
        """Check if HD signal already exists in cache."""
        filename = self._get_signal_filename(session_id, hd_dim, pattern_id)
        return filename and os.path.exists(filename)

    def _save_signal(self, session_id: int, hd_dim: int, pattern_id: int, rate_rnn_params: dict):
        """Save generated signal to cache."""
        if not self.signal_cache_dir:
            return

        signal_data = {
            'Y_base': self.Y_base,
            'n_timesteps': self.n_timesteps,
            'session_id': session_id,
            'hd_dim': hd_dim,
            'embed_dim': self.embed_dim,
            'pattern_id': pattern_id,
            'signal_type': self.signal_type,
            'rate_rnn_params': rate_rnn_params,
            'statistics': self.get_base_statistics()
        }

        filename = self._get_signal_filename(session_id, hd_dim, pattern_id)
        with open(filename, 'wb') as f:
            pickle.dump(signal_data, f)

    def _load_signal(self, session_id: int, hd_dim: int, pattern_id: int) -> bool:
        """Load signal from cache. Returns True if successful."""
        filename = self._get_signal_filename(session_id, hd_dim, pattern_id)
        if not filename or not os.path.exists(filename):
            return False

        with open(filename, 'rb') as f:
            signal_data = pickle.load(f)

        self.Y_base = signal_data['Y_base']
        self.n_timesteps = signal_data['n_timesteps']

        return True

    def initialize_base_input(self, session_id: int, hd_dim: int,
                             pattern_id: int = 0,
                             rate_rnn_params: dict = None):
        """
        Generate or load base HD input (fixed per session/hd_dim/embed_dim/pattern_id/signal_type).

        Args:
            session_id: Session ID
            hd_dim: Intrinsic dimensionality (d)
            pattern_id: Pattern identifier (default 0)
            rate_rnn_params: Parameters for rate RNN
                - n_neurons: default 1000
                - T: default 500ms (200ms transient + 300ms encoding)
                - g: default 2.0
        """
        # Try to load from cache first
        if self._load_signal(session_id, hd_dim, pattern_id):
            return

        # Generate new signal
        if rate_rnn_params is None:
            rate_rnn_params = {'n_neurons': 1000, 'T': 500.0, 'g': 2.0}

        # Run rate RNN with signal_type (ONCE per signal_type)
        rates, _ = run_rate_rnn(
            n_neurons=rate_rnn_params.get('n_neurons', 1000),
            T=rate_rnn_params.get('T', 500.0),
            dt=self.dt,
            g=rate_rnn_params.get('g', 2.0),
            session_id=session_id,
            signal_type=self.signal_type
        )

        # Generate embedding with projection method
        self.Y_base = make_embedding_projected(
            Rates=rates,
            k=self.embed_dim,
            d=hd_dim,
            session_id=session_id,
            pattern_id=pattern_id,
            signal_type=self.signal_type
        )

        self.n_timesteps = self.Y_base.shape[0]

        # Save to cache
        self._save_signal(session_id, hd_dim, pattern_id, rate_rnn_params)

    def generate_trial_input(self, session_id: int, v_th_std: float, g_std: float,
                            trial_id: int, hd_dim: int,
                            pattern_id: int = 0,
                            noise_std: float = 0.5,
                            rate_scale: float = 1.0,
                            static_input_rate: float = 0.0) -> np.ndarray:
        """
        Generate HD input for a single trial with added noise.

        Noise depends on: session_id, v_th_std, g_std, trial_id, pattern_id,
                        hd_dim, embed_dim, static_input_rate, AND signal_type.
        """
        if self.Y_base is None:
            raise ValueError("Must call initialize_base_input() first")

        # Get RNG for trial-specific noise (includes ALL parameters)
        rng = get_rng(session_id, v_th_std, g_std, trial_id,
                    f'{self.signal_type}_noise_{pattern_id}',
                    rate=static_input_rate,
                    hd_dim=hd_dim,
                    embed_dim=self.embed_dim)

        # Add Gaussian noise (independent per channel/neuron)
        noise = rng.normal(0, noise_std, self.Y_base.shape)
        Y_noisy = self.Y_base + noise

        # Shift to positive values (UNIFORM baseline shift)
        Y_positive = Y_noisy - np.min(Y_noisy)

        # Apply rate scaling
        Y_trial = Y_positive * rate_scale

        return Y_trial

    def initialize_and_get_patterns(self, session_id: int, hd_dim: int,
                                    n_patterns: int = 4,
                                    rate_rnn_params: dict = None) -> Dict[int, np.ndarray]:
        """
        Initialize and return multiple patterns at once.

        Args:
            session_id: Session ID
            hd_dim: Intrinsic dimensionality
            n_patterns: Number of patterns to generate (default 4)
            rate_rnn_params: Parameters for rate RNN (optional)

        Returns:
            Dictionary mapping pattern_id to pattern arrays
        """
        if rate_rnn_params is None:
            rate_rnn_params = {
                'n_neurons': 1000,
                'T': 500.0,  # 200ms transient + 300ms stimulus
                'g': 2.0
            }

        patterns = {}

        for pattern_id in range(n_patterns):
            self.initialize_base_input(
                session_id=session_id,
                hd_dim=hd_dim,
                pattern_id=pattern_id,
                rate_rnn_params=rate_rnn_params
            )
            patterns[pattern_id] = self.Y_base.copy()

        return patterns

    def get_base_statistics(self) -> dict:
        """Get statistics about the base HD input."""
        if self.Y_base is None:
            return {'error': 'Base input not initialized'}

        return {
            'n_timesteps': self.n_timesteps,
            'embed_dim': self.embed_dim,
            'signal_type': self.signal_type,
            'mean': float(np.mean(self.Y_base)),
            'std': float(np.std(self.Y_base)),
            'min': float(np.min(self.Y_base)),
            'max': float(np.max(self.Y_base))
        }
