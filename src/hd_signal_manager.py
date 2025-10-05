# src/hd_signal_manager.py - Manage HD signal generation and caching
"""
Utility for generating, saving, and loading HD base signals.
Avoids redundant storage by caching signals per (session_id, hd_dim, embed_dim).
"""

import numpy as np
import os
import pickle
from typing import Dict, Tuple
from hd_input_generator import HDInputGenerator


class HDSignalManager:
    """Manager for HD signal generation and caching."""

    def __init__(self, signal_dir: str = "hd_signals"):
        """
        Initialize HD signal manager.

        Args:
            signal_dir: Directory to store cached HD signals
        """
        self.signal_dir = signal_dir
        os.makedirs(signal_dir, exist_ok=True)

    def _get_signal_filename(self, session_id: int, hd_dim: int, embed_dim: int) -> str:
        """Generate filename for HD signal cache."""
        return os.path.join(self.signal_dir,
                          f"hd_signal_session_{session_id}_hd_{hd_dim}_k_{embed_dim}.pkl")

    def signal_exists(self, session_id: int, hd_dim: int, embed_dim: int) -> bool:
        """Check if HD signal already exists in cache."""
        filename = self._get_signal_filename(session_id, hd_dim, embed_dim)
        return os.path.exists(filename)

    def generate_and_save_signal(self, session_id: int, hd_dim: int, embed_dim: int,
                                 rate_rnn_params: dict = None) -> Dict[str, any]:
        """
        Generate HD signal and save to cache.

        Args:
            session_id: Session ID
            hd_dim: HD intrinsic dimensionality
            embed_dim: HD embedding dimensionality
            rate_rnn_params: Parameters for rate RNN generation

        Returns:
            Dictionary with signal data and metadata
        """
        # Initialize generator
        generator = HDInputGenerator(embed_dim=embed_dim, dt=0.1)

        # Generate base input
        generator.initialize_base_input(
            session_id=session_id,
            hd_dim=hd_dim,
            rate_rnn_params=rate_rnn_params
        )

        # Package data
        signal_data = {
            'Y_base': generator.Y_base,
            'chosen_components': generator.chosen_components,
            'n_timesteps': generator.n_timesteps,
            'session_id': session_id,
            'hd_dim': hd_dim,
            'embed_dim': embed_dim,
            'rate_rnn_params': rate_rnn_params or {'n_neurons': 1000, 'T': 350.0, 'g': 1.2},
            'statistics': generator.get_base_statistics()
        }

        # Save to file
        filename = self._get_signal_filename(session_id, hd_dim, embed_dim)
        with open(filename, 'wb') as f:
            pickle.dump(signal_data, f)

        print(f"Saved HD signal: session={session_id}, hd={hd_dim}, k={embed_dim}")
        print(f"  File: {filename}")

        return signal_data

    def load_signal(self, session_id: int, hd_dim: int, embed_dim: int) -> Dict[str, any]:
        """
        Load HD signal from cache.

        Args:
            session_id: Session ID
            hd_dim: HD intrinsic dimensionality
            embed_dim: HD embedding dimensionality

        Returns:
            Dictionary with signal data and metadata
        """
        filename = self._get_signal_filename(session_id, hd_dim, embed_dim)

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"HD signal not found: session={session_id}, hd={hd_dim}, k={embed_dim}\n"
                f"File: {filename}\n"
                f"Generate it first using generate_and_save_signal()"
            )

        with open(filename, 'rb') as f:
            signal_data = pickle.load(f)

        return signal_data

    def get_or_generate_signal(self, session_id: int, hd_dim: int, embed_dim: int,
                               rate_rnn_params: dict = None) -> Dict[str, any]:
        """
        Load signal if it exists, otherwise generate and save it.

        Args:
            session_id: Session ID
            hd_dim: HD intrinsic dimensionality
            embed_dim: HD embedding dimensionality
            rate_rnn_params: Parameters for rate RNN generation (only used if generating)

        Returns:
            Dictionary with signal data and metadata
        """
        if self.signal_exists(session_id, hd_dim, embed_dim):
            return self.load_signal(session_id, hd_dim, embed_dim)
        else:
            return self.generate_and_save_signal(session_id, hd_dim, embed_dim, rate_rnn_params)

    def list_cached_signals(self) -> list:
        """List all cached HD signals."""
        signals = []

        if not os.path.exists(self.signal_dir):
            return signals

        for filename in os.listdir(self.signal_dir):
            if filename.endswith('.pkl') and filename.startswith('hd_signal_'):
                # Parse filename
                parts = filename.replace('.pkl', '').split('_')
                try:
                    session_idx = parts.index('session') + 1
                    hd_idx = parts.index('hd') + 1
                    k_idx = parts.index('k') + 1

                    session_id = int(parts[session_idx])
                    hd_dim = int(parts[hd_idx])
                    embed_dim = int(parts[k_idx])

                    signals.append({
                        'session_id': session_id,
                        'hd_dim': hd_dim,
                        'embed_dim': embed_dim,
                        'filename': filename
                    })
                except (ValueError, IndexError):
                    continue

        return sorted(signals, key=lambda x: (x['session_id'], x['hd_dim'], x['embed_dim']))

    def clear_cache(self, session_id: int = None, hd_dim: int = None, embed_dim: int = None):
        """
        Clear cached signals with optional filtering.

        Args:
            session_id: If specified, only clear this session
            hd_dim: If specified, only clear this hd_dim
            embed_dim: If specified, only clear this embed_dim
        """
        signals = self.list_cached_signals()

        for signal in signals:
            should_delete = True
            if session_id is not None and signal['session_id'] != session_id:
                should_delete = False
            if hd_dim is not None and signal['hd_dim'] != hd_dim:
                should_delete = False
            if embed_dim is not None and signal['embed_dim'] != embed_dim:
                should_delete = False

            if should_delete:
                filepath = os.path.join(self.signal_dir, signal['filename'])
                os.remove(filepath)
                print(f"Deleted: {signal['filename']}")


def generate_all_signals_for_experiment(session_ids: list, hd_dims: list, embed_dims: list,
                                       signal_dir: str = "hd_signals",
                                       rate_rnn_params: dict = None) -> None:
    """
    Pre-generate all HD signals needed for an experiment.

    Args:
        session_ids: List of session IDs
        hd_dims: List of HD intrinsic dimensionalities
        embed_dims: List of HD embedding dimensionalities
        signal_dir: Directory for signal cache
        rate_rnn_params: Parameters for rate RNN generation
    """
    manager = HDSignalManager(signal_dir)

    total_signals = len(session_ids) * len(hd_dims) * len(embed_dims)
    count = 0

    print(f"Generating {total_signals} HD signals...")
    print(f"  Sessions: {session_ids}")
    print(f"  HD dims: {hd_dims}")
    print(f"  Embed dims: {embed_dims}")
    print()

    for session_id in session_ids:
        for hd_dim in hd_dims:
            for embed_dim in embed_dims:
                count += 1
                print(f"[{count}/{total_signals}] Generating signal...")

                if manager.signal_exists(session_id, hd_dim, embed_dim):
                    print(f"  Already exists: session={session_id}, hd={hd_dim}, k={embed_dim}")
                else:
                    manager.generate_and_save_signal(session_id, hd_dim, embed_dim, rate_rnn_params)
                print()

    print(f"Signal generation complete!")
    print(f"Cached signals: {len(manager.list_cached_signals())}")
