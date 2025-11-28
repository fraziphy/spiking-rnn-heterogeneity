# experiments/evoked_spike_to_hd_input_cache_experiment.py
"""
Generate and cache evoked spikes to HD inputs.
Uses cached transient states, then simulates response to HD patterns.
"""

import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from src.spiking_network import SpikingRNN
    from src.hd_input import HDInputGenerator
except ImportError:
    import sys
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    from src.spiking_network import SpikingRNN
    from src.hd_input import HDInputGenerator


class EvokedSpikeCache:
    """
    Generate and cache evoked network responses to HD inputs.

    Process:
    1. Load cached transient state (1000ms pre-simulation)
    2. Restore network state
    3. Simulate 300ms with HD input
    4. Store spike times
    """

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 stimulus_duration: float = 300.0, n_trials: int = 100,
                 transient_cache_dir: str = "results/cached_states",
                 spike_cache_dir: str = "results/cached_spikes",
                 hd_signal_cache_dir: str = "results/hd_signals"):
        """
        Initialize evoked spike cache experiment.

        Args:
            n_neurons: Number of neurons
            dt: Time step (ms)
            stimulus_duration: Duration of HD stimulus (ms)
            n_trials: Number of trials per pattern
            transient_cache_dir: Directory with cached transient states
            spike_cache_dir: Directory to store cached spikes
            hd_signal_cache_dir: Directory with pre-generated HD signals
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.stimulus_duration = stimulus_duration
        self.n_trials = n_trials
        self.transient_cache_dir = transient_cache_dir
        self.spike_cache_dir = spike_cache_dir
        self.hd_signal_cache_dir = hd_signal_cache_dir
        Path(spike_cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_spike_cache_filename(self, session_id: int, g_std: float,
                                  v_th_std: float, static_rate: float,
                                  hd_dim: int, embed_dim: int, pattern_id: int,
                                  hd_connection_mode: str) -> str:
        """
        Get cache filename for evoked spikes.

        Args:
            session_id: Session identifier
            g_std: Weight heterogeneity std
            v_th_std: Threshold heterogeneity std
            static_rate: Static input rate (Hz)
            hd_dim: Intrinsic HD dimension
            embed_dim: HD embedding dimension
            pattern_id: Pattern identifier
            hd_connection_mode: 'overlapping' or 'partitioned'

        Returns:
            Full path to cache file
        """
        mode_dir = os.path.join(self.spike_cache_dir, hd_connection_mode)
        Path(mode_dir).mkdir(parents=True, exist_ok=True)

        return os.path.join(mode_dir,
            f"session_{session_id}_g_{g_std:.3f}_vth_{v_th_std:.3f}_"
            f"rate_{static_rate:.1f}_h_{hd_dim}_d_{embed_dim}_"
            f"pattern_{pattern_id}_spikes.pkl")

    def generate_evoked_spikes(self, session_id: int, g_std: float, v_th_std: float,
                              static_rate: float, hd_dim: int, embed_dim: int,
                              pattern_id: int, hd_connection_mode: str = "overlapping",
                              signal_type: str = "hd_input",
                              static_input_mode: str = "common_tonic",
                              hd_input_mode: str = "common_tonic",
                              synaptic_mode: str = "filter",
                              v_th_distribution: str = "normal",
                              noise_std: float = 0.5, rate_scale: float = 1.0,
                              force_regenerate: bool = False) -> Dict[int, List[Tuple[float, int]]]:
        """
        Generate or load cached evoked spikes for HD pattern.

        Args:
            session_id: Session identifier
            g_std: Weight heterogeneity std
            v_th_std: Threshold heterogeneity std
            static_rate: Static input rate (Hz)
            hd_dim: Intrinsic HD dimension
            embed_dim: HD embedding dimension
            pattern_id: Pattern identifier
            hd_connection_mode: 'overlapping' or 'partitioned'
            signal_type: 'hd_input' or 'hd_output'
            static_input_mode: Mode for static input
            hd_input_mode: Mode for HD input
            synaptic_mode: Synaptic dynamics mode
            v_th_distribution: Threshold distribution type
            noise_std: Standard deviation of trial-to-trial noise
            rate_scale: Scaling factor for HD input rates
            force_regenerate: If True, regenerate even if cache exists

        Returns:
            Dictionary mapping trial_id to list of (time, neuron_id) spike tuples
        """
        filename = self._get_spike_cache_filename(
            session_id, g_std, v_th_std, static_rate,
            hd_dim, embed_dim, pattern_id, hd_connection_mode)

        if not force_regenerate and os.path.exists(filename):
            print(f"Loading cached: pattern={pattern_id}, mode={hd_connection_mode}")
            with open(filename, 'rb') as f:
                return pickle.load(f)['trial_spikes']

        print(f"Generating: session={session_id}, h={hd_dim}, k={embed_dim}, "
              f"pattern={pattern_id}, mode={hd_connection_mode}")

        # Load transient states
        transient_file = os.path.join(self.transient_cache_dir,
            f"session_{session_id}_g_{g_std:.3f}_vth_{v_th_std:.3f}_"
            f"rate_{static_rate:.1f}_trial_states.pkl")

        with open(transient_file, 'rb') as f:
            trial_states = pickle.load(f)['trial_states']

        # Create network
        network = SpikingRNN(
            n_neurons=self.n_neurons,
            dt=self.dt,
            synaptic_mode=synaptic_mode,
            static_input_mode=static_input_mode,
            hd_input_mode=hd_input_mode,
            n_hd_channels=embed_dim,
            hd_connection_mode=hd_connection_mode
        )

        # Initialize network
        network.initialize_network(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            v_th_distribution=v_th_distribution,
            hd_dim=hd_dim,
            embed_dim=embed_dim,
            static_input_strength=10.0,
            hd_connection_prob=0.3,
            hd_input_strength=50.0
        )

        # Load HD signal generator
        hd_generator = HDInputGenerator(
            embed_dim=embed_dim,
            dt=self.dt,
            signal_cache_dir=self.hd_signal_cache_dir,
            signal_type=signal_type
        )
        hd_generator.initialize_base_input(session_id, hd_dim, pattern_id)

        trial_spikes = {}

        for trial_id in range(self.n_trials):
            # Restore cached transient state
            initial_state = trial_states[trial_id]
            network.restore_state(initial_state)
            # Get the transient time from the restored state
            transient_time = initial_state['current_time']

            # Generate trial-specific HD input (with noise)
            hd_input_trial = hd_generator.generate_trial_input(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                hd_dim=hd_dim,
                pattern_id=pattern_id,
                noise_std=noise_std,
                rate_scale=rate_scale,
                static_input_rate=static_rate
            )

            # Simulate with HD input (continue from cached state)
            spikes = network.simulate(
                session_id=session_id,
                v_th_std=v_th_std,
                g_std=g_std,
                trial_id=trial_id,
                duration=self.stimulus_duration,
                static_input_rate=static_rate,
                hd_input_patterns=hd_input_trial,
                hd_dim=hd_dim,
                embed_dim=embed_dim,
                continue_from_state=True  # Use cached state
            )

            # Subtract transient time to make spikes relative to stimulus onset
            stimulus_spikes = [(t - transient_time, nid)
                                for t, nid in spikes
                                if t >= transient_time]

            trial_spikes[trial_id] = stimulus_spikes

            if (trial_id + 1) % 10 == 0:
                print(f"  Trial {trial_id + 1}/{self.n_trials}")

        # Cache results
        cache_data = {
            'session_id': session_id,
            'g_std': g_std,
            'v_th_std': v_th_std,
            'static_rate': static_rate,
            'hd_dim': hd_dim,
            'embed_dim': embed_dim,
            'pattern_id': pattern_id,
            'hd_connection_mode': hd_connection_mode,
            'signal_type': signal_type,
            'trial_spikes': trial_spikes
        }

        with open(filename, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Cached to {filename}")
        return trial_spikes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate cached evoked spikes")

    parser.add_argument('--session-start', type=int, default=0)
    parser.add_argument('--session-end', type=int, default=1)
    parser.add_argument('--g-std', type=float, nargs='+', default=[1.0])
    parser.add_argument('--v-th-std', type=float, nargs='+', default=[0.0])
    parser.add_argument('--static-rate', type=float, nargs='+', default=[30.0])
    parser.add_argument('--hd-dims', type=int, nargs='+', default=[2])
    parser.add_argument('--embed-dims', type=int, nargs='+', default=[5])
    parser.add_argument('--pattern-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--modes', type=str, nargs='+', default=['overlapping'])
    parser.add_argument('--signal-type', type=str, default='hd_input',
                       choices=['hd_input', 'hd_output'])

    args = parser.parse_args()

    exp = EvokedSpikeCache()

    for session in range(args.session_start, args.session_end):
        for g in args.g_std:
            for vth in args.v_th_std:
                for rate in args.static_rate:
                    for hd_dim in args.hd_dims:
                        for embed_dim in args.embed_dims:
                            if hd_dim <= embed_dim:
                                for pattern_id in args.pattern_ids:
                                    for mode in args.modes:
                                        exp.generate_evoked_spikes(
                                            session, g, vth, rate, hd_dim,
                                            embed_dim, pattern_id, mode,
                                            args.signal_type
                                        )
