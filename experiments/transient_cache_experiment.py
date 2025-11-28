# experiments/transient_cache_experiment.py
"""
Generate and cache transient final states WITHOUT spike_times.
Simulates 1000ms transient period and saves network states for all trials.
"""

import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, Any

try:
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng
except ImportError:
    import sys
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    from src.spiking_network import SpikingRNN
    from src.rng_utils import get_rng


class TransientCacheExperiment:
    """
    Generate and cache transient network states.

    Purpose: Pre-compute 1000ms transient period for all parameter combinations.
    This saves computation time in subsequent experiments that can load these states.
    """

    def __init__(self, n_neurons: int = 1000, dt: float = 0.1,
                 transient_duration: float = 1000.0, n_trials: int = 100,
                 cache_dir: str = "results/cached_states"):
        """
        Initialize transient cache experiment.

        Args:
            n_neurons: Number of neurons
            dt: Time step (ms)
            transient_duration: Duration of transient period (ms)
            n_trials: Number of trials to cache per parameter combination
            cache_dir: Directory to store cached states
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.transient_duration = transient_duration
        self.n_trials = n_trials
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_cache_filename(self, session_id: int, g_std: float,
                           v_th_std: float, static_rate: float) -> str:
        """
        Get cache filename for parameter combination.

        Args:
            session_id: Session identifier
            g_std: Weight heterogeneity std
            v_th_std: Threshold heterogeneity std
            static_rate: Static input rate (Hz)

        Returns:
            Full path to cache file
        """
        return os.path.join(self.cache_dir,
            f"session_{session_id}_g_{g_std:.3f}_vth_{v_th_std:.3f}_"
            f"rate_{static_rate:.1f}_trial_states.pkl")

    def run_transient_removal(self, session_id: int, g_std: float, v_th_std: float,
                             static_rate: float, static_input_mode: str = "common_tonic",
                             synaptic_mode: str = "filter",
                             v_th_distribution: str = "normal",
                             force_regenerate: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Generate or load cached transient states.

        Args:
            session_id: Session identifier
            g_std: Weight heterogeneity std
            v_th_std: Threshold heterogeneity std
            static_rate: Static input rate (Hz)
            static_input_mode: Mode for static input
            synaptic_mode: Synaptic dynamics mode
            v_th_distribution: Threshold distribution type
            force_regenerate: If True, regenerate even if cache exists

        Returns:
            Dictionary mapping trial_id to network state
        """
        filename = self._get_cache_filename(session_id, g_std, v_th_std, static_rate)

        if not force_regenerate and os.path.exists(filename):
            print(f"Loading cached: session={session_id}, g={g_std}, "
                  f"vth={v_th_std}, rate={static_rate}")
            with open(filename, 'rb') as f:
                return pickle.load(f)['trial_states']

        print(f"Generating: session={session_id}, g={g_std}, "
              f"vth={v_th_std}, rate={static_rate}")

        # Create network
        network = SpikingRNN(
            n_neurons=self.n_neurons,
            dt=self.dt,
            synaptic_mode=synaptic_mode,
            static_input_mode=static_input_mode,
            n_hd_channels=0  # No HD input for transient
        )

        # Initialize network
        network.initialize_network(
            session_id=session_id,
            v_th_std=v_th_std,
            g_std=g_std,
            v_th_distribution=v_th_distribution,
            static_input_strength=10.0
        )

        trial_states = {}
        n_steps = int(self.transient_duration / self.dt)

        for trial_id in range(self.n_trials):
            # Reset simulation for new trial
            network.reset_simulation(session_id, v_th_std, g_std, trial_id)

            # Simulate transient
            for step in range(n_steps):
                network.step(
                    session_id=session_id,
                    v_th_std=v_th_std,
                    g_std=g_std,
                    trial_id=trial_id,
                    static_input_rate=static_rate,
                    time_step=step
                )

            # Save state WITHOUT spike_times
            trial_states[trial_id] = {
                'current_time': network.current_time,
                'neuron_v_membrane': network.neurons.v_membrane.copy(),  # FIXED
                'neuron_last_spike': network.neurons.last_spike_time.copy(),  # FIXED
                'neuron_refractory': network.neurons.refractory_timer.copy(),  # FIXED
                'recurrent_synaptic_current': network.recurrent_synapses.synaptic_current.copy(),
                'static_synaptic_current': network.static_input_synapses.synaptic_current.copy()
            }

            if (trial_id + 1) % 10 == 0:
                print(f"  Trial {trial_id + 1}/{self.n_trials}")

        # Cache results
        cache_data = {
            'session_id': session_id,
            'g_std': g_std,
            'v_th_std': v_th_std,
            'static_rate': static_rate,
            'n_neurons': self.n_neurons,
            'dt': self.dt,
            'transient_duration': self.transient_duration,
            'n_trials': self.n_trials,
            'trial_states': trial_states,
            'network_params': {
                'static_input_mode': static_input_mode,
                'synaptic_mode': synaptic_mode,
                'v_th_distribution': v_th_distribution
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Cached to {filename}")
        return trial_states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate cached transient states")
    parser.add_argument('--session-start', type=int, default=0,
                       help="Starting session ID")
    parser.add_argument('--session-end', type=int, default=1,
                       help="Ending session ID (exclusive)")
    parser.add_argument('--g-std', type=float, nargs='+', default=[1.0],
                       help="Weight heterogeneity values")
    parser.add_argument('--v-th-std', type=float, nargs='+', default=[0.0],
                       help="Threshold heterogeneity values")
    parser.add_argument('--static-rate', type=float, nargs='+', default=[30.0],
                       help="Static input rates (Hz)")
    parser.add_argument('--n-trials', type=int, default=100,
                       help="Number of trials per combination")
    parser.add_argument('--cache-dir', type=str, default='results/cached_states',
                       help="Cache directory")
    parser.add_argument('--static-input-mode', type=str, default='common_tonic',
                       choices=['independent', 'common_stochastic', 'common_tonic'])
    parser.add_argument('--synaptic-mode', type=str, default='filter',
                       choices=['filter', 'pulse'])

    args = parser.parse_args()

    exp = TransientCacheExperiment(n_trials=args.n_trials, cache_dir=args.cache_dir)

    for session in range(args.session_start, args.session_end):
        for g in args.g_std:
            for vth in args.v_th_std:
                for rate in args.static_rate:
                    exp.run_transient_removal(
                        session, g, vth, rate,
                        static_input_mode=args.static_input_mode,
                        synaptic_mode=args.synaptic_mode
                    )
