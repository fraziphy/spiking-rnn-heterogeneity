# experiments/generate_hd_signals.py
"""
Pre-generate all HD input and output signals.
Creates deterministic HD patterns for all (session, hd_dim, embed_dim, pattern_id) combinations.
"""

import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.hd_input import HDInputGenerator


def generate_all_hd_signals(sessions: list, embed_dims: list,
                            n_patterns: int = 4,
                            signal_cache_dir: str = "results/hd_signals"):
    """Generate HD signals for given sessions and embed_dims."""

    # Generate BOTH hd_input and hd_output signals
    for signal_type in ['hd_input', 'hd_output']:
        print(f"\n{'='*60}")
        print(f"Generating {signal_type} signals")
        print(f"{'='*60}")

        for session in sessions:
            for k in embed_dims:
                for d in range(1, k + 1):  # d from 1 to k
                    print(f"  {signal_type}: Session {session}, d={d}, k={k}")

                    generator = HDInputGenerator(
                        embed_dim=k,
                        dt=0.1,
                        signal_cache_dir=signal_cache_dir,
                        signal_type=signal_type  # ADD THIS
                    )

                    patterns = generator.initialize_and_get_patterns(
                        session_id=session,
                        hd_dim=d,
                        n_patterns=n_patterns
                    )

    print(f"\nComplete! Generated both hd_input and hd_output signals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HD input and output signals")

    parser.add_argument('--sessions', type=int, nargs='+', required=True,
                       help="List of session IDs")
    parser.add_argument('--embed-dims', type=int, nargs='+', required=True,
                       help="List of embedding dimensions")
    parser.add_argument('--n-patterns', type=int, default=4,
                       help="Number of patterns per (session, hd_dim)")
    parser.add_argument('--cache-dir', type=str, default='results/hd_signals',
                       help="Cache directory for signals")

    args = parser.parse_args()

    generate_all_hd_signals(
        sessions=args.sessions,
        embed_dims=args.embed_dims,
        n_patterns=args.n_patterns,
        signal_cache_dir=args.cache_dir
    )
