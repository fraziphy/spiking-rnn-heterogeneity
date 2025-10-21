#!/usr/bin/env python3
"""
Generate linearly spaced values (like numpy.linspace).
Helper for bash scripts.
"""
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: linspace.py <start> <stop> <num> [--int]", file=sys.stderr)
        sys.exit(1)

    start = float(sys.argv[1])
    stop = float(sys.argv[2])
    num = int(sys.argv[3])
    as_int = len(sys.argv) == 5 and sys.argv[4] == '--int'

    if num < 1:
        print("Error: num must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Use numpy's linspace
    values = np.linspace(start, stop, num)

    # Convert to int if requested
    if as_int:
        values = np.round(values).astype(int)

    # Print space-separated values
    print(' '.join(str(v) for v in values))
