#!/bin/bash
set -e  # stop on first error

# 1. Run the spontaneous experiment and wait for it to finish
./runners/run_spontaneous_experiment.sh --session_ids "1" --n_v_th 2 --n_g 2 --n_input_rates 2 --nproc 8

# 2. After it completes, run the stability experiment
./runners/run_stability_experiment.sh --session_ids "1" --n_v_th 2 --n_g 2 --n_input_rates 2 --nproc 8
