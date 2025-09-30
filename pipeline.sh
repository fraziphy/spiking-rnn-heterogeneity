#!/bin/bash
set -e  # stop on first error

# 1. Run the spontaneous experiment and wait for it to finish
./runners/run_spontaneous_experiment.sh --session_ids "1" --v_th_std_min 0.0 --g_std_min 0.0 --n_v_th 10 --n_g 10 --n_input_rates 10 --nproc 50

# 2. After it completes, run the stability experiment
./runners/run_stability_experiment.sh --session_ids "1" --v_th_std_min 0.0 --g_std_min 0.0 --n_v_th 10 --n_g 10 --n_input_rates 10 --nproc 50
