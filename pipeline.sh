#!/bin/bash
set -e  # stop on first error

# 1. Run the spontaneous experiment and wait for it to finish
# ./runners/run_spontaneous_experiment.sh --session_ids "3" --v_th_distribution uniform --static_input_mode common_tonic --n_v_th 5 --n_g 5 --input_rate_min 30 --input_rate_max 40 --n_input_rates 2 --nproc 50
#
# # 2. After it completes, run the stability experiment
# ./runners/run_stability_experiment.sh --session_ids "3" --v_th_distribution uniform --static_input_mode common_tonic --n_v_th 5 --n_g 5 --input_rate_min 30 --input_rate_max 40 --n_input_rates 2 --nproc 50

# # 1. Run the spontaneous experiment and wait for it to finish
# ./runners/run_spontaneous_experiment.sh --session_ids "1" --v_th_distribution uniform --synaptic_mode pulse --static_input_mode common_stochastic --n_v_th 10 --n_g 10 --input_rate_min 10 --input_rate_max 30 --n_input_rates 3 --nproc 50
#
# # 2. After it completes, run the stability experiment
# ./runners/run_stability_experiment.sh --session_ids "1" --v_th_distribution uniform --synaptic_mode pulse --static_input_mode common_stochastic --n_v_th 10 --n_g 10 --input_rate_min 10 --input_rate_max 30 --n_input_rates 3 --nproc 50
#
# 1. Run the spontaneous experiment and wait for it to finish
./runners/run_encoding_experiment.sh --session_ids "4" --static_input_mode common_tonic --hd_input_mode common_tonic --v_th_distribution uniform --hd_dim_max 9 --n_hd 3 --n_v_th 3 --n_g 3 --input_rate_min 30 --input_rate_max 50 --n_input_rates 2 --nproc 54
# ./runners/run_spontaneous_experiment.sh --session_ids "4" --duration 1.0 --n_v_th 2 --n_g 2 --input_rate_min 10 --input_rate_max 12 --n_input_rates 2 --nproc 8
# ./runners/run_stability_experiment.sh --session_ids "4" --n_v_th 2 --n_g 2 --input_rate_min 10 --input_rate_max 12 --n_input_rates 2 --nproc 8

# 2. After it completes, run the stability experiment
# ./runners/run_stability_experiment.sh --session_ids "3" --v_th_distribution uniform --n_v_th 5 --n_g 5 --input_rate_min 30 --input_rate_max 40 --n_input_rates 2 --nproc 50
#
