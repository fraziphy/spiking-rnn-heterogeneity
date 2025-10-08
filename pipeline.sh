#!/bin/bash
set -e  # stop on first error


# 1. Run the encoding experiment and wait for it to finish
./runners/run_encoding_experiment.sh --static_input_mode common_tonic --hd_input_mode common_tonic --n_hd 5 --v_th_std_min 1.0 --v_th_std_max 2.0 --n_v_th 2 --g_std_min 1.0 --g_std_max 2.0 --n_g 2 --input_rate_min 30 --input_rate_max 50 --n_input_rates 5 --nproc 50
# ./runners/run_encoding_experiment.sh --static_input_mode common_tonic --hd_input_mode common_tonic --hd_dim_max 1 --embed_dim 1 --n_hd 1 --n_v_th 10 --n_g 10 --input_rate_min 30 --input_rate_max 50 --n_input_rates 6 --nproc 50
# ./runners/run_spontaneous_experiment.sh --duration 2.0 --static_input_mode common_tonic --n_v_th 10 --n_g 10 --input_rate_min 10 --input_rate_max 50 --n_input_rates 11 --nproc 50
# ./runners/run_stability_experiment.sh --static_input_mode common_tonic --n_v_th 10 --n_g 10 --input_rate_min 30 --input_rate_max 50 --n_input_rates 6 --nproc 50
