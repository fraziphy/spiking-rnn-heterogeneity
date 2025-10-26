#!/bin/bash
set -e  # stop on first error


# dnp3
./runners/run_autoencoding_task.sh \
    --n_sessions 20 \
    --n_trials_per_pattern 100 --n_input_patterns 1 \
    --hd_dim_input_min 1 --hd_dim_input_max 10 --n_hd_dim_input 10 --embed_dim_input 10 \
    --static_input_mode common_tonic --hd_input_mode common_tonic \
    --n_v_th_std 1 --g_std_min 1.0 --g_std_max 2.0 --n_g_std 1 \
    --static_input_rate_min 30.0 --static_input_rate_max 50.0 --n_static_input_rates 1 \
    --n_processes 10



./runners/run_temporal_task.sh \
    --n_sessions 20 \
    --n_trials_per_pattern 100 --n_input_patterns 1 \
    --hd_dim_input_min 1 --hd_dim_input_max 10 --n_hd_dim_input 10 --embed_dim_input 10 \
    --hd_dim_output_min 1 --hd_dim_output_max 2 --n_hd_dim_output 1 --embed_dim_output 3 \
    --static_input_mode common_tonic --hd_input_mode common_tonic \
    --n_v_th_std 1 --g_std_min 1.0 --g_std_max 2.0 --n_g_std 1 \
    --static_input_rate_min 30.0 --static_input_rate_max 50.0 --n_static_input_rates 1 \
    --n_processes 10


# dnp2
./runners/run_temporal_task.sh \
    --n_sessions 20 \
    --n_trials_per_pattern 100 --n_input_patterns 4 \
    --hd_dim_input_min 1 --hd_dim_input_max 10 --n_hd_dim_input 10 --embed_dim_input 10 \
    --hd_dim_output_min 1 --hd_dim_output_max 2 --n_hd_dim_output 1 --embed_dim_output 3 \
    --static_input_mode common_tonic --hd_input_mode common_tonic \
    --n_v_th_std 1 --g_std_min 1.0 --g_std_max 2.0 --n_g_std 1 \
    --static_input_rate_min 30.0 --static_input_rate_max 50.0 --n_static_input_rates 1 \
    --n_processes 10


# dnp4
./runners/run_categorical_task.sh \
    --n_sessions 20 \
    --n_trials_per_pattern 100 --n_input_patterns 4 \
    --hd_dim_input_min 1 --hd_dim_input_max 10 --n_hd_dim_input 10 --embed_dim_input 10 \
    --static_input_mode common_tonic --hd_input_mode common_tonic \
    --n_v_th_std 1 --g_std_min 1.0 --g_std_max 2.0 --n_g_std 1 \
    --static_input_rate_min 30.0 --static_input_rate_max 50.0 --n_static_input_rates 1 \
    --n_processes 10

