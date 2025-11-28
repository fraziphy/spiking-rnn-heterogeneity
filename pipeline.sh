#!/bin/bash
# run_pipeline.sh - Run all steps regardless of exit codes

echo "========================================="
echo "Starting Pipeline: $(date)"
echo "========================================="

echo "Step 1: Generating HD signals..."
# ./sweep/run_sweep_generate_hd_signals.sh > hd_signals.log 2>&1
echo "Step 1 complete: $(date)"

echo "Step 2: Generating transient states..."
# ./sweep/run_sweep_transient_cache.sh > transient_cache.log 2>&1
echo "Step 2 complete: $(date)"

echo "Step 3: Generating evoked spikes..."
./sweep/run_sweep_evoked_spike_cache.sh > evoked_spike_cache.log 2>&1
echo "Step 3 complete: $(date)"

echo "Step 4: Running categorical task..."
./sweep/run_sweep_categorical.sh > categorical.log 2>&1
echo "Step 4 complete: $(date)"

echo "========================================="
echo "Pipeline Complete: $(date)"
echo "========================================="
