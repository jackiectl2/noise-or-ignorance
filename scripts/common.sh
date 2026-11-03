#!/bin/bash
# Shared helpers. Source this file; do not run it.
# PROJECT_ROOT defaults to the package root (the parent of scripts/). NUM_GPUS sets how many GPUs to use (default 1).
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NUM_GPUS="${NUM_GPUS:-1}"
PYTHON="${PYTHON:-python}"
cd "$PROJECT_ROOT"

# run_pool LOGDIR CMD...: each element of the global array JOBS is one shell command. Worker g runs jobs g, g+N, ...
# on GPU g, one at a time, and writes logs to LOGDIR/<job index>.log. Returns non-zero if any job failed.
run_pool() {
  local logdir=$1; mkdir -p "$logdir"
  local fail_flag="$logdir/.failed"; rm -f "$fail_flag"
  for ((g = 0; g < NUM_GPUS; g++)); do
    (for ((i = g; i < ${#JOBS[@]}; i += NUM_GPUS)); do
       CUDA_VISIBLE_DEVICES=$g bash -c "${JOBS[$i]}" > "$logdir/$i.log" 2>&1 || { echo "FAILED: ${JOBS[$i]}"; touch "$fail_flag"; }
     done) &
  done
  wait
  [ ! -e "$fail_flag" ]
}
