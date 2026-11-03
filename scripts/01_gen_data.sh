#!/bin/bash
# Step 1 (CPU): noise-free state-action pools, 300k transitions per robot -> data_v2/<robot>_pool.npz
set -euo pipefail
source "$(dirname "$0")/common.sh"
for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
  $PYTHON code/gen_data.py --env $e --n 300000 --seed 0 --out data_v2
done
