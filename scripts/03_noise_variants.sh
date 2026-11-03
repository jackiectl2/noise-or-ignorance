#!/bin/bash
# Step 3 (GPU): noise shape and gate (Section 6). 3 variants x 3 robots x 3 seeds, n = 30k -> results/noise_v1/<noise>_<gate>
# About 3-10 min per job on one NVIDIA L40S; the contact gate adds one pass over the states.
set -uo pipefail
source "$(dirname "$0")/common.sh"
JOBS=()
for s in 0 1 2; do for v in "gauss contact" "t3 joint" "bimodal joint"; do for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
  set -- $v
  JOBS+=("$PYTHON code/run_batch.py --env $e --n 30000 --seed $s --noise $1 --gate $2 --data data_v2 --out results/noise_v1/$1_$2")
done; done; done
run_pool logs/03_noise_variants
n=$(ls results/noise_v1/*/*.json 2>/dev/null | wc -l); echo "noise variants: $n/405"; [ "$n" -eq 405 ]
