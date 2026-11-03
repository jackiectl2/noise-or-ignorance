#!/bin/bash
# Step 5 (GPU): the offline return screen of Appendix H, HalfCheetah only.
#   collect: kappa in {0, 1} x seed in {0, 1} -> results/return_data_v1/*.npz (about 35-40 min each on one NVIDIA L40S)
#   offline: arms MOPO, DIS, C3 on each dataset -> results/return_screen_v1/*.json (about 50 min each)
set -uo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p results/return_data_v1
JOBS=()
for k in 0.0 1.0; do for s in 0 1; do
  f=results/return_data_v1/HalfCheetah-v5_k${k}_s${s}.npz
  JOBS+=("[ -f $f ] || $PYTHON code/offline_mbrl.py --phase collect --kappa $k --seed $s --data data_v2 --dataset $f")
done; done
run_pool logs/05_return_collect || exit 1
JOBS=()
for s in 0 1; do for k in 0.0 1.0; do for arm in MOPO DIS C3; do
  JOBS+=("$PYTHON code/offline_mbrl.py --phase offline --kappa $k --seed $s --arm $arm --data data_v2 --dataset results/return_data_v1/HalfCheetah-v5_k${k}_s${s}.npz --out results/return_screen_v1")
done; done; done
run_pool logs/05_return_screen
n=$(ls results/return_screen_v1/*.json 2>/dev/null | wc -l); echo "return screen: $n/12"; [ "$n" -eq 12 ]
