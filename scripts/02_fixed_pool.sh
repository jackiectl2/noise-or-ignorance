#!/bin/bash
# Step 2 (GPU): the fixed-pool study.
#   main:  3 robots x n {100k, 30k, 10k} x 3 seeds, main feature pair     -> results/v2/main   (405 ensembles)
#   pair1: 3 robots x n 30k x 3 seeds, second-least-correlated pair       -> results/v2/pair1  (135 ensembles)
#   mi:    n 30k cells retrained with 128-sample Monte Carlo mutual information -> results/v2mi/main (135 ensembles)
# Each job trains the 15 ensembles of one (robot, n, seed) cell; about 2-3 min per job on one NVIDIA L40S.
# Refuses to overwrite existing output directories.
set -uo pipefail
source "$(dirname "$0")/common.sh"
for d in results/v2/main results/v2/pair1 results/v2mi/main; do
  [ -e "$d" ] && { echo "refusing to reuse $d"; exit 1; }
done
JOBS=()
for s in 0 1 2; do for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5; do for n in 100000 30000 10000; do
  JOBS+=("$PYTHON code/run_batch.py --env $e --n $n --seed $s --pair_rank 0 --data data_v2 --out results/v2/main"); done; done; done
for s in 0 1 2; do for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
  JOBS+=("$PYTHON code/run_batch.py --env $e --n 30000 --seed $s --pair_rank 1 --data data_v2 --out results/v2/pair1")
  JOBS+=("$PYTHON code/run_batch.py --env $e --n 30000 --seed $s --mi_samples 128 --data data_v2 --out results/v2mi/main")
done; done
run_pool logs/02_fixed_pool
nm=$(ls results/v2/main/*.json 2>/dev/null | wc -l); np1=$(ls results/v2/pair1/*.json 2>/dev/null | wc -l)
nmi=$(ls results/v2mi/main/*.json 2>/dev/null | wc -l)
echo "fixed pool: main $nm/405, pair1 $np1/135, mi $nmi/135"
[ "$nm" -eq 405 ] && [ "$np1" -eq 135 ] && [ "$nmi" -eq 135 ]
