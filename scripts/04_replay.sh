#!/bin/bash
# Step 4 (GPU): replay audit (Section 5). For each robot, noise level kappa in {0, 0.25, 1} and seed in {0, 1, 2}:
#   collect: 300k SAC steps in the noisy environment, saving the replay buffer -> results/audit_data_v1/*.npz
#            (about 35-40 min per dataset on one NVIDIA L40S; existing datasets are skipped)
#   audit:   five-member PE on the replay data, one-step and rollout metrics -> results/audit_policy_v1/*.json
set -uo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p results/audit_data_v1
JOBS=()
for s in 0 1 2; do for k in 0.0 0.25 1.0; do for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
  f=results/audit_data_v1/${e}_k${k}_s${s}.npz
  JOBS+=("[ -f $f ] || $PYTHON code/offline_mbrl.py --phase collect --env $e --kappa $k --seed $s --data data_v2 --dataset $f")
done; done; done
run_pool logs/04_replay_collect || exit 1
JOBS=()
for s in 0 1 2; do for k in 0.0 0.25 1.0; do for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
  JOBS+=("$PYTHON code/audit_policy.py --env $e --kappa $k --seed $s --dataset results/audit_data_v1/${e}_k${k}_s${s}.npz --data data_v2 --out results/audit_policy_v1")
done; done; done
run_pool logs/04_replay_audit
n=$(ls results/audit_policy_v1/*.json 2>/dev/null | wc -l); echo "replay audit: $n/27"; [ "$n" -eq 27 ]
