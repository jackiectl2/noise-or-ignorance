#!/bin/bash
# Step 6 (CPU): aggregate the per-ensemble outputs of step 2 into results/v2/agg, results/v2/agg_pair1 and
# results/v2mi/agg, and copy the per-ensemble JSON files to the layout the table and figure scripts read.
# Needs the NPZ files written by step 2 (not shipped in this package).
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd code
$PYTHON aggregate.py --runs ../results/v2/main --out ../results/v2/agg
$PYTHON aggregate.py --runs ../results/v2/pair1 --out ../results/v2/agg_pair1 --scatter_run none
$PYTHON aggregate.py --runs ../results/v2mi/main --out ../results/v2mi/agg --scatter_run none
cd ..
mkdir -p results/v2/json/main results/v2/json/pair1 results/v2mi/json
cp results/v2/main/*.json results/v2/json/main/
cp results/v2/pair1/*.json results/v2/json/pair1/
cp results/v2mi/main/*.json results/v2mi/json/
