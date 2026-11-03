#!/bin/bash
# Step 7 (CPU, minutes): every number, table and data figure of the paper from the stored results -> paper/
# Runs on the shipped results without any GPU step. The order matters: tables.py and ext_outputs.py write the
# numeric macros and early table drafts, figures_final.py and tables_final.py then write the final figures and tables.
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p paper/figures
cd code
$PYTHON tables.py --agg ../results/v2/agg --agg_pair ../results/v2/agg_pair1 --agg_mi ../results/v2mi/agg --paper ../paper
$PYTHON ext_outputs.py
$PYTHON figures_final.py
$PYTHON tables_final.py
$PYTHON forensics_paired.py > ../paper/paired_checks.txt
echo "wrote paper/numbers*.tex, paper/tab_*.tex, paper/figures/ and paper/paired_checks.txt"
