#!/bin/bash
# Create a Python 3.11 virtual environment in .venv with the versions used for the paper.
# Requires python3.11 on PATH and a CUDA 12.4 capable driver for the GPU steps.
set -euo pipefail
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch, gymnasium, mujoco, sklearn, matplotlib; print('torch', torch.__version__, 'gymnasium', gymnasium.__version__, 'mujoco', mujoco.__version__)"
