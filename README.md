# Noise or Ignorance?

Code and stored results for the paper *Noise or Ignorance? Auditing the Uncertainty Signals of Ensemble World Models against Known Noise and Error* (CoRL 2026 Workshop on Modeling Uncertainty in Robotic World Models).

The paper audits the uncertainty signals of probabilistic-ensemble world models on MuJoCo robots. Noise of known, state-dependent variance is added to the simulator step and a region of state space is withheld from training. For every test state the true noise variance, the true error of the ensemble mean and the coverage label are therefore exact, and each signal is scored against them.

## Two ways to use this repository

**Regenerate the paper's numbers, tables and data figures from the stored results (CPU, a few minutes).** The `results/` folder holds every per-ensemble result file and aggregate table that the paper uses. After installing the environment, run

```bash
bash scripts/07_paper_outputs.sh
```

It writes the numeric macros (`paper/numbers*.tex`), all tables (`paper/tab_*.tex`) and all data figures (`paper/figures/`) of the paper.

**Rerun the experiments (GPU).** Steps 1 to 6 below rebuild `results/` from scratch. Each script refuses to overwrite finished output, so move the shipped `results/` folder aside first if you want a full rerun.

## Setup

Python 3.11 and an NVIDIA GPU with a CUDA 12.4 capable driver for the training steps. The table and figure step needs no GPU.

```bash
bash scripts/00_setup_env.sh      # creates .venv with the pinned versions
source .venv/bin/activate
```

Versions used for the paper: PyTorch 2.6.0 (CUDA 12.4), Gymnasium 1.1.1, MuJoCo 3.14.0, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, pandas 3.0.6, matplotlib 3.11.2, Pillow 12.3.0. The figures use the serif font Nimbus Roman when it is installed (for example from the URW base35 fonts package) and fall back to Liberation Serif or DejaVu Serif otherwise. The paper's figures were rendered on macOS. With other fonts the text widths change slightly, and the layout check that `figures_final.py` prints may then report label overlaps; the plotted data are unchanged. The numeric macros and tables do not depend on fonts and match the paper exactly.

All scripts run from the repository root. Optional environment variables: `NUM_GPUS` (GPUs to use in parallel, default 1), `PYTHON` (interpreter, default `python`), `PROJECT_ROOT` (default: the repository root).

## Pipeline

| Step | Script | Hardware | Output |
|---|---|---|---|
| 1 | `scripts/01_gen_data.sh` | CPU | `data_v2/<robot>_pool.npz`: 300k noise-free transitions per robot (HalfCheetah, Hopper, Walker2d) with the noise-free state change in float64 |
| 2 | `scripts/02_fixed_pool.sh` | GPU | `results/v2/main` (405 ensembles: 3 robots, 5 noise levels, 3 dataset sizes, 3 objectives, 3 seeds), `results/v2/pair1` (second feature pair), `results/v2mi/main` (Monte Carlo mutual information) |
| 3 | `scripts/03_noise_variants.sh` | GPU | `results/noise_v1/` (contact-gated Gaussian, Student-t3 and bimodal noise) |
| 4 | `scripts/04_replay.sh` | GPU | SAC replay datasets, then `results/audit_policy_v1/` (one-step and rollout audit on replay data) |
| 5 | `scripts/05_return_screen.sh` | GPU | `results/return_screen_v1/` (offline return screen of Appendix H) |
| 6 | `scripts/06_aggregate.sh` | CPU | `results/v2/agg`, `results/v2/agg_pair1`, `results/v2mi/agg` and the JSON layout read by step 7 |
| 7 | `scripts/07_paper_outputs.sh` | CPU | `paper/`: numeric macros, tables and data figures |

Approximate GPU times on one NVIDIA L40S: 2 to 3 minutes per job of step 2 (63 jobs), 3 to 10 minutes per job of step 3 (27 jobs), 35 to 40 minutes per replay dataset plus about 20 minutes per audit in step 4 (27 each), and about 50 minutes per offline run in step 5 (12 runs). Step 2 also writes per-state NPZ files (several GB) that step 6 reads; they are not shipped here.

## Code

| File | Role |
|---|---|
| `code/gen_data.py` | Collects the state-action pools from deterministic MuJoCo dynamics |
| `code/run.py` | Benchmark construction (episode split, noise gate, withheld region, noise scale, evaluation units), models and metrics |
| `code/run_batch.py` | Trains the 15 ensembles of one (robot, dataset size, seed) cell and scores every signal against the simulator ground truth |
| `code/offline_mbrl.py` | SAC data collection in the noisy environment, and the penalized offline model-based SAC of the return screen |
| `code/audit_policy.py` | One-step and rollout audit on SAC replay data |
| `code/aggregate.py` | Per-run, per-robot and overall tables from the per-ensemble outputs |
| `code/tables.py`, `code/ext_outputs.py` | Numeric macros quoted in the text |
| `code/figures_final.py`, `code/figstyle.py`, `code/figures.py` | Data figures (figures.py only provides shared style constants) |
| `code/tables_final.py` | All paper tables |
| `code/forensics_paired.py` | Paired per-cell checks of the ordering statements in the text |

## Paper element to output

| Paper | File written by step 7 |
|---|---|
| Figures 2, 7, 8 (noise sweep at each dataset size) | `paper/figures/fig_dose_n30000.png`, `fig_dose_n10000.png`, `fig_dose_n100000.png` |
| Figure 3 (paired change per ensemble) | `paper/figures/fig_delta.png` |
| Figure 4 (error scale) | `paper/figures/fig_calib.png` |
| Figure 5 (replay data) | `paper/figures/fig_policy.png` |
| Figure 6 (noise shape and gate) | `paper/figures/fig_noise.png` |
| Tables 1 to 11 | `paper/tab_design.tex`, `tab_main.tex`, `tab_scale.tex`, `tab_policy.tex`, `tab_noise.tex`, `tab_env.tex`, `tab_obj.tex`, `tab_pair.tex`, `tab_tthree.tex`, `tab_rows.tex`, `tab_screen.tex` |
| Numbers in the text | `paper/numbers.tex`, `paper/numbers_ext.tex`, `paper/numbers_final.tex` |

Figure 1 is an illustration of the protocol and is not generated from data.

## License

MIT, see `LICENSE`.
