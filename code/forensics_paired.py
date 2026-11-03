"""Paired cell-level checks for the ordering claims flagged by the integrity forensics (findings BC003-BC010).

Reads stored results only and computes nothing new about the models. For each ordering the paper states, it pairs the two
signals (or noise variants) on the same cell and reports the mean paired difference, how many cells follow the stated
direction, and the direction per robot. A cell is one (robot, seed) combination. Signals of one cell come from the same
ensemble, so those comparisons are paired exactly. Noise variants share the robot, seed index, dataset size and pipeline but
not the noise draw, so those pairs are matched by seed index only. Writes results/forensics_paired.json and prints a summary.
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
SEEDS = [0, 1, 2]


def load():
    runs = list(csv.DictReader(open(ROOT / "results/v2/agg/runs.csv")))
    fixed = {}
    for r in runs:
        if r["model"] == "pe" and int(r["n"]) == 30000 and int(r["pair_rank"]) == 0:
            fixed[(r["env"], float(r["kappa"]), int(r["seed"]))] = r
    replay = {}
    for f in glob.glob(str(ROOT / "results/audit_policy_v1/*.json")):
        d = json.load(open(f))
        replay[(d["env"], float(d["kappa"]), int(d["seed"]))] = d
    noise = {}
    for f in glob.glob(str(ROOT / "results/noise_v1/*/*.json")):
        d = json.load(open(f))
        noise[(f.split("/")[-2], d["env"], d["model"], float(d["kappa"]), int(d["seed"]))] = d
    assert len(fixed) == 3 * 5 * 3, len(fixed)
    assert len(replay) == 27 and len(noise) == 405
    return fixed, replay, noise


def paired(get_a, get_b, higher_is_better, envs=ENVS):
    """Stated direction: a better than b. Returns paired summary over (robot, seed) cells."""
    diffs, per_env = [], {}
    for e in envs:
        de = []
        for s in SEEDS:
            a, b = float(get_a(e, s)), float(get_b(e, s))
            d = a - b if higher_is_better else b - a  # positive = stated direction holds
            de.append(d)
        per_env[e] = {"mean_diff": float(np.mean(de)), "cells_holding": int(sum(x > 0 for x in de)), "cells": len(de)}
        diffs += de
    return {"mean_diff": float(np.mean(diffs)), "cells_holding": int(sum(x > 0 for x in diffs)), "cells": len(diffs),
            "robots_holding": int(sum(v["mean_diff"] > 0 for v in per_env.values())), "per_robot": per_env}


def main():
    fixed, replay, noise = load()
    F = lambda key, k: (lambda e, s: fixed[(e, k, s)][key])
    R = lambda key, k=1.0: (lambda e, s: replay[(e, k, s)][key])
    N = lambda v, m, key, k=1.0: (lambda e, s: noise[(v, e, m, k, s)][key])
    out = {}
    # BC003: one-step replay error rank at kappa=1; stated leader per robot.
    out["BC003_one_step_EPI_over_MOPO_HalfCheetah"] = paired(R("spear/EPI"), R("spear/MOPO"), True, ["HalfCheetah-v5"])
    out["BC003_one_step_MOPO_over_EPI_Hopper_Walker2d"] = paired(R("spear/MOPO"), R("spear/EPI"), True, ["Hopper-v5", "Walker2d-v5"])
    # BC004: five-step rollout rank at kappa=1, EPI ahead of MOPO.
    out["BC004_h5_EPI_over_MOPO"] = paired(R("roll_spear/EPI/h5"), R("roll_spear/MOPO/h5"), True)
    # BC005: contact gating gives the lowest ALE error rank at kappa=1 (lower = harder).
    for m in ["pe", "pe_beta"]:
        for v in ["t3_joint", "bimodal_joint"]:
            out[f"BC005_{m}_contact_below_{v}"] = paired(N(v, m, "spear_err_in/ALE"), N("gauss_contact", m, "spear_err_in/ALE"), True)
    # BC006: without noise ALE ranks error above EPI and retains less error.
    out["BC006_k0_rank_ALE_over_EPI"] = paired(F("spear_err_in/ALE", 0.0), F("spear_err_in/EPI", 0.0), True)
    out["BC006_k0_retain_ALE_below_EPI"] = paired(F("retain80/ALE", 0.0), F("retain80/EPI", 0.0), False)
    # BC007: at kappa=1 ALE retains more error than EPI and kNN.
    for o in ["EPI", "kNN"]:
        out[f"BC007_k1_retain_ALE_above_{o}"] = paired(F(f"retain80/{o}", 1.0), F("retain80/ALE", 1.0), False)
    # BC008: at kappa=1 EPI and MaxDisc have the lowest retained error.
    for d in ["EPI", "MaxDisc"]:
        for o in ["ALE", "MOPO", "TOT", "MI", "kNN"]:
            out[f"BC008_k1_retain_{d}_below_{o}"] = paired(F(f"retain80/{d}", 1.0), F(f"retain80/{o}", 1.0), False)
    # BC009: at kappa=1 disagreement's noise AUROC exceeds the true error's.
    out["BC009_k1_noisyAUROC_EPI_over_trueerr"] = paired(F("auroc_noisy_in/EPI", 1.0), F("auroc_noisy_in/ORACLE_ERR", 1.0), True)
    # BC010: every signal's unvisited AUROC stays below the true error's (kappa 0 and 1).
    for k in [0.0, 1.0]:
        for o in ["ALE", "MOPO", "TOT", "EPI", "MaxDisc", "MI", "kNN"]:
            out[f"BC010_k{k:g}_unvisited_trueerr_over_{o}"] = paired(F("auroc_gap_all/ORACLE_ERR", k), F(f"auroc_gap_all/{o}", k), True)
    json.dump(out, open(ROOT / "results/forensics_paired.json", "w"), indent=1)
    for k, v in out.items():
        pr = " ".join(f"{e[:4]}:{x['cells_holding']}/{x['cells']}" for e, x in v["per_robot"].items())
        print(f"{k:45s} diff {v['mean_diff']:+.3f}  cells {v['cells_holding']}/{v['cells']}  robots {v['robots_holding']}/{len(v['per_robot'])}  [{pr}]")


if __name__ == "__main__":
    main()
