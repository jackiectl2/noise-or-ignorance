"""Analysis step: per-run JSON/NPZ -> results CSVs that the figures, tables and prose all read.

Writes to <out>/:
  runs.csv          one row per trained ensemble (all metrics, incl. ones derived from the npz files)
  by_env.csv        mean/std over seeds per (env, n, model, kappa)
  overall.csv       mean over envs of the per-env seed means, with the std over the 3 env means
  scatter.csv       a fixed 3000-point subsample of per-state values for the calibration figure
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SIGNALS = ["ALE", "MOPO", "TOT", "EPI", "MaxDisc", "MI", "MIMC", "kNN"]
COVERAGE = 0.8  # selective use: keep the 80% of test states a signal trusts most


def auroc(pos, s):
    ok = np.isfinite(s)
    if ok.sum() == 0 or pos[ok].all() or (~pos[ok]).all():
        return np.nan
    return roc_auc_score(pos[ok], s[ok])


def retained_error(err, s, cov=COVERAGE):
    """Mean true error of the cov-fraction of states with the lowest signal, relative to the mean over all
    states (1 = no better than random selection; lower is better). Ties are broken at random."""
    order = np.lexsort((np.random.default_rng(0).random(len(s)), s))
    keep = order[:int(cov * len(s))]
    return float(err[keep].mean() / err.mean())


def npz_metrics(path):
    z = np.load(path)
    err, ale_true, gap, noisy = z["true_err"], z["true_ale"], z["gap"], z["noisy"]
    ins = ~gap
    top = err >= np.quantile(err, 0.9)  # top-decile true error among all test states
    m = {}
    sigs = {s: z[f"sig_{s}"].astype(np.float64) for s in SIGNALS if f"sig_{s}" in z}
    # Oracle references: the ground-truth quantities scored as if they were signals.
    sigs["ORACLE_ERR"], sigs["ORACLE_NOISE"] = err, ale_true
    for s, v in sigs.items():
        m[f"auroc_toperr_all/{s}"] = auroc(top, v)
        m[f"retain80/{s}"] = retained_error(err, v)
        if s.startswith("ORACLE"):
            m[f"auroc_noisy_in/{s}"] = auroc(noisy[ins], v[ins])
            m[f"auroc_gap_all/{s}"] = auroc(gap, v)
    epi = sigs["EPI"]
    if "ALE" in sigs:
        ale = sigs["ALE"]
        # A member's variance head absorbs its own residual: E(y - mu_m)^2 = sigma^2 + (mu_m - f)^2, and
        # the mean over members of (mu_m - f)^2 equals e + EPI.  Hence the primary target is sigma^2 + e + EPI.
        m["ratio_ale_over_member_residual_in"] = float(np.median(ale[ins] / (ale_true[ins] + err[ins] + epi[ins])))
        nz = ins & noisy & (ale_true > 0)  # true noise is exactly 0 when kappa = 0
        m["ratio_ale_over_noise_in_noisy"] = float(np.median(ale[nz] / ale_true[nz])) if nz.any() else np.nan
    m["ratio_epi_over_err_in"] = float(np.median(epi[ins] / err[ins]))
    m["ratio_epi_over_err_top"] = float(np.median(epi[top] / err[top]))
    # Calibration of disagreement against the error, on subsets that do not condition on the error itself:
    # the design split (visited / unvisited) and the forecast's own top decile.
    top_epi = epi >= np.quantile(epi, 0.9)
    m["cal_err_over_epi_in"] = float(err[ins].mean() / epi[ins].mean())
    m["cal_err_over_epi_gap"] = float(err[gap].mean() / epi[gap].mean())
    m["cal_err_over_epi_topepi"] = float(err[top_epi].mean() / epi[top_epi].mean())
    m["ratio_epi_over_err_gap"] = float(np.median(epi[gap] / err[gap]))
    m["frac_err_gap_over_in"] = float(err[gap].mean() / err[ins].mean())
    m["frac_err_noisy_over_clean_in"] = float(err[ins & noisy].mean() / err[ins & ~noisy].mean())
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/v2/main")
    p.add_argument("--out", default="results/v2/agg")
    p.add_argument("--scatter_run", default="HalfCheetah-v5_p0_k{k}_n30000_pe_s0")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        r = json.load(open(f))
        r.update(npz_metrics(f[:-5] + ".npz"))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "runs.csv"), index=False)

    keys = ["env", "n", "model", "kappa"]
    drop = keys + ["run_id", "seed", "pair_rank"]
    num = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys)[num]
    by_env = g.mean().add_suffix("__mean").join(g.std().add_suffix("__std")).join(g.size().rename("n_seeds"))
    by_env.reset_index().to_csv(os.path.join(a.out, "by_env.csv"), index=False)

    env_means = g.mean().reset_index()
    g2 = env_means.groupby(["n", "model", "kappa"])[num]
    overall = g2.mean().add_suffix("__mean").join(g2.std().add_suffix("__envstd")).join(g2.size().rename("n_envs"))
    overall.reset_index().to_csv(os.path.join(a.out, "overall.csv"), index=False)

    parts = []
    for k in [0.0, 1.0]:
        path = os.path.join(a.runs, a.scatter_run.format(k=k) + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        idx = np.random.default_rng(0).choice(len(z["true_err"]), 3000, replace=False)
        parts.append(pd.DataFrame({"kappa": k, "gap": z["gap"][idx], "noisy": z["noisy"][idx],
                                   "true_err": z["true_err"][idx], "true_ale": z["true_ale"][idx],
                                   "ALE": z["sig_ALE"][idx], "EPI": z["sig_EPI"][idx]}))
    if parts:
        pd.concat(parts).to_csv(os.path.join(a.out, "scatter.csv"), index=False)
    print(f"{len(df)} runs -> {a.out} (by_env: {len(by_env)} cells, overall: {len(overall)} cells)")


if __name__ == "__main__":
    main()
