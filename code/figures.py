"""Figure step: results/agg/*.csv -> paper figures (PNG, 400 dpi, laid out at the printed width).

Computes nothing: every plotted number is a column of overall.csv / scatter.csv written by aggregate.py.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

# Signal families share a hue family: aleatoric-head signals warm, disagreement cool.
STYLE = {
    "ALE":     dict(color=fs.OKABE_ITO["vermil"], marker="o", label="ALE (mean var.)"),
    "MOPO":    dict(color=fs.OKABE_ITO["orange"], marker="s", ls="--", label="MOPO (max std)"),
    "EPI":     dict(color=fs.OKABE_ITO["blue"], marker="o", label="EPI (var. of means)"),
    "MaxDisc": dict(color=fs.OKABE_ITO["sky"], marker="s", ls="--", label="MaxDisc (MOReL)"),
    "MI":      dict(color=fs.OKABE_ITO["purple"], marker="^", label="MI proxy"),
    "kNN":     dict(color="#555555", marker="D", ls=":", label="kNN novelty"),
}


def dose_figure(ov, n, out, name):
    d = ov[(ov.model == "pe") & (ov.n == n)].sort_values("kappa")
    fig, ax = fs.panels("AB\nCD", figsize=(fs.TEXT_W_IN, 3.9))
    spec = [("A", "spear_err_in", "Spearman $\\rho$ with true error", "rank true error (visited)", 0.0),
            ("B", "retain80", "retained / mean error", "error kept when trusting 80%", 1.0),
            ("C", "auroc_noisy_in", "AUROC", "flag noisy states (visited)", 0.5),
            ("D", "auroc_gap_all", "AUROC", "flag unvisited states", 0.5)]
    for key, col, ylab, title, chance in spec:
        a = ax[key]
        for s, st in STYLE.items():
            c = f"{col}/{s}__mean"
            if c not in d or d[c].isna().all():
                continue
            m, e = d[c].to_numpy(), d[f"{col}/{s}__envstd"].to_numpy()
            a.plot(d.kappa, m, markersize=3.5, **st)
            a.fill_between(d.kappa, m - e, m + e, color=st["color"], alpha=0.05 if s == "kNN" else 0.10, lw=0)
        a.axhline(chance, color="#999999", lw=0.8, zorder=0)
        if col == "retain80":
            a.plot(d.kappa, d["retain80/ORACLE_ERR__mean"], color="#000000", lw=1.0, ls=(0, (4, 2)),
                   label="oracle (true error)")
        a.set_title(title)
        a.set_ylabel(ylab)
        a.set_xticks([0, 0.25, 0.5, 1.0], ["0", "0.25", "0.5", "1"])
        if key in "CD":
            a.set_xlabel("process-noise level $\\kappa$")
    fs.save(fig, out, name, width_in=fs.TEXT_W_IN, legend_from=ax["B"], legend_ncol=4,
            panel_labels=[ax["A"], ax["B"], ax["C"], ax["D"]])


def calib_figure(sc, out, name):
    d = sc[(sc.kappa == 1.0) & (~sc.gap)]
    fig, ax = fs.panels("AB", figsize=(fs.TEXT_W_IN, 2.4))
    a = ax["A"]
    for noisy, col, lab in [(False, fs.OKABE_ITO["blue"], "noise-free region"), (True, fs.OKABE_ITO["vermil"], "noisy region")]:
        q = d[d.noisy == noisy]
        a.scatter(q.true_ale + q.true_err + q.EPI, q.ALE, s=2, alpha=0.35, color=col, label=lab, rasterized=True)
    lim = [1e-2, 3e1]
    a.plot(lim, lim, color="#333333", lw=0.8)
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    a.set_xlabel("true noise var. + member error")
    a.set_ylabel("ALE")
    a.set_title("aleatoric head, $\\kappa=1$")
    a.legend(loc="upper left", frameon=False, markerscale=4)
    b = ax["B"]
    e = sc[sc.kappa == 1.0]
    for gap, col, lab in [(False, fs.OKABE_ITO["blue"], "visited"), (True, fs.OKABE_ITO["green"], "unvisited")]:
        q = e[e.gap == gap]
        b.scatter(q.true_err, q.EPI, s=2, alpha=0.35, color=col, label=lab, rasterized=True)
    lim2 = [1e-3, 3e1]
    b.plot(lim2, lim2, color="#333333", lw=0.8)
    b.set(xscale="log", yscale="log", xlim=lim2, ylim=lim2)
    b.set_xlabel("true error $\\|\\bar\\mu - f\\|^2$")
    b.set_ylabel("EPI")
    b.set_title("disagreement, $\\kappa=1$")
    b.legend(loc="upper left", frameon=False, markerscale=4)
    fs.save(fig, out, name, width_in=fs.TEXT_W_IN, panel_labels=[ax["A"], ax["B"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg", default="results/v2/agg")
    p.add_argument("--out", default="results/v2/figures")
    a = p.parse_args()
    fs.set_style()
    plt.rcParams["font.serif"] = ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
    out = Path(a.out)
    ov = pd.read_csv(Path(a.agg) / "overall.csv")
    for n in sorted(ov.n.unique()):
        dose_figure(ov, n, out, f"fig_dose_n{n}")
    sc_path = Path(a.agg) / "scatter.csv"
    if sc_path.exists():
        calib_figure(pd.read_csv(sc_path), out, "fig_calib")
    fs.qa_summary(out, tag="main")


if __name__ == "__main__":
    main()
