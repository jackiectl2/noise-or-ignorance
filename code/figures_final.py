"""Final-review figures: redraw every data figure from the stored result files.

Computes no new measurement. It reads results/v2/agg/*.csv and the per-run JSON files of the audit extensions, and the
only derived values are simple means, paired differences and binned medians of those stored numbers. Needs matplotlib and
numpy only, so it runs without the cluster environment.
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import figstyle as fs

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
AGG = ROOT / "results" / "v2" / "agg"
ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
SHORT = {"HalfCheetah-v5": "HalfCheetah", "Hopper-v5": "Hopper", "Walker2d-v5": "Walker2d"}
K3, K5 = [0.0, 0.25, 1.0], [0.0, 0.1, 0.25, 0.5, 1.0]
ROBOT_LS = {"HalfCheetah-v5": "-", "Hopper-v5": "--", "Walker2d-v5": ":"}
ROBOT_MK = {"HalfCheetah-v5": "o", "Hopper-v5": "s", "Walker2d-v5": "^"}
OI = fs.OKABE_ITO
STYLE = {
    "ALE": dict(color=OI["vermil"], marker="o", label="ALE (mean var.)"),
    "MOPO": dict(color=OI["orange"], marker="s", ls="--", label="MOPO (max std)"),
    "EPI": dict(color=OI["blue"], marker="v", label="EPI (var. of means)"),
    "MaxDisc": dict(color=OI["sky"], marker="s", ls="--", label="MaxDisc (MOReL)"),
    "MI": dict(color=OI["purple"], marker="^", label="MI"),
    "kNN": dict(color="#555555", marker="D", ls=":", label="kNN novelty"),
}
THIN = dict(lw=0.6, alpha=0.55)
KT = [0, 0.25, 0.5, 1.0]
KL = ["0", "0.25", "0.5", "1"]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(r, key):
    v = r.get(key, "")
    return float(v) if v not in ("", None) else np.nan


def style():
    fs.set_style()
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"


def flatten(name):
    """Drop the alpha channel: an opaque file renders the same in every PDF viewer."""
    p = FIG / f"{name}.png"
    Image.open(p).convert("RGB").save(p, dpi=(fs.PNG_DPI, fs.PNG_DPI))


def dose(n, name):
    ov = sorted([r for r in rows(AGG / "overall.csv") if r["model"] == "pe" and int(float(r["n"])) == n], key=lambda r: num(r, "kappa"))
    be = [r for r in rows(AGG / "by_env.csv") if r["model"] == "pe" and int(float(r["n"])) == n and int(float(r.get("pair_rank", 0) or 0)) == 0]
    ks = [num(r, "kappa") for r in ov]
    fig, ax = fs.panels("AB\nCD", figsize=(fs.TEXT_W_IN, 3.35))
    spec = [("A", "spear_err_in", "Spearman $\\rho$ with true error", "rank true error (visited)", 0.0),
            ("B", "retain80", "retained / mean error", "error kept when trusting 80%", 1.0),
            ("C", "auroc_noisy_in", "AUROC", "flag noisy states (visited)", 0.5),
            ("D", "auroc_gap_all", "AUROC", "flag unvisited states", 0.5)]
    for key, col, ylab, title, chance in spec:
        a = ax[key]
        focal = ("ALE", "EPI", "kNN") if key == "D" else ("ALE", "EPI")
        for s in focal:
            for e in ENVS:
                q = sorted([r for r in be if r["env"] == e], key=lambda r: num(r, "kappa"))
                y = [num(r, f"{col}/{s}__mean") for r in q]
                if not np.all(np.isnan(y)):
                    a.plot([num(r, "kappa") for r in q], y, color=STYLE[s]["color"], zorder=1, **THIN)
        for s, st in STYLE.items():
            y = [num(r, f"{col}/{s}__mean") for r in ov]
            if not np.all(np.isnan(y)):
                a.plot(ks, y, markersize=3.2, zorder=3, **st)
        a.axhline(chance, color="#999999", lw=0.8, zorder=0)
        if col == "retain80":
            a.plot(ks, [num(r, "retain80/ORACLE_ERR__mean") for r in ov], color="#000000", lw=1.0, ls=(0, (4, 2)), label="oracle (true error)")
        a.set_title(title)
        a.set_ylabel(ylab)
        a.set_xticks(KT, KL)
        if key in "CD":
            a.set_xlabel("process-noise level $\\kappa$")
    fs.save(fig, FIG, name, width_in=fs.TEXT_W_IN, legend_from=ax["B"], legend_ncol=4, panel_labels=[ax[k] for k in "ABCD"])
    flatten(name)


def binned_median(x, y, lim, bins=12, min_n=20):
    edges = np.logspace(np.log10(lim[0]), np.log10(lim[1]), bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            xs.append(np.sqrt(lo * hi))
            ys.append(np.median(y[m]))
    return xs, ys


def calib(name):
    sc = rows(AGG / "scatter.csv")
    arr = {k: np.array([num(r, k) for r in sc]) for k in ("kappa", "true_err", "true_ale", "ALE", "EPI")}
    flag = {k: np.array([r[k] == "True" for r in sc]) for k in ("gap", "noisy")}
    one = arr["kappa"] == 1.0
    vis = one & ~flag["gap"]
    target = arr["true_ale"] + arr["true_err"] + arr["EPI"]
    fig, ax = fs.panels("ABCD", figsize=(fs.TEXT_W_IN, 1.95))
    lim1, lim2 = (1e-2, 3e1), (1e-3, 3e1)
    panels = [("A", vis & ~flag["noisy"], target, arr["ALE"], lim1, "Blues", "noise-free", "noise var. + member error", "ALE"),
              ("B", vis & flag["noisy"], target, arr["ALE"], lim1, "Oranges", "noisy", "noise var. + member error", None),
              ("C", vis, arr["true_err"], arr["EPI"], lim2, "Blues", "visited", "true error $e$", "EPI"),
              ("D", one & flag["gap"], arr["true_err"], arr["EPI"], lim2, "Greens", "unvisited", "true error $e$", None)]
    for key, m, x, y, lim, cmap, title, xl, yl in panels:
        a = ax[key]
        xm, ym = np.clip(x[m], 1e-6, None), np.clip(y[m], 1e-6, None)
        a.hexbin(xm, ym, xscale="log", yscale="log", gridsize=22, extent=(*np.log10(lim), *np.log10(lim)), cmap=cmap, mincnt=1,
                 bins="log", linewidths=0, zorder=1)
        a.plot(lim, lim, color="#333333", lw=0.8, zorder=2)
        bx, by = binned_median(xm, ym, lim)
        a.plot(bx, by, color="black", lw=1.3, marker="o", ms=2, zorder=3)
        a.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
        a.set_title(title, loc="right", fontsize=8.5)
        a.set_xlabel(xl, fontsize=7.5)
        if yl:
            a.set_ylabel(yl)
        else:
            a.tick_params(labelleft=False)
        a.grid(False)
        a.text(0.96, 0.05, f"$n={int(m.sum())}$", transform=a.transAxes, ha="right", size=7, gid="data")
    fs.save(fig, FIG, name, width_in=fs.TEXT_W_IN, panel_labels=[ax[k] for k in "ABCD"])
    flatten(name)


def load_replay():
    E1 = {}
    for f in glob.glob(str(ROOT / "results/audit_policy_v1/*.json")):
        d = json.load(open(f))
        E1[(d["env"], d["kappa"], d["seed"])] = d
    assert len(E1) == 27
    return E1


def policy(name):
    E1 = load_replay()
    env_mean = lambda key, k, e: float(np.mean([E1[(e, k, s)][key] for s in range(3)]))
    all_mean = lambda key, k: float(np.mean([env_mean(key, k, e) for e in ENVS]))
    pool = sorted([r for r in rows(AGG / "overall.csv") if r["model"] == "pe" and int(float(r["n"])) == 30000], key=lambda r: num(r, "kappa"))
    fig, ax = fs.panels("AAABBB\nCCDDEE", figsize=(fs.TEXT_W_IN, 3.4), height_ratios=[1, 0.95])
    a = ax["A"]
    for s in ["ALE", "EPI"]:
        for e in ENVS:
            a.plot(K3, [env_mean(f"spear/{s}", k, e) for k in K3], color=STYLE[s]["color"], zorder=1, **THIN)
    for s in ["ALE", "MOPO", "EPI"]:
        st = {k: v for k, v in STYLE[s].items() if k != "label"}
        a.plot(K3, [all_mean(f"spear/{s}", k) for k in K3], markersize=3.2, label=s, zorder=3, **st)
        a.plot([num(r, "kappa") for r in pool], [num(r, f"spear_err_in/{s}__mean") for r in pool], color=st["color"], lw=0.9, ls=(0, (1, 1.5)), zorder=2)
    a.plot([], [], color="#555555", lw=0.9, ls=(0, (1, 1.5)), label="fixed pool, mean")
    a.set(title="error rank on replay data", ylabel="Spearman $\\rho$", xlabel="noise level $\\kappa$", ylim=(0.15, 1.0))
    a.set_xticks([0, 0.25, 1.0], ["0", "0.25", "1"])
    a.legend(loc="lower right", frameon=False, fontsize=7, ncol=2, columnspacing=0.8, handlelength=1.5)
    b = ax["B"]
    ys, labs, yy = [], [], 0
    for e in ENVS:
        cells = [s for s in range(3) if E1[(e, 1.0, s)]["testable"]]
        for s in cells:
            d = E1[(e, 1.0, s)]
            b.plot([d["top20_noisy/ERR"], d["top20_noisy/MOPO"]], [yy, yy], color="#BBBBBB", lw=1, zorder=1)
            b.scatter(d["top20_noisy/ERR"], yy, s=15, color="black", zorder=2, label="true error" if yy == 0 else None)
            b.scatter(d["top20_noisy/EPI"], yy, s=15, facecolors="none", edgecolors=OI["blue"], linewidths=1, zorder=3, label="EPI" if yy == 0 else None)
            b.scatter(d["top20_noisy/MOPO"], yy, s=15, color=OI["orange"], marker="s", zorder=2, label="MOPO" if yy == 0 else None)
            ys.append(yy)
            labs.append(f"{SHORT[e]} s{s}")
            yy += 1
        if cells:
            yy += 0.6
    b.set_yticks(ys, labs)
    b.invert_yaxis()
    b.set_title("top-20% rows, $\\kappa=1$", loc="right")
    b.set_xlabel("noisy share of top-20% rows")
    b.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.1)
    H = list(range(1, 6))
    c = ax["C"]
    for s in ["MOPO", "EPI"]:
        st = {k: v for k, v in STYLE[s].items() if k != "label"}
        c.plot(H, [all_mean(f"roll_spear/{s}/h{h}", 1.0) for h in H], markersize=3, label=f"{s}, $\\kappa=1$", **st)
        st2 = dict(st, ls=(0, (1, 1.5)), lw=1.1)
        c.plot(H, [all_mean(f"roll_spear/{s}/h{h}", 0.0) for h in H], markersize=2.5, label=f"{s}, $\\kappa=0$", **st2)
    c.set(xlabel="horizon $h$", ylabel="Spearman $\\rho$", ylim=(0.5, 0.95))
    c.set_title("mean rank", loc="right", fontsize=8.5)
    c.set_xticks(H)
    c.legend(loc="lower left", frameon=False, fontsize=6.5, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
    d = ax["D"]
    d.axhline(0, color="#999999", lw=0.8, zorder=0)
    for e in ENVS:
        pts = [[E1[(e, 1.0, s)][f"roll_spear/EPI/h{h}"] - E1[(e, 1.0, s)][f"roll_spear/MOPO/h{h}"] for s in range(3)] for h in H]
        for h, p in zip(H, pts):
            d.scatter([h] * 3, p, s=4, color="#AAAAAA", marker=ROBOT_MK[e], zorder=1, linewidths=0)
        d.plot(H, [np.mean(p) for p in pts], color="black", lw=1.1, ls=ROBOT_LS[e], marker=ROBOT_MK[e], ms=2.5, label=SHORT[e], zorder=2)
    d.set(xlabel="horizon $h$", ylabel="$\\rho$ difference, $\\kappa=1$", ylim=(-0.08, 0.14))
    d.set_title("EPI $-$ MOPO", loc="right", fontsize=8.5)
    d.set_xticks(H)
    d.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
    f = ax["E"]
    for e in ENVS:
        f.plot(H, [env_mean(f"roll_err_over_epi/h{h}", 1.0, e) for h in H], color="#444444", lw=0.9, ls=ROBOT_LS[e], marker=ROBOT_MK[e], ms=2.5, label=SHORT[e])
    f.plot(H, [all_mean(f"roll_err_over_epi/h{h}", 1.0) for h in H], color=OI["blue"], lw=1.8, label="mean")
    f.set(yscale="log", xlabel="horizon $h$", ylabel="ratio of means")
    f.set_title("err./EPI, $\\kappa=1$", loc="right", fontsize=8.5)
    f.set_xticks(H)
    f.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
    lo, hi = f.get_ylim()
    f.set_ylim(lo, hi * 8)
    fs.save(fig, FIG, name, width_in=fs.TEXT_W_IN, panel_labels=[ax[k] for k in "ABCDE"])
    flatten(name)


def noise(name):
    E3 = {}
    for p in glob.glob(str(ROOT / "results/noise_v1/*/*.json")):
        d = json.load(open(p))
        E3[(p.split("/")[-2], d["env"], d["model"], d["kappa"], d["seed"])] = d
    assert len(E3) == 3 * 135
    env = lambda v, e, key, k: float(np.mean([E3[(v, e, "pe", k, s)][key] for s in range(3)]))
    mean = lambda v, key, k: float(np.mean([env(v, e, key, k) for e in ENVS]))
    fig, ax = fs.panels("ABC\nDEF", figsize=(fs.TEXT_W_IN, 2.8), sharex=True)
    for col, low, (v, title) in zip("ABC", "DEF", [("gauss_contact", "contact gate"), ("t3_joint", "Student-$t_3$"), ("bimodal_joint", "bimodal")]):
        a, d = ax[col], ax[low]
        for s in ["ALE", "EPI"]:
            st = {k: x for k, x in STYLE[s].items() if k != "label"}
            for e in ENVS:
                a.plot(K5, [env(v, e, f"spear_err_in/{s}", k) for k in K5], color=st["color"], **THIN)
                d.plot(K5, [env(v, e, f"auroc_noisy_in/{s}", k) for k in K5], color=st["color"], **THIN)
            a.plot(K5, [mean(v, f"spear_err_in/{s}", k) for k in K5], markersize=3, label=s, **st)
            d.plot(K5, [mean(v, f"auroc_noisy_in/{s}", k) for k in K5], markersize=3, **st)
        a.set_title(title)
        a.set_ylim(0, 1)
        d.set_ylim(0.3, 1.02)
        d.axhline(0.5, color="#999999", lw=0.8, zorder=0)
        d.set_xlabel("noise level $\\kappa$")
        d.set_xticks(KT, KL)
    ax["A"].set_ylabel("$\\rho$ with true error")
    ax["D"].set_ylabel("noisy AUROC")
    ax["A"].legend(loc="lower left", frameon=False, fontsize=7)
    fs.save(fig, FIG, name, width_in=fs.TEXT_W_IN, panel_labels=[ax[k] for k in "ABCDEF"])
    flatten(name)


def delta(name):
    """Candidate from data not yet shown: the paired change in error rank for every ensemble of the main study."""
    rs = [r for r in rows(AGG / "runs.csv") if int(float(r["pair_rank"])) == 0]
    get = {(r["env"], int(float(r["n"])), r["model"], int(float(r["seed"])), float(r["kappa"])): r for r in rs}
    models = sorted({r["model"] for r in rs})
    print("models", models)
    order = [(m, n) for m in ("pe", "pe_beta", "de") if m in models for n in (10000, 30000, 100000)]
    names = {"pe": "PE", "pe_beta": "$\\beta$-NLL", "de": "DE"}
    fig, ax = fs.panels("A", figsize=(fs.TEXT_W_IN, 2.15))
    a = ax["A"]
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    out = {}
    for i, (m, n) in enumerate(order):
        for s, dx, col in [("ALE", -0.17, OI["vermil"]), ("EPI", 0.17, OI["blue"])]:
            vals = []
            for j, e in enumerate(ENVS):
                for seed in range(3):
                    r1, r0 = get.get((e, n, m, seed, 1.0)), get.get((e, n, m, seed, 0.0))
                    v = num(r1, f"spear_err_in/{s}") - num(r0, f"spear_err_in/{s}") if r1 and r0 else np.nan
                    if np.isnan(v):
                        continue
                    vals.append(v)
                    a.scatter(i + dx + (j - 1) * 0.055, v, s=9, marker=ROBOT_MK[e], facecolors="none", edgecolors=col, linewidths=0.6, zorder=2)
            if vals:
                a.plot([i + dx - 0.13, i + dx + 0.13], [np.mean(vals)] * 2, color=col, lw=1.8, zorder=3)
                out[f"{m}_{n}_{s}"] = float(np.mean(vals))
    a.set_xticks(range(len(order)), [f"{names[m]}\n{n // 1000}k" for m, n in order])
    a.set_ylabel("$\\Delta\\rho$, $\\kappa=0$ to $1$")
    a.grid(axis="x", visible=False)
    h = [plt.Line2D([], [], color=OI["vermil"], lw=1.8, label="ALE"), plt.Line2D([], [], color=OI["blue"], lw=1.8, label="EPI")]
    h += [plt.Line2D([], [], color="#444444", ls="", marker=ROBOT_MK[e], mfc="none", ms=3.5, mew=0.6, label=SHORT[e]) for e in ENVS]
    a.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=5, fontsize=7.5, columnspacing=1.2, handletextpad=0.4, borderaxespad=0.1)
    fs.save(fig, FIG, name, width_in=fs.TEXT_W_IN)
    flatten(name)
    json.dump(out, open(ROOT / "results" / "fig_delta_means.json", "w"), indent=1)


def main():
    style()
    for n in (10000, 30000, 100000):
        dose(n, f"fig_dose_n{n}")
    calib("fig_calib")
    policy("fig_policy")
    noise("fig_noise")
    delta("fig_delta")
    fs.qa_summary(FIG, tag="final")


if __name__ == "__main__":
    main()
