"""Final-review tables: rebuild every paper table from the stored results.

Selects and formats stored values only. All SDs are sample SDs (ddof=1), matching results/v2/agg. Bold marks the best value
of a defined comparison set at displayed precision, and ties share the mark. Needs numpy only. Replaces the table output of
code/tables.py and code/ext_outputs.py, which still produce the numeric macros.
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
AGG, PAIR = ROOT / "results/v2/agg", ROOT / "results/v2/agg_pair1"
ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
SHORT = {"HalfCheetah-v5": "HalfCheetah", "Hopper-v5": "Hopper", "Walker2d-v5": "Walker2d"}
OPEN = r"\tablebody"
SD1 = lambda v: float(np.std(v, ddof=1))
EPS = 1e-9


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def sel(rs, **kw):
    q = [r for r in rs if all(float(r[k]) == v if isinstance(v, (int, float)) else r[k] == v for k, v in kw.items())]
    assert len(q) == 1, (kw, len(q))
    return q[0]


def fmt(v):
    s = f"{v:.2f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def score(m, s=None, best=False):
    mac = r"\bestscore" if best else r"\score"
    return f"{mac}{{{fmt(m)}}}{{{'' if s is None else fmt(s)}}}"


def best_flags(vals, higher=True):
    disp = [round(v, 2) for v in vals]
    b = max(disp) if higher else min(disp)
    return [abs(d - b) < EPS for d in disp]


def tint(c, cell):
    return rf"\cellcolor{{{c}}}{cell}"


def write(name, lines):
    (PAPER / f"{name}.tex").write_text("\n".join([r"\begingroup\tablebody"] + lines + [r"\endgroup"]) + "\n")


def tab_main(ov):
    pe = dict(model="pe", n=30000)
    sigs = [("ALE", r"\ALE", "alefam"), ("MOPO", r"\MOPO", "alefam"), ("TOT", r"\TOT", "alefam"), ("EPI", r"\EPI", "epifam"),
            ("MaxDisc", r"\MaxDisc", "epifam"), ("MI", r"\MI", None), ("kNN", r"\kNN", None)]
    cols = [("spear_err_in", 0.0, True), ("spear_err_in", 1.0, True), ("retain80", 0.0, False), ("retain80", 1.0, False),
            ("auroc_noisy_in", 1.0, True), ("auroc_gap_all", 0.0, True), ("auroc_gap_all", 1.0, True)]
    data = {s[0]: [(float(sel(ov, kappa=k, **pe)[f"{c}/{s[0]}__mean"]), float(sel(ov, kappa=k, **pe)[f"{c}/{s[0]}__envstd"])) for c, k, _ in cols] for s in sigs}
    flags = [best_flags([data[s[0]][j][0] for s in sigs], cols[j][2]) for j in range(len(cols))]
    L = [r"\setlength{\scorewidth}{2.15em}\setlength{\tabcolsep}{2.7pt}", r"\begin{tabular}{l cc cc c cc}", r"\toprule",
         r" & \multicolumn{2}{c}{error rank $\rho$, visited $\uparrow$} & \multicolumn{2}{c}{retained error at 80\% $\downarrow$} & noisy AUROC $\uparrow$ & \multicolumn{2}{c}{unvisited AUROC $\uparrow$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}",
         r"Signal & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ \\", r"\midrule"]
    for i, (s, tex, fam) in enumerate(sigs):
        cells = [score(*data[s][j], best=flags[j][i]) for j in range(len(cols))]
        L.append((rf"\rowcolor{{{fam}}}" if fam else "") + " & ".join([tex] + cells) + r" \\")
    o = lambda c, k: score(float(sel(ov, kappa=k, **pe)[f"{c}/ORACLE_ERR__mean"]), float(sel(ov, kappa=k, **pe)[f"{c}/ORACLE_ERR__envstd"]))
    L += [r"\midrule", " & ".join([r"\textit{true error (reference)}", r"\score{1}{}", r"\score{1}{}", o("retain80", 0.0), o("retain80", 1.0),
                                   o("auroc_noisy_in", 1.0), o("auroc_gap_all", 0.0), o("auroc_gap_all", 1.0)]) + r" \\", r"\bottomrule", r"\end{tabular}"]
    write("tab_main", L)


def tab_policy(E1):
    g = lambda e, key, k=1.0: [E1[(e, k, s)][key] for s in range(3)]
    ms = lambda v, best=False: score(float(np.mean(v)), SD1(v), best)
    L = [r"\begin{tabular}{l cc cc ccc cc}", r"\toprule",
         r" & \multicolumn{2}{c}{noisy-region share} & \multicolumn{2}{c}{error rank $\rho$} & \multicolumn{3}{c}{noisy share of top 20\%} & \multicolumn{2}{c}{$\rho$ at $h{=}5$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
         r"Robot & $\kappa{=}0$ & $\kappa{=}1$ & \EPI & \MOPO & true error & \MOPO & \EPI & \EPI & \MOPO \\", r"\midrule"]
    for e in ENVS:
        testable = any(E1[(e, 1.0, s)]["testable"] for s in range(3))
        r1 = best_flags([np.mean(g(e, "spear/EPI")), np.mean(g(e, "spear/MOPO"))])
        r5 = best_flags([np.mean(g(e, "roll_spear/EPI/h5")), np.mean(g(e, "roll_spear/MOPO/h5"))])
        top = [ms(g(e, f"top20_noisy/{x}")) if testable else r"\missing" for x in ["ERR", "MOPO", "EPI"]]
        cells = [SHORT[e], ms(g(e, "frac_noisy", 0.0)), ms(g(e, "frac_noisy")),
                 tint("epifam", ms(g(e, "spear/EPI"), r1[0])), tint("alefam", ms(g(e, "spear/MOPO"), r1[1])),
                 top[0], tint("alefam", top[1]), tint("epifam", top[2]),
                 tint("epifam", ms(g(e, "roll_spear/EPI/h5"), r5[0])), tint("alefam", ms(g(e, "roll_spear/MOPO/h5"), r5[1]))]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_policy", L)


def tab_noise(E3):
    env = lambda v, e, key, k, m: float(np.mean([E3[(v, e, m, k, s)][key] for s in range(3)]))
    L = [r"\begin{tabular}{ll cc cc cc}", r"\toprule",
         r" & & \multicolumn{2}{c}{$\rho$ with $e$, $\kappa{=}0$} & \multicolumn{2}{c}{$\rho$ with $e$, $\kappa{=}1$} & \multicolumn{2}{c}{noisy AUROC, $\kappa{=}1$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Noise and gate & Objective & \ALE & \EPI & \ALE & \EPI & \ALE & \EPI \\", r"\midrule"]
    names = [("gauss_contact", "Gaussian, contact"), ("t3_joint", "Student-$t_3$, joint"), ("bimodal_joint", "bimodal, joint")]
    for v, name in names:
        for m, mn in [("pe", "PE"), ("pe_beta", r"$\beta$-NLL"), ("de", "DE")]:
            cells = [name if m == "pe" else "", mn]
            for key_fn, k in [(lambda s: f"spear_err_in/{s}", 0.0), (lambda s: f"spear_err_in/{s}", 1.0), (lambda s: f"auroc_noisy_in/{s}", 1.0)]:
                b = [env(v, e, key_fn("EPI"), k, m) for e in ENVS]
                if m == "de":  # no variance head, so no ALE and no ALE/EPI comparison to mark
                    cells += [tint("alefam", r"\missing"), tint("epifam", score(float(np.mean(b)), SD1(b)))]
                    continue
                a = [env(v, e, key_fn("ALE"), k, m) for e in ENVS]
                f = best_flags([np.mean(a), np.mean(b)])
                cells += [tint("alefam", score(float(np.mean(a)), SD1(a), f[0])), tint("epifam", score(float(np.mean(b)), SD1(b), f[1]))]
            L.append(" & ".join(cells) + r" \\")
        L.append(r"\midrule" if v != "bimodal_joint" else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("tab_noise", L)


def tab_env(be):
    pe = dict(model="pe", n=30000)
    L = [r"\begin{tabular}{ll ccc ccc ccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{error rank $\rho$, visited} & \multicolumn{3}{c}{noisy AUROC} & \multicolumn{3}{c}{unvisited AUROC} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Robot & $\kappa$ & \ALE & \EPI & \kNN & \ALE & \EPI & \kNN & \ALE & \EPI & \kNN \\", r"\midrule"]
    fam = {"ALE": "alefam", "EPI": "epifam", "kNN": None}
    for e in ENVS:
        for k in [0.0, 0.25, 1.0]:
            r = sel(be, env=e, kappa=k, **pe)
            cells = [SHORT[e] if k == 0.0 else "", f"{k:g}"]
            for col in ["spear_err_in", "auroc_noisy_in", "auroc_gap_all"]:
                f = best_flags([float(r[f"{col}/{s}__mean"]) for s in fam])
                for (s, c), b in zip(fam.items(), f):
                    x = score(float(r[f"{col}/{s}__mean"]), float(r[f"{col}/{s}__std"]), b)
                    cells.append(tint(c, x) if c else x)
            L.append(" & ".join(cells) + r" \\")
        L.append(r"\midrule" if e != "Walker2d-v5" else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("tab_env", L)


def tab_obj(ov):
    L = [r"\begin{tabular}{ll ccccc}", r"\toprule",
         r"Objective & $n$ & mean $e$, visited & mean $e$, unvisited & $\rho(\ALE,e)$ & $\rho(\EPI,e)$ & noisy AUROC (\EPI) \\", r"\midrule"]
    for m, name in [("pe", "PE (NLL)"), ("pe_beta", r"$\beta$-NLL"), ("de", "DE (MSE)")]:
        for n in [10000, 30000, 100000]:
            r = sel(ov, model=m, n=n, kappa=1.0)
            c = lambda key: score(float(r[f"{key}__mean"]), float(r[f"{key}__envstd"]))
            cells = [name if n == 10000 else "", f"{n // 1000}k", c("mse_in"), c("mse_gap"),
                     r"\missing" if m == "de" else c("spear_err_in/ALE"), c("spear_err_in/EPI"), c("auroc_noisy_in/EPI")]
            L.append((r"\rowcolor{mainrow}" if (m, n) == ("pe", 30000) else "") + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if m != "de" else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("tab_obj", L)


def tab_pair(ov, pov):
    pe = dict(model="pe", n=30000)
    cols = [("spear_err_in", "ALE"), ("spear_err_in", "EPI"), ("auroc_noisy_in", "ALE"), ("auroc_noisy_in", "EPI"), ("auroc_noisy_in", "MI"),
            ("auroc_gap_all", "EPI"), ("auroc_gap_all", "kNN")]
    L = [r"\begin{tabular}{lll cc ccc cc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{error rank $\rho$} & \multicolumn{3}{c}{noisy AUROC} & \multicolumn{2}{c}{unvisited AUROC} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
         r"Objective & Features & $\kappa$ & \ALE & \EPI & \ALE & \EPI & \MI & \EPI & \kNN \\", r"\midrule"]
    for m, mn in [("pe", "PE"), ("pe_beta", r"$\beta$-NLL"), ("de", "DE")]:
        for name, df in [("main pair", ov), ("second pair", pov)]:
            for k in [0.0, 1.0]:
                r = sel(df, kappa=k, model=m, n=30000)
                vals = [score(float(r[f"{c}/{s}__mean"]), float(r[f"{c}/{s}__envstd"])) if (m != "de" or s not in ("ALE", "MI")) else r"\missing"
                        for c, s in cols]
                if k == 0.0:
                    vals[2:5] = [r"\missing"] * 3
                first = mn if (name == "main pair" and k == 0.0) else ""
                L.append((r"\rowcolor{mainrow}" if name == "main pair" else "") + " & ".join([first, name if k == 0.0 else "", f"{k:g}"] + vals) + r" \\")
        L.append(r"\midrule" if m != "de" else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("tab_pair", L)


def tab_tthree(E3):
    get = lambda e, key: [E3[("t3_joint", e, "pe", 1.0, s)][key] for s in range(3)]
    L = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{$\rho(\ALE,e)$, $\kappa{=}1$} & \multicolumn{3}{c}{$\rho(\EPI,e)$, $\kappa{=}1$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Robot & seed 0 & seed 1 & seed 2 & seed 0 & seed 1 & seed 2 \\", r"\midrule"]
    for e in ENVS:
        L.append(" & ".join([SHORT[e]] + [tint("alefam", fmt(x)) for x in get(e, "spear_err_in/ALE")] + [tint("epifam", fmt(x)) for x in get(e, "spear_err_in/EPI")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_tthree", L)


def ratio(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def tab_scale(ov, be):
    """Candidate from results not yet tabulated: each signal against its exact reference, per robot."""
    pe = dict(model="pe", n=30000)
    cols = [("ratio_ale_over_member_residual_in", 0.0), ("ratio_ale_over_member_residual_in", 1.0), ("ratio_ale_over_noise_in_noisy", 1.0),
            ("ratio_epi_over_err_in", 0.0), ("ratio_epi_over_err_in", 1.0), ("cal_err_over_epi_in", 0.0), ("cal_err_over_epi_in", 1.0),
            ("cal_err_over_epi_gap", 0.0), ("cal_err_over_epi_gap", 1.0)]

    def cell(m, s):
        shade = 1 - 0.12 * min(abs(np.log10(m)) / 2, 1)
        return rf"\cellcolor[gray]{{{shade:.3f}}}\ratioscore{{{ratio(m)}}}{{{ratio(s)}}}"
    L = [r"\setlength{\tabcolsep}{1.6pt}", r"\begin{tabular}{l cc c cc cc cc}", r"\toprule",
         r" & \multicolumn{2}{c}{\ALE{} / target} & \ALE{} / noise & \multicolumn{2}{c}{\EPI{} / $e$} & \multicolumn{2}{c}{$e$ / \EPI{}, visited} & \multicolumn{2}{c}{$e$ / \EPI{}, unvisited} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Robot & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ \\", r"\midrule"]
    for e in ENVS:
        cells = [SHORT[e]]
        for c, k in cols:
            r = sel(be, env=e, kappa=k, **pe)
            cells.append(cell(float(r[f"{c}__mean"]), float(r[f"{c}__std"])))
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    cells = ["Mean"]
    for c, k in cols:
        r = sel(ov, kappa=k, **pe)
        cells.append(cell(float(r[f"{c}__mean"]), float(r[f"{c}__envstd"])))
    L += [" & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
    write("tab_scale", L)
    gap = lambda e: float(sel(be, env=e, kappa=1.0, **pe)["cal_err_over_epi_gap__mean"])
    (PAPER / "numbers_final.tex").write_text("% Generated by code/tables_final.py -- do not edit by hand.\n"
                                             + f"\\newcommand{{\\CalGapCheetahOne}}{{{gap('HalfCheetah-v5'):.1f}}}\n"
                                             + f"\\newcommand{{\\CalGapHopperOne}}{{{gap('Hopper-v5'):.1f}}}\n"
                                             + f"\\newcommand{{\\CalGapWalkerOne}}{{{gap('Walker2d-v5'):.1f}}}\n")


def tab_rows(E1):
    """Denominators of the replay noise claims, with the frozen 500-row rule as a visible status."""
    L = [r"\begin{tabular}{ll ccc c}", r"\toprule", r" & & \multicolumn{3}{c}{test rows with $w>0$} & \\", r"\cmidrule(lr){3-5}",
         r"Robot & $\kappa$ & seed 0 & seed 1 & seed 2 & testable seeds \\", r"\midrule"]
    for e in ENVS:
        for k in [0.0, 0.25, 1.0]:
            ds = [E1[(e, k, s)] for s in range(3)]
            assert all(d["n_test"] == 10000 for d in ds)
            cs = [str(d["n_noisy"]) if d["testable"] else rf"\cellcolor{{untest}}{d['n_noisy']}$^\dagger$" for d in ds]
            L.append(" & ".join([SHORT[e] if k == 0.0 else "", f"{k:g}"] + cs + [f"{sum(d['testable'] for d in ds)} of 3"]) + r" \\")
        L.append(r"\midrule" if e != "Walker2d-v5" else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("tab_rows", L)


def tab_design(ov, pov, E1, E3):
    """Study overview. Counts are checked against the result files, not typed from memory."""
    runs = rows(AGG / "runs.csv")
    pair_runs = rows(PAIR / "runs.csv")
    assert len(runs) == 405 and len(pair_runs) == 135 and len(E1) == 27 and len(E3) == 405
    assert {d["n_train"] for d in E1.values()} and len(glob.glob(str(ROOT / "results/v2mi/json/*"))) >= 135
    K5 = r"0, 0.1, 0.25, 0.5, 1"
    O3 = r"PE, $\beta$-NLL, DE"
    L = [r"\begin{tabular}{l l l l l r}", r"\toprule", r"Study & States & $\kappa$ & Training rows & Objectives & Ensembles \\", r"\midrule",
         rf"Fixed pool & random pool & {K5} & 10k, 30k, 100k & {O3} & 405 \\",
         rf"Second feature pair & random pool & {K5} & 30k & {O3} & 135 \\",
         rf"Monte Carlo \MI{{}} & random pool & {K5} & 30k & {O3} & 135 \\",
         rf"Noise shape and gate & random pool & {K5} & 30k & {O3} & 405 \\",
         r"Replay and rollouts & SAC replay & 0, 0.25, 1 & about 225k & PE & 27 \\", r"\bottomrule", r"\end{tabular}"]
    write("tab_design", L)


def tab_screen():
    """Returns of the earlier screen, moved from prose to a table. Normalized return, last five evaluations."""
    R = {}
    for f in glob.glob(str(ROOT / "results/return_screen_v1/*.json")):
        d = json.load(open(f))
        R[(d["kappa"], d["seed"], d["arm"])] = d["normalized"]
    assert len(R) == 12
    arms = [("MOPO", r"\MOPO"), ("DIS", "root disagreement"), ("C3", "gated penalty")]
    L = [r"\begin{tabular}{l cc cc}", r"\toprule", r" & \multicolumn{2}{c}{$\kappa{=}1$} & \multicolumn{2}{c}{$\kappa{=}0$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Arm & seed 0 & seed 1 & seed 0 & seed 1 \\", r"\midrule"]
    for a, tex in arms:
        L.append(" & ".join([tex] + [fmt(R[(k, s, a)]) for k in (1.0, 0.0) for s in (0, 1)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_screen", L)


def main():
    ov, be, pov = rows(AGG / "overall.csv"), rows(AGG / "by_env.csv"), rows(PAIR / "overall.csv")
    E1, E3 = {}, {}
    for f in glob.glob(str(ROOT / "results/audit_policy_v1/*.json")):
        d = json.load(open(f))
        E1[(d["env"], d["kappa"], d["seed"])] = d
    for f in glob.glob(str(ROOT / "results/noise_v1/*/*.json")):
        d = json.load(open(f))
        E3[(f.split("/")[-2], d["env"], d["model"], d["kappa"], d["seed"])] = d
    assert len(E1) == 27 and len(E3) == 3 * 135
    tab_main(ov)
    tab_policy(E1)
    tab_noise(E3)
    tab_env(be)
    tab_obj(ov)
    tab_pair(ov, pov)
    tab_tthree(E3)
    tab_scale(ov, be)
    tab_rows(E1)
    tab_screen()
    tab_design(ov, pov, E1, E3)


if __name__ == "__main__":
    main()
