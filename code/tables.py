"""Table/number step: results/agg/*.csv -> paper/numbers.tex (macros quoted in the prose) and
paper/tab_*.tex (LaTeX tables). Computes nothing beyond selecting and formatting CSV cells.

Main setting for quoted numbers: PE objective, n = 30k, mean over the three robots of the per-robot
seed means ("overall.csv").
"""
import argparse
from pathlib import Path

import pandas as pd

MAIN_N = 30000
SIG = ["ALE", "MOPO", "TOT", "EPI", "MaxDisc", "MI", "kNN"]
SIG_TEX = {"ALE": r"\ALE", "MOPO": r"\MOPO", "TOT": r"\TOT", "EPI": r"\EPI", "MaxDisc": r"\MaxDisc",
           "MI": r"\MI", "kNN": r"\kNN"}
ENV_TEX = {"HalfCheetah-v5": "HalfCheetah", "Hopper-v5": "Hopper", "Walker2d-v5": "Walker2d"}


def cell(df, col, **sel):
    q = df
    for k, v in sel.items():
        q = q[q[k] == v]
    assert len(q) == 1, (col, sel, len(q))
    return float(q[col].iloc[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg", default="results/v2/agg")
    p.add_argument("--agg_pair", default="results/v2/agg_pair1")
    p.add_argument("--agg_mi", default="")
    p.add_argument("--paper", default="paper")
    a = p.parse_args()
    ov = pd.read_csv(Path(a.agg) / "overall.csv")
    be = pd.read_csv(Path(a.agg) / "by_env.csv")
    runs = pd.read_csv(Path(a.agg) / "runs.csv")
    main_sel = dict(model="pe", n=MAIN_N)
    f2 = lambda v: f"{v:.2f}"

    num = {"NumEnsembles": str(len(runs)), "MainN": "30k"}
    for tag, k in [("zero", 0.0), ("one", 1.0)]:
        for s in ["ALE", "MOPO", "TOT", "EPI", "MaxDisc", "MI", "kNN"]:
            num[f"Sp{s}{tag}"] = f2(cell(ov, f"spear_err_in/{s}__mean", kappa=k, **main_sel))
            num[f"AUtop{s}{tag}"] = f2(cell(ov, f"auroc_toperr_all/{s}__mean", kappa=k, **main_sel))
            num[f"AUnoisy{s}{tag}"] = f2(cell(ov, f"auroc_noisy_in/{s}__mean", kappa=k, **main_sel))
            num[f"AUgap{s}{tag}"] = f2(cell(ov, f"auroc_gap_all/{s}__mean", kappa=k, **main_sel))
        num[f"RatioALE{tag}"] = f2(cell(ov, "ratio_ale_over_member_residual_in__mean", kappa=k, **main_sel))
        for s in ["ALE", "MOPO", "TOT", "EPI", "MaxDisc", "MI", "kNN", "ORACLE_ERR"]:
            num[f"Ret{s.replace('_', '')}{tag}"] = f2(cell(ov, f"retain80/{s}__mean", kappa=k, **main_sel))
        num[f"AUgapORACLEERR{tag}"] = f2(cell(ov, "auroc_gap_all/ORACLE_ERR__mean", kappa=k, **main_sel))
        r = cell(ov, "ratio_epi_over_err_in__mean", kappa=k, **main_sel)
        num[f"RatioEPI{tag}"] = f2(r)
        num[f"RatioEPIinv{tag}"] = f"{1 / r:.1f}"
        for c, nm in [("cal_err_over_epi_in", "CalEPIin"), ("cal_err_over_epi_gap", "CalEPIgap"),
                      ("cal_err_over_epi_topepi", "CalEPItop")]:
            num[f"{nm}{tag}"] = f"{cell(ov, c + '__mean', kappa=k, **main_sel):.1f}"
        rt = cell(ov, "ratio_epi_over_err_top__mean", kappa=k, **main_sel)
        num[f"RatioEPItop{tag}"] = f2(rt)
        num[f"RatioEPItopinv{tag}"] = f"{1 / rt:.0f}"
    for env, short in [("HalfCheetah-v5", "Cheetah"), ("Hopper-v5", "Hopper"), ("Walker2d-v5", "Walker")]:
        num[f"ErrRatio{short}"] = f"{cell(be, 'frac_err_gap_over_in__mean', env=env, kappa=0.0, **main_sel):.1f}"
        for tag, k in [("zero", 0.0), ("one", 1.0)]:
            num[f"AUgapKNN{short}{tag}"] = f2(cell(be, "auroc_gap_all/kNN__mean", env=env, kappa=k, **main_sel))
            num[f"AUgapEPI{short}{tag}"] = f2(cell(be, "auroc_gap_all/EPI__mean", env=env, kappa=k, **main_sel))
            num[f"AUgapALE{short}{tag}"] = f2(cell(be, "auroc_gap_all/ALE__mean", env=env, kappa=k, **main_sel))
    num["AUnoisyORACLEERRone"] = f2(cell(ov, "auroc_noisy_in/ORACLE_ERR__mean", kappa=1.0, **main_sel))
    num["ErrNoisyOverCleanone"] = f"{cell(ov, 'frac_err_noisy_over_clean_in__mean', kappa=1.0, **main_sel):.1f}"
    num["RatioALEnoiseone"] = f2(cell(ov, "ratio_ale_over_noise_in_noisy__mean", kappa=1.0, **main_sel))
    placebo = [cell(ov, f"auroc_noisy_in/{s}__mean", kappa=0.0, **main_sel) for s in ["ALE", "MOPO", "EPI", "MaxDisc", "MI", "kNN"]]
    num["PlaceboMin"], num["PlaceboMax"] = f2(min(placebo)), f2(max(placebo))
    for env, short in [("HalfCheetah-v5", "Cheetah"), ("Hopper-v5", "Hopper"), ("Walker2d-v5", "Walker")]:
        num[f"AUgapORACLEERR{short}zero"] = f2(cell(be, "auroc_gap_all/ORACLE_ERR__mean", env=env, kappa=0.0, **main_sel))
    # dataset-size and objective effects (kappa = 1, mean over robots)
    for n, tag in [(10000, "tenk"), (30000, "thirtyk"), (100000, "hundredk")]:
        num[f"SpALEone{tag}"] = f2(cell(ov, "spear_err_in/ALE__mean", kappa=1.0, model="pe", n=n))
        num[f"SpEPIone{tag}"] = f2(cell(ov, "spear_err_in/EPI__mean", kappa=1.0, model="pe", n=n))
        num[f"AUnoisyEPIone{tag}"] = f2(cell(ov, "auroc_noisy_in/EPI__mean", kappa=1.0, model="pe", n=n))
    for mdl, tag in [("pe", "PE"), ("pe_beta", "Beta"), ("de", "DE")]:
        for k, kt in [(0.0, "zero"), (1.0, "one")]:
            num[f"MSEin{tag}{kt}"] = f2(cell(ov, "mse_in__mean", kappa=k, model=mdl, n=MAIN_N))
            num[f"SpEPI{tag}{kt}"] = f2(cell(ov, "spear_err_in/EPI__mean", kappa=k, model=mdl, n=MAIN_N))
            if mdl != "de":
                num[f"SpMI{tag}{kt}"] = f2(cell(ov, "spear_err_in/MI__mean", kappa=k, model=mdl, n=MAIN_N))
                num[f"SpALE{tag}{kt}"] = f2(cell(ov, "spear_err_in/ALE__mean", kappa=k, model=mdl, n=MAIN_N))
                num[f"AUnoisyALE{tag}{kt}"] = f2(cell(ov, "auroc_noisy_in/ALE__mean", kappa=k, model=mdl, n=MAIN_N))
    Path(a.paper).mkdir(exist_ok=True)
    with open(Path(a.paper) / "numbers.tex", "w") as f:
        f.write("% Generated by code/tables.py from results/agg -- do not edit by hand.\n")
        for k, v in num.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # Table 1: main setting, kappa in {0, 1}, all signals (mean over robots; +- std over robots).
    def pm(col, s, k, df=ov, **sel):
        m = cell(df, f"{col}/{s}__mean", kappa=k, **sel)
        e = cell(df, f"{col}/{s}__envstd", kappa=k, **sel)
        return f"{m:.2f}\\,{{\\scriptsize$\\pm${e:.2f}}}"

    lines = [r"\begin{tabular}{l cc cc c cc}", r"\toprule",
             r" & \multicolumn{2}{c}{error rank $\rho$ (visited) $\uparrow$} & \multicolumn{2}{c}{retained error @80\% $\downarrow$}"
             r" & noisy AUROC & \multicolumn{2}{c}{unvisited AUROC} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}",
             r"Signal & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ & $\kappa{=}1$ & $\kappa{=}0$ & $\kappa{=}1$ \\",
             r"\midrule"]
    for s in SIG:
        row = [SIG_TEX[s], pm("spear_err_in", s, 0.0, **main_sel), pm("spear_err_in", s, 1.0, **main_sel),
               pm("retain80", s, 0.0, **main_sel), pm("retain80", s, 1.0, **main_sel),
               pm("auroc_noisy_in", s, 1.0, **main_sel),
               pm("auroc_gap_all", s, 0.0, **main_sel), pm("auroc_gap_all", s, 1.0, **main_sel)]
        lines.append(" & ".join(row) + r" \\")
    orc = [r"\midrule\textit{true error (oracle)}", "1", "1", pm("retain80", "ORACLE_ERR", 0.0, **main_sel),
           pm("retain80", "ORACLE_ERR", 1.0, **main_sel), pm("auroc_noisy_in", "ORACLE_ERR", 1.0, **main_sel),
           pm("auroc_gap_all", "ORACLE_ERR", 0.0, **main_sel), pm("auroc_gap_all", "ORACLE_ERR", 1.0, **main_sel)]
    lines.append(" & ".join(orc) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (Path(a.paper) / "tab_main.tex").write_text("\n".join(lines) + "\n")

    # Table 2 (appendix): per robot, main setting, kappa = 1 and 0, key signals.
    lines = [r"\begin{tabular}{ll ccc ccc ccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{error rank $\rho$ (visited)} & \multicolumn{3}{c}{noisy AUROC}"
             r" & \multicolumn{3}{c}{unvisited AUROC} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Robot & $\kappa$ & \ALE & \EPI & \kNN & \ALE & \EPI & \kNN & \ALE & \EPI & \kNN \\", r"\midrule"]
    for env in ENV_TEX:
        for k in [0.0, 0.25, 1.0]:
            vals = []
            for col in ["spear_err_in", "auroc_noisy_in", "auroc_gap_all"]:
                for s in ["ALE", "EPI", "kNN"]:
                    m = cell(be, f"{col}/{s}__mean", env=env, kappa=k, **main_sel)
                    sd = cell(be, f"{col}/{s}__std", env=env, kappa=k, **main_sel)
                    vals.append(f"{m:.2f}\\,{{\\scriptsize$\\pm${sd:.2f}}}")
            lines.append(" & ".join([ENV_TEX[env] if k == 0.0 else "", f"{k:g}"] + vals) + r" \\")
        lines.append(r"\midrule" if env != "Walker2d-v5" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (Path(a.paper) / "tab_env.tex").write_text("\n".join(lines) + "\n")

    # Table 3 (appendix): objectives and dataset sizes, kappa = 1 (mean over robots).
    lines = [r"\begin{tabular}{ll ccccc}", r"\toprule",
             r"Objective & $n$ & mean error (visited) & mean error (unvisited) & $\rho$(\ALE, $e$) & $\rho$(\EPI, $e$) & noisy AUROC (\EPI) \\",
             r"\midrule"]
    for mdl, name in [("pe", "PE (NLL)"), ("pe_beta", r"$\beta$-NLL"), ("de", "DE (MSE)")]:
        for n in [10000, 30000, 100000]:
            sel = dict(model=mdl, n=n, kappa=1.0)
            ale = cell(ov, "spear_err_in/ALE__mean", **sel) if mdl != "de" else float("nan")
            row = [name if n == 10000 else "", f"{n // 1000}k",
                   f"{cell(ov, 'mse_in__mean', **sel):.2f}", f"{cell(ov, 'mse_gap__mean', **sel):.2f}",
                   "--" if mdl == "de" else f"{ale:.2f}", f"{cell(ov, 'spear_err_in/EPI__mean', **sel):.2f}",
                   f"{cell(ov, 'auroc_noisy_in/EPI__mean', **sel):.2f}"]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule" if mdl != "de" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (Path(a.paper) / "tab_obj.tex").write_text("\n".join(lines) + "\n")
    # Table 4 (appendix): robustness to the feature pair (second-least-correlated pair; PE, n = 30k).
    pov = pd.read_csv(Path(a.agg_pair) / "overall.csv")
    cols = [("spear_err_in", "ALE"), ("spear_err_in", "EPI"), ("auroc_noisy_in", "ALE"), ("auroc_noisy_in", "EPI"),
            ("auroc_noisy_in", "MI"), ("auroc_gap_all", "EPI"), ("auroc_gap_all", "kNN")]
    lines = [r"\begin{tabular}{ll ccccccc}", r"\toprule",
             r"Features & $\kappa$ & $\rho$(\ALE) & $\rho$(\EPI) & noisy (\ALE) & noisy (\EPI) & noisy (\MI)"
             r" & unvisited (\EPI) & unvisited (\kNN) \\", r"\midrule"]
    for name, df in [("main pair", ov), ("second pair", pov)]:
        for k in [0.0, 1.0]:
            vals = [f"{cell(df, f'{c}/{s}__mean', kappa=k, **main_sel):.2f}" for c, s in cols]
            if k == 0.0:
                vals[2:5] = ["--"] * 3  # nothing is noisy at kappa = 0
            lines.append(" & ".join([name if k == 0.0 else "", f"{k:g}"] + vals) + r" \\")
        lines.append(r"\midrule" if name == "main pair" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (Path(a.paper) / "tab_pair.tex").write_text("\n".join(lines) + "\n")
    num_pair = {"PairSpALEone": f2(cell(pov, "spear_err_in/ALE__mean", kappa=1.0, **main_sel)),
                "PairSpALEzero": f2(cell(pov, "spear_err_in/ALE__mean", kappa=0.0, **main_sel)),
                "PairSpEPIone": f2(cell(pov, "spear_err_in/EPI__mean", kappa=1.0, **main_sel)),
                "PairAUnoisyALEone": f2(cell(pov, "auroc_noisy_in/ALE__mean", kappa=1.0, **main_sel)),
                "PairAUgapKNNone": f2(cell(pov, "auroc_gap_all/kNN__mean", kappa=1.0, **main_sel)),
                "PairAUgapEPIone": f2(cell(pov, "auroc_gap_all/EPI__mean", kappa=1.0, **main_sel))}
    with open(Path(a.paper) / "numbers.tex", "a") as f:
        for k, v in num_pair.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # Supplementary slice: Monte Carlo mutual information (re-trained n = 30k cells, PE).
    num_mi = {}
    if a.agg_mi and (Path(a.agg_mi) / "overall.csv").exists():
        mov = pd.read_csv(Path(a.agg_mi) / "overall.csv")
        for tag, k in [("zero", 0.0), ("one", 1.0)]:
            for s in ["MI", "MIMC", "EPI"]:
                num_mi[f"Sp{s}{tag}mc"] = f2(cell(mov, f"spear_err_in/{s}__mean", kappa=k, **main_sel))
                num_mi[f"AUnoisy{s}{tag}mc"] = f2(cell(mov, f"auroc_noisy_in/{s}__mean", kappa=k, **main_sel))
                num_mi[f"Ret{s}{tag}mc"] = f2(cell(mov, f"retain80/{s}__mean", kappa=k, **main_sel))
                num_mi[f"AUgap{s}{tag}mc"] = f2(cell(mov, f"auroc_gap_all/{s}__mean", kappa=k, **main_sel))
        with open(Path(a.paper) / "numbers.tex", "a") as f:
            for k, v in num_mi.items():
                f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(num) + len(num_pair) + len(num_mi)} macros and 4 tables to {a.paper}")


if __name__ == "__main__":
    main()
