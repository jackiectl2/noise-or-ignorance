"""Train every (noise level, model type) ensemble for one (env, n, seed) inside a single batched
network, then score each ensemble against the simulator ground truth.

Our GPUs ran in exclusive-process mode, so one process per GPU was all we had; stacking the
ensembles along the member dimension keeps each GPU busy. Members never share parameters, data
order or targets, so this is numerically the same experiment as separate runs.

Data budget: the model sees only its n training transitions (a nested prefix of a seed-specific
permutation of training-episode states); its input/target normalization comes from the first
ANCHOR of them, which every dataset size contains. Scores are computed in fixed evaluation units
(noise-free statistics of the training episodes) so they are comparable across settings.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from run import N_VAL, Ensemble, auroc, build_world, contact_gate, noisy_targets, spear

KAPPAS = [0.0, 0.1, 0.25, 0.5, 1.0]
MODELS = ["pe", "pe_beta", "de"]
BETA = 0.5
ANCHOR = 10_000


def mc_mutual_information(mu, lv, samples, chunk=1024):
    """Monte Carlo I(y; member) for the uniform mixture of the members' diagonal Gaussians, i.e. the
    epistemic term of Depeweg et al. (2018): H[mixture] - mean_m H[member m]. Bounded by log(M)."""
    M, N, D = mu.shape
    gen = torch.Generator(device=mu.device).manual_seed(0)
    h_cond = 0.5 * lv.mean(0).sum(-1) + 0.5 * D * np.log(2 * np.pi * np.e)
    out = []
    for c in range(0, N, chunk):
        m_c, lv_c = mu[:, c:c + chunk], lv[:, c:c + chunk]
        sd_c = (0.5 * lv_c).exp()
        n = m_c.shape[1]
        pick = torch.randint(0, M, (samples, n), device=mu.device, generator=gen)
        cols = torch.arange(n, device=mu.device)[None]
        y = m_c[pick, cols] + sd_c[pick, cols] * torch.randn(samples, n, D, device=mu.device, dtype=mu.dtype, generator=gen)
        z = (y[None] - m_c[:, None]) / sd_c[:, None]                      # (M, S, n, D)
        logp_m = (-0.5 * z ** 2 - 0.5 * lv_c[:, None] - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = torch.logsumexp(logp_m, 0) - np.log(M)                     # (S, n)
        out.append(-logp.mean(0) - h_cond[c:c + chunk])
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pair_rank", type=int, default=0)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--mi_samples", type=int, default=0, help="if > 0, also score the Monte Carlo mutual information")
    p.add_argument("--noise", default="gauss", choices=["gauss", "t3", "bimodal"], help="E3 noise shape")
    p.add_argument("--gate", default="joint", choices=["joint", "contact"], help="E3 noise gate")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tag = f"{a.env}_p{a.pair_rank}_n{a.n}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.1f}s] {s}", flush=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")

    W = build_world(np.load(os.path.join(a.data, f"{a.env}_pool.npz")), a.pair_rank)
    if a.gate == "contact":
        W["w_noise"] = contact_gate(a.env, W["X"])
    rs = np.random.default_rng(a.seed)
    perm = rs.permutation(W["tr_pool"])
    assert len(perm) >= a.n, (len(perm), a.n)
    tr, anchor = perm[:a.n], perm[:ANCHOR]
    va = rs.permutation(W["va_pool"])[:N_VAL]
    te = W["te"]

    Z = np.concatenate([W["X"], W["A"]], 1).astype(np.float64)
    z_mu, z_sd = Z[anchor].mean(0), Z[anchor].std(0) + 1e-6
    Zn = (Z - z_mu) / z_sd
    T = lambda arr: torch.as_tensor(arr, dtype=torch.float32, device=dev)
    Zt, Zv, Zte = T(Zn[tr]), T(Zn[va]), T(Zn[te])

    targets = [noisy_targets(W, k, a.seed, a.noise) for k in KAPPAS]           # (Y, sigma) per noise level
    ym = np.stack([Y[anchor].mean(0) for Y, _ in targets])           # model target scaling per noise level
    ys = np.stack([Y[anchor].std(0) + 1e-6 for Y, _ in targets])
    Yt = torch.stack([T((Y[tr] - ym[i]) / ys[i]) for i, (Y, _) in enumerate(targets)])  # (K, n, D)
    Yv = torch.stack([T((Y[va] - ym[i]) / ys[i]) for i, (Y, _) in enumerate(targets)])

    groups = [(ki, m) for ki in range(len(KAPPAS)) for m in MODELS]
    E = a.members
    Mt = len(groups) * E
    kidx = torch.tensor([ki for ki, _ in groups for _ in range(E)], device=dev)
    is_de = torch.tensor([m == "de" for _, m in groups for _ in range(E)], device=dev)
    is_beta = torch.tensor([m == "pe_beta" for _, m in groups for _ in range(E)], device=dev)
    D = Yt.shape[-1]
    model = Ensemble(Mt, Zt.shape[1], D, prob=True).to(dev)  # "de" members ignore their variance head

    def losses(z, y, train):
        mu, lv = model(z)
        mse = ((mu - y) ** 2).mean((1, 2))
        nll_el = (mu - y) ** 2 * torch.exp(-lv) + lv
        if train:
            w = torch.where(is_beta[:, None, None], torch.exp(lv).detach() ** BETA, torch.ones_like(lv))
            nll_el = nll_el * w
        return torch.where(is_de, mse, nll_el.mean((1, 2)))

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = torch.full((Mt,), float("inf"), device=dev)
    prob = (~is_de).float()[:, None, None]
    stale = 0
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, a.n, (Mt, a.bs), device=dev)
        loss = losses(Zt[idx], Yt[kidx[:, None], idx], True).sum()
        loss = loss + 0.01 * ((model.max_lv - model.min_lv) * prob).sum() / D
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                val = losses(Zv.expand(Mt, -1, -1), Yv[kidx], False)
            better = val < best_val
            if better.any():
                stale = 0
                best_val = torch.where(better, val, best_val)
                for k, v in model.state_dict().items():
                    best[k][better] = v[better]
            else:
                stale += 1
            if step % 5000 == 0:
                log(f"step {step}, members improved at last eval: {int(better.sum())}/{Mt}")
            if stale >= 8:
                log(f"early stop at step {step}")
                break
    model.load_state_dict(best)

    with torch.no_grad():
        outs = [model(c.expand(Mt, -1, -1)) for c in torch.split(Zte, 4096)]
        MU = torch.cat([o[0] for o in outs], 1).double()
        LV = torch.cat([o[1] for o in outs], 1).double()
        knn = torch.cat([torch.cdist(c, Zt).topk(a.knn, largest=False).values[:, -1] for c in torch.split(Zte, 2048)])
    e_mu, e_sd = W["e_mu"], W["e_sd"]
    Fn = (W["DF"][te] - e_mu) / e_sd                                   # noise-free change, evaluation units
    gap, noisy = W["gap"][te], W["w_noise"][te] > 0
    ins = ~gap
    GRP = [("in_clean", ins & ~noisy), ("in_noisy", ins & noisy), ("gap_clean", gap & ~noisy), ("gap_noisy", gap & noisy)]

    for g, (ki, mname) in enumerate(groups):
        Y, sigma = targets[ki]
        scale = torch.as_tensor(ys[ki] / e_sd, device=dev)             # model units -> evaluation units
        shift = torch.as_tensor((ym[ki] - e_mu) / e_sd, device=dev)
        with torch.no_grad():
            mu = MU[g * E:(g + 1) * E] * scale + shift
            lv = LV[g * E:(g + 1) * E] + 2 * torch.log(scale)
            mbar = mu.mean(0)
            epi_d = mu.var(0, unbiased=False)
            sig = {"EPI": epi_d.sum(-1), "MaxDisc": (mu[:, None] - mu[None]).norm(dim=-1).amax((0, 1)), "kNN": knn}
            if mname != "de":
                ale_d = lv.exp().mean(0)
                sig["ALE"] = ale_d.sum(-1)
                sig["TOT"] = sig["ALE"] + sig["EPI"]
                sig["MOPO"] = lv.exp().sum(-1).sqrt().amax(0)                  # max_m ||std_m||_2 (official MOPO code)
                sig["MI"] = 0.5 * ((ale_d + epi_d).log() - lv.mean(0)).sum(-1)  # diagonal moment-matched MI proxy
                if a.mi_samples:
                    sig["MIMC"] = mc_mutual_information(mu, lv, a.mi_samples)
            sig = {k: v.double().cpu().numpy() for k, v in sig.items()}
            true_err = ((mbar.cpu().numpy() - Fn) ** 2).sum(-1)
        true_ale = ((sigma[te] / e_sd) ** 2).sum(-1)
        run_id = f"{a.env}_p{a.pair_rank}_k{KAPPAS[ki]}_n{a.n}_{mname}_s{a.seed}"
        m = {"run_id": run_id, "env": a.env, "pair_rank": a.pair_rank, "kappa": KAPPAS[ki], "n": a.n, "model": mname,
             "seed": a.seed, "noise": a.noise, "gate": a.gate, "frac_noisy_test": float(noisy.mean()), "n_noisy_in": int((noisy & ins).sum()), "n_clean_in": int((~noisy & ins).sum()), "steps": step, "best_val": float(best_val[g * E:(g + 1) * E].mean()),
             "noise_feat": int(W["ig"]), "support_feat": int(W["ih"]), "n_train_pool": int(len(W["tr_pool"])),
             "n_test_in": int(ins.sum()), "n_test_gap": int(gap.sum()), "seconds": time.time() - t0,
             "mse_in": float(true_err[ins].mean()), "mse_gap": float(true_err[gap].mean())}
        for k, s in sig.items():
            m[f"auroc_gap_all/{k}"] = auroc(gap, s)
            m[f"auroc_gap_noisy/{k}"] = auroc(gap[noisy], s[noisy])
            m[f"auroc_gap_clean/{k}"] = auroc(gap[~noisy], s[~noisy])
            m[f"auroc_noisy_in/{k}"] = auroc(noisy[ins], s[ins])
            m[f"spear_err_in/{k}"] = spear(s[ins], true_err[ins])
            m[f"spear_err_all/{k}"] = spear(s, true_err)
            m[f"spear_ale_in/{k}"] = spear(s[ins], true_ale[ins])
            for grp, sel in GRP:
                m[f"mean_{grp}/{k}"] = float(s[sel].mean())
        for grp, sel in GRP:
            m[f"mean_{grp}/TRUE_ERR"] = float(true_err[sel].mean())
            m[f"mean_{grp}/TRUE_ALE"] = float(true_ale[sel].mean())
        m = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in m.items()}
        with open(os.path.join(a.out, f"{run_id}.json"), "w") as f:
            json.dump(m, f, indent=1)
        np.savez_compressed(os.path.join(a.out, f"{run_id}.npz"), gap=gap, noisy=noisy, w_noise=W["w_noise"][te],
                            h=W["h"][te], true_err=true_err, true_ale=true_ale,
                            **{f"sig_{k}": v.astype(np.float32) for k, v in sig.items()})
    log(f"done: {len(groups)} ensembles x {E} members, {step} steps")


if __name__ == "__main__":
    main()
