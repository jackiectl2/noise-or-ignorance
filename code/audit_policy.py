"""Audit extensions E1 and E2 (frozen in a pre-registered plan): one-step and rollout audit on policy data.

The dataset comes from offline_mbrl.py --phase collect, where noise propagates into later states. The reference for
every logged (x, a) is the noise-free simulator step started from x, so all metrics compare with the error of the
ensemble mean against noise-free dynamics, as in the main audit.
"""
import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from offline_mbrl import gate_params
from run import Ensemble, auroc, spear

HORIZON = 5
MIN_CLASS = 500  # frozen in the pre-registered plan: fewer test rows in either gate class makes the noise claims untestable


def episodes(S, S2, D):
    """Episode ids recovered from discontinuities in the replay stream."""
    brk = (D[:-1] > 0) | (np.abs(S[1:] - S2[:-1]).max(1) > 0)
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)


class Sim:
    """Noise-free simulator restarted from a logged state; the root x-position is translation invariant."""

    def __init__(self, env_id, scale):
        self.u = gym.make(env_id).unwrapped
        self.u.reset(seed=0)
        self.nq, self.scale = self.u.model.nq, scale

    def set(self, x):
        q = np.concatenate([[0.0], x[: self.nq - 1]])
        self.u.set_state(q, x[self.nq - 1 :])

    def get(self):
        d = self.u.data
        return np.concatenate([d.qpos[1:], d.qvel]).astype(np.float64)

    def step(self, a):
        self.u.do_simulation(a * self.scale, self.u.frame_skip)
        return self.get()


def signals(mu, lv):
    """Same definitions as run_batch.py. mu, lv: (M, N, D) in evaluation units."""
    epi = mu.var(0, unbiased=False)
    ale = lv.exp().mean(0)
    out = {"EPI": epi.sum(-1), "ALE": ale.sum(-1), "MaxDisc": (mu[:, None] - mu[None]).norm(dim=-1).amax((0, 1)),
           "MOPO": lv.exp().sum(-1).sqrt().amax(0), "MI": 0.5 * ((ale + epi).log() - lv.mean(0)).sum(-1)}
    out["TOT"] = out["ALE"] + out["EPI"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data", default="data_v2")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--n_test", type=int, default=10_000)
    p.add_argument("--n_roll", type=int, default=5_000)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rs = np.random.default_rng(a.seed + 77)
    dev = torch.device("cuda")
    t0 = time.time()

    d = np.load(a.dataset)
    S, Ac, S2, W_log = d["S64"], d["A64"], d["S2_64"], d["W"]  # full precision, and the gate actually applied
    Dn = d["D"].astype(np.float64)
    EP = episodes(S, S2, Dn)
    ids = rs.permutation(np.unique(EP))
    n_te, n_va = int(0.2 * len(ids)), int(0.05 * len(ids))
    is_te, is_va = np.isin(EP, ids[:n_te]), np.isin(EP, ids[n_te:n_te + n_va])
    tr, va, te_all = np.where(~is_te & ~is_va)[0], np.where(is_va)[0], np.where(is_te)[0]

    G = gate_params(a.env, a.data)
    env0 = gym.make(a.env)
    sim = Sim(a.env, float(env0.action_space.high[0]))

    def f_change(rows):
        out = np.zeros((len(rows), S.shape[1]))
        for i, r in enumerate(rows):
            sim.set(S[r])
            out[i] = sim.step(Ac[r]) - S[r]
        return out

    # Evaluation units come from noise-free changes on training rows.
    ref = f_change(rs.choice(tr, 20_000, replace=False))
    e_mu, e_sd = ref.mean(0), ref.std(0) + 1e-6
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    Z = np.concatenate([S, Ac], 1)
    Y = S2 - S
    zm, zs, ym, ys = Z[tr].mean(0), Z[tr].std(0) + 1e-6, Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Zn, Yn = T((Z - zm) / zs), T((Y - ym) / ys)
    M = a.members
    model = Ensemble(M, Z.shape[1], Y.shape[1], prob=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va[rs.permutation(len(va))[:5000]], device=dev)

    def nll(z, y):
        mu, lv = model(z)
        return ((mu - y) ** 2 * torch.exp(-lv) + lv).mean((1, 2))

    best = {k: v.clone() for k, v in model.state_dict().items()}
    best_val, stale = torch.full((M,), float("inf"), device=dev), 0
    for step in range(1, a.steps + 1):
        i = tr_t[torch.randint(0, len(tr_t), (M, 1024), device=dev)]
        loss = nll(Zn[i], Yn[i]).sum() + 0.01 * (model.max_lv - model.min_lv).sum() / Y.shape[1]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                val = nll(Zn[va_t].expand(M, -1, -1), Yn[va_t].expand(M, -1, -1))
            better = val < best_val
            stale = 0 if better.any() else stale + 1
            best_val = torch.where(better, val, best_val)
            for k, v in model.state_dict().items():
                best[k][better] = v[better]
            if stale >= 8:
                break
    model.load_state_dict(best)

    scale, shift = torch.as_tensor(ys / e_sd, device=dev), torch.as_tensor((ym - e_mu) / e_sd, device=dev)
    zm_t, zs_t, ym_t, ys_t = (torch.as_tensor(v, device=dev) for v in (zm, zs, ym, ys))

    @torch.no_grad()
    def query(z_raw):
        """z_raw: float64 raw state-action tensor. Returns evaluation-unit member means and log-variances."""
        out = [model((c.float()).expand(M, -1, -1)) for c in torch.split((z_raw - zm_t) / zs_t, 4096)]
        mu = torch.cat([o[0] for o in out], 1).double() * scale + shift
        lv = torch.cat([o[1] for o in out], 1).double() + 2 * torch.log(scale)
        return mu, lv

    @torch.no_grad()
    def knn(z_raw):
        ztr = Zn[tr_t]
        return torch.cat([torch.cdist(c.float(), ztr).topk(10, largest=False).values[:, -1]
                          for c in torch.split((z_raw - zm_t) / zs_t, 1024)]).double()

    # E1: one-step audit.
    te = rs.choice(te_all, min(a.n_test, len(te_all)), replace=False)
    z_te = torch.as_tensor(Z[te], device=dev)
    mu, lv = query(z_te)
    Fn = (f_change(te) - e_mu) / e_sd
    mbar = mu.mean(0)
    err = ((mbar.cpu().numpy() - Fn) ** 2).sum(-1)
    sig = signals(mu, lv)
    sig["kNN"] = knn(z_te)
    sig = {k: v.cpu().numpy() for k, v in sig.items()}
    w = W_log[te]
    noisy = w > 0
    qv = np.arange(S.shape[1] - G["nv"], S.shape[1])
    true_ale = np.zeros(len(te))
    true_ale += ((a.kappa * w[:, None] * G["s"][None]) ** 2 / e_sd[qv][None] ** 2).sum(-1)
    # Replay check: rows the gate left noise-free must reproduce the logged next state under Sim.
    clean_rows = te[w == 0]
    replay = np.abs((f_change(clean_rows[:2000]) - Y[clean_rows[:2000]]) / e_sd).max(1) if len(clean_rows) else np.zeros(1)
    n_noisy, n_clean = int(noisy.sum()), int((~noisy).sum())
    m = {"env": a.env, "n_noisy": n_noisy, "n_clean": n_clean, "testable": bool(min(n_noisy, n_clean) >= MIN_CLASS),
         "replay_median": float(np.median(replay)), "replay_p99": float(np.quantile(replay, 0.99)),
         "breaks_term": int((Dn[:-1] > 0).sum()), "breaks_total": int(EP.max()), "kappa": a.kappa, "seed": a.seed, "n_train": int(len(tr)), "n_test": int(len(te)), "steps": step,
         "episodes": int(len(ids)), "frac_noisy": float(noisy.mean()), "mean_err": float(err.mean()),
         "mean_err_noisy": float(err[noisy].mean()) if noisy.any() else None,
         "mean_err_clean": float(err[~noisy].mean()) if (~noisy).any() else None,
         "ratio_ale_target": float(np.median(sig["ALE"] / (true_ale + err + sig["EPI"]))),
         "mean_err_over_epi": float(err.mean() / sig["EPI"].mean())}
    k_top = int(0.2 * len(te))
    top = lambda s: np.argsort(-s)[:k_top]
    m["top20_noisy/ERR"] = float(noisy[top(err)].mean())
    keep = int(0.8 * len(te))
    for k, s in sig.items():
        m[f"spear/{k}"] = spear(s, err)
        m[f"retained/{k}"] = float(err[np.argsort(s)[:keep]].mean() / err.mean())
        m[f"auroc_noisy/{k}"] = auroc(noisy, s)
        m[f"top20_noisy/{k}"] = float(noisy[top(s)].mean())
    m["retained/ORACLE"] = float(err[np.argsort(err)[:keep]].mean() / err.mean())

    # E2: open-loop rollouts with logged actions; rows must have HORIZON further rows in the same episode.
    ok = te_all[te_all + HORIZON - 1 < len(S)]  # same eligible rows at every h
    ok = ok[EP[ok + HORIZON - 1] == EP[ok]]
    st = rs.choice(ok, min(a.n_roll, len(ok)), replace=False)
    x_model = torch.as_tensor(S[st], device=dev)
    x_true = S[st].copy()
    cum = {k: 0.0 for k in ("EPI", "ALE", "TOT", "MOPO", "MaxDisc", "MI")}
    for h in range(1, HORIZON + 1):
        act = Ac[st + h - 1]
        mu, lv = query(torch.cat([x_model, torch.as_tensor(act, device=dev)], 1))
        s_h = signals(mu, lv)
        cum = {k: cum[k] + s_h[k] for k in cum}
        x_model = x_model + (mu.mean(0) * torch.as_tensor(e_sd, device=dev) + torch.as_tensor(e_mu, device=dev))
        for i in range(len(st)):
            sim.set(x_true[i])
            x_true[i] = sim.step(act[i])
        herr = (((x_model.cpu().numpy() - x_true) / e_sd) ** 2).sum(-1)
        fin = np.isfinite(herr)
        m[f"roll_valid/h{h}"] = float(fin.mean())
        m[f"roll_median_err/h{h}"] = float(np.median(herr[fin]))
        for k in cum:
            m[f"roll_spear/{k}/h{h}"] = spear(cum[k].cpu().numpy(), herr)
        c_epi = cum["EPI"].cpu().numpy()[fin]
        m[f"roll_err_over_epi/h{h}"] = float(herr[fin].mean() / c_epi.mean()) if fin.any() and c_epi.mean() > 0 else None
    m["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.env}_k{a.kappa}_s{a.seed}"
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in m.items()}, f, indent=1)
    np.savez_compressed(os.path.join(a.out, f"{tag}.npz"), err=err, noisy=noisy, w=w, true_ale=true_ale,
                        **{f"sig_{k}": v.astype(np.float32) for k, v in sig.items()})
    print(tag, "done", round(m["seconds"]), flush=True)


if __name__ == "__main__":
    main()
