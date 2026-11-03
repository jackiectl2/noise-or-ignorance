"""Benchmark construction and shared model / metric code (used by run_batch.py).

Ground truth available only because the world is a simulator:
  * true noise variance   = the injected process-noise variance sigma^2(x, a)
  * true model error      = || ensemble mean - noise-free simulator state change ||^2
  * support label         = whether (x, a) lies in a region removed from the training data
Everything that defines the benchmark (episode split, noise / support features, thresholds, noise
scale, evaluation units) is computed from training episodes only and frozen before any model is fit.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

TEST_EP_FRAC, VAL_EP_FRAC = 0.2, 0.05  # episode-level split; test and validation episodes are never trained on
TRAIN_Q, GAP_Q = 0.5, 0.75             # train on h <= q50; unvisited test states have h >= q75 (q50-q75 unused)
N_TEST = 10_000                        # per test group (visited / unvisited), both from test episodes
N_VAL = 5_000                          # early-stopping transitions, from validation episodes
GATE_W = 0.05                          # width of the noise ramp, in IQRs of the noise feature


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_world(pool, pair_rank=0):
    """Frozen benchmark construction (independent of noise level, dataset size and seed)."""
    X, A, DF, EP, nv = pool["X"], pool["A"], pool["DF"].astype(np.float64), pool["EP"], int(pool["nv"])
    D = X.shape[1]
    qv = np.arange(D - nv, D)
    rs = np.random.default_rng(12345)
    eps = rs.permutation(np.unique(EP))
    n_te, n_va = int(TEST_EP_FRAC * len(eps)), int(VAL_EP_FRAC * len(eps))
    is_te, is_va = np.isin(EP, eps[:n_te]), np.isin(EP, eps[n_te:n_te + n_va])
    is_tr = ~is_te & ~is_va

    # Two joint-angle coordinates (qpos[1:] minus root height and pitch), ranked by |corr| on training
    # episodes: g decides where the dynamics are noisy, h decides which region is never visited.
    joints = list(range(2, D - nv))
    C = np.corrcoef(X[is_tr][:, joints].T)
    pairs = sorted((abs(C[i, j]), joints[i], joints[j]) for i in range(len(joints)) for j in range(i + 1, len(joints)))
    _, ig, ih = pairs[pair_rank]
    g, h = X[:, ig].astype(np.float64), X[:, ih].astype(np.float64)
    g_med = np.median(g[is_tr])
    g_iqr = np.subtract(*np.percentile(g[is_tr], [75, 25]))
    w_noise = np.clip(2 * sigmoid((g - g_med) / (GATE_W * g_iqr)) - 1, 0, 1)  # exactly 0 on the clean half
    h_lo, h_hi = np.quantile(h[is_tr], [TRAIN_Q, GAP_Q])
    support, gap = h <= h_lo, h >= h_hi

    te_in = rs.choice(np.where(is_te & support)[0], N_TEST, replace=False)
    te_gap = rs.choice(np.where(is_te & gap)[0], N_TEST, replace=False)

    # Evaluation units and noise scale: noise-free state-change statistics of the training episodes.
    ref = is_tr & support
    e_mu, e_sd = DF[ref].mean(0), DF[ref].std(0) + 1e-6
    return dict(X=X, A=A, DF=DF, qv=qv, s=e_sd[qv], e_mu=e_mu, e_sd=e_sd, w_noise=w_noise, gap=gap,
                te=np.concatenate([te_in, te_gap]), tr_pool=np.where(is_tr & support)[0],
                va_pool=np.where(is_va & support)[0], ig=ig, ih=ih, h=h)


def unit_noise(rn, shape, shape_name):
    """Zero-mean unit-variance noise; "gauss" reproduces the main study's draws exactly."""
    if shape_name == "gauss":
        return rn.standard_normal(shape)
    if shape_name == "t3":
        return rn.standard_t(3, shape) / np.sqrt(3.0)
    if shape_name == "bimodal":
        return rn.choice([-1.0, 1.0], shape) * np.sqrt(0.99) + 0.1 * rn.standard_normal(shape)
    raise ValueError(shape_name)


def noisy_targets(W, kappa, seed, shape="gauss"):
    """Observed state changes: noise-free change + noise of variance sigma^2 on velocity coordinates."""
    sigma = np.zeros_like(W["DF"])
    sigma[:, W["qv"]] = kappa * W["w_noise"][:, None] * W["s"][None, :]
    rn = np.random.default_rng(1000 + seed)
    return W["DF"] + sigma * unit_noise(rn, sigma.shape, shape), sigma


def contact_gate(env_id, X):
    """1 where the pre-step state has a floor contact, else 0: a proxy for contact-driven stochasticity."""
    import gymnasium as gym
    import mujoco
    u = gym.make(env_id).unwrapped
    u.reset(seed=0)
    floor = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0
    nq, out = u.model.nq, np.zeros(len(X))
    for i, x in enumerate(X.astype(np.float64)):
        u.set_state(np.concatenate([[0.0], x[: nq - 1]]), x[nq - 1:])
        d = u.data
        out[i] = any(floor in (d.contact[c].geom1, d.contact[c].geom2) for c in range(d.ncon))
    return out


class EnsembleLinear(nn.Module):
    def __init__(self, M, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(M, i, o).clamp(-2, 2) / (2 * i ** 0.5))
        self.b = nn.Parameter(torch.zeros(M, 1, o))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class Ensemble(nn.Module):
    def __init__(self, M, din, dout, prob, hid=200, layers=4):
        super().__init__()
        dims = [din] + [hid] * layers
        self.body = nn.ModuleList(EnsembleLinear(M, a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = EnsembleLinear(M, hid, dout * (2 if prob else 1))
        self.prob = prob
        self.max_lv = nn.Parameter(torch.full((M, 1, dout), 0.5))
        self.min_lv = nn.Parameter(torch.full((M, 1, dout), -10.0))

    def forward(self, x):
        for layer in self.body:
            x = F.silu(layer(x))
        out = self.head(x)
        if not self.prob:
            return out, None
        mu, lv = out.chunk(2, -1)
        lv = self.max_lv - F.softplus(self.max_lv - lv)
        lv = self.min_lv + F.softplus(lv - self.min_lv)
        return mu, lv


def auroc(pos, score):
    ok = np.isfinite(score)
    if ok.sum() == 0 or pos[ok].all() or (~pos[ok]).all():
        return float("nan")
    return float(roc_auc_score(pos[ok], score[ok]))


def spear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() <= 10 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])
