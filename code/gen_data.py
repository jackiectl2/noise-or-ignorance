"""Collect state-action pools from deterministic MuJoCo dynamics.

For every visited (x, a) we store the noise-free state change f(x, a) - x (float64). Stochasticity is injected
later (run.py), per transition, so the state-action distribution is identical across noise levels
and the exact mean function f is known for every sample.

State x = (qpos[1:], qvel): the root x-position is dropped (the dynamics are translation invariant).
"""
import argparse
import os

import gymnasium as gym
import numpy as np


def state_of(d):
    return np.concatenate([d.qpos[1:], d.qvel]).astype(np.float64)


def collect(env_id, n_steps, seed, max_ep_len):
    env = gym.make(env_id)
    env.reset(seed=seed)
    u = env.unwrapped
    rng = np.random.default_rng(seed)
    lo, hi = env.action_space.low, env.action_space.high
    X, A, FX, EP = [], [], [], []
    ep = 0
    while len(X) < n_steps:
        env.reset(seed=int(rng.integers(2**31 - 1)))
        repeat = int(rng.integers(1, 11))  # sticky random actions broaden state coverage
        a = rng.uniform(lo, hi)
        for k in range(max_ep_len):
            if k % repeat == 0:
                a = rng.uniform(lo, hi)
            x = state_of(u.data)
            u.do_simulation(a, u.frame_skip)
            X.append(x)
            A.append(a.copy())
            FX.append(state_of(u.data))
            EP.append(ep)
            if hasattr(u, "is_healthy") and not u.is_healthy:
                break
        ep += 1
    n = n_steps
    X64, FX64 = np.asarray(X[:n]), np.asarray(FX[:n])
    # The noise-free state change is differenced in float64 before any rounding.
    return (X64.astype(np.float32), np.asarray(A[:n], np.float32), FX64 - X64,
            np.asarray(EP[:n], np.int32), u.model.nv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--n", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_ep_len", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    X, A, DF, EP, nv = collect(args.env, args.n, args.seed, args.max_ep_len)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.env}_pool.npz")
    np.savez(path, X=X, A=A, DF=DF, EP=EP, nv=nv)
    print(f"{args.env}: {len(X)} transitions, {EP.max() + 1} episodes, state dim {X.shape[1]}, "
          f"nv {nv}, action dim {A.shape[1]} -> {path}")
