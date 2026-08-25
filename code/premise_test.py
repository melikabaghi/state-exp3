"""Stage 1 premise test for proxy drift.

Implements prereg/PREREGISTERED_PREMISE_TEST.md (frozen 2026-08-21, amendment A1).

No proposed method appears here. Every algorithm is an existing or trivial baseline, or a
hindsight reference used to bound recoverable value.

Vectorization: algorithms x seeds are carried as a leading (A, S) batch so the Python loop
runs only over rounds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- frozen constants

D_CTX = 5
K_ACT = 5
P_FEAT = K_ACT * D_CTX          # disjoint features, amendment A1
SIGMA_S = 0.3
SIGMA_L = 0.5
LAMBDA = 1.0
EPS_GREEDY = 0.05
GAP_TARGET = 0.5
ENV_SEED = 1234
N_SEEDS = 30
T_HORIZON = 5000
DELAY_MAIN = 50

WINDOWS = (50, 200, 1000, 5000)
DISCOUNTS = (0.9, 0.99, 0.999, 0.9999)

PROXY_ALGOS = (["STATIONARY-PROXY"]
               + [f"WINDOW-{w}" for w in WINDOWS]
               + [f"DISCOUNT-{g}" for g in DISCOUNTS]
               + ["ORACLE-BETA"])
LT_ALGOS = (["LT-ONLY-STAT"]
            + [f"LT-WINDOW-{w}" for w in WINDOWS]
            + [f"LT-DISCOUNT-{g}" for g in DISCOUNTS])
ALL_ALGOS = PROXY_ALGOS + LT_ALGOS

PHI_GRID = (0.0, np.pi / 8, np.pi / 4, np.pi / 2, np.pi)
RHO_GRID = (1.0, 0.3, 0.1, 0.03, 0.01)
K_GRID = (3, 10)
REGIMES = ("ABRUPT", "GRADUAL")


# ---------------------------------------------------------------- environment


@dataclass
class Env:
    k: int
    Theta: np.ndarray       # (k, P_FEAT)
    beta0: np.ndarray       # (k,)
    plane: np.ndarray       # (2, k), plane[0] == beta0
    gap: float
    n_resamples: int


def _sample_contexts(rng, *shape):
    x = rng.standard_normal(shape + (D_CTX,))
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _proj_all(Tb, x):
    """Tb (k,K,D), x (...,D) -> (...,K,k) = Theta @ phi(x,a) for every a."""
    return np.einsum("jkd,...d->...kj", Tb, x)


def _mean_reward_all(Theta, beta, x):
    return np.einsum("...kj,j->...k", _proj_all(Theta.reshape(-1, K_ACT, D_CTX), x), beta)


def build_env(k: int, env_seed: int = ENV_SEED) -> Env:
    """Whiten Theta, equalize the optimal-action gap across k, run the degeneracy guard."""
    rng = np.random.default_rng(env_seed + 7919 * k)
    xn = _sample_contexts(rng, 50_000)
    a_unif = rng.integers(0, K_ACT, size=50_000)
    phi_flat = np.zeros((50_000, P_FEAT))
    idx = (a_unif * D_CTX)[:, None] + np.arange(D_CTX)[None, :]
    np.put_along_axis(phi_flat, idx, xn, axis=1)

    for n_resamples in range(11):
        Theta_raw = rng.standard_normal((k, P_FEAT))
        C = np.atleast_2d(np.cov(phi_flat @ Theta_raw.T, rowvar=False))
        ev, evec = np.linalg.eigh(C)
        Theta_white = (evec @ np.diag(ev ** -0.5) @ evec.T) @ Theta_raw

        beta0 = rng.standard_normal(k)
        beta0 /= np.linalg.norm(beta0)

        m = _mean_reward_all(Theta_white, beta0, xn)
        gap_unit = float(np.mean(m.max(axis=1) - m.mean(axis=1)))
        Theta = (GAP_TARGET / gap_unit) * Theta_white

        v = rng.standard_normal(k)
        v -= (v @ beta0) * beta0
        v /= np.linalg.norm(v)
        plane = np.stack([beta0, v])

        xg = _sample_contexts(rng, 10_000)
        a_start = _mean_reward_all(Theta, beta0, xg).argmax(axis=1)
        ok = np.max(np.bincount(a_start, minlength=K_ACT)) / len(a_start) <= 0.60
        for Phi in (np.pi / 4, np.pi / 2, np.pi):
            be = np.cos(Phi) * plane[0] + np.sin(Phi) * plane[1]
            a_end = _mean_reward_all(Theta, be, xg).argmax(axis=1)
            if np.mean(a_start != a_end) < 0.30:
                ok = False
            if np.max(np.bincount(a_end, minlength=K_ACT)) / len(a_end) > 0.60:
                ok = False
        if ok:
            mm = _mean_reward_all(Theta, beta0, xn)
            gap = float(np.mean(mm.max(axis=1) - mm.mean(axis=1)))
            return Env(k, Theta, beta0, plane, gap, n_resamples)

    raise RuntimeError(f"degeneracy guard failed after 10 resamples for k={k}")


def beta_trajectory(env: Env, regime: str, Phi: float, T: int) -> np.ndarray:
    t = np.arange(T)
    if regime == "STATIONARY" or Phi == 0.0:
        ang = np.zeros(T)
    elif regime == "ABRUPT":
        ang = np.where(t < T // 2, 0.0, Phi)
    elif regime == "GRADUAL":
        ang = Phi * t / max(T - 1, 1)
    elif regime == "RECURRING":
        ang = np.where((t // (T // 4)) % 2 == 0, 0.0, Phi)
    else:
        raise ValueError(regime)
    return np.cos(ang)[:, None] * env.plane[0] + np.sin(ang)[:, None] * env.plane[1]


# ---------------------------------------------------------------- runner


def run_config(env, regime, Phi, rho, delay, T, n_seeds,
               seed_offset=0, theta_known=False):
    """Return (n_algos, n_seeds) long-term dynamic regret per round."""
    k, S_ = env.k, n_seeds
    Tb = env.Theta.reshape(k, K_ACT, D_CTX)
    betas = beta_trajectory(env, regime, Phi, T)

    rng = np.random.default_rng(20260821 + seed_offset)
    X = _sample_contexts(rng, T, S_)
    epsS = SIGMA_S * rng.standard_normal((T, S_, k))
    epsL = SIGMA_L * rng.standard_normal((T, S_))
    labelled = rng.random((T, S_)) < rho
    do_expl = rng.random((T, S_)) < EPS_GREEDY
    expl_a = rng.integers(0, K_ACT, size=(T, S_))

    proj_t = _proj_all(Tb, X)                                   # (T,S,K,k)
    m_true = np.einsum("tskj,tj->tsk", proj_t, betas)           # (T,S,K)
    best = m_true.max(axis=2)

    A_p, A_l = len(PROXY_ALGOS), len(LT_ALGOS)
    ar = np.arange(S_)

    AinvT = np.broadcast_to(np.eye(P_FEAT) / LAMBDA, (A_p, S_, P_FEAT, P_FEAT)).copy()
    BT = np.zeros((A_p, S_, P_FEAT, k))
    Mb = np.zeros((A_p, S_, k, k))
    cb = np.zeros((A_p, S_, k))
    hS = np.zeros((A_p, S_, T, k), dtype=np.float32)
    hL = np.zeros((A_p, S_, T), dtype=np.float32)
    p_win = [WINDOWS[i - 1] if 1 <= i <= 4 else 0 for i in range(A_p)]
    p_gam = np.array([DISCOUNTS[i - 5] if 5 <= i <= 8 else 1.0 for i in range(A_p)])
    is_or = np.array([n == "ORACLE-BETA" for n in PROXY_ALGOS])

    ML = np.zeros((A_l, S_, P_FEAT, P_FEAT))
    cL = np.zeros((A_l, S_, P_FEAT))
    hX = np.zeros((A_l, S_, T, D_CTX), dtype=np.float32)
    hA = np.zeros((A_l, S_, T), dtype=np.int8)
    hLL = np.zeros((A_l, S_, T), dtype=np.float32)
    l_win = [WINDOWS[i - 1] if 1 <= i <= 4 else 0 for i in range(A_l)]
    l_gam = np.array([DISCOUNTS[i - 5] if 5 <= i <= 8 else 1.0 for i in range(A_l)])

    reg_p = np.zeros((A_p, S_))
    reg_l = np.zeros((A_l, S_))
    eye_k, eye_p = np.eye(k), np.eye(P_FEAT)
    dr = np.arange(D_CTX)

    for t in range(T):
        x = X[t]
        proj_true = proj_t[t]                                   # (S,K,k)

        # decay (discount family) every round
        Mb *= p_gam[:, None, None, None]
        cb *= p_gam[:, None, None]
        ML *= l_gam[:, None, None, None]
        cL *= l_gam[:, None, None]

        # ---- proxy family action
        if theta_known:
            proj = np.broadcast_to(proj_true, (A_p, S_, K_ACT, k))
        else:
            u = np.einsum("aspkd,sd->aspk",
                          AinvT.reshape(A_p, S_, P_FEAT, K_ACT, D_CTX), x)
            proj = np.einsum("aspj,aspk->askj", BT, u)
        bhat = np.linalg.solve(Mb + LAMBDA * eye_k, cb[..., None])[..., 0]
        bhat = np.where(is_or[:, None, None], betas[t], bhat)
        act_p = np.einsum("askj,asj->ask", proj, bhat).argmax(axis=2)
        act_p = np.where(do_expl[t], expl_a[t], act_p)

        # ---- lt-only family action
        what = np.linalg.solve(ML + LAMBDA * eye_p, cL[..., None])[..., 0]   # (A,S,P)
        act_l = np.einsum("askd,sd->ask",
                          what.reshape(A_l, S_, K_ACT, D_CTX), x).argmax(axis=2)
        act_l = np.where(do_expl[t], expl_a[t], act_l)

        # ---- regret
        reg_p += best[t][None, :] - m_true[t][ar[None, :], act_p]
        reg_l += best[t][None, :] - m_true[t][ar[None, :], act_l]

        # ---- observations along each family's own trajectory
        Sp = proj_true[ar[None, :], act_p] + epsS[t][None, :, :]      # (A_p,S,k)
        hS[:, :, t, :] = Sp
        hL[:, :, t] = Sp @ betas[t] + epsL[t][None, :]

        Sl = proj_true[ar[None, :], act_l] + epsS[t][None, :, :]      # (A_l,S,k)
        hX[:, :, t, :] = x[None, :, :]
        hA[:, :, t] = act_l
        hLL[:, :, t] = Sl @ betas[t] + epsL[t][None, :]

        # ---- Theta ridge (abundant stream, Sherman-Morrison, every round)
        if not theta_known:
            v = np.zeros((A_p, S_, P_FEAT))
            np.put_along_axis(v, (act_p * D_CTX)[:, :, None] + dr,
                              np.broadcast_to(x, (A_p, S_, D_CTX)), axis=2)
            Av = np.einsum("aspq,asq->asp", AinvT, v)
            den = 1.0 + np.einsum("asp,asp->as", v, Av)
            AinvT -= np.einsum("asp,asq->aspq", Av, Av) / den[:, :, None, None]
            BT += np.einsum("asp,asj->aspj", v, Sp)

        # ---- delayed long-term labels
        src = t - delay
        if src < 0:
            continue
        nm = labelled[src]
        if nm.any():
            Ms = hS[:, :, src, :].astype(np.float64)
            Ls = hL[:, :, src].astype(np.float64)
            Mb += nm[None, :, None, None] * np.einsum("asj,asl->asjl", Ms, Ms)
            cb += nm[None, :, None] * Ms * Ls[:, :, None]

            phiL = np.zeros((A_l, S_, P_FEAT))
            np.put_along_axis(phiL, (hA[:, :, src].astype(np.int64) * D_CTX)[:, :, None] + dr,
                              hX[:, :, src, :].astype(np.float64), axis=2)
            Ll = hLL[:, :, src].astype(np.float64)
            ML += nm[None, :, None, None] * np.einsum("asp,asq->aspq", phiL, phiL)
            cL += nm[None, :, None] * phiL * Ll[:, :, None]

        # ---- window expiry
        for i, w in enumerate(p_win):
            old = src - w
            if w and old >= 0 and labelled[old].any():
                om = labelled[old]
                Mo = hS[i, :, old, :].astype(np.float64)
                Lo = hL[i, :, old].astype(np.float64)
                Mb[i] -= om[:, None, None] * np.einsum("sj,sl->sjl", Mo, Mo)
                cb[i] -= om[:, None] * Mo * Lo[:, None]
        for i, w in enumerate(l_win):
            old = src - w
            if w and old >= 0 and labelled[old].any():
                om = labelled[old]
                po = np.zeros((S_, P_FEAT))
                np.put_along_axis(po, (hA[i, :, old].astype(np.int64) * D_CTX)[:, None] + dr,
                                  hX[i, :, old, :].astype(np.float64), axis=1)
                Lo = hLL[i, :, old].astype(np.float64)
                ML[i] -= om[:, None, None] * np.einsum("sp,sq->spq", po, po)
                cL[i] -= om[:, None] * po * Lo[:, None]

    return np.concatenate([reg_p, reg_l], axis=0) / T


# ---------------------------------------------------------------- sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    envs = {k: build_env(k) for k in K_GRID}
    meta = {"gap_target": GAP_TARGET, "envs": {}}
    for k, e in envs.items():
        rel = abs(e.gap - GAP_TARGET) / GAP_TARGET
        meta["envs"][str(k)] = {"gap": e.gap, "gap_rel_err": rel,
                                "resamples": e.n_resamples}
        if rel > 0.02:
            raise RuntimeError(f"prereg 2.5 SNR check failed k={k}: gap={e.gap:.4f}")
    print(json.dumps(meta, indent=2), flush=True)

    if args.mode == "smoke":
        cells = [(3, "ABRUPT", 0.0, 0.3), (3, "ABRUPT", float(np.pi), 0.3)]
        n_seeds = 5
    else:
        cells = [(k, r, float(p), rho) for k in K_GRID for r in REGIMES
                 for p in PHI_GRID for rho in RHO_GRID]
        n_seeds = N_SEEDS

    rows, t0 = [], time.time()
    for i, (k, regime, Phi, rho) in enumerate(cells):
        ts = time.time()
        reg = run_config(envs[k], regime, Phi, rho, DELAY_MAIN, T_HORIZON, n_seeds)
        for ai, name in enumerate(ALL_ALGOS):
            for s in range(n_seeds):
                rows.append({"k": k, "regime": regime, "Phi": Phi, "rho": rho,
                             "algo": name, "seed": s, "regret": float(reg[ai, s])})
        print(f"[{i+1}/{len(cells)}] k={k} {regime} Phi={Phi:.3f} rho={rho} "
              f"({time.time()-ts:.1f}s)", flush=True)

    elapsed = time.time() - t0
    with open(out / f"raw_regret_{args.mode}.json", "w") as f:
        json.dump({"meta": meta, "elapsed_s": elapsed, "n_seeds": n_seeds,
                   "T": T_HORIZON, "delay": DELAY_MAIN, "rows": rows}, f)
    print(f"\nwrote {out}/raw_regret_{args.mode}.json  ({elapsed:.1f}s)")

    if args.mode == "smoke":
        n_full = len(K_GRID) * len(REGIMES) * len(PHI_GRID) * len(RHO_GRID)
        proj = elapsed / len(cells) * n_full * (N_SEEDS / n_seeds)
        print(f"projected full-grid runtime: {proj/3600:.2f} h (prereg 7.4 requires < 4 h)")


if __name__ == "__main__":
    main()
