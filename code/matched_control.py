"""Matched control: does pooling help because of overlap, or because overlap made the task easy?

Study 6 varies the overlap between the rows of P and finds the pooling advantage falling from
206.6 to 1.6 as the mean v_t falls from 2.719 to 1.009.  Read naively that says pooling helps when
v_t is LARGE, which is the opposite of what Theorem 1 predicts, since the bound
sqrt(2(d+1) vbar T log K) improves as vbar falls.

The sweep is confounded.  Raising the overlap also makes every row of P alike, so c_t = P theta_t
becomes nearly constant across actions, the comparator gap collapses, and there is nothing left
for any algorithm to win.  Both regrets go to zero together and the advantage goes with them.

This file removes the confound.  It holds K, |S|, T, d and the *difficulty* fixed and varies only
vbar.  Difficulty is measured by the uniform-play regret

    R_unif = sum_t mean_a c_t(a)  -  min_a sum_t c_t(a),

a property of the instance alone, with no algorithm in it.  For each overlap level the amplitude
of theta is bisected until R_unif hits a common target, so every environment presents the same
comparator gap and differs only in how much the rows of P share.

The construction keeps a contrast direction alive at every overlap.  Rows are

    P(.|a)  =  base  +  spread_a * contrast,

where the base is shared and the per-action coefficients spread_a stay fixed while `overlap`
shrinks the *other* directions in which the rows differ.  So a high-overlap map still separates
the actions along one direction, and theta is aligned with that direction, which is what lets the
comparator gap survive while vbar falls to nearly one.

Prediction under test: at matched difficulty the pooling advantage should GROW as vbar falls.

Writes matched_control.npz and prints the table.
"""
from __future__ import annotations

import numpy as np

K, S, T, D = 40, 8, 10000, 50
OVERLAP = (0.00, 0.35, 0.65, 0.85, 0.95)
SEEDS = 20
TARGET = 900.0          # common uniform-play regret, in the middle of the reachable range


def build_P(overlap: float, seed: int) -> np.ndarray:
    """Rows share `overlap` of their mass but keep a fixed contrast direction at every level."""
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(S) * 2.0)
    idiosyncratic = rng.dirichlet(np.ones(S) * 0.25, size=K)
    # the contrast: a fixed signed direction summing to zero, so rows stay distributions
    contrast = np.zeros(S)
    contrast[0], contrast[1] = 1.0, -1.0
    spread = np.linspace(-1.0, 1.0, K)
    rng.shuffle(spread)
    P = overlap * base[None, :] + (1.0 - overlap) * idiosyncratic
    P = P + 0.12 * spread[:, None] * contrast[None, :]
    P = np.clip(P, 1e-4, None)
    return P / P.sum(1, keepdims=True), contrast


def theta_of(scale: float, contrast: np.ndarray, seed: int) -> np.ndarray:
    """Loss means aligned with the contrast direction, so the gap survives high overlap.

    The coefficient 0.6 + 0.4 sin(.) is deliberately of one sign.  A pure sine integrates to zero
    over the horizon, every action then has the same cumulative loss, the comparator gap is
    exactly zero and there is nothing to match.  The persistent part is what creates a best
    action; the oscillating part is the drift on top of it.
    """
    rng = np.random.default_rng(seed + 555)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(T)[:, None] / T
    wave = 0.6 + 0.4 * np.sin(2.0 * np.pi * t + phase)
    return np.clip(0.5 + scale * wave * contrast[None, :], 0.0, 1.0)


def unif_regret(P: np.ndarray, theta: np.ndarray) -> float:
    c = theta @ P.T
    return float(c.mean(1).sum() - c.sum(0).min())


def match_scale(P, contrast, seed, target=TARGET):
    """Bisect the theta amplitude until the uniform-play regret equals the common target."""
    lo, hi = 1e-4, 0.80
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = unif_regret(P, theta_of(mid, contrast, seed))
        if r < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vbar(P, rng, draws=4000):
    best = 1.0
    for _ in range(draws):
        x = rng.dirichlet(np.ones(K) * rng.choice([0.2, 1.0, 5.0]))
        q = x @ P
        best = max(best, float(np.sum(x[:, None] * P ** 2 / np.maximum(q, 1e-300))))
    return best


def run(P, theta, d, mode, eta, seed):
    rng = np.random.default_rng(seed + 20_000)
    c = theta @ P.T
    astar = int(c.sum(0).argmin())
    L = np.zeros(K)
    pend, loss = {}, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            qs, s_r, X_r, a_r, xa = pend.pop(r)
            if mode == "state":
                L += P[:, s_r] * X_r / max(qs, 1e-12)
            else:
                L[a_r] += X_r / max(xa, 1e-12)
        z = -eta * (L - L.min())
        x = np.exp(z)
        x /= x.sum()
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(S, p=P[a]))
        X = rng.binomial(1, theta[t, s])
        pend[t] = (float(x @ P[:, s]), s, X, a, float(x[a]))
        loss += c[t, a]
    return loss - c[:, astar].sum()


def main() -> None:
    rows = []
    print(f"K={K} |S|={S} T={T} d={D}, uniform-play regret matched to {TARGET:.0f} in every cell\n")
    print(f"{'overlap':>8}{'vbar':>8}{'R_unif':>9}{'gap/round':>11}"
          f"{'state':>9}{'action':>9}{'gain':>9}{'gain %':>8}")
    for ov in OVERLAP:
        st, ac, ru, sc = [], [], [], []
        for sd in range(SEEDS):
            P, contrast = build_P(ov, sd)
            scale = match_scale(P, contrast, sd)
            theta = theta_of(scale, contrast, sd)
            ru.append(unif_regret(P, theta))
            sc.append(scale)
            vb = vbar(P, np.random.default_rng(sd + 3), draws=600)
            eta_s = np.sqrt(2.0 * np.log(K) / (vb * T + D * T))
            eta_a = np.sqrt(2.0 * np.log(K) / (K * T + D * T))
            st.append(run(P, theta, D, "state", eta_s, sd))
            ac.append(run(P, theta, D, "action", eta_a, sd))
        vb = vbar(build_P(ov, 0)[0], np.random.default_rng(11))
        st, ac = np.array(st), np.array(ac)
        gain = ac - st
        rows.append(dict(overlap=ov, vbar=vb, r_unif=float(np.mean(ru)),
                         state=float(st.mean()), action=float(ac.mean()),
                         gain=float(gain.mean()),
                         gain_se=float(gain.std(ddof=1) / np.sqrt(SEEDS)),
                         scale=float(np.mean(sc))))
        print(f"{ov:8.2f}{vb:8.3f}{np.mean(ru):9.1f}{np.mean(ru)/T:11.4f}"
              f"{st.mean():9.1f}{ac.mean():9.1f}"
              f"{gain.mean():6.1f}+-{gain.std(ddof=1)/np.sqrt(SEEDS):<4.1f}"
              f"{100*gain.mean()/ac.mean():7.1f}", flush=True)

    np.savez("matched_control.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    g = np.array([r["gain"] for r in rows])
    v = np.array([r["vbar"] for r in rows])
    ru = np.array([r["r_unif"] for r in rows])
    print(f"\nuniform-play regret across cells: {ru.min():.1f} to {ru.max():.1f} "
          f"(spread {ru.max()/ru.min():.3f}x), so difficulty is held fixed")
    print(f"corr(vbar, absolute gain) = {np.corrcoef(v, g)[0, 1]:+.3f}")
    print(f"gain at highest vbar {v.max():.2f}: {g[np.argmax(v)]:.1f}   "
          f"at lowest vbar {v.min():.2f}: {g[np.argmin(v)]:.1f}")
    print("\nStudy 6 sweeps overlap WITHOUT matching difficulty and finds the gain falling with "
          "vbar.  If the gain here rises as vbar falls, that trend was the confound and not the "
          "mechanism.")


if __name__ == "__main__":
    main()
