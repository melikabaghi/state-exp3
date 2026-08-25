"""Scaling of State-EXP3 across effective dimension, delay, and horizon.

The point of this study is not another win over the action-level baseline.  It is whether
Theorem 1 describes the right *dependence*.  Theorem 1 bounds the analysed m = d+1 variant by

    sqrt( 2 (d+1) V log K ),      V = vbar T,

so if the theorem captures the dependence on all three of vbar, d and T, then the normalised
quantity

    R / sqrt( (d+1) vbar T log K )

should stay roughly flat over a grid in (vbar, d, T) rather than trending in any of them.  A raw
regret table cannot show that, and "realised regret stayed under the bound" cannot either, since
the bound is loose by a large constant.

Two variants are run at every cell.  m = d+1 is the analysed one and is the one the normalisation
is entitled to describe.  m = 1 is the practical variant, which carries no guarantee, and its
normalised curve is reported only to show how the two differ as d grows.

Environments hold K and |S| fixed and vary only the overlap between the rows of P, so the raw
state count cannot explain any trend.  Paired seeds: a seed fixes the environment and the draws,
and only the estimator changes.

Writes scaling_3d.npz for the figure and prints the table.
"""
from __future__ import annotations

import numpy as np

K, S = 40, 8
OVERLAP = (0.00, 0.40, 0.70, 0.95)
DELAY = (10, 50, 200)
HORIZON = (2500, 5000, 10000, 20000, 40000)
SEEDS = 5
MC_PLAYS = 4000


def overlap_environment(K: int, S: int, T: int, overlap: float, seed: int):
    """Same construction as enrichment_experiments.py, so the two studies are comparable."""
    rng = np.random.default_rng(seed)
    common = rng.dirichlet(np.ones(S) * 2.0)
    individual = rng.dirichlet(np.ones(S) * 0.25, size=K)
    P = overlap * common[None, :] + (1.0 - overlap) * individual
    base = rng.uniform(0.15, 0.85, size=S)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=S)
    time = np.arange(T)[:, None] / T
    theta = np.clip(base + 0.12 * np.sin(2.0 * np.pi * time + phase), 0.0, 1.0)
    return P, theta


def vbar(P: np.ndarray, rng: np.random.Generator, draws: int = MC_PLAYS) -> float:
    """Monte-Carlo sup over play distributions of v(x) = sum_s q(s)^-1 sum_a x_a P(s|a)^2."""
    best = 1.0
    for _ in range(draws):
        x = rng.dirichlet(np.ones(P.shape[0]) * rng.choice([0.2, 1.0, 5.0]))
        q = x @ P
        best = max(best, float(np.sum(x[:, None] * P ** 2 / np.maximum(q, 1e-300))))
    return best


def run(P, theta, T, d, m, mode, eta, seed):
    """One paired run.  mode in {'state','action'}; m interleaved copies.

    Timing matches Section 3: round r's outcome is consumed at round r + d + 1.
    """
    rng = np.random.default_rng(seed + 10_000)
    c = theta @ P.T
    astar = int(c.sum(0).argmin())
    L = np.zeros((m, K))
    pend: dict[int, tuple] = {}
    loss = 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            i, qs, s_r, X_r, a_r, x_a = pend.pop(r)
            if mode == "state":
                L[i] += P[:, s_r] * X_r / max(qs, 1e-12)
            else:
                e = np.zeros(K)
                e[a_r] = X_r / max(x_a, 1e-12)
                L[i] += e
        i = t % m
        z = -eta * (L[i] - L[i].min())
        x = np.exp(z)
        x /= x.sum()
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(S, p=P[a]))
        X = rng.binomial(1, theta[t, s])
        pend[t] = (i, float(x @ P[:, s]), s, X, a, float(x[a]))
        loss += c[t, a]
    return loss - c[:, astar].sum()


def main() -> None:
    rows = []
    for ov in OVERLAP:
        vb = vbar(overlap_environment(K, S, HORIZON[0], ov, 0)[0], np.random.default_rng(1))
        for d in DELAY:
            for T in HORIZON:
                acc = {key: [] for key in ("s_int", "a_int", "s_one", "a_one")}
                for sd in range(SEEDS):
                    P, theta = overlap_environment(K, S, T, ov, sd)
                    mm = d + 1
                    # each variant is tuned with the budget its own analysis gives it
                    eta_s_int = np.sqrt(2.0 * np.log(K) / (vb * (T / mm)))
                    eta_a_int = np.sqrt(2.0 * np.log(K) / (K * (T / mm)))
                    eta_s_one = np.sqrt(2.0 * np.log(K) / (vb * T))
                    eta_a_one = np.sqrt(2.0 * np.log(K) / (K * T))
                    acc["s_int"].append(run(P, theta, T, d, mm, "state", eta_s_int, sd))
                    acc["a_int"].append(run(P, theta, T, d, mm, "action", eta_a_int, sd))
                    acc["s_one"].append(run(P, theta, T, d, 1, "state", eta_s_one, sd))
                    acc["a_one"].append(run(P, theta, T, d, 1, "action", eta_a_one, sd))
                norm = np.sqrt((d + 1) * vb * T * np.log(K))
                rows.append(
                    dict(
                        overlap=ov, vbar=vb, d=d, T=T, norm=norm,
                        **{k: float(np.mean(v)) for k, v in acc.items()},
                        **{k + "_se": float(np.std(v, ddof=1) / np.sqrt(SEEDS))
                           for k, v in acc.items()},
                    )
                )
                print(
                    f"  ov={ov:.2f} vbar={vb:5.3f} d={d:3d} T={T:5d} | "
                    f"state m=d+1 {np.mean(acc['s_int']):8.1f} "
                    f"action m=d+1 {np.mean(acc['a_int']):8.1f} | "
                    f"state m=1 {np.mean(acc['s_one']):8.1f} "
                    f"action m=1 {np.mean(acc['a_one']):8.1f} | "
                    f"R/bound(state,m=d+1) {np.mean(acc['s_int'])/norm:6.4f}",
                    flush=True,
                )

    keys = rows[0].keys()
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez("scaling_3d.npz", **out)

    print("\nNormalised R / sqrt((d+1) vbar T log K), analysed variant m = d+1")
    print(f"{'vbar':>7}{'d':>6}" + "".join(f"{T:>9}" for T in HORIZON))
    for ov in OVERLAP:
        for d in DELAY:
            sel = [r for r in rows if r["overlap"] == ov and r["d"] == d]
            sel.sort(key=lambda r: r["T"])
            print(f"{sel[0]['vbar']:7.3f}{d:6d}"
                  + "".join(f"{r['s_int']/r['norm']:9.4f}" for r in sel))
    vals = np.array([r["s_int"] / r["norm"] for r in rows])
    print(f"\nacross all {len(rows)} cells: min {vals.min():.4f}  max {vals.max():.4f}  "
          f"ratio {vals.max()/vals.min():.2f}x")
    print("A flat row means the theorem's T dependence is right; a flat column means its "
          "(d+1) and vbar dependence are.")


if __name__ == "__main__":
    main()
