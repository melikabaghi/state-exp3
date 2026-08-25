"""Scaling of the drift lower bound across delay, drift budget, and drifting coordinates.

Theorem 3 says every algorithm, including one handed the exact stale action-loss vector m_t,
suffers

    Omega( sqrt( d E_2 min{1 + log q, T/d} ) )

on the hard family.  The paper checks this at one point.  This study varies all three of d, E_2
and q and asks whether the measured regret collapses onto the predicted scaling once divided by
it.  A collapse is the informative outcome, since the three arguments enter very differently.

The construction is the one in Section 6.  Each of q actions gets a private state, the remaining
K - q actions share one inert state, and each private state is driven by its own independent
Rademacher sign of amplitude eps held for blocks of h = d rounds.

Four policies are run, chosen so the claim "every algorithm" is what is being tested and not just
one learner:

    uniform       ignores everything
    exp3          action-level EXP3 under the delay, which does not see m_t
    state_exp3    the paper's estimator, which also does not see m_t
    greedy_hint   plays argmin of the exact stale vector m_t, the oracle-assisted policy

E_2 = sum_t || c_t - m_t ||_inf^2 is measured on the realised instance rather than predicted, so
the normalisation uses the same quantity the theorem does.

Writes drift_scaling.npz for the figure and prints the table.
"""
from __future__ import annotations

import numpy as np

K = 40
DELAY = (10, 25, 50, 100, 200)
EPS = (0.02, 0.05, 0.10, 0.20)
QQ = (1, 2, 4, 8, 16)
T = 8000
SEEDS = 12


def instance(K, q, d, eps, T, rng):
    """Return per-round action losses c and the stale vector m = c_{t-d}, plus measured E_2.

    Action j < q reaches its own private state; the rest reach one inert state of loss 1/2.
    """
    B = int(np.ceil(T / d))
    sign = rng.choice([-1.0, 1.0], size=(B, q))
    sign[0] = 0.0                      # neutral first block, so c_1 = (1/2)1 and the stale
                                       # vector handed out over rounds t <= d reveals nothing
                                       # about the block the learner is currently acting in
    blk = np.arange(T) // d
    c = np.full((T, K), 0.5)
    c[:, :q] = 0.5 - eps * sign[blk]
    m = np.vstack([np.tile(c[0], (d, 1)), c[:-d]])   # m_t = c_1 for t <= d, the paper's convention
    e2 = float(np.sum(np.max(np.abs(c - m), axis=1) ** 2))
    return c, m, e2


def play(policy, c, m, d, eps, rng, K, q):
    """Return realised expected loss sum_t c_t(A_t) for the named policy."""
    T = c.shape[0]
    if policy == "uniform":
        A = rng.integers(K, size=T)
        return float(c[np.arange(T), A].sum())
    if policy == "greedy_hint":
        A = m.argmin(axis=1)
        return float(c[np.arange(T), A].sum())
    # the two learners share a delayed exponential-weights loop
    eta = np.sqrt(2.0 * np.log(K) / (K * T))
    L = np.zeros(K)
    pend: dict[int, tuple] = {}
    loss = 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            a_r, x_a, X_r, s_r, xs = pend.pop(r)
            if policy == "state_exp3":
                # private state s = a for a < q, inert state s = q otherwise
                col = np.zeros(K)
                if s_r < q:
                    col[s_r] = 1.0
                else:
                    col[q:] = 1.0
                L += col * X_r / max(xs, 1e-12)
            else:
                e = np.zeros(K)
                e[a_r] = X_r / max(x_a, 1e-12)
                L += e
        z = -eta * (L - L.min())
        x = np.exp(z)
        x /= x.sum()
        a = int(rng.choice(K, p=x))
        s = a if a < q else q
        X = rng.binomial(1, c[t, a])
        xs = float(x[s]) if s < q else float(x[q:].sum())
        pend[t] = (a, float(x[a]), X, s, xs)
        loss += c[t, a]
    return loss


def cell(q, d, eps, T, seeds=SEEDS):
    out = {p: [] for p in ("uniform", "exp3", "state_exp3", "greedy_hint")}
    e2s, regs = [], {p: [] for p in out}
    for sd in range(seeds):
        rng = np.random.default_rng(1000 * sd + 7)
        c, m, e2 = instance(K, q, d, eps, T, rng)
        best = float(c.sum(0).min())
        e2s.append(e2)
        for p in out:
            loss = play(p, c, m, d, eps, np.random.default_rng(1000 * sd + 13), K, q)
            regs[p].append(loss - best)
    return np.mean(e2s), {p: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(seeds)))
                          for p, v in regs.items()}


def predicted(d, e2, q, T):
    return np.sqrt(d * e2 * min(1.0 + np.log(max(q, 1)), T / d))


def main() -> None:
    rows = []
    print("sweeping d at eps=0.10, q=4 ; eps at d=50, q=4 ; q at d=50, eps=0.10", flush=True)
    grid = ([(q, d, 0.10) for d in DELAY for q in (4,)]
            + [(4, 50, e) for e in EPS]
            + [(q, 50, 0.10) for q in QQ])
    seen = set()
    for q, d, eps in grid:
        if (q, d, eps) in seen:
            continue
        seen.add((q, d, eps))
        e2, regs = cell(q, d, eps, T)
        pred = predicted(d, e2, q, T)
        rows.append(dict(q=q, d=d, eps=eps, e2=e2, pred=pred,
                         **{p: regs[p][0] for p in regs},
                         **{p + "_se": regs[p][1] for p in regs}))
        print(f"  q={q:2d} d={d:3d} eps={eps:.2f} E2={e2:8.1f} pred={pred:8.1f} | "
              + " ".join(f"{p}={regs[p][0]:7.1f}/{pred:.0f}={regs[p][0]/pred:5.3f}"
                         for p in ("uniform", "exp3", "state_exp3", "greedy_hint")),
              flush=True)

    keys = rows[0].keys()
    np.savez("drift_scaling.npz", **{k: np.array([r[k] for r in rows]) for k in keys})

    print("\nnormalised regret R / sqrt(d E_2 min{1+log q, T/d})")
    for p in ("uniform", "exp3", "state_exp3", "greedy_hint"):
        v = np.array([r[p] / r["pred"] for r in rows])
        print(f"  {p:<12} min {v.min():.3f}  max {v.max():.3f}  "
              f"mean {v.mean():.3f}  spread {v.max()/v.min():.2f}x")
    print("\nA narrow spread across a grid that varies d by 20x, E_2 by 100x and q by 16x is the "
          "collapse the theorem predicts.  The oracle-assisted greedy_hint policy is included "
          "because Theorem 3 claims to survive it.")
    print("Scope: every cell has T/d > 1 + log q, so the sweep exercises the 1 + log q branch of "
          "the minimum only.  The T/d saturation branch needs d comparable to T and is not "
          "probed here.")


if __name__ == "__main__":
    main()
