"""A recommendation funnel, to check that "many actions resolved by few states" is not contrived.

Honest labelling first.  No real data is used here.  There is no licensed interaction log in this
environment, so nothing about this file is fitted to one, and it is NOT a semi-synthetic benchmark
in the usual sense of that phrase.  What it is: an environment whose *structure* is built to the
shape a recommendation funnel has, rather than to the shape that flatters the method.  Every
number below is generated.  The claim it supports is about a regime being natural, not about
performance on a real system.

The shape being imposed:

  action     one of K = 200 recommendable items, far more actions than the earlier studies use
  state      one of |S| = 6 ordered engagement depths, from no-engagement to add-to-cart, which
             is few, because engagement taxonomies in such systems are small and hand-designed
  outcome    a delayed conversion, observed d rounds later, coded as a loss in [0,1]

Three structural facts drive the environment, each chosen because funnels have it:

  1. items fall into a modest number of categories that share an engagement profile, so the rows
     of P overlap heavily even though K is large.  This is the mechanism the paper is about, and
     it arrives here from the catalogue structure rather than being assumed;
  2. item quality is heavy-tailed, so a few items engage often and most rarely do;
  3. the loss decreases with engagement depth and drifts slowly, since what a deep engagement is
     worth changes across a season but not adversarially.

Reported are vbar against K, the favourable-regime condition of Section 4, and the three arms:
state-pooled at m = 1, action-level, and the plug-in with Phat_t estimated online, since a real
system would not be handed P.

Writes funnel.npz and prints the table.
"""
from __future__ import annotations

import numpy as np

K, S = 200, 6
CATEGORIES = 12
T = 20000
DELAYS = (10, 50, 200)
SEEDS = 8
ALPHA = 0.01


def funnel_environment(seed: int):
    """P from catalogue structure, theta from a depth-ordered conversion model."""
    rng = np.random.default_rng(seed)

    # category-level engagement profiles over the six depths, tilted toward shallow engagement
    depth = np.arange(S)
    cat_tilt = rng.uniform(0.5, 2.6, size=CATEGORIES)
    cat_profile = np.exp(-cat_tilt[:, None] * depth[None, :])
    cat_profile /= cat_profile.sum(1, keepdims=True)

    # heavy-tailed item quality shifts mass toward deeper engagement within a category
    assign = rng.integers(CATEGORIES, size=K)
    quality = rng.lognormal(mean=0.0, sigma=0.6, size=K)
    quality /= quality.mean()

    P = np.empty((K, S))
    for a in range(K):
        prof = cat_profile[assign[a]] * np.exp(0.45 * (quality[a] - 1.0) * depth)
        prof = prof * rng.dirichlet(np.ones(S) * 40.0)      # small idiosyncratic jitter
        P[a] = prof / prof.sum()

    # loss falls with engagement depth and drifts slowly, no adversarial component
    t = np.arange(T)[:, None] / T
    base = np.linspace(0.92, 0.28, S)[None, :]
    season = 0.06 * np.sin(2.0 * np.pi * t + rng.uniform(0, 2 * np.pi, size=S)[None, :])
    theta = np.clip(base + season, 0.0, 1.0)
    return P, theta, theta @ P.T, assign


def vbar_of(P, rng, draws=4000):
    best = 1.0
    for _ in range(draws):
        x = rng.dirichlet(np.ones(P.shape[0]) * rng.choice([0.2, 1.0, 5.0]))
        q = x @ P
        best = max(best, float(np.sum(x[:, None] * P ** 2 / np.maximum(q, 1e-300))))
    return best


def draw(cum, u):
    return int(np.searchsorted(cum, u))


def run(P, theta, c, d, mode, seed):
    """mode in {'state','action','plugin'}, m = 1 throughout, the practical variant."""
    rng = np.random.default_rng(seed + 80_000)
    astar = int(c.sum(0).argmin())
    Vb = S if mode != "action" else K
    eta = np.sqrt(2.0 * np.log(K) / (Vb * T + d * T))
    L = np.zeros(K)
    counts = np.zeros((K, S))
    Pcum = np.cumsum(P, axis=1)
    pend, loss = {}, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            x, Pr_, a_r, s_r, X_r = pend.pop(r)
            if mode == "action":
                L[a_r] += X_r / max(x[a_r], 1e-300)
            else:
                q = x @ Pr_
                L += Pr_[:, s_r] * X_r / max(q[s_r], 1e-300)
        z = -eta * (L - L.min())
        x = np.exp(z)
        x /= x.sum()
        Pr_ = ((counts + ALPHA) / (counts.sum(1, keepdims=True) + ALPHA * S)
               if mode == "plugin" else P)
        a = draw(np.cumsum(x), rng.random())
        s = draw(Pcum[a], rng.random())
        counts[a, s] += 1
        pend[t] = (x, Pr_, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def main() -> None:
    rng = np.random.default_rng(0)
    P0, theta0, c0, assign = funnel_environment(0)
    vb = vbar_of(P0, rng)
    occupancy = np.bincount(assign, minlength=CATEGORIES)
    print(f"K={K} items, |S|={S} engagement depths, {CATEGORIES} categories, T={T}")
    print(f"category occupancy {occupancy.min()} to {occupancy.max()} items")
    print(f"vbar = {vb:.3f} against K = {K}, so the effective dimension is {K/vb:.0f}x smaller "
          f"than the action count\n")

    rows = []
    print(f"{'d':>5}{'regime holds':>14}{'state m=1':>12}{'plug-in':>11}{'action':>10}"
          f"{'gain %':>9}{'plug-in gain %':>16}")
    for d in DELAYS:
        st, pl, ac = [], [], []
        for sd in range(SEEDS):
            P, theta, c, _ = funnel_environment(sd)
            st.append(run(P, theta, c, d, "state", sd))
            pl.append(run(P, theta, c, d, "plugin", sd))
            ac.append(run(P, theta, c, d, "action", sd))
        st, pl, ac = np.array(st), np.array(pl), np.array(ac)
        lhs, rhs = vb * (d + 1) * np.log(K), K + d
        rows.append(dict(d=d, vbar=vb, lhs=float(lhs), rhs=float(rhs),
                         state=float(st.mean()), plugin=float(pl.mean()),
                         action=float(ac.mean()),
                         state_se=float(st.std(ddof=1) / np.sqrt(SEEDS)),
                         plugin_se=float(pl.std(ddof=1) / np.sqrt(SEEDS)),
                         action_se=float(ac.std(ddof=1) / np.sqrt(SEEDS))))
        print(f"{d:5d}{('yes' if lhs <= rhs else 'no'):>8}"
              f" {lhs:5.0f}/{rhs:<4.0f}{st.mean():12.1f}{pl.mean():11.1f}{ac.mean():10.1f}"
              f"{100*(ac.mean()-st.mean())/ac.mean():9.1f}"
              f"{100*(ac.mean()-pl.mean())/ac.mean():16.1f}", flush=True)

    np.savez("funnel.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print("\nThe regime condition vbar (d+1) log K <= K + d is the one Theorem 1 needs for the")
    print("ANALYSED m = d+1 variant.  It fails at the delays a funnel actually has, because that")
    print("variant charges the delay multiplicatively.  The arms above run m = 1, which carries")
    print("no guarantee and is the subject of Conjecture 4, so what the table shows is the")
    print("conjectured regime being the realistic one, not a theorem being confirmed.")


if __name__ == "__main__":
    main()
