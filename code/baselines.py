"""A wider baseline set, to separate which structural assumption produces which benefit.

Most of this appendix compares state-pooled weighting against action-level exponential weights.
That is the necessary comparison but not a sufficient one, since it leaves open whether the gain
comes from the state structure or from some detail of the implementation.  This file runs five
policies on the same paired seeds.

  action      delayed exponential weights, rate tuned for the total delay, the standard baseline
  ix          the same with implicit exploration, chat(a) = 1{A=a} X / (x(a) + gamma), which is
              the usual robustification when importance weights blow up
  naive       state sharing with the WRONG denominator.  It credits P(S_t|a) X_t / pbar(S_t) with
              pbar the unconditional state distribution, ignoring that the play distribution is
              what induces the states.  This isolates the value of the denominator q_t = x_t^T P
              from the value of sharing at all
  beststate   estimate thetahat(s) online from the delayed outcomes, take the best state, and play
              the action most likely to reach it, with epsilon-greedy exploration.  A heuristic a
              practitioner might write down before reading the paper
  state       State-EXP3

Not included, with reasons.  "Run a bandit over states and map the values back through P" is not
a separate method: sum_s P(s|a) 1{S_t=s} X_t / q_t(s) equals P(S_t|a) X_t / q_t(S_t) identically,
so that baseline IS State-EXP3, checked numerically to 4e-15.  The optimism-under-delay methods
need a predictor the base model does not supply, so they appear only in the drift family below,
where the stale vector m_t is available.

Every arm is reported whatever it does.  Where a baseline wins, that is the informative outcome.

Writes baselines.npz and prints the tables.
"""
from __future__ import annotations

import numpy as np

SEEDS = 10
ALPHA_IX = 0.5          # gamma_ix = ALPHA_IX * eta, the standard choice
EPS_GREEDY = 0.08


# ----------------------------------------------------------------- environments
def dirichlet_family(seed, K=40, S=8, T=10000):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1.0, 1.0, size=S)
    theta = np.clip(base + 0.05 * slope * (np.arange(T)[:, None] / T), 0.0, 1.0)
    return P, theta, theta @ P.T


def funnel_family(seed, K=200, S=6, T=10000, categories=12):
    rng = np.random.default_rng(seed)
    depth = np.arange(S)
    cat_tilt = rng.uniform(0.5, 2.6, size=categories)
    cat_profile = np.exp(-cat_tilt[:, None] * depth[None, :])
    cat_profile /= cat_profile.sum(1, keepdims=True)
    assign = rng.integers(categories, size=K)
    quality = rng.lognormal(0.0, 0.6, size=K)
    quality /= quality.mean()
    P = np.empty((K, S))
    for a in range(K):
        prof = cat_profile[assign[a]] * np.exp(0.45 * (quality[a] - 1.0) * depth)
        prof = prof * rng.dirichlet(np.ones(S) * 40.0)
        P[a] = prof / prof.sum()
    t = np.arange(T)[:, None] / T
    base = np.linspace(0.92, 0.28, S)[None, :]
    theta = np.clip(base + 0.06 * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi, S)[None, :]),
                    0.0, 1.0)
    return P, theta, theta @ P.T


def drift_family(seed, K=40, S=17, T=8000, q=8, d=50, eps=0.10):
    """The construction of Theorem 3: q private states, the rest inert."""
    rng = np.random.default_rng(seed)
    P = np.zeros((K, S))
    for a in range(K):
        P[a, a if a < q else q] = 1.0
    B = int(np.ceil(T / d))
    sign = rng.choice([-1.0, 1.0], size=(B, q))
    sign[0] = 0.0                      # neutral first block, matching the repaired construction
    blk = np.arange(T) // d
    theta = np.full((T, S), 0.5)
    theta[:, :q] = 0.5 - eps * sign[blk]
    c = theta @ P.T
    m = np.vstack([np.tile(c[0], (d, 1)), c[:-d]])
    return P, theta, c, m


def spread_family(seed, K=40, S=6, T=10000):
    """The best action mixes over several good states rather than maxing the single best one.

    One decoy action puts the most mass of any action on the single best state and the rest on the
    worst, so a rule that plays argmax_a P(s*|a) targets it and loses.  The optimal action spreads
    over the good states.  theta is near-stationary, so this isolates the misspecification of the
    heuristic from any nonstationarity.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((K, S))
    P[0, 0], P[0, S - 1] = 0.62, 0.38                 # decoy: most mass on the best state
    P[1, :4] = np.array([0.30, 0.28, 0.24, 0.18])     # optimum: spread over the good states
    for a in range(2, K):
        P[a] = rng.dirichlet(np.ones(S) * 0.7)
    t = np.arange(T)[:, None] / T
    base = np.array([0.10, 0.16, 0.22, 0.30, 0.60, 0.95])[None, :]
    theta = np.clip(base + 0.03 * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi, S)[None, :]),
                    0.0, 1.0)
    return P, theta, theta @ P.T


def switching_family(seed, K=40, S=8, T=10000, switches=5):
    """Obliviously nonstationary: the ranking of the states swaps abruptly a few times.

    Still oblivious, so still inside the model of Section 3.  A rule that averages all history
    tracks the time-average rather than the current best.
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 0.6, size=K)
    theta = np.empty((T, S))
    edges = np.linspace(0, T, switches + 1).astype(int)
    levels = np.linspace(0.12, 0.88, S)
    for b in range(switches):
        perm = rng.permutation(S)
        theta[edges[b]:edges[b + 1], :] = levels[perm][None, :]
    return P, theta, theta @ P.T


# ----------------------------------------------------------------- the policies
def run(P, theta, c, d, policy, seed, m_hint=None):
    rng = np.random.default_rng(seed + 90_000)
    T, S = theta.shape
    K = P.shape[0]
    astar = int(c.sum(0).argmin())
    pbar = P.mean(0)
    Vb = {"action": K, "ix": K, "state": S, "naive": S, "beststate": K,
          "optimistic": K}[policy]
    eta = np.sqrt(2.0 * np.log(K) / (Vb * T + d * T))
    gamma_ix = ALPHA_IX * eta
    L = np.zeros(K)
    th_sum, th_cnt = np.zeros(S), np.zeros(S)
    Pcum = np.cumsum(P, axis=1)
    pend, loss = {}, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            x, a_r, s_r, X_r = pend.pop(r)
            if policy in ("action", "optimistic"):
                L[a_r] += X_r / max(x[a_r], 1e-300)
            elif policy == "ix":
                L[a_r] += X_r / (x[a_r] + gamma_ix)
            elif policy == "state":
                L += P[:, s_r] * X_r / max(float(x @ P[:, s_r]), 1e-300)
            elif policy == "naive":
                L += P[:, s_r] * X_r / max(pbar[s_r], 1e-300)
            elif policy == "beststate":
                th_sum[s_r] += X_r
                th_cnt[s_r] += 1.0

        if policy == "beststate":
            thhat = np.where(th_cnt > 0, th_sum / np.maximum(th_cnt, 1.0), 0.5)
            sstar = int(thhat.argmin())
            x = np.full(K, EPS_GREEDY / K)
            x[int(P[:, sstar].argmax())] += 1.0 - EPS_GREEDY
        else:
            score = L.copy()
            if policy == "optimistic" and m_hint is not None:
                score = score + m_hint[t]          # optimistic step on the stale vector
            z = -eta * (score - score.min())
            x = np.exp(z)
            x /= x.sum()

        a = int(np.searchsorted(np.cumsum(x), rng.random()))
        s = int(np.searchsorted(Pcum[a], rng.random()))
        pend[t] = (x, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def growth_study(Ts=(2500, 5000, 10000, 20000, 40000), d=10, seeds=6):
    """How does regret grow in T where the beststate heuristic's assumption is false?

    The empirical worst case over the families above is a worst case over a family set chosen
    here, not over the model.  A rule that commits to one action suffers regret linear in T
    whenever that action is not optimal, and no setting of its exploration rate prevents that.
    Theorem 1 bounds State-EXP3 by sqrt(.) on every oblivious instance.  This measures both.
    """
    out = {}
    for p in ("beststate", "state", "action"):
        out[p] = np.array([float(np.mean([run(*spread_family(sd, T=T), d, p, sd)
                                          for sd in range(seeds)])) for T in Ts])
    lt = np.log(np.array(Ts, dtype=float))
    exponents = {p: float(np.polyfit(lt, np.log(out[p]), 1)[0]) for p in out}
    print("\ngrowth on the spread family, fitted exponent b in R ~ T^b")
    for p, b in exponents.items():
        print(f"  {p:<11} b = {b:.2f}" + ("   <-- linear" if b > 0.9 else ""))
    np.savez("baselines_growth.npz", T=np.array(Ts),
             **{p: out[p] for p in out},
             **{p + "_b": exponents[p] for p in exponents})
    return exponents


def table(name, builder, d, policies, hint=False):
    acc = {p: [] for p in policies}
    for sd in range(SEEDS):
        out = builder(sd)
        P, theta, c = out[0], out[1], out[2]
        mh = out[3] if hint and len(out) > 3 else None
        for p in policies:
            acc[p].append(run(P, theta, c, d, p, sd, mh))
    print(f"\n{name}, d={d}, {SEEDS} paired seeds")
    ref = float(np.mean(acc["action"]))
    for p in policies:
        v = np.array(acc[p])
        mark = "  <-- ours" if p == "state" else ""
        print(f"  {p:<11}{v.mean():9.1f} +- {v.std(ddof=1)/np.sqrt(SEEDS):5.1f}"
              f"   vs action {100*(ref - v.mean())/ref:+7.1f}%{mark}", flush=True)
    return {p: (float(np.mean(acc[p])), float(np.std(acc[p], ddof=1) / np.sqrt(SEEDS)))
            for p in policies}


def main() -> None:
    POL = ("action", "ix", "naive", "beststate", "state")
    res = {}
    for d in (10, 50):
        res[f"dirichlet_d{d}"] = table("Dirichlet family, K=40 |S|=8", dirichlet_family, d, POL)
    for d in (10, 50):
        res[f"funnel_d{d}"] = table("funnel family, K=200 |S|=6", funnel_family, d, POL)
    for d in (10, 50):
        res[f"spread_d{d}"] = table("spread family, best action mixes over good states",
                                    spread_family, d, POL)
    for d in (10, 50):
        res[f"switching_d{d}"] = table("switching family, oblivious but nonstationary",
                                       switching_family, d, POL)
    res["drift_d50"] = table("drift construction of Theorem 3, K=40 q=8", drift_family, 50,
                             POL + ("optimistic",), hint=True)

    flat = {}
    for k, v in res.items():
        for p, (mu, se) in v.items():
            flat[f"{k}__{p}"] = mu
            flat[f"{k}__{p}__se"] = se
    np.savez("baselines.npz", **flat)

    print("\nwhere does State-EXP3 not come first?")
    any_loss = False
    for k, v in res.items():
        best = min(v, key=lambda p: v[p][0])
        if best != "state":
            any_loss = True
            print(f"  {k}: {best} leads at {v[best][0]:.1f} against state {v['state'][0]:.1f}")
    if not any_loss:
        print("  it leads in every family run here")
    growth_study()


if __name__ == "__main__":
    main()
