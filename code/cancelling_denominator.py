"""A denominator the learner can certify: count the states instead of predicting them.

Two exact facts about the plug-in chat_t(a) = Phat_t(S_t|a) X_t / qhat_t(S_t):

  (E1) it is play-normalised, sum_a x_t(a) chat_t(a) = X_t, for ANY Phat, because
       sum_a x_t(a) Phat_t(s|a) = qhat_t(s).  Hence the PLAYED bias is identically zero and the
       regret's whole bias charge is the single comparator term |b_t(a*)|.
  (E2) b_t(a) = <Delta_t(.|a), theta_t> - <Phat_t(.|a), u_t>,  u_t = theta_t delta_t / qhat_t,
       delta_t = qhat_t - q_t.  The second term is where rho_t enters and it is the only obstacle.

The observation this file tests: q_t is the distribution of the state the learner actually sees.
Predicting it as x_t^T Phat needs every row of P, at cost sum_a x_a r_a ~ sqrt(K log/t).
**Counting it needs no rows at all.**  Freeze x over a block, count the states in the first half,
and use those frequencies as the denominator in the second half.  The resulting deviation is a
plain multinomial one, sqrt(q(s) log/n), with no K in it and no dependence on P, and it is
certifiable from the counts themselves.

Measured here: the comparator bias and the certifiable bound for both denominators, then regret.
"""
import numpy as np

LOG = np.log


def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T


def run(P, theta, c, d, denom, seed, B=400, alpha=0.01, measure=False):
    """denom in {'predict', 'count'}.  Blocks of length B with x frozen; the first half counts."""
    rng = np.random.default_rng(seed)
    T, S = theta.shape
    K = P.shape[0]
    astar = int(c.sum(0).argmin())
    eta = np.sqrt(2 * LOG(K) / ((S + d) * T))
    L = np.zeros(K); N = np.zeros(K); C = np.zeros((K, S))
    pend, loss = {}, 0.0
    x = np.full(K, 1.0 / K)
    counts = np.zeros(S)
    stats = dict(comp=0.0, cert=0.0, n=0)
    half = B // 2
    for t in range(T):
        r = t - d - 1
        if r in pend:
            x_o, Pm_, den_o, a_o, s_o, X_o, pooled = pend.pop(r)
            if pooled:
                L += Pm_[:, s_o] * X_o / max(den_o[s_o], 1e-300)
            else:
                L[a_o] += X_o / max(x_o[a_o], 1e-300)
        if t % B == 0:                                  # refresh the frozen play at block start
            z = -eta * (L - L.min()); x = np.exp(z); x /= x.sum()
            counts = np.zeros(S)
        Pm = (C + alpha) / (N[:, None] + alpha * S)
        qh = x @ Pm
        phase_pool = (t % B) >= half
        if denom == 'predict' or not phase_pool:
            den = qh
        else:
            n = max(counts.sum(), 1.0)
            den = np.maximum(counts / n, 1e-6)
        if measure and phase_pool:
            q = x @ P
            thr = theta[t]
            b = ((thr * q / den)[None, :] * (Pm - P)).sum(1) - ((thr * (den - q) / den)[None, :] * P).sum(1)
            stats["comp"] += abs(float(b[astar]))
            lg = LOG(2 * K * S * T)
            r1 = np.minimum(2.0, np.sqrt(2 * (S * LOG(2) + LOG(2 * K * T)) / np.maximum(N, 1.0)))
            if denom == 'predict':
                rB = np.minimum(1.0, np.sqrt(2 * Pm * (1 - Pm) * lg / np.maximum(N, 1.0)[:, None])
                                + 3 * lg / np.maximum(N, 1.0)[:, None])
                u = (x[:, None] * rB).sum(0)
            else:                                        # multinomial, no K and no P
                n = max(counts.sum(), 1.0)
                u = np.sqrt(2 * den * LOG(2 * S * T) / n) + 3 * LOG(2 * S * T) / n
            slack = den - u
            rho = np.inf if (slack <= 0).any() else float((u / slack).max())
            stats["cert"] += min(4.0, float(r1.max()) + 2 * rho)
            stats["n"] += 1
        a = int(rng.choice(K, p=x)); s = int(rng.choice(S, p=P[a]))
        N[a] += 1; C[a, s] += 1
        if (t % B) < half:
            counts[s] += 1
        pend[t] = (x, Pm, den, a, s, float(rng.binomial(1, theta[t, s])), phase_pool)
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum()), stats


if __name__ == "__main__":
    K, S, T, d, seeds = 40, 8, 8000, 50, 12
    eta = np.sqrt(2 * LOG(K) / ((K + d) * T))
    budget = 0.5 * eta * (K - 2 * S) * (T // 2)          # only the pooled half is charged
    print(f"K={K} |S|={S} T={T} d={d}, {seeds} seeds, blocks of 400 with x frozen\n")
    print(f"  budget over the pooled half of the horizon   {budget:8.1f}\n")
    print(f"  {'denominator':<12} {'regret':>9} {'comparator bias':>17} {'certifiable':>13}")
    for denom in ('predict', 'count'):
        R, cb, ce = [], [], []
        for sd in range(seeds):
            P, th, c = env(K, S, T, 0.05, np.random.default_rng(sd))
            reg, st = run(P, th, c, d, denom, 10_000 + sd, measure=True)
            R.append(reg); cb.append(st["comp"]); ce.append(st["cert"])
        print(f"  {denom:<12} {np.mean(R):9.1f} {np.mean(cb):11.1f} "
              f"{np.mean(cb)/budget:5.2f}x {np.mean(ce):9.1f} {np.mean(ce)/budget:5.1f}x")
