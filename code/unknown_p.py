"""State-EXP3 when the action-to-state map is not given.

The pairs (A_t, S_t) arrive immediately, with no delay, so P can be estimated online from the
learner's own play.  At round t the learner holds

    Phat_t(s | a) = (N_a(s) + alpha) / (N_a + alpha |S|)          counts from rounds r < t

with add-alpha smoothing, which keeps every entry strictly positive and therefore keeps
qhat_t = x_t^T Phat_t strictly positive.  It plays x_t mixed with the uniform distribution at rate
gamma, and when X_t arrives it forms the plug-in estimate

    chat_t(a) = Phat_t(S_t | a) X_t / qhat_t(S_t).

Both Phat_t and x_t are measurable before the action, so the estimate is well defined; it is no
longer unbiased, and the question this file answers is how much that costs.

Baselines: the same algorithm with P given, and action-level delayed EXP3, which needs no P.
"""
import numpy as np


def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T


def run(P, theta, c, d, m, mode, gamma, seed, alpha=0.01):
    """mode in {'known', 'unknown', 'action'}.  m interleaved copies."""
    rng = np.random.default_rng(seed)
    T, S = theta.shape
    K = P.shape[0]
    astar = int(c.sum(0).argmin())
    Ti = T / m
    Vb = S if mode != 'action' else K
    eta = np.sqrt(2.0 * np.log(K) / (Vb * Ti + (0.0 if m == d + 1 else d * Ti)))
    L = np.zeros((m, K))
    counts = np.zeros((K, S))                      # immediate (A_r, S_r), no delay
    pend, loss = {}, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:                              # round r's outcome is usable from r+d+1
            i, x, Pr_, a_r, s_r, X_r = pend.pop(r)
            if mode == 'action':
                L[i, a_r] += X_r / max(x[a_r], 1e-300)
            else:
                q = x @ Pr_
                L[i] += Pr_[:, s_r] * X_r / max(q[s_r], 1e-300)
        i = t % m
        z = -eta * (L[i] - L[i].min())
        x = np.exp(z); x /= x.sum()
        if gamma > 0:
            x = (1.0 - gamma) * x + gamma / K
        if mode == 'unknown':
            Pr_ = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * S)
        else:
            Pr_ = P
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(S, p=P[a]))
        counts[a, s] += 1                          # observed at once, before the outcome
        pend[t] = (i, x, Pr_, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def row_error(P, theta, d, gamma, seed, alpha=0.01):
    """Realised max_a ||Phat(.|a) - P(.|a)||_1 at the end of a run, for context."""
    rng = np.random.default_rng(seed)
    T, S = theta.shape
    K = P.shape[0]
    counts = np.zeros((K, S))
    for _ in range(T):
        x = np.full(K, 1.0 / K) if gamma >= 1 else None
        a = int(rng.choice(K, p=x)) if x is not None else int(rng.integers(K))
        counts[a, int(rng.choice(S, p=P[a]))] += 1
    Ph = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * S)
    return float(np.abs(Ph - P).sum(1).max())


if __name__ == "__main__":
    T, d, S, V, seeds = 10000, 50, 4, 0.05, 60
    print(f"|S|={S} T={T} d={d} drift={V}, {seeds} paired seeds, m=1 (the practical variant)\n")
    print(f"  {'K':>4} {'gamma':>7} {'known P':>10} {'unknown P':>11} {'action':>9} "
          f"{'unknown-action':>15}")
    for K in (8, 20, 40, 80):
        for gamma in (0.0, 0.02, 0.10):
            kn, un, ac = [], [], []
            for sd in range(seeds):
                P, theta, c = env(K, S, T, V, np.random.default_rng(sd))
                kn.append(run(P, theta, c, d, 1, 'known', gamma, 10_000 + sd))
                un.append(run(P, theta, c, d, 1, 'unknown', gamma, 10_000 + sd))
                ac.append(run(P, theta, c, d, 1, 'action', gamma, 10_000 + sd))
            kn, un, ac = map(np.asarray, (kn, un, ac))
            df = ac - un
            print(f"  {K:4d} {gamma:7.2f} {kn.mean():10.1f} {un.mean():11.1f} {ac.mean():9.1f} "
                  f"{df.mean():8.1f} +- {df.std(ddof=1)/np.sqrt(seeds):4.1f}")
