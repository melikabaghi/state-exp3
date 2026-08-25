"""Verification of the proposed State-Delayed-EXP3.

Checks, in order:
  1. the state-based estimator  chat_t(a) = P(S_t|a) X_t / q_t(S_t)  is unbiased for c_t(a)
  2. its weighted second moment  sum_a x_t(a) E[chat_t(a)^2]  is bounded by |S|, not K
  3. interleaved State-EXP3 versus interleaved action-level EXP3 under delay d
"""
import numpy as np

def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T          # c[t, a] = sum_s P[a,s] theta[t,s]

# ---------------------------------------------------------------- 1 and 2
def moments(K, S, reps, seed=0):
    rng = np.random.default_rng(seed)
    P, theta, c = env(K, S, 1, 0.0, rng)
    th = theta[0]
    x = rng.dirichlet(np.ones(K) * 0.3)     # a deliberately lopsided play distribution
    q = x @ P                               # induced state distribution
    est = np.zeros((reps, K))
    for r in range(reps):
        a = rng.choice(K, p=x)
        s = rng.choice(S, p=P[a])
        Xt = rng.binomial(1, th[s])
        est[r] = P[:, s] * Xt / q[s]
    bias = np.abs(est.mean(0) - c[0]).max()
    se = (est.std(0, ddof=1) / np.sqrt(reps)).max()
    second_state = float(np.sum(x * (est ** 2).mean(0)))
    # the action-level estimator, for contrast
    est_a = np.zeros((reps, K))
    for r in range(reps):
        a = rng.choice(K, p=x)
        s = rng.choice(S, p=P[a])
        Xt = rng.binomial(1, th[s])
        e = np.zeros(K); e[a] = Xt / x[a]
        est_a[r] = e
    second_action = float(np.sum(x * (est_a ** 2).mean(0)))
    return bias, se, second_state, second_action

# ---------------------------------------------------------------- 3
def run(K, S, T, d, V, seed, mode):
    """One seed. mode in {'state', 'action'}. d+1 interleaved EXP3 copies."""
    rng = np.random.default_rng(seed)
    P, theta, c = env(K, S, T, V, rng)
    astar = int(c.sum(0).argmin())
    M = d + 1
    Ti = T / M
    Vbound = S if mode == 'state' else K
    eta = np.sqrt(2.0 * np.log(K) / (Vbound * Ti))
    L = np.zeros((M, K))                      # cumulative loss estimates, one row per copy
    pend = {}                                 # round -> (copy, x, A, S, X)
    loss = 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:                         # round r's outcome is usable from r+d+1
            i, x, a_r, s_r, X_r = pend.pop(r)
            if mode == 'state':
                q = x @ P
                chat = P[:, s_r] * X_r / max(q[s_r], 1e-12)
            else:
                chat = np.zeros(K); chat[a_r] = X_r / max(x[a_r], 1e-12)
            L[i] += chat
        i = t % M
        z = -eta * (L[i] - L[i].min())
        x = np.exp(z); x /= x.sum()
        a = rng.choice(K, p=x)
        s = rng.choice(S, p=P[a])
        X = rng.binomial(1, theta[t, s])
        pend[t] = (i, x, a, s, X)
        loss += c[t, a]
    return loss - c[:, astar].sum()

if __name__ == "__main__":
    print("1-2. estimator moments  (K=40, |S|=4, 400000 draws)")
    b, se, ms, ma = moments(40, 4, 400000)
    print(f"     max |bias|                 {b:.5f}   (2 s.e. = {2*se:.5f})")
    print(f"     sum_a x_a E[chat_a^2]      state {ms:6.2f}   action {ma:7.2f}"
          f"   |S|={4}  K={40}")
    print()

    T, S, d, seeds = 5000, 4, 50, 24
    print(f"3. interleaved EXP3 under delay   T={T} |S|={S} d={d}  {seeds} seeds, "
          f"common random numbers")
    print(f"   {'setting':<22} {'state-aware':>12} {'action-level':>13} {'paired diff':>16}")
    for label, V in (("stationary", 0.0), ("drifting", 0.3)):
        for K in (40, 80):
            st = np.array([run(K, S, T, d, V, s, 'state') for s in range(seeds)])
            ac = np.array([run(K, S, T, d, V, s, 'action') for s in range(seeds)])
            df = ac - st
            print(f"   {label+f', K={K}':<22} {st.mean():12.1f} {ac.mean():13.1f} "
                  f"{df.mean():9.1f} +- {df.std(ddof=1)/np.sqrt(seeds):5.1f}")
