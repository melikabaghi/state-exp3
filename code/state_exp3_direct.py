"""Does the state estimator plug into PLAIN delayed EXP3 (no interleaving)?

If the delay penalty is additive and estimator-independent, state-aware and action-level
regret should differ by roughly sqrt(KT)-sqrt(|S|T), a gap that does NOT grow with d.
If instead the delay multiplies the bandit term, the gap should grow like sqrt(d).
"""
import numpy as np
from state_exp3 import env

def run_direct(K, S, T, d, V, seed, mode):
    rng = np.random.default_rng(seed)
    P, theta, c = env(K, S, T, V, rng)
    astar = int(c.sum(0).argmin())
    Vb = S if mode == 'state' else K
    eta = np.sqrt(np.log(K) / (Vb * T + d * T))     # Thune-style tuning
    L = np.zeros(K); pend = {}; loss = 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            x_r, a_r, s_r, X_r = pend.pop(r)
            if mode == 'state':
                q = x_r @ P
                L += P[:, s_r] * X_r / max(q[s_r], 1e-12)
            else:
                L[a_r] += X_r / max(x_r[a_r], 1e-12)
        z = -eta * (L - L.min()); x = np.exp(z); x /= x.sum()
        a = rng.choice(K, p=x); s = rng.choice(S, p=P[a])
        pend[t] = (x, a, s, rng.binomial(1, theta[t, s]))
        loss += c[t, a]
    return loss - c[:, astar].sum()

if __name__ == "__main__":
    T, S, K, V, seeds = 5000, 4, 40, 0.3, 24
    print(f"plain delayed EXP3, T={T} |S|={S} K={K} drift={V}, {seeds} seeds")
    print(f"   {'d':>5} {'state-aware':>12} {'action-level':>13} {'gap':>16}")
    for d in (0, 10, 50, 200, 800):
        st = np.array([run_direct(K, S, T, d, V, s, 'state') for s in range(seeds)])
        ac = np.array([run_direct(K, S, T, d, V, s, 'action') for s in range(seeds)])
        g = ac - st
        print(f"   {d:5d} {st.mean():12.1f} {ac.mean():13.1f} "
              f"{g.mean():9.1f} +- {g.std(ddof=1)/np.sqrt(seeds):5.1f}")
