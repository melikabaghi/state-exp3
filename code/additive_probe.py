"""Does the additive route close?  Probe the one step that is not routine.

Decomposition for plain delayed EXP3 with x_t ~ exp(-eta * sum_{s<=t-d-1} chat_s):

  R_T <= logK/eta
         + (eta/2) sum_r E[ sum_a x_{r+d}(a) chat_r(a)^2 ]     <-- QUADRATIC
         + eta * sum_r E[ <x_r, G_r> ],  G_r = sum_{s=r-d}^{r-1} chat_s   <-- DELAY

The DELAY term is fine: E[<x_r, chat_s>] = <x_r, c_s> <= 1 because x_r depends only on
rounds <= r-d-1 <= s-1.  So it is at most eta*d*T for ANY unbiased estimator.

The QUADRATIC term is the problem.  The second-moment bound |S| holds against x_r, the
distribution that generated the importance weights, but the analysis needs it against
x_{r+d}.  Since x_{r+d} = x_r * exp(-eta G_r)/Z_r with Z_r <= 1, the honest bound is

  sum_a x_{r+d}(a) E[chat_r(a)^2] <= (1/Z_r) * |S|,   Z_r = E_{a~x_r}[exp(-eta G_r(a))].

This measures 1/Z_r and the realised ratio directly.  If 1/Z_r is O(1) the additive bound
follows; if it is not, the route needs a device the paper does not have.
"""
import numpy as np
from state_exp3 import env

def probe(K, S, T, d, V, seed, eta):
    rng = np.random.default_rng(seed)
    P, theta, c = env(K, S, T, V, rng)
    L = np.zeros(K); hist = {}
    inv_Z, ratio = [], []
    for t in range(T):
        r = t - d - 1
        if r in hist:                      # chat_r enters L at the start of round r+d+1
            L += hist[r][1]
        x = np.exp(-eta * (L - L.min())); x /= x.sum()
        q = x @ P
        a = rng.choice(K, p=x); s = rng.choice(S, p=P[a])
        X = float(rng.binomial(1, theta[t, s]))
        chat = P[:, s] * X / max(q[s], 1e-300)
        hist[t] = (x, chat)
        if t >= d + 1:                     # G_t = sum of the d outstanding estimates
            G = np.zeros(K)
            for u in range(t - d, t):
                if u in hist: G += hist[u][1]
            Z = float(np.dot(x, np.exp(-eta * G)))
            inv_Z.append(1.0 / max(Z, 1e-300))
            xd = x * np.exp(-eta * G); xd /= xd.sum()
            qd = xd @ P
            # the quantity the analysis actually needs, against its claimed bound |S|
            ratio.append(float(np.sum(qd / np.maximum(q, 1e-300))) / S)
    return np.array(inv_Z), np.array(ratio)

if __name__ == "__main__":
    T, S, K, V = 4000, 4, 40, 0.3
    print(f"plain delayed EXP3, T={T} |S|={S} K={K}   1/Z_r and the quadratic-term inflation")
    print(f"   {'d':>5} {'eta':>9} {'mean 1/Z':>10} {'p99 1/Z':>10} {'max 1/Z':>12} "
          f"{'mean infl':>10} {'max infl':>10}")
    for d in (10, 50, 200):
        eta = np.sqrt(np.log(K) / (S * T + d * T))
        iz, rt = [], []
        for sd in range(6):
            a, b = probe(K, S, T, d, V, sd, eta)
            iz.append(a); rt.append(b)
        iz = np.concatenate(iz); rt = np.concatenate(rt)
        print(f"   {d:5d} {eta:9.5f} {iz.mean():10.3f} {np.quantile(iz,.99):10.3f} "
              f"{iz.max():12.2f} {rt.mean():10.3f} {rt.max():10.2f}")
