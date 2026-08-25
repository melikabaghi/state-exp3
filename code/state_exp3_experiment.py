"""Paired evaluation of State-EXP3: state-pooled versus action-level importance weighting.

Three policies on identical environments and common random numbers:

  state, m=d+1   the analysed algorithm: d+1 interleaved copies, each undelayed
  state, m=1     the same estimator in plain delayed EXP3, the practical variant
  action, m=1    plain delayed EXP3 with the action-level estimator, the baseline

A seed fixes the environment, so differences are taken within a seed before averaging.
"""
import numpy as np

def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T

def run(P, theta, c, T, d, m, mode, rng):
    """m interleaved copies; m = d+1 makes every copy undelayed, m = 1 is plain delayed."""
    K, S = P.shape
    astar = int(c.sum(0).argmin())
    Ti = T / m
    Vb = S if mode == 'state' else K
    eta = np.sqrt(2.0 * np.log(K) / (Vb * Ti + (0.0 if m == d + 1 else d * Ti)))
    L = np.zeros((m, K))
    pend = {}
    loss = 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:                        # round r's outcome is usable from r+d+1
            i, x, a_r, s_r, X_r = pend.pop(r)
            if mode == 'state':
                q = x @ P
                L[i] += P[:, s_r] * X_r / max(q[s_r], 1e-300)
            else:
                L[i, a_r] += X_r / max(x[a_r], 1e-300)
        i = t % m
        x = np.exp(-eta * (L[i] - L[i].min())); x /= x.sum()
        a = rng.choice(K, p=x)
        s = rng.choice(S, p=P[a])
        pend[t] = (i, x, a, s, float(rng.binomial(1, theta[t, s])))
        loss += c[t, a]
    return loss - c[:, astar].sum()

def trial(K, S, T, d, V, seed):
    P, theta, c = env(K, S, T, V, np.random.default_rng(seed))
    out = {}
    for name, m, mode in (("state_full", d + 1, 'state'),
                          ("state_plain", 1, 'state'),
                          ("action_plain", 1, 'action')):
        out[name] = run(P, theta, c, T, d, m, mode, np.random.default_rng(10_000 + seed))
    return out

def bound_full(K, S, T, d):
    return np.sqrt(2.0 * (d + 1) * S * T * np.log(K))

if __name__ == "__main__":
    T, d, S, V, seeds = 10000, 50, 4, 0.05, 100
    print(f"|S|={S}  T={T}  d={d}  drift band={V}   paired, common random numbers, "
          f"{seeds} seeds\n")
    hdr = (f"  {'K':>4}  {'state m=d+1':>12}  {'state m=1':>11}  {'action m=1':>11}  "
           f"{'plain gain':>15}  {'win':>7}  {'Thm bound':>10}")
    print(hdr)
    for K in (8, 20, 40, 80):
        R = [trial(K, S, T, d, V, s) for s in range(seeds)]
        sf = np.array([r["state_full"] for r in R])
        sp = np.array([r["state_plain"] for r in R])
        ap = np.array([r["action_plain"] for r in R])
        g = ap - sp
        ok = int((sf <= bound_full(K, S, T, d)).sum())
        print(f"  {K:4d}  {sf.mean():12.1f}  {sp.mean():11.1f}  {ap.mean():11.1f}  "
              f"{g.mean():7.1f} +- {g.std(ddof=1)/np.sqrt(seeds):4.1f}  "
              f"{int((g > 0).sum()):3d}/{seeds:<3d}  {bound_full(K,S,T,d):10.0f}"
              f"   [{ok}/{seeds} under]")
    print("\n  delay sweep at K=40 (the analysed variant pays d multiplicatively,")
    print("  the plain variant is conjectured to pay it additively)")
    print(f"  {'d':>5}  {'state m=d+1':>12}  {'state m=1':>11}  {'action m=1':>11}")
    for dd in (10, 50, 200):
        R = [trial(40, S, T, dd, V, s) for s in range(40)]
        print(f"  {dd:5d}  {np.mean([r['state_full'] for r in R]):12.1f}  "
              f"{np.mean([r['state_plain'] for r in R]):11.1f}  "
              f"{np.mean([r['action_plain'] for r in R]):11.1f}")
