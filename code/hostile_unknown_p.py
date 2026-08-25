"""Hostile environments for the plug-in estimate of P.

The friendly Dirichlet sweep showed gamma = 0 losing nothing.  That is not evidence that forced
exploration is unnecessary, only that those instances do not punish its absence.  These five do:

  rare-optimal      the best action is reachable only through a state the others rarely produce
  unbalanced        one state carries almost all the mass, so kappa is large
  near-unreachable  one state has probability ~1e-3 under every action
  deterministic     every row is a point mass, so v = |S| and rows share nothing
  misleading        the first visits of the good action are unrepresentative by construction

Reported against known P and against action-level exponential weights, over several mixing rates.
"""
import numpy as np


def _theta(S, T, rng, band=0.12):
    base = rng.uniform(0.15, 0.85, size=S)
    ph = rng.uniform(0, 2 * np.pi, size=S)
    tt = np.arange(T)[:, None] / T
    return np.clip(base + band * np.sin(2 * np.pi * tt + ph), 0.0, 1.0)


def make(kind, K, S, T, rng):
    if kind == "rare-optimal":
        P = np.full((K, S), 1e-3)
        P[:, 0] = 1.0
        P[0, :] = 1e-3; P[0, 1] = 1.0            # only action 0 reaches state 1
        P /= P.sum(1, keepdims=True)
        th = _theta(S, T, rng)
        th[:, 1] = 0.05                           # and state 1 is the good one
        return P, th
    if kind == "unbalanced":
        w = np.full(S, 1e-2); w[0] = 1.0
        P = rng.dirichlet(w * 4.0, size=K)
        return P, _theta(S, T, rng)
    if kind == "near-unreachable":
        P = rng.dirichlet(np.ones(S) * 1.5, size=K)
        P[:, -1] = 1e-3
        P /= P.sum(1, keepdims=True)
        return P, _theta(S, T, rng)
    if kind == "deterministic":
        P = np.zeros((K, S))
        P[np.arange(K), rng.integers(0, S, size=K)] = 1.0
        P[np.arange(min(K, S)), np.arange(min(K, S))] = 1.0   # every state reachable
        P /= P.sum(1, keepdims=True)
        return P, _theta(S, T, rng)
    if kind == "misleading":
        P = rng.dirichlet(np.ones(S) * 0.3, size=K)           # sparse rows, slow to learn
        th = _theta(S, T, rng)
        best = int((th.mean(0) @ P.T).argmin())
        P[best] = 0.85 * P[best] + 0.15 * rng.dirichlet(np.ones(S) * 0.2)
        P /= P.sum(1, keepdims=True)
        return P, th
    raise ValueError(kind)


def kappa(P):
    pbar = P.mean(0)
    return 1.0 / (P.shape[1] * pbar.min())


def run(P, theta, c, d, mode, gamma, seed, alpha=0.01):
    rng = np.random.default_rng(seed)
    T_, S_ = theta.shape
    K_ = P.shape[0]
    astar = int(c.sum(0).argmin())
    Vb = K_ if mode == 'action' else S_
    eta = np.sqrt(2.0 * np.log(K_) / ((Vb + d) * T_))
    L = np.zeros(K_); counts = np.zeros((K_, S_)); pend = {}; loss = 0.0
    for t in range(T_):
        r = t - d - 1
        if r in pend:
            x_o, Pm_, a_o, s_o, X_o = pend.pop(r)
            if mode == 'action':
                L[a_o] += X_o / max(x_o[a_o], 1e-300)
            else:
                q = x_o @ Pm_
                L += Pm_[:, s_o] * X_o / max(q[s_o], 1e-300)
        z = -eta * (L - L.min()); x = np.exp(z); x /= x.sum()
        if gamma > 0:
            x = (1.0 - gamma) * x + gamma / K_
        Pm_ = ((counts + alpha) / (counts.sum(1, keepdims=True) + alpha * S_)
               if mode == 'plugin' else P)
        a = int(rng.choice(K_, p=x)); s = int(rng.choice(S_, p=P[a]))
        counts[a, s] += 1
        pend[t] = (x, Pm_, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


if __name__ == "__main__":
    K, S, T, d, seeds = 40, 8, 8000, 50, 30
    print(f"K={K} |S|={S} T={T} d={d}, {seeds} paired seeds\n")
    print(f"  {'environment':<18} {'kappa':>7} {'known P':>9} {'action':>8} "
          f"{'plug g=0':>9} {'g=0.01':>8} {'g=0.05':>8} {'g=0.20':>8}")
    for kind in ("rare-optimal", "unbalanced", "near-unreachable", "deterministic", "misleading"):
        acc = {k: [] for k in ('known', 'action', 0.0, 0.01, 0.05, 0.20)}
        kaps = []
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            P, th = make(kind, K, S, T, rng)
            c = th @ P.T
            kaps.append(kappa(P))
            acc['known'].append(run(P, th, c, d, 'state', 0.0, 10_000 + sd))
            acc['action'].append(run(P, th, c, d, 'action', 0.0, 10_000 + sd))
            for g in (0.0, 0.01, 0.05, 0.20):
                acc[g].append(run(P, th, c, d, 'plugin', g, 10_000 + sd))
        print(f"  {kind:<18} {np.mean(kaps):7.1f} {np.mean(acc['known']):9.1f} "
              f"{np.mean(acc['action']):8.1f} {np.mean(acc[0.0]):9.1f} "
              f"{np.mean(acc[0.01]):8.1f} {np.mean(acc[0.05]):8.1f} {np.mean(acc[0.20]):8.1f}",
              flush=True)
