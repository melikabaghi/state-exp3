"""Explore-then-commit with state pooling. Verifies the regret identity and the bound."""
import numpy as np

def run(K, S, T, d, n, V, seed, pool=True, P_known=True):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)          # row-stochastic K x |S|
    base = rng.uniform(0.25, 0.75, size=S)
    # drift: each state moves monotonically within a band of width V
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base[None, :] + V * slope[None, :] * (np.arange(T)[:, None] / T), 0, 1)
    c = theta @ P.T                                        # T x K, c[t,a]
    cbar = c.mean(0)
    astar = cbar.argmin()

    N = n + d
    A = rng.integers(0, K, size=N)                         # explore: uniform
    Sv = np.array([rng.choice(S, p=P[a]) for a in A])
    X = rng.binomial(1, theta[np.arange(N), Sv]).astype(float)

    use = np.arange(n)                                     # outcomes arrived by round N
    if pool:
        th = np.full(S, 0.5)
        for s in range(S):
            m = use[Sv[use] == s]
            if len(m): th[s] = X[m].mean()
        Pm = P if P_known else np.array([
            (np.bincount(Sv[use][A[use] == a], minlength=S) / max((A[use] == a).sum(), 1))
            if (A[use] == a).sum() else np.ones(S)/S for a in range(K)])
        chat = Pm @ th
    else:                                                  # action-level pooling
        chat = np.array([X[use][A[use] == a].mean() if (A[use] == a).sum() else 0.5
                         for a in range(K)])
    ahat = int(chat.argmin())

    played = np.concatenate([A, np.full(T - N, ahat)])
    regret = c[np.arange(T), played].sum() - c[:, astar].sum()
    return dict(regret=regret, N=N, gap=cbar[ahat] - cbar[astar],
                est_err=np.abs(chat - cbar).max(),
                Vinf=(theta.max(0) - theta.min(0)).max())

K, S, T, d, V = 40, 4, 20000, 50, 0.06
n = int((T * np.sqrt(S * np.log(2*S/0.05)))**(2/3))
print(f"K={K} |S|={S} T={T} d={d} drift band V={V}   n*={n}\n")
for tag, kw in [("state pooling, P known", dict(pool=True,  P_known=True)),
                ("state pooling, P learned", dict(pool=True, P_known=False)),
                ("action pooling", dict(pool=False, P_known=True))]:
    r = [run(K, S, T, d, n, V, s, **kw) for s in range(24)]
    reg = np.array([x['regret'] for x in r]); ee = np.array([x['est_err'] for x in r])
    gap = np.array([x['gap'] for x in r]);    Vi = r[0]['Vinf']
    ident = np.all(reg <= n + d + T*gap + 1e-6)
    print(f"  {tag:26s} regret {reg.mean():8.1f}+-{reg.std()/len(reg)**.5:5.1f}"
          f"   est.err {ee.mean():.4f}   identity R<=N+T*gap: {ident}")
print(f"\n  V_inf (measured) = {r[0]['Vinf']:.3f}")
bound = n + d + 2*T*(r[0]['Vinf'] + np.sqrt(S*np.log(2*S/0.05)/ (2*n/(2*S)) ))
print(f"  loose bound n+d+2T(V+sqrt(kappa|S|L/n)) = {bound:.0f}")
