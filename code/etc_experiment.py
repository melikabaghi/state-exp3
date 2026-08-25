"""Paired evaluation of Theorem 3: pooling by state vs by action, common random numbers."""
import numpy as np

def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T

def trial(K, S, T, d, n, V, seed):
    """One seed, both estimators on the SAME environment and the SAME explore trajectory."""
    rng = np.random.default_rng(seed)
    P, theta, c = env(K, S, T, V, rng)
    cbar = c.mean(0); astar = int(cbar.argmin())
    N = n + d
    A = rng.integers(0, K, size=N)
    Sv = np.array([rng.choice(S, p=P[a]) for a in A])
    X = rng.binomial(1, theta[np.arange(N), Sv]).astype(float)
    u = np.arange(n)
    out = {}
    # pool by state
    th = np.full(S, 0.5)
    for s in range(S):
        m = u[Sv[u] == s]
        if len(m): th[s] = X[m].mean()
    out['state'] = P @ th
    # pool by action
    out['action'] = np.array([X[u][A[u] == a].mean() if (A[u] == a).sum() else 0.5
                              for a in range(K)])
    res = {}
    for k, chat in out.items():
        ah = int(chat.argmin())
        played = np.concatenate([A, np.full(T - N, ah)])
        res[k] = dict(regret=c[np.arange(T), played].sum() - c[:, astar].sum(),
                      err=np.abs(chat - cbar).max(),
                      ok=bool(c[np.arange(T), played].sum() - c[:, astar].sum()
                              <= N + T * (cbar[ah] - cbar[astar]) + 1e-6))
    res['Vinf'] = float((theta.max(0) - theta.min(0)).max())
    return res

def paired(K, S, T, d, V, seeds=200):
    n = int((T * np.sqrt(S * np.log(2 * S / 0.05))) ** (2 / 3))
    R = [trial(K, S, T, d, n, V, s) for s in range(seeds)]
    ds = np.array([r['action']['regret'] - r['state']['regret'] for r in R])
    st = np.array([r['state']['regret'] for r in R])
    ac = np.array([r['action']['regret'] for r in R])
    ident = all(r['state']['ok'] and r['action']['ok'] for r in R)
    return dict(n=n, state=st.mean(), action=ac.mean(), diff=ds.mean(),
                se=ds.std(ddof=1)/np.sqrt(seeds), wins=int((ds > 0).sum()), seeds=seeds,
                ident=ident, Vinf=R[0]['Vinf'],
                err_s=np.mean([r['state']['err'] for r in R]),
                err_a=np.mean([r['action']['err'] for r in R]))

T, d, S, V = 20000, 50, 4, 0.05
print(f"|S|={S}  T={T}  d={d}  drift band={V}   paired, common random numbers, 200 seeds\n")
print(f"  {'K':>4}  {'by state':>10}  {'by action':>10}  {'paired diff':>16}  {'wins':>8}  identity")
for K in [8, 20, 40, 80]:
    r = paired(K, S, T, d, V)
    print(f"  {K:4d}  {r['state']:10.1f}  {r['action']:10.1f}  "
          f"{r['diff']:8.1f} +- {r['se']:5.1f}  {r['wins']:4d}/{r['seeds']:<4d}  {r['ident']}")
print(f"\n  estimation error at K=40: state {paired(40,S,T,d,V)['err_s']:.4f}  "
      f"action {paired(40,S,T,d,V)['err_a']:.4f}")
print("\n  drift sweep at K=40 (Theorem 3 predicts a term linear in V_inf)")
for V2 in [0.0, 0.1, 0.3, 0.6]:
    r = paired(40, S, T, d, V2, seeds=200)
    print(f"    V_inf={r['Vinf']:.3f}   state-pooled regret {r['state']:8.1f}")
