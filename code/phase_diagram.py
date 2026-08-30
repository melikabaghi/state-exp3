"""Phase diagram: when does pooling through the state actually pay?

Relative gain  (R_action - R_state) / R_action  over a grid of effective dimension (varied by
the overlap between the rows of P) against delay d, in three panels:

    known P        Theorem 1's setting
    estimated P    the plug-in of Appendix D, gamma = 0
    grouped        the same algorithm run on a coarsening of the states, Theorem 6

The predicted boundary is (d+1) vbar log K <= K + d, which is where the bound of Theorem 1 falls
below the Otilde(sqrt((K+d)T)) available without a stationary sensor.  It is a comparison of two
upper bounds, so it is sufficient for a gain and not necessary.

Writes phase.npz for the figure.
"""
import numpy as np

OVERLAP = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
DELAY = (5, 10, 25, 50, 100, 200)
K, S, T, SEEDS = 40, 8, 5000, 12


def env(K, S, T, overlap, rng):
    common = rng.dirichlet(np.ones(S) * 2.0)
    individual = rng.dirichlet(np.ones(S) * 0.25, size=K)
    P = overlap * common[None, :] + (1.0 - overlap) * individual
    base = rng.uniform(0.15, 0.85, size=S)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=S)
    tt = np.arange(T)[:, None] / T
    theta = np.clip(base + 0.12 * np.sin(2.0 * np.pi * tt + phase), 0.0, 1.0)
    return P, theta, theta @ P.T


def aggregate(P, groups):
    G = int(groups.max()) + 1
    out = np.zeros((P.shape[0], G))
    for s, z in enumerate(groups):
        out[:, z] += P[:, s]
    return out


def vbar(P, samples=4000, seed=0):
    """Largest v over play distributions, estimated by sampling the simplex plus the vertices."""
    rng = np.random.default_rng(seed)
    best = 1.0
    for _ in range(samples):
        x = rng.dirichlet(np.ones(P.shape[0]) * rng.choice([0.1, 0.5, 2.0]))
        q = x @ P
        num = (x[:, None] * P ** 2).sum(0)
        best = max(best, float((num[q > 0] / q[q > 0]).sum()))
    return best


def run(P, theta, c, d, mode, groups=None, gamma=0.0, seed=0, alpha=0.01):
    rng = np.random.default_rng(seed)
    T_, S_ = theta.shape
    K_ = P.shape[0]
    astar = int(c.sum(0).argmin())
    Pg = P if groups is None else aggregate(P, groups)
    G = Pg.shape[1]
    Vb = K_ if mode == 'action' else G
    eta = np.sqrt(2.0 * np.log(K_) / ((Vb + d) * T_))
    L = np.zeros(K_)
    counts = np.zeros((K_, G))
    pend, loss = {}, 0.0
    for t in range(T_):
        r = t - d - 1
        if r in pend:
            x_o, Pm_, a_o, z_o, X_o = pend.pop(r)
            if mode == 'action':
                L[a_o] += X_o / max(x_o[a_o], 1e-300)
            else:
                q = x_o @ Pm_
                L += Pm_[:, z_o] * X_o / max(q[z_o], 1e-300)
        z = -eta * (L - L.min())
        x = np.exp(z); x /= x.sum()
        if gamma > 0:
            x = (1.0 - gamma) * x + gamma / K_
        Pm_ = ((counts + alpha) / (counts.sum(1, keepdims=True) + alpha * G)
               if mode == 'plugin' else Pg)
        a = int(rng.choice(K_, p=x))
        s = int(rng.choice(S_, p=P[a]))
        zz = s if groups is None else int(groups[s])
        counts[a, zz] += 1
        pend[t] = (x, Pm_, a, zz, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


if __name__ == "__main__":
    coarse = np.arange(S) // 2                      # |S| -> |S|/2, a generic coarsening
    gain = np.zeros((3, len(OVERLAP), len(DELAY)))
    vb = np.zeros(len(OVERLAP))
    for i, ov in enumerate(OVERLAP):
        vs = []
        for j, d in enumerate(DELAY):
            acc = {k: [] for k in ('known', 'plugin', 'grouped', 'action')}
            for sd in range(SEEDS):
                P, th, c = env(K, S, T, ov, np.random.default_rng(sd))
                if j == 0:
                    vs.append(vbar(P, seed=sd))
                acc['known'].append(run(P, th, c, d, 'state', None, 0.0, 10_000 + sd))
                acc['plugin'].append(run(P, th, c, d, 'plugin', None, 0.0, 10_000 + sd))
                acc['grouped'].append(run(P, th, c, d, 'state', coarse, 0.0, 10_000 + sd))
                acc['action'].append(run(P, th, c, d, 'action', None, 0.0, 10_000 + sd))
            base = np.mean(acc['action'])
            for k, name in enumerate(('known', 'plugin', 'grouped')):
                gain[k, i, j] = (base - np.mean(acc[name])) / base
            print(f"  overlap {ov:4.2f}  d {d:4d}   known {gain[0,i,j]:+.3f}  "
                  f"plugin {gain[1,i,j]:+.3f}  grouped {gain[2,i,j]:+.3f}", flush=True)
        vb[i] = float(np.mean(vs))
        print(f"overlap {ov:4.2f}: vbar = {vb[i]:.3f}", flush=True)
    np.savez("phase.npz", gain=gain, vbar=vb, overlap=np.array(OVERLAP),
             delay=np.array(DELAY), K=K, S=S, T=T, seeds=SEEDS)
    print("\nwrote phase.npz")
    print(f"predicted boundary (d+1) vbar log K <= K + d, log K = {np.log(K):.3f}")
    for j, d in enumerate(DELAY):
        print(f"  d={d:4d}: gain predicted for vbar <= {(K + d)/((d+1)*np.log(K)):.3f}")
