"""Safe-Pool State-EXP3: blend action-level and state-pooled weights, per action, per round.

The plug-in estimate is better than action-level weighting when the map is well estimated and
worse when it is not.  Rather than choose once, blend:

    gcheck_t(a) = (1 - lam_t(a)) * ctilde_t(a) + lam_t(a) * chat_t(a),      lam_t(a) in [0,1]

with ctilde the action-level estimate and chat the plug-in.  Two facts make this work.

  bias.      E[gcheck_t(a)] - c_t(a) = lam_t(a) * (plug-in bias at a), so shrinking lam on an
             action shrinks its bias exactly there.  lam = 0 is unbiased.  NOTE the play-weighted
             cancellation sum_a x_a b_a = 0 does NOT survive per-action weights: at x = (1/2,1/2),
             P = I, Phat = [[.9,.1],[.2,.8]], theta = (0,1) we get b = (1/9,-1/9) with x.b = 0 but
             x.(lam*b) = 1/18 for lam = (1,0).  Both biases are therefore charged.
  variance.  sum_a x_t(a) E[gcheck_t(a)^2] <= (K - Lam_t) + Vhat_t(lam), Lam_t = sum_a lam_t(a),
             which is K at lam = 0 and at most 2*vhat_t at lam = 1.

So the per-round cost of pooling a set of actions is certifiable from quantities the learner
holds.  A numerical candidate tau sets lam_t(a) = 1{r_t(a) <= tau} and costs

    (eta/2) [ (K - Lam_t(tau)) + Vhat_t(tau) ] + 4 tau + 4 rhohat_t

and a separate abstain candidate sets lam_t = 0 and costs (eta/2) K, the action-level value.
Abstain is kept apart from tau = 0 because a valid radius may itself be zero, in which case
tau = 0 would pool the actions attaining it.  Below, abstain is best_k = 0.
The algorithm therefore never pays more per round than action-level weighting would, whatever P
turns out to be, and pays less exactly when the certificate says pooling is worth it.

r_t(a) is the l1 confidence radius for row a and rhohat_t certifies the relative error of the
induced state distribution.  Both come from the immediate (A_r, S_r) pairs, which arrive with no
delay, and both carry the add-alpha smoothing bias as well as the sampling error.
"""
import numpy as np

LOG2 = np.log(2.0)


def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T


def radii(N, K, S, T, alpha=0.01, certified=True):
    """l1 radius per row and per-entry radius, both from N_a alone.

    certified=True uses the worst-case multinomial and Hoeffding constants, which is what the
    theorem needs.  certified=False drops the confidence logs and keeps only the sqrt(|S|/N)
    shape, which is what the same rule looks like with a realistic constant.
    """
    n = np.maximum(N, 1.0)
    if certified:
        # sampling error plus the add-alpha smoothing bias.  Against the empirical row the
        # smoothing moves an entry by at most alpha(S-1)/(N+alpha S), attained when all N
        # observations sit on one state, and the l1 distance by twice that.
        smooth_inf = alpha * (S - 1.0) / (n + alpha * S)
        smooth_l1 = 2.0 * smooth_inf
        # Simultaneous validity over all t <= T needs a union bound over K*T (row, time) pairs
        # for the l1 family and K*S*T (entry, time) triples for the entrywise one.  Allocating
        # 1/(2T) to each family gives per-pair 1/(2 K T^2) and per-triple 1/(2 K S T^2), so
        # Weissman needs log(2 K T^2) and Hoeffding needs log(4 K S T^2).  The earlier logs
        # spent 1/2 and 1 respectively, which certified nothing.
        r1 = np.minimum(
            2.0,
            np.sqrt(2.0 * (S * LOG2 + np.log(2.0 * K * T * T)) / n) + smooth_l1,
        )
        rinf = np.minimum(
            1.0,
            np.sqrt(np.log(4.0 * K * S * T * T) / (2.0 * n)) + smooth_inf,
        )
    else:
        r1 = np.minimum(2.0, np.sqrt(S / n))
        rinf = np.minimum(1.0, np.sqrt(1.0 / n))
    return r1, rinf


def choose_lambda(x, Phat, N, eta, K, S, T, alpha=0.01, certified=True):
    """Return lam in {0,1}^K minimising the certified per-round cost, and that cost."""
    r1, rinf = radii(N, K, S, T, alpha, certified)
    qhat = x @ Phat
    uhat = np.full(S, float(x @ rinf))                    # sum_a x_a rinf_a, equal across states
    slack = qhat - uhat
    if np.any(slack <= 0):
        rho = np.inf
    else:
        rho = float((uhat / slack).max())
    base = 0.5 * eta * K                                   # the abstain candidate, lam = 0
    if not np.isfinite(rho) or rho > 0.5:
        return np.zeros(K), base
    # per-action contribution to the pooled second moment, with the q/qhat factor of two
    contrib = 2.0 * x * (Phat ** 2 / np.maximum(qhat, 1e-300)).sum(1)
    order = np.argsort(r1)
    best_cost, best_k = base, 0                            # best_k = 0 is abstain
    Vh, Lam = 0.0, 0.0
    for k, a in enumerate(order, start=1):
        Vh += contrib[a]
        Lam += 1.0
        # A threshold candidate pools every action with r_t(a) <= tau, so a prefix that splits a
        # group of equal radii is not expressible as a threshold.  Score only at the last member
        # of each group, which is where the prefix and the threshold agree.
        if k < K and r1[order[k]] == r1[a]:
            continue
        cost = 0.5 * eta * ((K - Lam) + Vh) + 4.0 * r1[a] + 4.0 * rho
        if cost < best_cost:
            best_cost, best_k = cost, k
    lam = np.zeros(K)
    if best_k:
        lam[order[:best_k]] = 1.0
    return lam, best_cost


def run(P, theta, c, d, mode, seed, alpha=1.0, gamma=0.0):
    """mode in {'action', 'plugin', 'safe', 'known'}."""
    rng = np.random.default_rng(seed)
    T, S = theta.shape
    K = P.shape[0]
    astar = int(c.sum(0).argmin())
    Vb = S if mode in ('plugin', 'known') else K   # safe-pool is capped at K by construction
    eta = np.sqrt(2.0 * np.log(K) / ((Vb + d) * T))
    L = np.zeros(K)
    N = np.zeros(K)
    C = np.zeros((K, S))
    pend, loss, pooled_rounds, pooled_actions = {}, 0.0, 0, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            x_o, Pm, lam, a_o, s_o, X_o = pend.pop(r)
            act = np.zeros(K); act[a_o] = X_o / max(x_o[a_o], 1e-300)
            if mode == 'action':
                L += act
            else:
                q = x_o @ Pm
                st = Pm[:, s_o] * X_o / max(q[s_o], 1e-300)
                L += st if lam is None else (1.0 - lam) * act + lam * st
        z = -eta * (L - L.min()); x = np.exp(z); x /= x.sum()
        if gamma > 0:
            x = (1.0 - gamma) * x + gamma / K
        if mode in ('plugin', 'safe', 'safeh'):
            Pm = (C + alpha) / (N[:, None] + alpha * S)
        else:
            Pm = P
        if mode in ('safe', 'safeh'):
            lam, _ = choose_lambda(x, Pm, N, eta, K, S, T, alpha,
                                   certified=(mode == 'safe'))
            pooled_rounds += int(lam.sum() > 0); pooled_actions += float(lam.sum())
        else:
            lam = None
        a = int(rng.choice(K, p=x)); s = int(rng.choice(S, p=P[a]))
        N[a] += 1; C[a, s] += 1
        pend[t] = (x, Pm, lam, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return (loss - float(c[:, astar].sum()),
            pooled_rounds / T, pooled_actions / (T * K))


if __name__ == "__main__":
    from hostile_unknown_p import make
    K, S, d, seeds = 40, 8, 50, 20
    print("Safe-Pool against its two ingredients.  'pooled' is the fraction of (round, action)")
    print("pairs the rule cleared.  safe = worst-case certificate, safe-h = same rule with the")
    print("confidence logs dropped.\n")
    for T in (8000, 50000):
        print(f"  T={T}")
        print(f"    {'environment':<16} {'alpha':>6} {'action':>8} {'plug-in':>9} {'safe':>8} "
              f"{'pooled':>7} {'safe-h':>8} {'pooled':>7}")
        for kind in ("dirichlet", "deterministic", "rare-optimal"):
            for alpha in (1.0, 0.01):
                ac, pl, sf, sh, f1, f2 = [], [], [], [], [], []
                for sd in range(seeds):
                    rng = np.random.default_rng(sd)
                    if kind == "dirichlet":
                        P, th, c = env(K, S, T, 0.05, rng)
                    else:
                        P, th = make(kind, K, S, T, rng); c = th @ P.T
                    if alpha == 1.0:
                        ac.append(run(P, th, c, d, 'action', 10_000 + sd)[0])
                    pl.append(run(P, th, c, d, 'plugin', 10_000 + sd, alpha)[0])
                    R, _, fa = run(P, th, c, d, 'safe', 10_000 + sd, alpha); sf.append(R); f1.append(fa)
                    R, _, fb = run(P, th, c, d, 'safeh', 10_000 + sd, alpha); sh.append(R); f2.append(fb)
                a_str = f"{np.mean(ac):8.1f}" if alpha == 1.0 else f"{'':>8}"
                print(f"    {kind:<16} {alpha:6.2f} {a_str} {np.mean(pl):9.1f} {np.mean(sf):8.1f} "
                      f"{np.mean(f1):7.3f} {np.mean(sh):8.1f} {np.mean(f2):7.3f}", flush=True)
        print()
