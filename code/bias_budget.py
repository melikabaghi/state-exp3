"""Is the played-weighted bias actually too large, or only too hard to certify?

Safe-Pool clears pooling when the per-round bias it must charge is below
(eta/2)(Lam_t - Vhat_t), the variance it saves.  Four quantities per round:

    truth       |<x_t, b_t> - b_t(a*)|                     what regret really charges
    level       max_a |b_t(a)|                             what the analysis charges
    cert-H      2 r_max + 2 rhohat, Hoeffding radii        what the algorithm can verify
    cert-B      the same with Bernstein radii              a tighter attempt

against the budget.  If truth < budget < level the algorithm should pool and only the analysis
stops it; if truth > budget the design is wrong, not the analysis.
"""
import numpy as np

K, S, T, d, SEEDS = 40, 8, 8000, 50, 12
LOG = np.log


def env(K, S, T, V, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1, 1, size=S)
    theta = np.clip(base + V * slope * (np.arange(T)[:, None] / T), 0, 1)
    return P, theta, theta @ P.T


def main():
    eta = np.sqrt(2 * LOG(K) / ((K + d) * T))          # the safe-pool tuning
    budget = 0.5 * eta * (K - 2 * S)                   # (eta/2)(Lam - Vhat), best case
    acc = dict(truth=0.0, level=0.0, certH=0.0, certB=0.0)
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        P, th, c = env(K, S, T, 0.05, rng)
        astar = int(c.sum(0).argmin())
        eta_run = np.sqrt(2 * LOG(K) / ((S + d) * T))
        L = np.zeros(K); N = np.zeros(K); C = np.zeros((K, S)); pend = {}
        for t in range(T):
            r = t - d - 1
            if r in pend:
                x_o, Pm, a_o, s_o, X_o = pend.pop(r)
                q_o = x_o @ Pm
                L += Pm[:, s_o] * X_o / max(q_o[s_o], 1e-300)
            z = -eta_run * (L - L.min()); x = np.exp(z); x /= x.sum()
            Pm = (C + 0.01) / (N[:, None] + 0.01 * S)
            q, qh = x @ P, x @ Pm
            D, dl, thr = Pm - P, qh - q, th[t]
            b = ((thr * q / qh)[None, :] * D).sum(1) - ((thr * dl / qh)[None, :] * P).sum(1)
            acc["truth"] += abs(float(x @ b - b[astar]))
            acc["level"] += float(np.abs(b).max())
            n = np.maximum(N, 1.0)
            r1 = np.minimum(2.0, np.sqrt(2 * (S * LOG(2) + LOG(2 * K * T)) / n))
            rH = np.minimum(1.0, np.sqrt(LOG(2 * K * S * T) / (2 * n)))
            uH = np.full(S, float(x @ rH))
            lg = LOG(2 * K * S * T)
            rB = np.minimum(1.0, np.sqrt(2 * Pm * (1 - Pm) * lg / n[:, None]) + 3 * lg / n[:, None])
            uB = (x[:, None] * rB).sum(0)
            for key, u in (("certH", uH), ("certB", uB)):
                slack = qh - u
                rho = np.inf if (slack <= 0).any() else float((u / slack).max())
                acc[key] += min(4.0, 2 * float(r1.max()) + 2 * rho)
            a_ = int(rng.choice(K, p=x)); s_ = int(rng.choice(S, p=P[a_]))
            N[a_] += 1; C[a_, s_] += 1
            pend[t] = (x, Pm, a_, s_, float(rng.binomial(1, th[t, s_])))
    for k in acc:
        acc[k] /= SEEDS
    tot = budget * T
    print(f"K={K} |S|={S} T={T} d={d}, {SEEDS} seeds, plug-in alpha=0.01\n")
    print(f"  per-round budget (eta/2)(K - 2|S|)      {budget:.5f}")
    print(f"  summed over the horizon                 {tot:10.1f}\n")
    for k, lab in (("truth", "what regret charges"),
                   ("level", "what the analysis charges"),
                   ("certH", "what the algorithm can verify, Hoeffding"),
                   ("certB", "the same with Bernstein radii")):
        print(f"  {acc[k]:10.1f}  {acc[k]/tot:7.2f}x budget   {lab}")
    print("\n  pooling is worth it exactly when the first line is below the budget.")


if __name__ == "__main__":
    main()
