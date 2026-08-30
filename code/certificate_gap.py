"""How far is the algorithm from its own certificate, and how far is the certificate from useful?

Study 2 reports that the continuously updated plug-in lands within half a per cent of the same
algorithm given P, while Theorem 9's warm-start rate is an order of magnitude above both.  Those
three facts sit in different columns of a table and the reader has to assemble them.  This file
puts the four quantities on one axis across a sweep in K:

    action baseline     realised regret of action-level delayed EXP3, the thing to beat
    known P             realised regret of State-EXP3 with P supplied
    plug-in P           realised regret of the same estimator with Phat_t updated every round
    certificate         what Theorem 9 promises for the warm-start algorithm it analyses

The first three are measured on paired seeds at m = 1, the practical variant.  The fourth is a
bound for a different algorithm, the warm-start one that freezes Phat and restarts, and it is
plotted to show the size of the gap rather than to be compared like for like.  Theorem 1's own
bound for the analysed m = d+1 variant is also drawn, since it is the certificate for known P.

sqrt(KT) is marked because the warm-start rate never reaches it, which is what stops it
certifying the channel it analyses.

Writes certificate_gap.npz and prints the table.
"""
from __future__ import annotations

import numpy as np

S, T, D = 8, 5000, 50
KS = (8, 20, 40, 80, 160)
SEEDS = 12
ALPHA = 0.01


def env(K, S, T, drift, rng):
    P = rng.dirichlet(np.ones(S) * 1.5, size=K)
    base = rng.uniform(0.25, 0.75, size=S)
    slope = rng.uniform(-1.0, 1.0, size=S)
    theta = np.clip(base + drift * slope * (np.arange(T)[:, None] / T), 0.0, 1.0)
    return P, theta, theta @ P.T


def run(P, theta, c, K, d, mode, seed):
    """mode in {'known','plugin','action'}, m = 1 throughout."""
    rng = np.random.default_rng(seed + 60_000)
    astar = int(c.sum(0).argmin())
    Vb = S if mode != "action" else K
    eta = np.sqrt(2.0 * np.log(K) / (Vb * T + d * T))
    L = np.zeros(K)
    counts = np.zeros((K, S))
    pend, loss = {}, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            x, Pr_, a_r, s_r, X_r = pend.pop(r)
            if mode == "action":
                L[a_r] += X_r / max(x[a_r], 1e-300)
            else:
                q = x @ Pr_
                L += Pr_[:, s_r] * X_r / max(q[s_r], 1e-300)
        z = -eta * (L - L.min())
        x = np.exp(z)
        x /= x.sum()
        Pr_ = ((counts + ALPHA) / (counts.sum(1, keepdims=True) + ALPHA * S)
               if mode == "plugin" else P)
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(S, p=P[a]))
        counts[a, s] += 1
        pend[t] = (x, Pr_, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def kappa_of(P):
    return 1.0 / (S * P.mean(0).min())


def main() -> None:
    rows = []
    print(f"|S|={S} T={T} d={D} alpha={ALPHA}, m=1 for every measured arm, {SEEDS} paired seeds\n")
    print(f"{'K':>5}{'kappa':>7}{'action':>9}{'known P':>9}{'plug-in':>9}"
          f"{'Thm 1 bnd':>11}{'Thm 9 bnd':>11}{'sqrt(KT)':>10}")
    for K in KS:
        ac, kn, pl, kaps = [], [], [], []
        for sd in range(SEEDS):
            rng = np.random.default_rng(sd + 5)
            P, theta, c = env(K, S, T, 0.05, rng)
            kaps.append(kappa_of(P))
            ac.append(run(P, theta, c, K, D, "action", sd))
            kn.append(run(P, theta, c, K, D, "known", sd))
            pl.append(run(P, theta, c, K, D, "plugin", sd))
        kap = float(np.mean(kaps))
        # Theorem 5, the certificate for the rotating known-P variant
        thm1 = np.sqrt(2.0 * (D + 1) * S * T * np.log(K))
        # Theorem 9, the warm-start rate, exploration term as reported in Appendix D
        thm9 = (kap * S) ** 0.4 * K ** 0.2 * T ** 0.8
        rows.append(dict(K=K, kappa=kap, action=float(np.mean(ac)), known=float(np.mean(kn)),
                         plugin=float(np.mean(pl)),
                         action_se=float(np.std(ac, ddof=1) / np.sqrt(SEEDS)),
                         known_se=float(np.std(kn, ddof=1) / np.sqrt(SEEDS)),
                         plugin_se=float(np.std(pl, ddof=1) / np.sqrt(SEEDS)),
                         thm1=float(thm1), thm9=float(thm9), sqrtKT=float(np.sqrt(K * T))))
        print(f"{K:5d}{kap:7.2f}{np.mean(ac):9.1f}{np.mean(kn):9.1f}{np.mean(pl):9.1f}"
              f"{thm1:11.0f}{thm9:11.0f}{np.sqrt(K*T):10.0f}", flush=True)

    np.savez("certificate_gap.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print("\nratios, all against the measured plug-in")
    for r in rows:
        print(f"  K={r['K']:3d}: known/plug-in {r['known']/r['plugin']:5.3f}   "
              f"Thm 5 / plug-in {r['thm1']/r['plugin']:6.1f}x   "
              f"Thm 9 / plug-in {r['thm9']/r['plugin']:6.1f}x   "
              f"action / plug-in {r['action']/r['plugin']:5.3f}")
    kn = np.array([r["known"] for r in rows]); pl = np.array([r["plugin"] for r in rows])
    print(f"\nknown P against plug-in, worst cell: {100*abs(pl/kn - 1).max():.2f} per cent")
    print("The distance from the measured arms to either certificate is the analysis, not the "
          "algorithm.  Theorem 9 also stays above sqrt(KT) at every K here, which is the sense "
          "in which it does not certify the channel it analyses.")


if __name__ == "__main__":
    main()
