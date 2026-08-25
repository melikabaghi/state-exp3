"""Is the plug-in robust or fragile in the smoothing constant, and does that depend on coverage?

Study 3 finds one case where the plug-in loses to action-level weighting at the Laplace default
alpha = 1, on a deterministic map, and that sweeping alpha down to 0.01 fixes it.  That is an
anecdote about one map.  This file turns it into a surface.

Two axes:

  alpha   the add-alpha smoothing pseudocount in Phat_t(s|a) = (N_a(s)+alpha)/(N_a+alpha|S|)

  kappa   the sensor balance of Theorem 9, defined through pbar(s) >= 1/(kappa |S|) for
          pbar = K^-1 sum_a P(.|a).  Large kappa means some state is reached rarely on average,
          so the rows are hard to estimate where it matters.

kappa is controlled directly.  The target average state distribution puts mass rho on one state
and spreads the rest, so kappa = 1/(|S| min_s pbar(s)); rows are then drawn Dirichlet around that
mean, which leaves pbar in place while letting rows differ.

Reported per cell is the plug-in's regret and its gain over action-level exponential weights on
the same paired seeds, plus the known-P run as the floor the plug-in is trying to reach.  A cell
where the gain is negative is one where estimating P has cost more than pooling bought.

Writes alpha_kappa.npz and prints the two tables.
"""
from __future__ import annotations

import numpy as np

K, S, T, D = 40, 8, 8000, 50
ALPHA = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)
RHO = (0.115, 0.050, 0.020, 0.008, 0.003)      # mass on the scarce state -> kappa = 1/(S rho)
SEEDS = 10
CONC = 3.0                                      # Dirichlet concentration around the target mean


def env(rho: float, seed: int):
    """Rows drawn around a mean pbar whose scarcest state carries mass rho."""
    rng = np.random.default_rng(seed)
    pbar = np.full(S, (1.0 - rho) / (S - 1))
    pbar[0] = rho
    P = rng.dirichlet(CONC * S * pbar, size=K)
    P = np.clip(P, 1e-9, None)
    P /= P.sum(1, keepdims=True)
    base = rng.uniform(0.2, 0.8, size=S)
    slope = rng.uniform(-1.0, 1.0, size=S)
    theta = np.clip(base + 0.25 * slope * (np.arange(T)[:, None] / T), 0.0, 1.0)
    kappa = 1.0 / (S * P.mean(0).min())
    return P, theta, theta @ P.T, kappa


def run(P, theta, c, d, mode, alpha, seed):
    """mode in {'known','plugin','action'}, all at m = 1, the practical variant."""
    rng = np.random.default_rng(seed + 7)
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
        Pr_ = ((counts + alpha) / (counts.sum(1, keepdims=True) + alpha * S)
               if mode == "plugin" else P)
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(S, p=P[a]))
        counts[a, s] += 1                       # the pair arrives with no delay
        pend[t] = (x, Pr_, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def main() -> None:
    kap = np.zeros(len(RHO))
    plug = np.zeros((len(ALPHA), len(RHO)))
    gain = np.zeros((len(ALPHA), len(RHO)))
    known = np.zeros(len(RHO))
    action = np.zeros(len(RHO))

    for j, rho in enumerate(RHO):
        ks, kn, ac = [], [], []
        for sd in range(SEEDS):
            P, theta, c, kk = env(rho, sd)
            ks.append(kk)
            kn.append(run(P, theta, c, D, "known", 0.0, sd))
            ac.append(run(P, theta, c, D, "action", 0.0, sd))
        kap[j], known[j], action[j] = np.mean(ks), np.mean(kn), np.mean(ac)
        for i, al in enumerate(ALPHA):
            pv = []
            for sd in range(SEEDS):
                P, theta, c, _ = env(rho, sd)
                pv.append(run(P, theta, c, D, "plugin", al, sd))
            plug[i, j] = np.mean(pv)
            gain[i, j] = action[j] - plug[i, j]
        print(f"  rho={rho:.3f} kappa={kap[j]:7.2f}  known {known[j]:7.1f}  "
              f"action {action[j]:7.1f}  plug-in " +
              " ".join(f"{plug[i, j]:7.1f}" for i in range(len(ALPHA))), flush=True)

    np.savez("alpha_kappa.npz", alpha=np.array(ALPHA), rho=np.array(RHO), kappa=kap,
             plug=plug, gain=gain, known=known, action=action)

    print("\nplug-in regret")
    print(f"{'alpha':>8}" + "".join(f"{k:9.1f}" for k in kap))
    for i, al in enumerate(ALPHA):
        print(f"{al:8.3f}" + "".join(f"{plug[i, j]:9.1f}" for j in range(len(RHO))))
    print(f"{'known P':>8}" + "".join(f"{known[j]:9.1f}" for j in range(len(RHO))))
    print(f"{'action':>8}" + "".join(f"{action[j]:9.1f}" for j in range(len(RHO))))

    print("\ngain over action-level weighting, negative means estimating P cost more than "
          "pooling bought")
    print(f"{'alpha':>8}" + "".join(f"{k:9.1f}" for k in kap))
    for i, al in enumerate(ALPHA):
        print(f"{al:8.3f}" + "".join(f"{gain[i, j]:9.1f}" for j in range(len(RHO))))

    best = [ALPHA[int(np.argmin(plug[:, j]))] for j in range(len(RHO))]
    print(f"\nbest alpha per kappa column: {best}")
    print(f"worst-to-best plug-in regret within a column: "
          f"{np.max(plug.max(0) / plug.min(0)):.2f}x")
    print(f"negative-gain cells: {int((gain < 0).sum())} of {gain.size}")


if __name__ == "__main__":
    main()
