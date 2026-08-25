"""Focused tests for two State-EXP3 enrichments.

1. Effective sensor dimension: the exact variance proxy

       v_t = sum_s [sum_a x_t(a) P(s|a)^2] / q_t(s)

   lies in [1, |S|].  We vary overlap among the rows of P and check whether
   measured estimator complexity and regret track v_t rather than raw |S|.

2. State abstraction: replace S by a fixed grouping g(S).  Coarser groups
   reduce variance but can bias the estimator when losses differ within a
   group.  We generate a four-cluster sensor and sweep the representation
   size to look for the predicted bias--variance tradeoff.

Feedback timing matches paper/main.tex: the outcome from round r arrives
after action r+d, and is therefore usable before action r+d+1.
"""
from __future__ import annotations

import numpy as np


def _softmax_loss(loss: np.ndarray, eta: float) -> np.ndarray:
    z = -eta * (loss - loss.min())
    z = np.clip(z, -700.0, 0.0)
    x = np.exp(z)
    return x / x.sum()


def _aggregate_transition(P: np.ndarray, groups: np.ndarray) -> np.ndarray:
    n_groups = int(groups.max()) + 1
    out = np.zeros((P.shape[0], n_groups))
    for s, group in enumerate(groups):
        out[:, group] += P[:, s]
    return out


def run_state_exp3(
    P: np.ndarray,
    theta: np.ndarray,
    delay: int,
    seed: int,
    groups: np.ndarray | None = None,
    interleaved: bool = False,
) -> tuple[float, float]:
    """Return static pseudo-regret and the average exact variance proxy."""
    rng = np.random.default_rng(seed)
    T, S = theta.shape
    K = P.shape[0]
    if groups is None:
        groups = np.arange(S)
    Pg = _aggregate_transition(P, groups)
    G = Pg.shape[1]
    copies = delay + 1 if interleaved else 1
    local_horizon = T / copies
    local_delay_charge = 0 if interleaved else delay
    eta = np.sqrt(2.0 * np.log(K) / ((G + local_delay_charge) * local_horizon))
    losses = np.zeros((copies, K))
    pending: dict[int, tuple[int, np.ndarray, int, float]] = {}
    c = theta @ P.T
    best = float(c.sum(axis=0).min())
    learner = 0.0
    variance_sum = 0.0

    for t in range(T):
        # The feedback produced at r is usable only after action r+d, hence
        # before action r+d+1.  This is the manuscript's filtration.
        arrived = t - delay - 1
        if arrived in pending:
            copy, x_old, group_old, outcome_old = pending.pop(arrived)
            q_old = x_old @ Pg
            losses[copy] += Pg[:, group_old] * outcome_old / max(q_old[group_old], 1e-300)

        copy = t % copies
        x = _softmax_loss(losses[copy], eta)
        q = x @ Pg
        numerator = (x[:, None] * Pg**2).sum(axis=0)
        variance_sum += float((numerator[q > 0] / q[q > 0]).sum())

        action = int(rng.choice(K, p=x))
        state = int(rng.choice(S, p=P[action]))
        outcome = float(rng.binomial(1, theta[t, state]))
        pending[t] = (copy, x, int(groups[state]), outcome)
        learner += float(c[t, action])

    return learner - best, variance_sum / T


def run_action_exp3(P: np.ndarray, theta: np.ndarray, delay: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    T = theta.shape[0]
    K = P.shape[0]
    eta = np.sqrt(2.0 * np.log(K) / ((K + delay) * T))
    losses = np.zeros(K)
    pending: dict[int, tuple[np.ndarray, int, float]] = {}
    c = theta @ P.T
    learner = 0.0

    for t in range(T):
        arrived = t - delay - 1
        if arrived in pending:
            x_old, action_old, outcome_old = pending.pop(arrived)
            losses[action_old] += outcome_old / max(x_old[action_old], 1e-300)
        x = _softmax_loss(losses, eta)
        action = int(rng.choice(K, p=x))
        state = int(rng.choice(P.shape[1], p=P[action]))
        outcome = float(rng.binomial(1, theta[t, state]))
        pending[t] = (x, action, outcome)
        learner += float(c[t, action])
    return learner - float(c.sum(axis=0).min())


def overlap_environment(K: int, S: int, T: int, overlap: float, seed: int):
    rng = np.random.default_rng(seed)
    common = rng.dirichlet(np.ones(S) * 2.0)
    individual = rng.dirichlet(np.ones(S) * 0.25, size=K)
    P = overlap * common[None, :] + (1.0 - overlap) * individual
    base = rng.uniform(0.15, 0.85, size=S)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=S)
    time = np.arange(T)[:, None] / T
    theta = np.clip(base + 0.12 * np.sin(2.0 * np.pi * time + phase), 0.0, 1.0)
    return P, theta


def clustered_environment(K: int, S: int, T: int, heterogeneity: float, seed: int):
    if S % 4:
        raise ValueError("S must be divisible by four")
    rng = np.random.default_rng(seed)
    per_cluster = S // 4
    cluster_mass = rng.dirichlet(np.ones(4) * 0.7, size=K)
    P = np.zeros((K, S))
    for action in range(K):
        for cluster in range(4):
            within = rng.dirichlet(np.ones(per_cluster) * 1.2)
            lo = cluster * per_cluster
            P[action, lo : lo + per_cluster] = cluster_mass[action, cluster] * within

    cluster_base = np.array([0.15, 0.38, 0.62, 0.85])
    cluster_phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    offsets = rng.uniform(-1.0, 1.0, size=S)
    time = np.arange(T)[:, None] / T
    theta = np.empty((T, S))
    for state in range(S):
        cluster = state // per_cluster
        theta[:, state] = (
            cluster_base[cluster]
            + 0.08 * np.sin(2.0 * np.pi * time[:, 0] + cluster_phase[cluster])
            + heterogeneity * offsets[state]
        )
    return P, np.clip(theta, 0.0, 1.0)


def nested_groups(S: int, n_groups: int) -> np.ndarray:
    """Nested, cluster-aligned partitions for S=16 and group counts 1,2,4,8,16."""
    if S % n_groups:
        raise ValueError("n_groups must divide S")
    return np.arange(S) // (S // n_groups)


def group_diameter(theta: np.ndarray, groups: np.ndarray) -> float:
    total = 0.0
    for t in range(theta.shape[0]):
        diameter = 0.0
        for group in range(int(groups.max()) + 1):
            values = theta[t, groups == group]
            diameter = max(diameter, float(values.max() - values.min()))
        total += diameter
    return total / theta.shape[0]


def test_effective_dimension():
    K, S, T, delay, seeds = 40, 8, 5000, 20, 24
    print("\nEFFECTIVE SENSOR DIMENSION")
    print(f"K={K} S={S} T={T} d={delay}, {seeds} seeds; manuscript timing")
    print(f"{'overlap':>8} {'d_eff':>9} {'state regret':>14} {'action regret':>15} {'gain':>11}")
    for overlap in (0.0, 0.25, 0.5, 0.75, 0.95):
        state_regret, action_regret, dimensions = [], [], []
        for seed in range(seeds):
            P, theta = overlap_environment(K, S, T, overlap, seed)
            regret, dimension = run_state_exp3(P, theta, delay, 10_000 + seed)
            state_regret.append(regret)
            dimensions.append(dimension)
            action_regret.append(run_action_exp3(P, theta, delay, 10_000 + seed))
        state_regret = np.asarray(state_regret)
        action_regret = np.asarray(action_regret)
        gain = action_regret - state_regret
        print(
            f"{overlap:8.2f} {np.mean(dimensions):9.3f} {state_regret.mean():14.1f} "
            f"{action_regret.mean():15.1f} {gain.mean():7.1f}+-{gain.std(ddof=1)/np.sqrt(seeds):3.1f}"
        )


def test_state_abstraction():
    K, S, T, delay, seeds = 40, 16, 6000, 20, 24
    print("\nSTATE ABSTRACTION")
    print(f"K={K} raw-S={S} T={T} d={delay}, {seeds} seeds; four latent clusters")
    for heterogeneity in (0.0, 0.04, 0.12):
        print(f"\nwithin-cluster heterogeneity={heterogeneity:.2f}")
        print(f"{'groups':>8} {'diameter':>10} {'d_eff':>9} {'plain regret':>14} {'interleaved':>14}")
        for n_groups in (1, 2, 4, 8, 16):
            groups = nested_groups(S, n_groups)
            plain, safe, dimensions, diameters = [], [], [], []
            for seed in range(seeds):
                P, theta = clustered_environment(K, S, T, heterogeneity, seed)
                regret, dimension = run_state_exp3(
                    P, theta, delay, 20_000 + seed, groups=groups, interleaved=False
                )
                safe_regret, _ = run_state_exp3(
                    P, theta, delay, 20_000 + seed, groups=groups, interleaved=True
                )
                plain.append(regret)
                safe.append(safe_regret)
                dimensions.append(dimension)
                diameters.append(group_diameter(theta, groups))
            print(
                f"{n_groups:8d} {np.mean(diameters):10.3f} {np.mean(dimensions):9.3f} "
                f"{np.mean(plain):14.1f} {np.mean(safe):14.1f}"
            )


if __name__ == "__main__":
    test_effective_dimension()
    test_state_abstraction()
