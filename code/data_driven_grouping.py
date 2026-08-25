"""Choosing the state coarsening from data rather than from the latent truth.

Study 7 evaluates nested groupings of sizes 1, 2, 4, 8, 16 on an environment built from four
latent clusters, and reports that the regret minimum sits at four.  The experimenter knew the
cluster count; no algorithm did.  The paper flags this as a limitation of Theorem 2, which
analyses a representation fixed before the run and gives no rule for choosing one.

This file supplies a crude rule and measures what it costs.  It is exploratory.  There is no
theorem behind the selector, and none is claimed.

The rule.  Spend a prefix of length T0 on the identity representation.  From it estimate the
state-loss means thetahat(s) by averaging the outcomes observed at each state.  Sort the states by
thetahat and merge adjacent pairs greedily, which for a one-dimensional key is the family that
keeps within-group spread small.  That yields one candidate grouping at every granularity from
|S| down to 1.  Score each candidate by the quantity Theorem 2 bounds,

    sqrt( 2 (d+1) V(g) log K )  +  2 sum_t deltahat_t(g),

with V(g) = vbar(g) (T - T0) from the known P and deltahat from thetahat, then run the remaining
rounds on the argmin.  Everything the selector reads comes from the prefix.

Four arms, all on the same paired seeds and the same environment:

    identity     no coarsening, the raw |S| states
    oracle       the true latent clusters, which is study 7's winner and needs outside knowledge
    selected     the rule above
    hindsight    the best fixed nested grouping scored on the realised full-horizon regret,
                 which no causal algorithm can attain and which bounds what any selector could

Writes data_driven_grouping.npz and prints the table.
"""
from __future__ import annotations

import numpy as np

K, S, T, D = 40, 16, 6000, 20
PREFIX = 1200
SEEDS = 24
HETERO = (0.0, 0.12)


def clustered_environment(K, S, T, heterogeneity, seed):
    """Same construction as enrichment_experiments.py study 7, so the two are comparable."""
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
            P[action, lo:lo + per_cluster] = cluster_mass[action, cluster] * within
    cluster_base = np.array([0.15, 0.38, 0.62, 0.85])
    cluster_phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    offsets = rng.uniform(-1.0, 1.0, size=S)
    time = np.arange(T)[:, None] / T
    theta = np.empty((T, S))
    for state in range(S):
        cluster = state // per_cluster
        theta[:, state] = (cluster_base[cluster]
                           + 0.08 * np.sin(2.0 * np.pi * time[:, 0] + cluster_phase[cluster])
                           + heterogeneity * offsets[state])
    return P, np.clip(theta, 0.0, 1.0)


def aggregate(P, groups):
    Z = int(groups.max()) + 1
    out = np.zeros((P.shape[0], Z))
    for z in range(Z):
        out[:, z] = P[:, groups == z].sum(1)
    return out


def oracle_groups(S):
    return np.arange(S) // (S // 4)


def greedy_ladder(key):
    """Groupings at every granularity, merging adjacent pairs in the order of `key`."""
    order = np.argsort(key)
    blocks = [[i] for i in order]
    ladder = {}
    while True:
        g = np.empty(len(key), dtype=int)
        for bi, b in enumerate(blocks):
            for s in b:
                g[s] = bi
        ladder[len(blocks)] = g.copy()
        if len(blocks) == 1:
            break
        widths = [max(key[blocks[i] + blocks[i + 1]]) - min(key[blocks[i] + blocks[i + 1]])
                  for i in range(len(blocks) - 1)]
        j = int(np.argmin(widths))
        blocks[j:j + 2] = [blocks[j] + blocks[j + 1]]
    return ladder


def vbar_of(P, rng, draws=400):
    best = 1.0
    for _ in range(draws):
        x = rng.dirichlet(np.ones(P.shape[0]) * rng.choice([0.2, 1.0, 5.0]))
        q = x @ P
        best = max(best, float(np.sum(x[:, None] * P ** 2 / np.maximum(q, 1e-300))))
    return best


def run(P, theta, groups, d, m, eta, seed, t0=0, t1=None, counts=None):
    """State-EXP3 on the grouped states over rounds [t0, t1); optionally record state visits."""
    t1 = T if t1 is None else t1
    rng = np.random.default_rng(seed + 40_000)
    Pg = aggregate(P, groups)
    c = theta @ P.T
    L = np.zeros((m, K))
    pend, loss = {}, 0.0
    for t in range(t0, t1):
        r = t - d - 1
        if r in pend:
            i, qs, z_r, X_r = pend.pop(r)
            L[i] += Pg[:, z_r] * X_r / max(qs, 1e-12)
        i = t % m
        zz = -eta * (L[i] - L[i].min())
        x = np.exp(zz)
        x /= x.sum()
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(S, p=P[a]))
        X = rng.binomial(1, theta[t, s])
        if counts is not None:
            counts[0][s] += X
            counts[1][s] += 1
        z = int(groups[s])
        pend[t] = (i, float(x @ Pg[:, z]), z, X)
        loss += c[t, a]
    return loss


def select(P, thetahat, d, rng):
    """Score every candidate on the quantity Theorem 2 bounds, using prefix estimates only."""
    ladder = greedy_ladder(thetahat)
    rest = T - PREFIX
    best, best_g, table = np.inf, None, {}
    for nz, g in ladder.items():
        vb = vbar_of(aggregate(P, g), rng, draws=250)
        delta = 0.0
        for z in range(int(g.max()) + 1):
            members = thetahat[g == z]
            if members.size:
                delta = max(delta, float(members.max() - members.min()))
        score = np.sqrt(2.0 * (d + 1) * vb * rest * np.log(K)) + 2.0 * rest * delta
        table[nz] = score
        if score < best:
            best, best_g = score, g
    return best_g, table


def main() -> None:
    rows = []
    for het in HETERO:
        acc = {k: [] for k in ("identity", "oracle", "selected", "hindsight")}
        picked = []
        for sd in range(SEEDS):
            P, theta = clustered_environment(K, S, T, het, sd)
            c = theta @ P.T
            best_fixed = c.sum(0).min()
            rng = np.random.default_rng(sd + 91)

            # --- prefix on the identity representation, shared by every arm
            counts = [np.zeros(S), np.zeros(S)]
            idg = np.arange(S)
            vb_id = vbar_of(P, rng, draws=250)
            eta_id = np.sqrt(2.0 * np.log(K) / (vb_id * (T / (D + 1))))
            pre_loss = run(P, theta, idg, D, D + 1, eta_id, sd, 0, PREFIX, counts)
            thetahat = np.where(counts[1] > 0, counts[0] / np.maximum(counts[1], 1), 0.5)

            sel_g, _ = select(P, thetahat, D, rng)
            picked.append(int(sel_g.max()) + 1)

            arms = {"identity": idg, "oracle": oracle_groups(S), "selected": sel_g}
            ladder = greedy_ladder(thetahat)
            for name, g in list(arms.items()):
                vb = vbar_of(aggregate(P, g), rng, draws=250)
                eta = np.sqrt(2.0 * np.log(K) / (vb * (T / (D + 1))))
                tail = run(P, theta, g, D, D + 1, eta, sd, PREFIX, T)
                acc[name].append(pre_loss + tail - best_fixed)
            # hindsight over the same candidate ladder, scored on realised regret
            hs = []
            for nz, g in ladder.items():
                vb = vbar_of(aggregate(P, g), rng, draws=250)
                eta = np.sqrt(2.0 * np.log(K) / (vb * (T / (D + 1))))
                hs.append(pre_loss + run(P, theta, g, D, D + 1, eta, sd, PREFIX, T) - best_fixed)
            acc["hindsight"].append(min(hs))

        row = dict(hetero=het, picked_mean=float(np.mean(picked)),
                   picked_mode=int(np.bincount(picked).argmax()))
        for k, v in acc.items():
            row[k] = float(np.mean(v))
            row[k + "_se"] = float(np.std(v, ddof=1) / np.sqrt(SEEDS))
        rows.append(row)
        print(f"heterogeneity {het:.2f}  groups picked: mean {np.mean(picked):.2f}, "
              f"mode {np.bincount(picked).argmax()}", flush=True)
        for k in ("identity", "oracle", "selected", "hindsight"):
            print(f"    {k:<10} {row[k]:8.1f} +- {row[k+'_se']:.1f}", flush=True)

    np.savez("data_driven_grouping.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    for r in rows:
        gap = 100.0 * (r["selected"] - r["oracle"]) / r["oracle"]
        vs_id = 100.0 * (r["identity"] - r["selected"]) / r["identity"]
        print(f"\nheterogeneity {r['hetero']:.2f}: selected is {gap:+.1f}% against the oracle "
              f"grouping and {vs_id:+.1f}% better than no coarsening")


if __name__ == "__main__":
    main()
