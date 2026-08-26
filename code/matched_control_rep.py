"""Study 10b: does the matched-difficulty overlap trend replicate at a second configuration?

Study 10 runs one design point, K=40, |S|=8, T=10000, d=50.  Three review rounds flagged that the
paper's structural claim -- that the pooling advantage GROWS as vbar falls once task difficulty is
held fixed -- rests on that single cell, and that it reverses the sign of study 6's unmatched
sweep.  A trend supported by one configuration is not a trend.

This re-runs the identical procedure at a second, deliberately different point: a smaller catalogue,
a larger state space and a shorter delay.  Nothing else changes -- same construction, same matching
routine, same estimator, same paired seeds -- so a sign flip here would mean study 10 is not
reporting a mechanism.

Writes matched_control_rep.npz and prints the table.
"""
from __future__ import annotations

import numpy as np

import matched_control as mc

# Second design point: K down, |S| up, d down.  TARGET2 must be passed explicitly to
# match_scale -- its `target=TARGET` default binds at definition time, so rebinding
# mc.TARGET does nothing and the bisection would silently saturate.
K2, S2, T2, D2 = 20, 12, 8000, 20
TARGET2 = 500.0


def main() -> None:
    mc.K, mc.S, mc.T, mc.D, mc.TARGET = K2, S2, T2, D2, TARGET2
    K, S, T, D = K2, S2, T2, D2

    rows = []
    print(f"K={K} |S|={S} T={T} d={D}, uniform-play regret matched to {TARGET2:.0f} in every cell\n")
    print(f"{'overlap':>8}{'vbar':>8}{'R_unif':>9}{'gap/round':>11}"
          f"{'state':>9}{'action':>9}{'gain':>9}{'gain %':>8}")
    for ov in mc.OVERLAP:
        st, ac, ru = [], [], []
        for sd in range(mc.SEEDS):
            P, contrast = mc.build_P(ov, sd)
            scale = mc.match_scale(P, contrast, sd, target=TARGET2)
            theta = mc.theta_of(scale, contrast, sd)
            ru.append(mc.unif_regret(P, theta))
            vb = mc.vbar(P, np.random.default_rng(sd + 3), draws=600)
            eta_s = np.sqrt(2.0 * np.log(K) / (vb * T + D * T))
            eta_a = np.sqrt(2.0 * np.log(K) / (K * T + D * T))
            st.append(mc.run(P, theta, D, "state", eta_s, sd))
            ac.append(mc.run(P, theta, D, "action", eta_a, sd))
        vb = mc.vbar(mc.build_P(ov, 0)[0], np.random.default_rng(11))
        st, ac = np.array(st), np.array(ac)
        gain = ac - st
        rows.append(dict(overlap=ov, vbar=vb, r_unif=float(np.mean(ru)),
                         state=float(st.mean()), action=float(ac.mean()),
                         gain=float(gain.mean()),
                         gain_se=float(gain.std(ddof=1) / np.sqrt(mc.SEEDS))))
        print(f"{ov:8.2f}{vb:8.3f}{np.mean(ru):9.1f}{np.mean(ru)/T:11.4f}"
              f"{st.mean():9.1f}{ac.mean():9.1f}"
              f"{gain.mean():6.1f}+-{gain.std(ddof=1)/np.sqrt(mc.SEEDS):<4.1f}"
              f"{100*gain.mean()/ac.mean():7.1f}", flush=True)

    np.savez("matched_control_rep.npz",
             **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    g = np.array([r["gain"] for r in rows])
    v = np.array([r["vbar"] for r in rows])
    ru = np.array([r["r_unif"] for r in rows])
    print(f"\nuniform-play regret across cells: {ru.min():.1f} to {ru.max():.1f} "
          f"(spread {ru.max()/ru.min():.3f}x)")
    print(f"corr(vbar, absolute gain) = {np.corrcoef(v, g)[0, 1]:+.3f}")
    print(f"relative gain at highest vbar {v.max():.3f}: "
          f"{100*g[np.argmax(v)]/rows[int(np.argmax(v))]['action']:.1f} per cent")
    print(f"relative gain at lowest  vbar {v.min():.3f}: "
          f"{100*g[np.argmin(v)]/rows[int(np.argmin(v))]['action']:.1f} per cent")
    print("\nReplication verdict: the sign of corr(vbar, gain) should match study 10's -0.929. "
          "A positive correlation here would refute the paper's structural claim.")


if __name__ == "__main__":
    main()
