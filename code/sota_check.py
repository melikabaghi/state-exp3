"""SOTA check, not yet in the paper: tuned Zimmert-Seldin against State-EXP3.

Every reported gain so far compares State-EXP3 against action-level EXP3, EXP3-IX, naive
sharing, or the best-state rule.  None of those is the strongest published algorithm for
delayed bandits: that is the hybrid Tsallis+entropy FTRL of Zimmert and Seldin (2020), whose
delay term is additive rather than multiplicative.  algo.py already implements it faithfully
(class ZS / ftrl_hybrid); it was used in the lower-bound scaling study but never run on the
headline families.  This script runs it there.

Three tunings are reported, and only the third is symmetric:
  1. ZS at the best of a bounded grid around its own theory value; State-EXP3 at the single
     tuning used everywhere else in the paper.  This favors ZS.
  2. ZS additionally given an unbounded oracle sweep over the learning rate, State-EXP3 still
     at its paper tuning.  This favors ZS more, and ZS wins four of the six cells.
  3. Both given the same unbounded oracle sweep.  Symmetric.
Oracle tuning is not an algorithm; it is reported to bound how much of the gap is tuning.

Families:
  funnel          -- the 79 per cent headline claim (K=200, |S|=6)
  matched control -- the overlap sweep behind the abstract's structural claim (K=40, |S|=8)
Sanity: ZS ignores states, so on both families it should behave like a well-tuned
action-level method, and its regret should be sublinear-looking, not broken.

Writes sota_check.npz and prints tables.  Nothing here feeds the paper yet.
"""
from __future__ import annotations

import numpy as np

import funnel as fn
import matched_control as mc
from algo import ftrl_hybrid

ZS_GRID = [(em, sh) for em in (0.5, 1.0, 2.0) for sh in (True, False)]
ZS_ORACLE = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)     # shift=False won every bounded cell
ST_ORACLE = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def run_zs(P, theta, c, d, T, K, seed, rng_offset, eta_mult, shift):
    """Zimmert-Seldin hybrid FTRL, action-level IW estimator, same delay convention
    and pairing offsets as the study drivers it is compared against."""
    rng = np.random.default_rng(seed + rng_offset)
    astar = int(c.sum(0).argmin())
    eta = eta_mult * np.sqrt(np.log(max(K, 2)) / (max(d, 1) * T))
    t0 = 64.0 * d * d if shift else 0.0
    L = np.zeros(K)
    u = mu = None
    pend, loss = {}, 0.0
    for t in range(T):
        r = t - d - 1
        if r in pend:
            xa, a_r, X_r = pend.pop(r)
            L[a_r] += X_r / max(xa, 1e-300)
        alpha = 1.0 / np.sqrt(t + 1.0 + t0)
        x, u, mu = ftrl_hybrid(L, alpha, eta, u, mu)
        a = int(rng.choice(K, p=x))
        s = int(rng.choice(P.shape[1], p=P[a]))
        X = float(rng.binomial(1, theta[t, s]))
        pend[t] = (float(x[a]), a, X)
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def run_state_funnel(P, theta, c, d, seed, eta_mult):
    """fn.run fixes eta internally; this mirrors it exactly with eta scaled by eta_mult."""
    rng = np.random.default_rng(seed + 80_000)
    astar = int(c.sum(0).argmin())
    eta = eta_mult * np.sqrt(2.0 * np.log(fn.K) / (fn.S * fn.T + d * fn.T))
    L = np.zeros(fn.K); Pcum = np.cumsum(P, axis=1); pend, loss = {}, 0.0
    for t in range(fn.T):
        r = t - d - 1
        if r in pend:
            x, a_r, s_r, X_r = pend.pop(r)
            L += P[:, s_r] * X_r / max(float(x @ P[:, s_r]), 1e-300)
        z = -eta * (L - L.min()); x = np.exp(z); x /= x.sum()
        a = fn.draw(np.cumsum(x), rng.random()); s = fn.draw(Pcum[a], rng.random())
        pend[t] = (x, a, s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum())


def funnel_block(seeds=fn.SEEDS):
    print(f"== funnel family: K={fn.K} |S|={fn.S} T={fn.T}, {seeds} paired seeds ==")
    rows = {}
    for d in fn.DELAYS:
        st, ac = [], []
        zs = {g: [] for g in ZS_GRID}
        for sd in range(seeds):
            P, theta, c, _ = fn.funnel_environment(sd)
            st.append(fn.run(P, theta, c, d, "state", sd))
            ac.append(fn.run(P, theta, c, d, "action", sd))
            for g in ZS_GRID:
                zs[g].append(run_zs(P, theta, c, d, fn.T, fn.K, sd, 80_000, *g))
        zs_orc = {em: np.mean([run_zs(*fn.funnel_environment(sd)[:3], d, fn.T, fn.K,
                                      sd, 80_000, em, False) for sd in range(seeds)])
                  for em in ZS_ORACLE}
        st_orc = {em: np.mean([run_state_funnel(*fn.funnel_environment(sd)[:3], d, sd, em)
                               for sd in range(seeds)]) for em in ST_ORACLE}
        best_g = min(ZS_GRID, key=lambda g: np.mean(zs[g]))
        b = np.array(zs[best_g]); st, ac = np.array(st), np.array(ac)
        rows[d] = dict(state=st.mean(), action=ac.mean(), zs=b.mean(),
                       zs_oracle=min(zs_orc.values()), state_oracle=min(st_orc.values()),
                       zs_grid=best_g,
                       gain_vs_action=100 * (1 - st.mean() / ac.mean()),
                       gain_vs_zs=100 * (1 - st.mean() / b.mean()),
                       se=float((b - st).std(ddof=1) / np.sqrt(seeds)))
        r = rows[d]
        print(f"  d={d:3d}  state {r['state']:8.1f}   action {r['action']:8.1f}   "
              f"ZS* {r['zs']:8.1f} (eta x{best_g[0]}, shift={best_g[1]})   "
              f"cut vs action {r['gain_vs_action']:5.1f}%   cut vs ZS* {r['gain_vs_zs']:5.1f}%")
        print(f"        oracle: ZS {r['zs_oracle']:8.1f}   state {r['state_oracle']:8.1f}")
    return rows


def matched_block(seeds=10, overlaps=(0.00, 0.65, 0.95)):
    print(f"\n== matched-control family: K={mc.K} |S|={mc.S} T={mc.T} d={mc.D}, "
          f"{seeds} paired seeds, difficulty matched to {mc.TARGET:.0f} ==")
    rows = {}
    for ov in overlaps:
        st, ac = [], []
        zs = {g: [] for g in ZS_GRID}
        for sd in range(seeds):
            P, contrast = mc.build_P(ov, sd)
            scale = mc.match_scale(P, contrast, sd)
            theta = mc.theta_of(scale, contrast, sd)
            c = theta @ P.T
            vb = mc.vbar(P, np.random.default_rng(sd + 3), draws=600)
            eta_s = np.sqrt(2.0 * np.log(mc.K) / (vb * mc.T + mc.D * mc.T))
            st.append(mc.run(P, theta, mc.D, "state", eta_s, sd))
            eta_a = np.sqrt(2.0 * np.log(mc.K) / (mc.K * mc.T + mc.D * mc.T))
            ac.append(mc.run(P, theta, mc.D, "action", eta_a, sd))
            for g in ZS_GRID:
                zs[g].append(run_zs(P, theta, c, mc.D, mc.T, mc.K, sd, 20_000, *g))
        def _mc(em, mode):
            out = []
            for sd in range(seeds):
                P, con = mc.build_P(ov, sd); sc = mc.match_scale(P, con, sd)
                th = mc.theta_of(sc, con, sd)
                if mode == "zs":
                    out.append(run_zs(P, th, th @ P.T, mc.D, mc.T, mc.K, sd, 20_000, em, False))
                else:
                    vb = mc.vbar(P, np.random.default_rng(sd + 3), draws=600)
                    eta = em * np.sqrt(2.0 * np.log(mc.K) / (vb * mc.T + mc.D * mc.T))
                    out.append(mc.run(P, th, mc.D, "state", eta, sd))
            return float(np.mean(out))
        zs_orc = min(_mc(em, "zs") for em in ZS_ORACLE)
        st_orc = min(_mc(em, "state") for em in ST_ORACLE)
        best_g = min(ZS_GRID, key=lambda g: np.mean(zs[g]))
        b = np.array(zs[best_g]); st, ac = np.array(st), np.array(ac)
        rows[ov] = dict(state=st.mean(), action=ac.mean(), zs=b.mean(), zs_grid=best_g,
                        zs_oracle=zs_orc, state_oracle=st_orc,
                        gain_vs_zs=100 * (1 - st.mean() / b.mean()),
                        se=float((b - st).std(ddof=1) / np.sqrt(seeds)))
        r = rows[ov]
        print(f"  overlap {ov:.2f}  state {r['state']:7.1f}   action {r['action']:7.1f}   "
              f"ZS* {r['zs']:7.1f} (eta x{best_g[0]}, shift={best_g[1]})   "
              f"cut vs ZS* {r['gain_vs_zs']:5.1f}%")
        print(f"              oracle: ZS {r['zs_oracle']:7.1f}   state {r['state_oracle']:7.1f}")
    return rows


def main() -> None:
    fr = funnel_block()
    mr = matched_block()
    np.savez("sota_check.npz",
             funnel_d=np.array(sorted(fr)),
             funnel_state=np.array([fr[d]["state"] for d in sorted(fr)]),
             funnel_action=np.array([fr[d]["action"] for d in sorted(fr)]),
             funnel_zs=np.array([fr[d]["zs"] for d in sorted(fr)]),
             funnel_zs_oracle=np.array([fr[d]["zs_oracle"] for d in sorted(fr)]),
             funnel_state_oracle=np.array([fr[d]["state_oracle"] for d in sorted(fr)]),
             mc_overlap=np.array(sorted(mr)),
             mc_state=np.array([mr[o]["state"] for o in sorted(mr)]),
             mc_action=np.array([mr[o]["action"] for o in sorted(mr)]),
             mc_zs=np.array([mr[o]["zs"] for o in sorted(mr)]),
             mc_zs_oracle=np.array([mr[o]["zs_oracle"] for o in sorted(mr)]),
             mc_state_oracle=np.array([mr[o]["state_oracle"] for o in sorted(mr)]))
    print("\nsanity: ZS* should sit near (or below) action-level everywhere, since both "
          "ignore the state; a ZS* far above action-level would mean the implementation "
          "or tuning is broken.")


if __name__ == "__main__":
    main()
