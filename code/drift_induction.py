"""The deterministic drift induction behind Theorem 1 (Appendix J).

Theorem 2 needs eta <= 1/(e(d+1)), and its proof rests on a lemma that is claimed to hold
SURELY, not in expectation:

    x_{t+1}(a) <= (1 + 1/d) x_t(a)   for every t and every a,   hence  x_{t+d} <= e x_t.

The lemma turns on an identity peculiar to the state-pooled estimator, which this script also
checks pathwise:

    <x_t, chat_t> = sum_a x_t(a) P(S_t|a) X_t / q_t(S_t) = q_t(S_t) X_t / q_t(S_t) = X_t <= 1,

so the play-weighted mass of the estimate is bounded by one no matter how small q_t(S_t) gets.
The action-level estimator has K in that position, which is where Cesa-Bianchi, Gentile and
Mansour (2019) pick up the factor K in their learning-rate condition.

Reproduces the two numeric claims of Appendix J:
  * the funnel study's own learning rates satisfy the cap at every delay it reports;
  * the one-step ratio stays far below the permitted 1 + 1/d.

Nothing here is tuned: eta is exactly the rate funnel.py uses, so the runs are the published ones.
"""
from __future__ import annotations

import numpy as np

import funnel as fn

E = np.e
AUDIT_SEEDS = 3          # the appendix reports the maximum over these


def run_audit(P, theta, c, d, eta, seed):
    """Single-copy State-EXP3, instrumented for the two ratios the lemma bounds."""
    rng = np.random.default_rng(seed + 80_000)
    K, S = P.shape
    astar = int(c.sum(0).argmin())
    L = np.zeros(K)
    pend, xs = {}, {}
    loss, step, lag, inner = 0.0, 0.0, 0.0, 0.0
    prev = None
    for t in range(fn.T):
        r = t - d - 1
        chat = None
        if r in pend:
            qs, s_r, X_r = pend.pop(r)
            chat = P[:, s_r] * X_r / qs
            L += chat
        x = np.exp(-eta * (L - L.min())); x /= x.sum()
        if prev is not None:
            step = max(step, float(np.max(x / np.maximum(prev, 1e-300))))
        if chat is not None:
            inner = max(inner, float(x @ chat))          # <x_t, chat_{t-d-1}>, bounded by e
        if t - d in xs:
            lag = max(lag, float(np.max(x / np.maximum(xs[t - d], 1e-300))))
        xs[t] = x
        for k in [k for k in xs if k < t - d - 2]:
            del xs[k]
        prev = x
        q = x @ P
        a = int(rng.choice(K, p=x)); s = int(rng.choice(S, p=P[a]))
        pend[t] = (float(q[s]), s, float(rng.binomial(1, theta[t, s])))
        loss += float(c[t, a])
    return loss - float(c[:, astar].sum()), step, lag, inner


def main() -> None:
    logK = np.log(fn.K)
    print(f"funnel: K={fn.K} |S|={fn.S} T={fn.T}, eta as in funnel.py, "
          f"max over {AUDIT_SEEDS} seeds\n")
    print(f"{'d':>5}{'eta':>9}{'cap 1/(e(d+1))':>16}{'ok':>5}"
          f"{'step ratio':>12}{'<= 1+1/d':>10}{'lag ratio':>11}{'<= e':>8}{'<x,chat>':>10}")
    for d in fn.DELAYS:
        eta = float(np.sqrt(2.0 * logK / (fn.S * fn.T + d * fn.T)))
        cap = 1.0 / (E * (d + 1))
        step = lag = inner = 0.0
        for sd in range(AUDIT_SEEDS):
            P, theta, c, _ = fn.funnel_environment(sd)
            _, a, b, ci = run_audit(P, theta, c, d, eta, sd)
            step, lag, inner = max(step, a), max(lag, b), max(inner, ci)
        assert eta <= cap, f"eta exceeds the cap at d={d}"
        assert step <= 1.0 + 1.0 / d + 1e-9, f"one-step ratio violated at d={d}"
        assert lag <= E + 1e-9, f"d-step ratio violated at d={d}"
        assert inner <= E + 1e-9, f"play-weighted mass exceeded e at d={d}"
        print(f"{d:5d}{eta:9.4f}{cap:16.4f}{'yes':>5}"
              f"{step:12.3f}{1 + 1 / d:10.3f}{lag:11.3f}{E:8.3f}{inner:10.3f}")
    print("\nAll pathwise assertions held.  The lemma is claimed surely, so a single violation on "
          "any seed would refute it; none occurred.")


if __name__ == "__main__":
    main()
