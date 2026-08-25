"""Algorithms for the drifting-intermediate-observation bandit.

Model (BIO): at round t the learner picks A_t, immediately observes a state
S_t ~ P(.|A_t) with P stationary, and observes the loss X_t (mean theta_t(S_t))
at time t+d.  Induced action-loss vector c_t = P theta_t.  Oracle hint m_t = c_{t-d}.

Implements our optimistic hybrid-FTRL method and the baselines it is compared against.
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------- hybrid FTRL solver

def _g_and_dg(u, inv2a, inv_eta):
    """g(e^u) and d/du g(e^u) for g(x) = -(1/(2a)) x^{-1/2} + (1/eta) log x."""
    e = np.exp(-0.5 * u)
    return -inv2a * e + inv_eta * u, 0.5 * inv2a * e + inv_eta


def ftrl_hybrid(L, alpha, eta, u0=None, mu0=None, outer=60, inner=40, tol=1e-11):
    """argmin_{x in simplex} <x,L> + F(x),  F = -(1/alpha) sum sqrt(x) + (1/eta) sum x log x.

    Stationarity: g(x_a) = mu - L_a with g increasing and onto R, so the inner solve
    is a monotone 1-D root find (done in u = log x for stability) and the outer solve
    is a monotone root find on mu enforcing sum_a x_a = 1.  Both are warm-started.
    Returns (x, u, mu).
    """
    inv2a, inv_eta = 1.0 / (2.0 * alpha), 1.0 / eta
    K = L.shape[0]
    mu = float(np.median(L) if mu0 is None else mu0)
    u = (np.full(K, -np.log(K)) if u0 is None else u0).copy()

    for _ in range(outer):
        nu = mu - L
        for _ in range(inner):                       # inner Newton in u (monotone)
            g, dg = _g_and_dg(u, inv2a, inv_eta)
            step = (g - nu) / dg
            step = np.clip(step, -4.0, 4.0)          # damping
            u -= step
            if np.max(np.abs(step)) < tol:
                break
        x = np.exp(u)
        s = x.sum()
        if abs(s - 1.0) < tol:
            break
        _, dg = _g_and_dg(u, inv2a, inv_eta)
        dsdmu = float(np.sum(x / dg))                # d(sum x)/d(mu) > 0
        mu -= (s - 1.0) / max(dsdmu, 1e-300)
    x = np.exp(u)
    x = np.maximum(x, 1e-300)
    return x / x.sum(), u, mu


# --------------------------------------------------------------- algorithms

class Base:
    name = "base"
    def __init__(self, K, S, d, T, P=None, **kw):
        self.K, self.S, self.d, self.T, self.P = K, S, d, T, P
    def act(self, t, rng, hint=None):  raise NotImplementedError
    def update(self, t, a, s, loss_feedback, hint=None):  pass


class Ours(Base):
    """Hybrid Tsallis+entropy FTRL, optimistic step at gamma=(2d+1)eta on the hint,
    centered importance-weighted estimator (unbiased for c_t)."""
    name = "Ours"
    def __init__(self, K, S, d, T, eta=None, shift=True, optimistic=True,
                 centered=True, E1=None, **kw):
        super().__init__(K, S, d, T, **kw)
        self.t0 = 64.0 * d * d if shift else 0.0
        # Theorem tuning: eta* = sqrt(log K / (d * E1)).  With no hint the delay term
        # is eta * d * T, recovering Zimmert-Seldin's eta* = sqrt(log K / (d*T)).
        # Tuning to E1 rather than T is precisely the mechanism that separates the two.
        if eta is None:
            budget = max(d, 1) * (T if E1 is None else max(E1, 1.0))
            eta = np.sqrt(np.log(max(K, 2)) / budget)
        self.eta = float(eta)
        self.gamma_mult = (2.0 * d + 1.0) if optimistic else 1.0
        self.centered = centered
        # Before act(t), L contains every past hint and every residual whose delayed
        # feedback arrived before t.  Hints are immediate; only residuals are pending.
        self.L = np.zeros(K)
        self.pend = {}                # s -> (a, x_s, m_s)
        self._u = None; self._mu = None; self._x = np.full(K, 1.0 / K)

    def _alpha(self, t):
        return 1.0 / np.sqrt(t + 1.0 + self.t0)

    def act(self, t, rng, hint=None):
        Lopt = self.L.copy()
        if hint is not None and self.gamma_mult > 1.0:
            Lopt = Lopt + self.gamma_mult * hint          # aggressive optimistic step
        elif hint is not None:
            Lopt = Lopt + hint
        x, self._u, self._mu = ftrl_hybrid(Lopt, self._alpha(t), self.eta,
                                           self._u, self._mu)
        self._x = x
        return int(rng.choice(self.K, p=x))

    def update(self, t, a, s, loss_feedback, hint=None):
        """Register round ``t`` and process the loss from round ``t-d``, if any."""
        self.pend[t] = (a, self._x[a], None if hint is None else hint.copy())
        arr = t - self.d
        if arr in self.pend:
            if loss_feedback is None:
                raise ValueError(f"missing loss feedback for round {arr} at time {t}")
            aa, xa, ms = self.pend.pop(arr)
            if self.centered and ms is not None:
                # hat_ell_s = m_s + 1{A_s=a}(X_s - m_s(a))/x_{s,a}  -- unbiased for c_s.
                # m_s entered L immediately after round s.  At arrival, add only its
                # importance-weighted correction; this is why only residuals are delayed.
                self.L[aa] += (loss_feedback - ms[aa]) / max(xa, 1e-12)
            else:
                self.L[aa] += loss_feedback / max(xa, 1e-12)
        if self.centered and hint is not None:
            # The current hint is known now and is part of the cumulative estimate for
            # the next decision, without waiting d rounds for its residual.
            self.L += hint


class ZS(Ours):
    """Zimmert-Seldin hybrid FTRL with plain IW estimator and no hint."""
    name = "ZS (no hint)"
    def __init__(self, K, S, d, T, **kw):
        kw.update(optimistic=False, centered=False)
        super().__init__(K, S, d, T, **kw)
    def act(self, t, rng, hint=None):
        return super().act(t, rng, hint=None)
    def update(self, t, a, s, loss_feedback, hint=None):
        super().update(t, a, s, loss_feedback, hint=None)


class Exp3Delay(Base):
    """Exponential weights with delayed IW feedback (entropy only)."""
    name = "Exp3-delay"
    def __init__(self, K, S, d, T, eta=None, **kw):
        super().__init__(K, S, d, T, **kw)
        self.eta = eta if eta is not None else np.sqrt(np.log(max(K, 2)) / (max(T, 2) * (d + 1)))
        self.L = np.zeros(K); self.pend = {}; self._x = np.full(K, 1.0 / K)
    def act(self, t, rng, hint=None):
        w = np.exp(-self.eta * (self.L - self.L.min()))
        self._x = w / w.sum()
        return int(rng.choice(self.K, p=self._x))
    def update(self, t, a, s, loss_feedback, hint=None):
        self.pend[t] = (a, self._x[a])
        arr = t - self.d
        if arr in self.pend:
            if loss_feedback is None:
                raise ValueError(f"missing loss feedback for round {arr} at time {t}")
            aa, xa = self.pend.pop(arr)
            self.L[aa] += loss_feedback / max(xa, 1e-12)


class GreedyHint(Base):
    """Play argmin of the stale hint."""
    name = "Greedy-hint"
    def act(self, t, rng, hint=None):
        return int(np.argmin(hint)) if hint is not None else int(rng.integers(self.K))


class MetaBIO(Base):
    """Esposito-style state imputation: pool ALL observations per state to estimate
    theta(s), impute the action loss through the (known) stationary P.  Assumes the
    state-to-loss map is time-homogeneous -- exactly the assumption drift violates."""
    name = "MetaBIO"
    def __init__(self, K, S, d, T, P, window=None, **kw):
        super().__init__(K, S, d, T, P=P, **kw)
        self.window = window
        self.buf = [[] for _ in range(S)]
        self.pend = {}
        self.eps = 0.05
    def act(self, t, rng, hint=None):
        if rng.random() < self.eps:
            return int(rng.integers(self.K))
        th = np.array([np.mean(b[-self.window:] if self.window else b) if b else 0.5
                       for b in self.buf])
        return int(np.argmin(self.P @ th))
    def update(self, t, a, s, loss_feedback, hint=None):
        self.pend[t] = s
        arr = t - self.d
        if arr in self.pend:
            if loss_feedback is None:
                raise ValueError(f"missing loss feedback for round {arr} at time {t}")
            self.buf[self.pend.pop(arr)].append(loss_feedback)


class OracleC(Base):
    """Cheating reference: knows c_t exactly."""
    name = "Oracle-c"
    def __init__(self, K, S, d, T, c=None, **kw):
        super().__init__(K, S, d, T, **kw); self.c = c
    def act(self, t, rng, hint=None):
        return int(np.argmin(self.c[t]))


# --------------------------------------------------------------- environment / runner

def make_env(K, S, d, T, drift, rng, seed_env=0):
    """Stationary P, drifting theta_t.  drift in {'none','jump','smooth'} with a
    magnitude set by `drift[1]`; returns P, theta, c, hint."""
    r = np.random.default_rng(seed_env)
    P = r.random((K, S)) + 0.1
    P /= P.sum(1, keepdims=True)
    kind, mag = drift
    th = np.empty((T, S))
    base = r.random(S)
    if kind == "none":
        th[:] = base
    elif kind == "jump":                       # piecewise constant, blocks of length d
        B = max(T // max(d, 1), 1)
        for b in range(B + 1):
            lo, hi = b * d, min((b + 1) * d, T)
            if lo >= T: break
            th[lo:hi] = np.clip(base + mag * r.choice([-1.0, 1.0], size=S), 0, 1)
    elif kind == "smooth":
        ph = r.random(S) * 2 * np.pi
        tt = np.arange(T)[:, None] / max(T, 1)
        th[:] = np.clip(base + mag * np.sin(2 * np.pi * tt + ph), 0, 1)
    c = th @ P.T                                # (T,K) action-loss means
    hint = np.vstack([np.tile(c[0], (min(d, T), 1)), c[:-d]]) if d > 0 else c.copy()
    return P, th, c, hint


def run(alg, P, th, c, hint, d, rng):
    T, K = c.shape
    A = np.empty(T, dtype=int)
    feedback_due = {}
    for t in range(T):
        a = alg.act(t, rng, hint[t])
        s = int(rng.choice(P.shape[1], p=P[a]))
        loss = float(np.clip(th[t, s] + 0.1 * rng.standard_normal(), 0, 1))
        feedback_due[t + d] = loss
        # A loss produced at round s is revealed only at time s+d.  The queue belongs
        # to the environment so algorithms cannot access an unrevealed value, and the
        # feedback remains aligned with the stored action/state from that round.
        loss_feedback = feedback_due.pop(t, None)
        alg.update(t, a, s, loss_feedback, hint[t])
        A[t] = a
    loss = c[np.arange(T), A].sum()
    return loss - c.sum(0).min(), A


def E1_E2(c, hint):
    e = np.abs(c - hint).max(1)
    return float(e.sum()), float((e ** 2).sum())
