"""The delay-window quantity has mean <= 1 but no exponential moment UNIFORM IN q.

For the state estimator, <x_r, chat_s> = q_r(S_s) X_s / q_s(S_s), so the whole delay
analysis turns on 1/q_s(S_s), the reciprocal probability of the state actually observed.

  E[1/q_s(S_s)] = sum_s q_s(s)/q_s(s) = |S|                      <- mean is fine
  P(1/q_s(S_s) > x) = sum_{s: q_s(s) < 1/x} q_s(s) <= |S|/x      <- 1/x tail

For any FIXED q with full support 1/q(S) is bounded and its MGF is finite; what fails is a bound
uniform in q.  At |S| = 2 with q = (eps, 1-eps), E[exp(eta/q(S))] >= eps*exp(eta/eps) -> infinity
as eps -> 0 while the mean stays at 2.  Since q_s is set by the algorithm's own past play, no
bound depending only on |S| and eta is available, and Z_r >= exp(-eta <x_r,G_r>) needs exactly
such a bound.
Any device that supplies one -- uniform mixing, clipping, implicit exploration -- floors
q at gamma and costs gamma*T, and gamma must be ~ eta*d*kappa*|S| to make eta*G = O(1),
which reinstates the very product d*|S| the additive bound was meant to remove.

This measures the tail directly on adversarially skewed but legal state distributions.
"""
import numpy as np

def tail(S, skew, reps=400000, seed=0):
    rng = np.random.default_rng(seed)
    q = np.array([skew ** i for i in range(S)], float); q /= q.sum()
    obs = rng.choice(S, size=reps, p=q)
    inv = 1.0 / q[obs]
    return q, inv

if __name__ == "__main__":
    S = 4
    print(f"|S|={S}.  1/q(S_s) for the observed state, over skewed state distributions.")
    print(f"   {'skew':>6} {'min q':>10} {'mean':>8} {'p99':>10} {'p99.9':>11} {'max':>11} "
          f"{'E[e^(0.05 Y)]':>14}")
    for skew in (1.0, 0.3, 0.1, 0.03, 0.01):
        q, inv = tail(S, skew)
        mgf = float(np.mean(np.exp(0.05 * inv)))
        print(f"   {skew:6.2f} {q.min():10.5f} {inv.mean():8.3f} {np.quantile(inv,.99):10.2f} "
              f"{np.quantile(inv,.999):11.2f} {inv.max():11.2f} {mgf:14.2f}")
    print()
    print("   mean stays at |S| for every skew, which is why the DELAY term is fine.")
    print("   each value is finite, but the family is unbounded, which is why the QUADRATIC")
    print("   term cannot be transferred from x_r to x_{r+d} without a floor on q.")
