"""Regression tests for the certified Safe-Pool machinery.

These cover the three properties that were wrong at some point and are easy to break again:
the union-bound allocation inside the confidence radii, the tie handling in the threshold search,
and the availability of the abstain candidate.
"""
import unittest

import numpy as np

import safe_pool as sp

LOG2 = np.log(2.0)


class CertifiedRadii(unittest.TestCase):
    """The logs must pay for a union bound over all rows, states and rounds."""

    GRID = [(40, 8, 8000), (40, 8, 50000), (200, 6, 5000), (1000, 8, 100000)]

    def test_row_l1_log_is_2KT2(self):
        """Weissman with delta = 1/(2 K T^2) needs log(2 K T^2), not log(2 K T)."""
        for K, S, T in self.GRID:
            n = np.array([7.0, 55.0, 900.0])
            r1, _ = sp.radii(n, K, S, T, alpha=0.0)
            want = np.sqrt(2.0 * (S * LOG2 + np.log(2.0 * K * T * T)) / n)
            np.testing.assert_allclose(r1, np.minimum(2.0, want), rtol=0, atol=1e-12)

    def test_entrywise_log_is_4KST2(self):
        """Hoeffding with delta = 1/(2 K S T^2) needs log(4 K S T^2)."""
        for K, S, T in self.GRID:
            n = np.array([7.0, 55.0, 900.0])
            _, rinf = sp.radii(n, K, S, T, alpha=0.0)
            want = np.sqrt(np.log(4.0 * K * S * T * T) / (2.0 * n))
            np.testing.assert_allclose(rinf, np.minimum(1.0, want), rtol=0, atol=1e-12)

    def test_union_bound_totals_at_most_one_over_T(self):
        """Each family gets 1/(2T), so the two together stay under 1/T."""
        for K, S, T in self.GRID:
            l1_total = K * T * np.exp(-np.log(2.0 * K * T * T))
            inf_total = K * S * T * 2.0 * np.exp(-np.log(4.0 * K * S * T * T))
            self.assertLessEqual(l1_total, 1.0 / (2.0 * T) + 1e-15)
            self.assertLessEqual(inf_total, 1.0 / (2.0 * T) + 1e-15)
            self.assertLessEqual(l1_total + inf_total, 1.0 / T + 1e-15)

    def test_smoothing_bias_is_covered(self):
        """The add-alpha shift is alpha(S-1)/(N+alpha S) entrywise and twice that in l1."""
        for K, S, T in self.GRID:
            for alpha in (1.0, 0.1, 0.01):
                n = np.array([1.0, 12.0, 400.0])
                r1, rinf = sp.radii(n, K, S, T, alpha=alpha)
                r1_0, rinf_0 = sp.radii(n, K, S, T, alpha=0.0)
                worst_inf = alpha * (S - 1.0) / (n + alpha * S)
                self.assertTrue(np.all(rinf + 1e-12 >= np.minimum(1.0, rinf_0 + worst_inf)))
                self.assertTrue(np.all(r1 + 1e-12 >= np.minimum(2.0, r1_0 + 2.0 * worst_inf)))


class ThresholdSearch(unittest.TestCase):
    """A threshold pools every action with r_t(a) <= tau, so tied radii move together."""

    def _instance(self, seed, K=6, S=4, counts=(10.0, 10.0, 10.0, 37.0, 37.0, 91.0)):
        rng = np.random.default_rng(seed)
        x = rng.dirichlet(np.ones(K))
        Phat = rng.dirichlet(np.ones(S), size=K)
        return x, Phat, np.array(counts)

    def test_tied_radii_are_never_split(self):
        for seed in range(300):
            x, Phat, N = self._instance(seed)
            lam, _ = sp.choose_lambda(x, Phat, N, 0.05, 6, 4, 1000)
            r1, _ = sp.radii(N, 6, 4, 1000)
            pooled = set(np.round(r1[lam > 0], 12))
            abstained = set(np.round(r1[lam == 0], 12))
            self.assertEqual(pooled & abstained, set(),
                             msg=f"seed {seed} split a group of equal radii")

    def test_abstain_candidate_is_always_available(self):
        """Whatever the radii, the per-round cost never exceeds the action-level value."""
        for seed in range(200):
            x, Phat, N = self._instance(seed)
            eta = 0.05
            _, cost = sp.choose_lambda(x, Phat, N, eta, 6, 4, 1000)
            self.assertLessEqual(cost, 0.5 * eta * 6 + 1e-12)

    def test_zero_radius_does_not_force_pooling(self):
        """With alpha = 0 and a saturated count a radius can hit zero; abstain must survive."""
        x = np.full(6, 1.0 / 6.0)
        Phat = np.eye(6, 4) + 1e-3
        Phat /= Phat.sum(1, keepdims=True)
        N = np.full(6, 1e12)                       # radii underflow toward zero
        eta = 0.05
        lam, cost = sp.choose_lambda(x, Phat, N, eta, 6, 4, 1000, alpha=0.0)
        self.assertLessEqual(cost, 0.5 * eta * 6 + 1e-12)

    def test_wider_radii_cannot_create_pooling(self):
        """Enlarging the certified radii can only shrink the pooled set."""
        for seed in range(150):
            x, Phat, N = self._instance(seed)
            lam_tight, _ = sp.choose_lambda(x, Phat, N, 0.05, 6, 4, 1000, certified=False)
            lam_wide, _ = sp.choose_lambda(x, Phat, N, 0.05, 6, 4, 1000, certified=True)
            if lam_tight.sum() == 0:
                self.assertEqual(lam_wide.sum(), 0.0,
                                 msg=f"seed {seed}: wider radii introduced pooling")


if __name__ == "__main__":
    unittest.main(verbosity=2)
