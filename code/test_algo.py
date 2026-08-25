"""Regression tests for the delayed-feedback algorithms."""

import unittest

import numpy as np

from algo import Base, MetaBIO, Ours, run


class _ZeroNoiseRng:
    """Minimal deterministic RNG used to expose feedback timing."""

    def choice(self, n, p=None):
        return 0

    def standard_normal(self):
        return 0.0


class _Probe(Base):
    def __init__(self, K, S, d, T):
        super().__init__(K, S, d, T)
        self.feedback = []

    def act(self, t, rng, hint=None):
        return 0

    def update(self, t, a, s, loss_feedback, hint=None):
        self.feedback.append(loss_feedback)


class DelayedFeedbackTests(unittest.TestCase):
    def test_runner_reveals_the_matching_past_loss(self):
        P = np.array([[1.0]])
        th = np.array([[0.1], [0.2], [0.3]])
        c = th.copy()
        hint = np.zeros((3, 1))
        alg = _Probe(K=1, S=1, d=1, T=3)

        run(alg, P, th, c, hint, d=1, rng=_ZeroNoiseRng())

        self.assertIsNone(alg.feedback[0])
        np.testing.assert_allclose(alg.feedback[1:], [0.1, 0.2])

    def test_centered_update_accumulates_hint_immediately_and_residual_on_arrival(self):
        alg = Ours(K=2, S=1, d=1, T=3, eta=0.1)
        m0 = np.array([0.2, 0.4])
        m1 = np.array([0.3, 0.1])
        alg._x = np.array([0.25, 0.75])
        alg.update(0, a=0, s=0, loss_feedback=None, hint=m0)
        np.testing.assert_allclose(alg.L, m0)
        alg._x = np.array([0.6, 0.4])

        alg.update(1, a=1, s=0, loss_feedback=0.5, hint=m1)

        expected = m0 + m1
        expected[0] += (0.5 - m0[0]) / 0.25
        np.testing.assert_allclose(alg.L, expected)

    def test_metabio_pairs_feedback_with_the_past_state(self):
        P = np.array([[0.5, 0.5]])
        alg = MetaBIO(K=1, S=2, d=1, T=3, P=P)
        alg.update(0, a=0, s=1, loss_feedback=None)
        alg.update(1, a=0, s=0, loss_feedback=0.7)

        self.assertEqual(alg.buf[0], [])
        self.assertEqual(alg.buf[1], [0.7])


if __name__ == "__main__":
    unittest.main()
