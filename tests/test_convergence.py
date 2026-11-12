"""The main claim of the program: λ_k(A)/n of sampled graphs converges to the graphon operator eigenvalues."""

import numpy as np
import pytest

from graphon_lab import Graphon, eigenvalues, sample_adjacency

CASES = ["x*y", "Piecewise((0.8, (x<0.5) & (y<0.5)), (0.1, True))", "exp(-abs(x-y))"]


def mean_top(W, n, k, trials=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.mean([eigenvalues(sample_adjacency(W, n, rng=rng)[0])[:k] / n for _ in range(trials)], axis=0)


@pytest.mark.parametrize("formula", CASES)
def test_top_eigenvalues_near_limit(formula):
    W = Graphon(formula)
    limit = W.operator_eigenvalues()[:3]
    assert mean_top(W, 1000, 3) == pytest.approx(limit, abs=0.03)


@pytest.mark.parametrize("formula", CASES)
def test_error_shrinks_with_n(formula):
    W = Graphon(formula)
    limit = W.operator_eigenvalues()[:3]
    small = np.abs(mean_top(W, 100, 3, trials=10) - limit).max()
    large = np.abs(mean_top(W, 1000, 3, trials=10) - limit).max()
    assert large < small
