"""Sampling: the graphs have the right structure and edges appear at the rates W says."""

import networkx as nx
import numpy as np
import pytest

from graphon_lab import Graphon, sample_adjacency, sample_graph
from graphon_lab.sampling import latent_points


@pytest.mark.parametrize("method", ["random", "grid"])
def test_adjacency_is_simple_graph(method):
    A, U = sample_adjacency(Graphon("exp(-abs(x-y))"), 200, method, rng=1)
    assert A.shape == (200, 200)
    assert np.array_equal(A, A.T), "must be symmetric (undirected)"
    assert set(np.unique(A)) <= {0.0, 1.0}, "must be 0/1"
    assert np.all(np.diag(A) == 0), "must have no self-loops"
    assert len(U) == 200


def test_random_latent_points_are_sorted_uniforms():
    U = latent_points(5000, "random", np.random.default_rng(0))
    assert np.all((U >= 0) & (U <= 1))
    assert np.all(np.diff(U) >= 0)
    # Uniform[0,1] has mean 1/2 and variance 1/12; tolerances are ~5 standard errors
    assert U.mean() == pytest.approx(0.5, abs=0.02)
    assert U.var() == pytest.approx(1 / 12, abs=0.006)


def test_grid_latent_points():
    assert latent_points(4, "grid", None) == pytest.approx([0.125, 0.375, 0.625, 0.875])


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        latent_points(5, "nope", np.random.default_rng(0))


def test_same_seed_same_graph():
    W = Graphon("x*y")
    A1, U1 = sample_adjacency(W, 100, rng=42)
    A2, U2 = sample_adjacency(W, 100, rng=42)
    A3, _ = sample_adjacency(W, 100, rng=43)
    assert np.array_equal(A1, A2) and np.array_equal(U1, U2)
    assert not np.array_equal(A1, A3)


def test_zero_and_one_graphons():
    assert sample_adjacency(Graphon("0"), 50, rng=0)[0].sum() == 0
    A, _ = sample_adjacency(Graphon("1"), 50, rng=0)
    assert np.array_equal(A, 1 - np.eye(50))


def test_edge_frequencies_match_W():
    # With fixed grid points, every pair (i, j) is an independent coin with probability W(x_i, x_j).
    # Averaging many samples, the observed frequency must match W within binomial error.
    W = Graphon("exp(-3*abs(x-y)) * (0.2 + 0.6*x*y)")
    n, trials = 30, 4000
    rng = np.random.default_rng(7)
    mean = sum(sample_adjacency(W, n, "grid", rng)[0] for _ in range(trials)) / trials
    P = W.matrix(latent_points(n, "grid", None))
    off = ~np.eye(n, dtype=bool)
    z = (mean[off] - P[off]) / np.sqrt(P[off] * (1 - P[off]) / trials)
    # 870 pairs: the largest of 870 standard normals is almost never above 4.5
    assert np.abs(z).max() < 4.5
    assert np.abs(z).mean() == pytest.approx(np.sqrt(2 / np.pi), abs=0.1)


def test_mean_degree_matches_integral():
    # E[degree] = (n-1) * ∫∫ W, which is (n-1)/4 for W = x*y
    n = 2000
    degrees = [sample_adjacency(Graphon("x*y"), n, rng=s)[0].sum(axis=1).mean() for s in range(5)]
    assert np.mean(degrees) == pytest.approx((n - 1) / 4, rel=0.03)


def test_sample_graph_matches_adjacency():
    W = Graphon("x*y")
    A, U = sample_adjacency(W, 80, rng=3)
    G = sample_graph(W, 80, rng=3)
    assert np.array_equal(nx.to_numpy_array(G, nodelist=range(80)), A)
    assert [G.nodes[v]["u"] for v in range(80)] == pytest.approx(U)
