"""Eigenvalues: exact identities, known graphs, an independent cross-check, and multiplicity grouping."""

import networkx as nx
import numpy as np
import pytest
import scipy.linalg

from graphon_lab import Graphon, eigenvalues, sample_adjacency, with_multiplicity

BIPARTITE = "Piecewise((1, ((x<0.5) & (y>=0.5)) | ((x>=0.5) & (y<0.5))), (0, True))"


@pytest.fixture(params=[("x*y", 1), ("exp(-abs(x-y))", 2), ("0.1", 3), (BIPARTITE, 4)])
def sampled(request):
    formula, seed = request.param
    A, _ = sample_adjacency(Graphon(formula), 150, "random", rng=seed)
    return A, eigenvalues(A)


def test_count_and_order(sampled):
    A, eigs = sampled
    assert len(eigs) == len(A)
    assert np.all(np.diff(eigs) <= 0), "largest first"


def test_trace_identities(sampled):
    # tr(A) = 0 (no self-loops) and tr(A^2) = 2 * number of edges
    A, eigs = sampled
    assert eigs.sum() == pytest.approx(0, abs=1e-8)
    assert (eigs**2).sum() == pytest.approx(A.sum(), rel=1e-10)


def test_trace_cubed_counts_triangles(sampled):
    # tr(A^3) = 6 * number of triangles
    A, eigs = sampled
    triangles = sum(nx.triangles(nx.from_numpy_array(A)).values()) / 3
    assert (eigs**3).sum() == pytest.approx(6 * triangles, abs=1e-6)


def test_matches_independent_implementations(sampled):
    A, eigs = sampled
    assert eigs == pytest.approx(scipy.linalg.eigh(A, eigvals_only=True)[::-1], abs=1e-9)
    nx_eigs = np.sort(nx.adjacency_spectrum(nx.from_numpy_array(A)).real)[::-1]
    assert eigs == pytest.approx(nx_eigs, abs=1e-8)


def test_complete_graph():
    # K_n: eigenvalue n-1 once, -1 with multiplicity n-1
    A, _ = sample_adjacency(Graphon("1"), 12, "grid", rng=0)
    assert with_multiplicity(eigenvalues(A)) == [(pytest.approx(11), 1), (pytest.approx(-1), 11)]


def test_complete_bipartite_graph():
    # K_{5,5}: eigenvalues 5, 0 (multiplicity 8), -5
    A, _ = sample_adjacency(Graphon(BIPARTITE), 10, "grid", rng=0)
    groups = with_multiplicity(eigenvalues(A))
    assert groups == [(pytest.approx(5), 1), (pytest.approx(0, abs=1e-8), 8), (pytest.approx(-5), 1)]


def test_empty_graph():
    A, _ = sample_adjacency(Graphon("0"), 7, "grid", rng=0)
    assert with_multiplicity(eigenvalues(A)) == [(0.0, 7)]


def test_erdos_renyi_spectrum():
    # W = p gives the Erdős–Rényi graph G(n, p): λ1 ≈ np, and all other eigenvalues lie
    # within about 2 sqrt(np(1-p)) of zero (semicircle law)
    n, p = 1000, 0.3
    eigs = eigenvalues(sample_adjacency(Graphon(str(p)), n, rng=5)[0])
    edge = 2 * np.sqrt(n * p * (1 - p))
    assert eigs[0] / n == pytest.approx(p, abs=0.01)
    assert eigs[1] < 1.05 * edge
    assert eigs[-1] > -1.05 * edge
    assert eigs[1] > 0.9 * edge, "bulk should reach close to the semicircle edge"


def test_with_multiplicity_grouping():
    eigs = [3.0, 3.0 + 1e-10, 1.0, 1.0, 1.0, -2.0]
    assert with_multiplicity(eigs) == [(pytest.approx(3.0), 2), (1.0, 3), (-2.0, 1)]


def test_with_multiplicity_tolerance():
    eigs = [1.0, 1.001, 2.0]
    assert [m for _, m in with_multiplicity(eigs, tol=1e-8)] == [1, 1, 1]
    assert [m for _, m in with_multiplicity(eigs, tol=0.01)] == [1, 2]


def test_multiplicities_add_up_to_n(sampled):
    _, eigs = sampled
    assert sum(m for _, m in with_multiplicity(eigs)) == len(eigs)
