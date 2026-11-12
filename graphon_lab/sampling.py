"""Sampling W-random graphs from a graphon.

The standard two-step process for a graph on n vertices:
  1. Latent positions: draw U_1, ..., U_n i.i.d. Uniform[0,1]   (method="random")
     or use the fixed grid x_i = (i - 1/2) / n                   (method="grid")
  2. Edges: for each pair i < j, independently include edge {i,j} with
     probability W(U_i, U_j). No self-loops.
"""

import networkx as nx
import numpy as np

METHODS = ("random", "grid")


def latent_points(n: int, method: str, rng: np.random.Generator):
    if method == "random":
        return np.sort(rng.uniform(0.0, 1.0, size=n))
    if method == "grid":
        return (np.arange(n) + 0.5) / n
    raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")


def sample_adjacency(graphon, n: int, method: str = "random", rng=None):
    """Sample one graph. Returns (A, U): the 0/1 symmetric adjacency matrix and latent points."""
    rng = np.random.default_rng(rng)
    U = latent_points(n, method, rng)
    P = graphon.matrix(U)
    coins = rng.uniform(size=(n, n))
    upper = np.triu(coins < P, k=1)
    A = (upper | upper.T).astype(float)
    return A, U


def sample_graph(graphon, n: int, method: str = "random", rng=None) -> nx.Graph:
    """Same as sample_adjacency but returns a networkx Graph with each node's latent position stored as 'u'."""
    A, U = sample_adjacency(graphon, n, method, rng)
    G = nx.from_numpy_array(A)
    nx.set_node_attributes(G, dict(enumerate(U)), "u")
    return G
