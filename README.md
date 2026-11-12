# Graphon eigenvalue lab

Sample random graphs from a graphon and study the eigenvalues of the sampled graphs.

## The model

A **graphon** is a symmetric function `W: [0,1]² → [0,1]`. A graph on `n` vertices is sampled in two steps:

1. Give each vertex a latent position: `U_i ~ Uniform[0,1]` i.i.d. (`--method random`), or the fixed grid `(i - ½)/n` (`--method grid`).
2. For each pair `i < j`, add the edge independently with probability `W(U_i, U_j)`.

**Parameter studied: the eigenvalues of the adjacency matrix A**, all `n` of them, with multiplicity.
For large `n`, `λ_k(A) / n` converges to the k-th eigenvalue of the integral operator
`(T_W f)(x) = ∫ W(x,y) f(y) dy`. The code computes that limit numerically, so you can watch the sampled values approach it.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```bash
.venv\Scripts\python.exe run.py --formula "x*y" --n 100 200 400 800 --trials 20
```

| Option | Meaning |
|---|---|
| `--formula` | The graphon in `x` and `y` (sympy syntax): `"x*y"`, `"exp(-abs(x-y))"`, `"0.5"`, `"Piecewise((0.8, (x<0.5) & (y<0.5)), (0.1, True))"` |
| `--n` | The graph sizes to sample |
| `--trials` | How many graphs to sample at each size |
| `--method` | `random` or `grid` latent points |
| `--top-k` | How many leading eigenvalues to track |
| `--tol` | Tolerance for treating two eigenvalues as equal when counting multiplicity |
| `--seed` | Random seed, for reproducible runs |
| `--draw-n` | Number of vertices in the graph that gets drawn (default 60; large graphs become a hairball) |
| `--show` | Open the plots in a window as well as saving them |

The files written to `results/`:

- `eigenvalues.csv`: every eigenvalue of every sampled graph
- `multiplicities_n<N>.txt`: the distinct eigenvalues of one sampled graph, with their multiplicities
- `summary.csv`: mean ± std of `λ_k/n` at each `n`, next to the graphon limit
- `graph.png`: one sampled graph drawn as a network, with each vertex coloured by its latent position `u` and sized by its degree, next to its adjacency matrix with vertices sorted by `u`, which looks like a noisy picture of W
- `graphon.png`, `top_eigenvalues.png`, `spectrum_n<N>.png`: plots

## Verifying correctness

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

The suite (about 1½ minutes) checks the program against answers known from mathematics:

| File | What it checks |
|---|---|
| `tests/test_graphon.py` | Formulas are parsed and validated. The limit eigenvalues match closed-form answers for `p`, `x*y` (1/3), a two-block graphon (2×2 matrix), and `exp(-abs(x-y))` (roots of a transcendental equation) |
| `tests/test_sampling.py` | Sampled graphs are simple, undirected 0/1 graphs. Latent points are Uniform[0,1]. Over 4000 samples each edge appears at rate W(x_i, x_j) within binomial error. Mean degree is (n−1)∫∫W. Seeds reproduce |
| `tests/test_spectrum.py` | Exact identities: Σλ = 0, Σλ² = 2·edges, Σλ³ = 6·triangles. Agreement with scipy and networkx. Exact spectra of K_n, K_{5,5} and the empty graph. Erdős–Rényi semicircle bounds. Multiplicity grouping |
| `tests/test_convergence.py` | λ_k/n approaches the graphon limit, and the error shrinks as n grows |
| `tests/test_run.py` | `run.py` writes every output file, with contents that agree with each other, reproducibly, and rejects invalid formulas |

The tests were themselves checked by planting six deliberate bugs, such as edges 10% too rare, self-loops, non-uniform latent points, a 0.05% error in the limit, wrong multiplicity grouping, and the wrong eigenvalue order. The suite caught every one.

**Known limitations**
- Sampled graphs almost never have exactly repeated eigenvalues. Multiplicities are found by grouping values within `--tol`, so they are meaningful mainly for structured graphs (grid sampling, block graphons).
- Limit eigenvalues come from a 2000-point midpoint rule. They are accurate to about 1e-6 for smooth W, and less accurate for W with jumps that don't fall on grid cells.
- Each graph needs O(n²) memory and O(n³) time for its eigenvalues, so n of a few thousand is the practical limit.

## Use it from Python

```python
from graphon_lab import Graphon, sample_adjacency, sample_graph, eigenvalues, with_multiplicity

W = Graphon("exp(-abs(x-y))")
A, U = sample_adjacency(W, n=500, method="random", rng=1)   # numpy adjacency + latent points
G = sample_graph(W, n=500)                                   # networkx Graph
print(with_multiplicity(eigenvalues(A)))                     # [(value, multiplicity), ...]
print(W.operator_eigenvalues()[:5])                          # limits of λ_k / n
```
