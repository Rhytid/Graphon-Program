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

The files written to `results/`:

- `eigenvalues.csv`: every eigenvalue of every sampled graph
- `multiplicities_n<N>.txt`: the distinct eigenvalues of one sampled graph, with their multiplicities
- `summary.csv`: mean ± std of `λ_k/n` at each `n`, next to the graphon limit
- `graphon.png`, `top_eigenvalues.png`, `spectrum_n<N>.png`: plots

## Use it from Python

```python
from graphon_lab import Graphon, sample_adjacency, sample_graph, eigenvalues, with_multiplicity

W = Graphon("exp(-abs(x-y))")
A, U = sample_adjacency(W, n=500, method="random", rng=1)   # numpy adjacency + latent points
G = sample_graph(W, n=500)                                   # networkx Graph
print(with_multiplicity(eigenvalues(A)))                     # [(value, multiplicity), ...]
print(W.operator_eigenvalues()[:5])                          # limits of λ_k / n
```
