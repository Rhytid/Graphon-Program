# Graphon eigenvalue lab

Sample random graphs from a graphon and study the eigenvalues of the sampled graphs.

## The model

A **graphon** is a symmetric function `W: [0,1]² → [0,1]`. A graph on `n` vertices is sampled in two steps:

1. Give each vertex a latent position: `U_i ~ Uniform[0,1]` i.i.d. (`--method random`), or the fixed grid `(i - ½)/n` (`--method grid`).
2. For each pair `i < j`, add the edge independently with probability `W(U_i, U_j)`.

**Parameter studied: the eigenvalues of the adjacency matrix A**, all `n` of them, with multiplicity.
For large `n`, `λ_k(A) / n` converges to the k-th eigenvalue of the integral operator
`(T_W f)(x) = ∫ W(x,y) f(y) dy`. The code computes that limit numerically, so you can watch the sampled values approach it.

## Getting started (no experience needed)

You only do steps 1–3 once. After that, step 4 is all you need.

### 1. Install Python

You need Python 3.10 or newer. It's free.

- **Windows**
  1. Go to https://www.python.org/downloads/ and click the yellow **Download Python** button.
  2. Run the installer. On the first screen, **tick "Add python.exe to PATH"** at the bottom. This step is easy to miss, and nothing works without it.
  3. Click **Install Now**.
- **Mac**
  1. Go to https://www.python.org/downloads/ and download the macOS installer.
  2. Open it and click through the steps.

To check the install, open a terminal:
- **Windows:** press the Windows key, type `cmd`, and press Enter.
- **Mac:** open **Terminal** from Applications → Utilities.

Then type this and press Enter:

```bash
python --version
```

On Mac, type `python3 --version` instead. You should see something like `Python 3.12.4`. If you get "not recognized" on Windows, reinstall Python and make sure you tick the PATH box.

### 2. Download this program

1. Go to https://github.com/Rhytid/Graphon-Program
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip it. On Windows, right-click the file and choose **Extract All**. On Mac, double-click it. You'll get a folder called `Graphon-Program-main`.

### 3. Install the libraries the program uses

In the terminal, go into that folder. The easiest way is to type `cd ` (with a space after it), drag the folder onto the terminal window, and press Enter. Then run these two commands.

**Windows:**

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Mac:**

```bash
python3 -m venv .venv
```
```bash
.venv/bin/python -m pip install -r requirements.txt
```

The commands create a private Python environment for this folder, called `.venv`, and download the libraries into it. This takes a minute or two and needs an internet connection.

### 4. Run it

**Windows:**

```bash
.venv\Scripts\python.exe run.py --formula "x*y"
```

**Mac:**

```bash
.venv/bin/python run.py --formula "x*y"
```

The program prints a table of eigenvalues and saves its results and pictures in a new `results` folder. Change the formula to try other graphons, such as `--formula "exp(-abs(x-y))"`. Add `--show` to open the pictures in a window.

The rest of this README uses the Windows commands. On Mac, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

**If something goes wrong**
- *"python is not recognized"*: Python isn't on your PATH. Reinstall it and tick **Add python.exe to PATH**.
- *"No such file run.py"*: you're not in the program's folder. Repeat the `cd` step in step 3.
- *"No module named numpy"* (or sympy, networkx and so on): run step 3 again. Also make sure you start the command with `.venv\Scripts\python.exe` (Mac: `.venv/bin/python`), not a plain `python`.

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
