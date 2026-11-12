"""A graphon W: [0,1]^2 -> [0,1] built from a formula string in x and y."""

import numpy as np
import sympy as sp

X, Y = sp.symbols("x y")


class Graphon:
    """Wraps a symbolic formula like "x*y" or "exp(-abs(x-y))" as a vectorized function.

    A graphon must be symmetric (W(x,y) = W(y,x)) and take values in [0,1].
    Both are checked numerically on a grid when the graphon is created.
    """

    def __init__(self, formula: str, check_grid: int = 200):
        self.formula = formula
        self.expr = sp.sympify(formula, locals={"x": X, "y": Y})
        extra = self.expr.free_symbols - {X, Y}
        if extra:
            raise ValueError(f"Formula may only use x and y, found: {sorted(map(str, extra))}")
        self._f = sp.lambdify((X, Y), self.expr, modules="numpy")
        self._check(check_grid)

    def __call__(self, x, y):
        # Broadcast inputs first: Piecewise formulas need x and y of equal shape
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        # broadcast_to handles constant formulas such as "0.5", which lambdify returns as a scalar
        return np.broadcast_to(self._f(x, y), x.shape).astype(float)

    def matrix(self, points):
        """Return the matrix [W(p_i, p_j)] for a 1-D array of latent points."""
        points = np.asarray(points, dtype=float)
        return self(points[:, None], points[None, :])

    def operator_eigenvalues(self, m: int = 2000):
        """Approximate eigenvalues of the integral operator (T_W f)(x) = ∫ W(x,y) f(y) dy.

        Uses the midpoint rule on an m-point grid, so the result is eig([W(x_i,x_j)]) / m,
        sorted from largest to smallest. These are the limits of eig(A)/n for sampled graphs.
        """
        grid = (np.arange(m) + 0.5) / m
        return np.linalg.eigvalsh(self.matrix(grid) / m)[::-1]

    def _check(self, n):
        grid = (np.arange(n) + 0.5) / n
        W = self.matrix(grid)
        if not np.all(np.isfinite(W)):
            raise ValueError("Formula gives non-finite values on [0,1]^2")
        if not np.allclose(W, W.T, atol=1e-9):
            raise ValueError("Formula is not symmetric: W(x,y) != W(y,x)")
        if W.min() < -1e-12 or W.max() > 1 + 1e-12:
            raise ValueError(f"Formula must lie in [0,1], got range [{W.min():.4g}, {W.max():.4g}]")

    def __repr__(self):
        return f"Graphon({self.formula!r})"
