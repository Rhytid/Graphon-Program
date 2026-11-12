"""The graphon: formula parsing, validation, and the operator eigenvalues used as limits.

Operator eigenvalues are checked against closed-form answers, so these tests confirm
the "limit" lines in the plots are the true mathematical limits.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from graphon_lab import Graphon

TWO_BLOCK = "Piecewise((0.8, (x<0.5) & (y<0.5)), (0.1, True))"


def test_evaluates_formula():
    W = Graphon("x*y")
    assert W(0.5, 0.4) == pytest.approx(0.2)
    assert W.matrix([0.2, 0.5]) == pytest.approx(np.array([[0.04, 0.1], [0.1, 0.25]]))


def test_constant_formula_broadcasts():
    W = Graphon("0.3")
    assert W.matrix(np.linspace(0, 1, 4)) == pytest.approx(np.full((4, 4), 0.3))


def test_piecewise_formula():
    W = Graphon(TWO_BLOCK)
    assert W(0.2, 0.3) == pytest.approx(0.8)
    assert W(0.2, 0.7) == pytest.approx(0.1)
    assert W(0.7, 0.7) == pytest.approx(0.1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # 1/(x-y) divides by zero on the diagonal, by design
@pytest.mark.parametrize(
    "formula, message",
    [
        ("x", "not symmetric"),
        ("x*y + 0.5", "must lie in"),
        ("x - y", "not symmetric"),
        ("-x*y", "must lie in"),
        ("x*y*z", "only use x and y"),
        ("1/(x-y)", "non-finite"),
    ],
)
def test_rejects_invalid_graphons(formula, message):
    with pytest.raises(ValueError, match=message):
        Graphon(formula)


def test_operator_eigenvalues_constant():
    # W = p is rank one: the only nonzero eigenvalue is p (eigenfunction f = 1)
    mu = Graphon("0.3").operator_eigenvalues()
    assert mu[0] == pytest.approx(0.3)
    assert np.abs(mu[1:]).max() < 1e-10


def test_operator_eigenvalues_product():
    # W = x*y is rank one: eigenvalue ∫ x^2 dx = 1/3 (eigenfunction f = x); midpoint rule error ~ 1/m^2
    mu = Graphon("x*y").operator_eigenvalues()
    assert mu[0] == pytest.approx(1 / 3, abs=1e-6)
    assert np.abs(mu[1:]).max() < 1e-10


def test_operator_eigenvalues_two_block():
    # Step graphon on blocks of size 1/2 acts like the 2x2 matrix [[0.8, 0.1], [0.1, 0.1]] * 1/2
    expected = np.linalg.eigvalsh(np.array([[0.8, 0.1], [0.1, 0.1]]) / 2)[::-1]
    mu = Graphon(TWO_BLOCK).operator_eigenvalues()
    assert mu[:2] == pytest.approx(expected, abs=1e-6)
    assert np.abs(mu[2:]).max() < 1e-10


def test_operator_eigenvalues_exponential_kernel():
    # For W = exp(-|x-y|) on [0,1] the eigenvalues are 2 / (1 + w^2), where w > 0 solves
    # (w^2 - 1) sin w = 2 w cos w  (the classical solution of the Fredholm equation for this kernel)
    def f(w):
        return (w**2 - 1) * np.sin(w) - 2 * w * np.cos(w)

    grid = np.linspace(0.01, 20, 20000)
    roots = [brentq(f, a, b) for a, b in zip(grid[:-1], grid[1:]) if f(a) * f(b) < 0]
    expected = 2 / (1 + np.array(roots[:4]) ** 2)

    mu = Graphon("exp(-abs(x-y))").operator_eigenvalues()
    assert mu[:4] == pytest.approx(expected, abs=1e-5)
