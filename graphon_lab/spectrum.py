"""Eigenvalues of sampled graphs, with multiplicities."""

import numpy as np


def eigenvalues(A):
    """All n eigenvalues of a symmetric adjacency matrix, largest first, repeated by multiplicity."""
    return np.linalg.eigvalsh(A)[::-1]


def with_multiplicity(eigs, tol: float = 1e-8):
    """Group eigenvalues that agree within tol. Returns a list of (value, multiplicity), largest first.

    Floating-point eigenvalues of a repeated root differ in the last few digits,
    so equal values are detected by a tolerance rather than exact comparison.
    """
    eigs = np.sort(np.asarray(eigs))[::-1]
    groups = []
    start = 0
    for i in range(1, len(eigs) + 1):
        if i == len(eigs) or eigs[start] - eigs[i] > tol:
            groups.append((float(np.mean(eigs[start:i])), i - start))
            start = i
    return groups


def format_multiplicities(groups, decimals: int = 6):
    lines = [f"{'eigenvalue':>18}  multiplicity"]
    lines += [f"{v:>18.{decimals}f}  {m}" for v, m in groups]
    return "\n".join(lines)
