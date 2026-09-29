"""Objective function and move evaluation for the Linear Ordering Problem.

Notation: ``M`` is the weight matrix and ``pi`` a permutation listing the elements
from first to last. ``A = M[pi][:, pi]`` is the matrix with rows and columns permuted,
and the objective is ``f(pi) = sum_{i<j} A[i, j]``, the weight above the diagonal.

All move gains are computed from ``B = A.T - A``. ``B[i, k]`` is what the
objective gains if the elements at positions ``i`` and ``k`` swap their relative order.
"""

from __future__ import annotations

import numpy as np


def objective(matrix: np.ndarray, perm: np.ndarray) -> int:
    """Weight above the diagonal after reordering ``matrix`` by ``perm``."""
    a = matrix[np.ix_(perm, perm)]
    return int(np.triu(a, 1).sum())


def insert_gains(matrix: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Gains of every insertion move, in ``O(n^2)``.

    ``G[i, j]`` is ``f(insert(perm, i, j)) - f(perm)``, where ``insert`` takes the
    element at position ``i`` out and puts it back at position ``j`` (see
    :func:`apply_insert`). ``G[i, i] == 0``.

    Moving the element at ``i`` forward to ``j > i`` puts it after positions
    ``i+1..j``, so the gain is ``sum_{k=i+1}^{j} B[i, k]``. Moving it backward to
    ``j < i`` puts it before positions ``j..i-1``, which reverses those pairs the
    other way, so the gain is ``-sum_{k=j}^{i-1} B[i, k]``. With inclusive and
    exclusive row-wise prefix sums ``C`` and ``E = C - B`` (so ``C[i, i] = E[i, i]``,
    as ``B`` has a zero diagonal) these are ``C[i, j] - E[i, i]`` (forward) and
    ``E[i, j] - E[i, i]`` (backward).
    """
    a = matrix[np.ix_(perm, perm)]
    b = a.T - a
    inclusive = np.cumsum(b, axis=1)
    exclusive = inclusive - b
    base = np.diagonal(exclusive)[:, None]
    upper = np.triu(np.ones_like(b, dtype=bool), 1)
    return np.where(upper, inclusive, exclusive) - base


def insert_gains_row(matrix: np.ndarray, perm: np.ndarray, i: int) -> np.ndarray:
    """Row ``i`` of :func:`insert_gains` (all moves of one element), in ``O(n)``."""
    e = perm[i]
    b = matrix[perm, e] - matrix[e, perm]  # B[i, k] for every position k
    b[i] = 0
    inclusive = np.cumsum(b)
    exclusive = inclusive - b
    gains = exclusive - exclusive[i]
    gains[i + 1 :] = inclusive[i + 1 :] - exclusive[i]
    return gains


def apply_insert(perm: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return a copy of ``perm`` with the element at position ``i`` moved to position ``j``."""
    out = perm.copy()
    if i < j:
        out[i:j] = perm[i + 1 : j + 1]
    elif i > j:
        out[j + 1 : i + 1] = perm[j:i]
    out[j] = perm[i]
    return out


def adjacent_swap_gains(matrix: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Gains of swapping positions ``i`` and ``i+1`` for every ``i`` (length ``n-1``)."""
    first, second = perm[:-1], perm[1:]
    return matrix[second, first] - matrix[first, second]


def kendall_distance(p: np.ndarray, q: np.ndarray) -> int:
    """Number of element pairs ordered differently by ``p`` and ``q``, in ``O(n^2)``.

    This is the minimum number of adjacent swaps turning one ordering into the
    other and is the natural metric of the LOP search space: ``f`` only depends
    on the relative order of each pair.
    """
    n = len(p)
    pos_p = np.empty(n, dtype=np.int64)
    pos_q = np.empty(n, dtype=np.int64)
    pos_p[p] = np.arange(n)
    pos_q[q] = np.arange(n)
    dp = np.sign(pos_p[:, None] - pos_p[None, :])
    dq = np.sign(pos_q[:, None] - pos_q[None, :])
    return int(np.triu(dp != dq, 1).sum())
