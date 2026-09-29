"""Constructive heuristics: build a permutation from first to last position."""

from __future__ import annotations

import numpy as np


def random_permutation(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(matrix.shape[0])


def becker(matrix: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Becker's (1967) quotient heuristic.

    Repeatedly place first the remaining element with the largest ratio of outgoing
    to incoming weight, ``q_e = sum_k m_ek / sum_k m_ke``, where both sums run only
    over the elements not yet placed. Ties are broken by the smallest index.
    ``rng`` is accepted for a uniform signature and ignored.
    """
    n = matrix.shape[0]
    m = matrix.astype(np.float64)
    np.fill_diagonal(m, 0.0)
    out_w, in_w = m.sum(axis=1), m.sum(axis=0)
    remaining = np.ones(n, dtype=bool)
    perm = np.empty(n, dtype=np.int64)
    for pos in range(n):
        with np.errstate(divide="ignore", invalid="ignore"):
            # An element with no incoming weight goes first; 0/0 sorts last.
            q = np.where(in_w > 0, out_w / np.where(in_w > 0, in_w, 1.0), np.where(out_w > 0, np.inf, 0.0))
        q[~remaining] = -np.inf
        e = int(np.argmax(q))
        perm[pos] = e
        remaining[e] = False
        # Remove e: its outgoing weight no longer counts as incoming weight of the others, and vice versa.
        in_w -= m[e]
        out_w -= m[:, e]
    return perm


def grasp_construct(matrix: np.ndarray, rng: np.random.Generator, alpha: float = 0.1) -> np.ndarray:
    """Greedy randomised construction with a value-based restricted candidate list.

    The greedy value of a remaining element ``e`` is its net outflow towards the
    other remaining elements, ``g_e = sum_k (m_ek - m_ke)``. Placing it next
    commits exactly that much weight above the diagonal relative to the reverse
    choice. The RCL keeps the elements with ``g_e >= g_max - alpha (g_max - g_min)``,
    and one of them is drawn uniformly. ``alpha = 0`` is pure greedy and
    ``alpha = 1`` is a uniformly random permutation.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    n = matrix.shape[0]
    m = matrix.astype(np.float64)
    np.fill_diagonal(m, 0.0)
    net = m.sum(axis=1) - m.sum(axis=0)
    remaining = np.ones(n, dtype=bool)
    perm = np.empty(n, dtype=np.int64)
    for pos in range(n):
        values = net[remaining]
        idx = np.flatnonzero(remaining)
        g_max, g_min = values.max(), values.min()
        rcl = idx[values >= g_max - alpha * (g_max - g_min)]
        e = int(rng.choice(rcl))
        perm[pos] = e
        remaining[e] = False
        net -= m[:, e] - m[e]  # drop e's contribution to every other element's net outflow
    return perm


CONSTRUCTORS = {
    "random": random_permutation,
    "becker": becker,
    "grasp": grasp_construct,
}
