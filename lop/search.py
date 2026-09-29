"""Local search and the three metaheuristics: GRASP, ILS and Tabu Search.

Every metaheuristic runs under the same stopping rule. It stops as soon as its
CPU-time budget is spent or it reaches ``target`` (usually the proven optimum),
whichever comes first. The budget is also checked *inside* each local search, so
no descent can overrun it. Each run returns a :class:`Result`.

* ``trace`` records every improvement of the best-so-far value as
  ``(cpu seconds, value, gain evaluations so far)``, including improvements found
  mid-descent. The anytime curves and the time-to-target statistics are built from it.
* ``evaluations`` counts inspected move gains. It is a machine-independent effort
  measure to set against CPU time.
* ``diagnostics`` holds trajectory statistics used to explain the behaviour of
  each method (see each function).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .construct import becker, grasp_construct
from .objective import adjacent_swap_gains, apply_insert, insert_gains, insert_gains_row, objective


# --------------------------------------------------------------------------- bookkeeping
class Budget:
    """CPU-time budget with an optional target value.

    CPU time (``time.process_time``) rather than wall-clock time keeps runs comparable
    when many single-threaded runs share a node.
    """

    def __init__(self, seconds: float, target: int | None = None):
        self.seconds = seconds
        self.target = target
        self.start = time.process_time()

    def elapsed(self) -> float:
        return time.process_time() - self.start

    def reached(self, value: float) -> bool:
        return self.target is not None and value >= self.target

    def exhausted(self, value: float) -> bool:
        return self.reached(value) or self.elapsed() >= self.seconds


@dataclass
class Result:
    perm: np.ndarray
    value: int
    trace: list[tuple[float, int, int]] = field(default_factory=list)
    iterations: int = 0
    evaluations: int = 0
    elapsed: float = 0.0
    time_to_target: float | None = None
    diagnostics: dict = field(default_factory=dict)
    visited: list[np.ndarray] = field(default_factory=list)  # optional search trajectory


class Incumbent:
    """Best-so-far solution, its improvement trace, and the run's effort counter."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.perm: np.ndarray | None = None
        self.value = -np.inf
        self.trace: list[tuple[float, int, int]] = []
        self.time_to_target: float | None = None
        self.evaluations = 0

    def offer(self, perm: np.ndarray, value: int) -> bool:
        if value <= self.value:
            return False
        self.perm, self.value = perm.copy(), int(value)
        t = self.budget.elapsed()
        self.trace.append((t, self.value, self.evaluations))
        # a hit only counts if it happened within the budget
        if self.time_to_target is None and self.budget.reached(self.value) and t <= self.budget.seconds:
            self.time_to_target = t
        return True

    def done(self) -> bool:
        return self.budget.exhausted(self.value)

    def result(self, iterations: int, diagnostics: dict | None = None,
               visited: list[np.ndarray] | None = None) -> Result:
        return Result(
            perm=self.perm,
            value=int(self.value),
            trace=self.trace,
            iterations=iterations,
            evaluations=self.evaluations,
            elapsed=self.budget.elapsed(),
            time_to_target=self.time_to_target,
            diagnostics=diagnostics or {},
            visited=visited or [],
        )


def _argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> int:
    """Flat index of a maximum of ``values``; ties are broken uniformly at random."""
    flat = values.ravel()
    best = np.flatnonzero(flat == flat.max())
    return int(best[0] if best.size == 1 else rng.choice(best))


# --------------------------------------------------------------------------- local search
@dataclass
class LocalSearchStats:
    moves: int = 0
    evaluations: int = 0  # move gains inspected
    interrupted: bool = False  # stopped by the budget before reaching a local optimum


def local_search(
    matrix: np.ndarray,
    perm: np.ndarray,
    rng: np.random.Generator,
    neighbourhood: str = "insert",
    pivot: str = "element",
    value: int | None = None,
    incumbent: Incumbent | None = None,
) -> tuple[np.ndarray, int, LocalSearchStats]:
    """Hill-climb to a local optimum of the chosen neighbourhood.

    Args:
        neighbourhood: ``"insert"`` (move one element to another position; ``n(n-1)``
            move descriptions, ``(n-1)^2`` distinct neighbours because swapping two
            adjacent elements can be described in two ways) or ``"swap"`` (exchange two
            adjacent elements; ``n-1`` neighbours).
        pivot: which improving move to apply.

            * ``"best"``: the best move of the whole neighbourhood (steepest ascent).
            * ``"first"``: first improvement. Visit elements (insert) or adjacent pairs
              (swap) in a random order and apply an improving move of the first one
              that has any, drawn uniformly among its improving positions.
            * ``"element"`` (insert only): sweep the elements in a random order and move
              each one to its *best* position if that improves. This is the local
              search of Schiavinotto & Stützle (2004), the standard for the LOP.

            Sweeps repeat until one full sweep finds no improvement.
        incumbent: if given, every new best solution is reported to it as soon as it
            is found, gain evaluations are charged to it, and the search stops early
            once its budget is exhausted (``stats.interrupted`` is then set).

    Returns:
        The final permutation, its objective value, and move and evaluation counts.
    """
    perm = perm.copy()
    n = len(perm)
    value = objective(matrix, perm) if value is None else int(value)
    stats = LocalSearchStats()

    def moved() -> bool:
        """Bookkeeping after a move; returns True if the search must stop."""
        stats.moves += 1
        if incumbent is None:
            return False
        if value > incumbent.value:
            incumbent.offer(perm, value)
        if incumbent.done():
            stats.interrupted = True
            return True
        return False

    def charge(evals: int) -> None:
        stats.evaluations += evals
        if incumbent is not None:
            incumbent.evaluations += evals

    if neighbourhood == "insert" and pivot == "best":
        while True:
            gains = insert_gains(matrix, perm)
            charge(n * (n - 1))
            i, j = divmod(_argmax_random_tie(gains, rng), n)
            if gains[i, j] <= 0:
                break
            perm = apply_insert(perm, i, j)
            value += int(gains[i, j])
            if moved():
                break

    elif neighbourhood == "insert" and pivot in ("first", "element"):
        position = np.empty(n, dtype=np.int64)
        improved = True
        while improved and not stats.interrupted:
            improved = False
            for e in rng.permutation(n):
                position[perm] = np.arange(n)
                i = int(position[e])
                gains = insert_gains_row(matrix, perm, i)
                charge(n - 1)
                if pivot == "element":
                    j = _argmax_random_tie(gains, rng)
                    if gains[j] <= 0:
                        continue
                else:
                    better = np.flatnonzero(gains > 0)
                    if better.size == 0:
                        continue
                    j = int(rng.choice(better))
                perm = apply_insert(perm, i, j)
                value += int(gains[j])
                improved = True
                if moved():
                    break
                if pivot == "first":
                    break  # restart the scan from a fresh random order

    elif neighbourhood == "swap" and pivot == "best":
        while True:
            gains = adjacent_swap_gains(matrix, perm)
            charge(n - 1)
            i = _argmax_random_tie(gains, rng)
            if gains[i] <= 0:
                break
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            value += int(gains[i])
            if moved():
                break

    elif neighbourhood == "swap" and pivot in ("first", "element"):
        improved = True
        while improved and not stats.interrupted:
            improved = False
            for i in rng.permutation(n - 1):
                a, b = perm[i], perm[i + 1]
                gain = int(matrix[b, a] - matrix[a, b])
                charge(1)
                if gain > 0:
                    perm[i], perm[i + 1] = b, a
                    value += gain
                    improved = True
                    if moved():
                        break
    else:
        raise ValueError(f"unknown local search {neighbourhood!r}/{pivot!r}")

    return perm, value, stats


# --------------------------------------------------------------------------- GRASP
def grasp(
    matrix: np.ndarray,
    rng: np.random.Generator,
    budget: Budget,
    alpha: float = 0.1,
    neighbourhood: str = "insert",
    pivot: str = "element",
) -> Result:
    """Greedy Randomised Adaptive Search Procedure (Feo & Resende, 1995).

    Independent restarts: build a solution with :func:`grasp_construct` at
    greediness ``alpha``, improve it with local search, keep the best. With
    ``alpha = 1`` the restricted candidate list always holds every remaining element,
    so the construction is a uniformly random permutation. It is then drawn directly
    with ``rng.permutation``, which gives random-restart local search (RRLS), the
    control for both GRASP's construction and ILS's perturbation.

    Diagnostics: ``distinct_local_optima`` (fraction of restarts that ended at a local
    optimum not seen before) and ``mean_descent_moves``.
    """
    n = matrix.shape[0]
    best = Incumbent(budget)
    seen: set[bytes] = set()
    moves, it = 0, 0
    while not best.done():
        perm = rng.permutation(n) if alpha >= 1.0 else grasp_construct(matrix, rng, alpha)
        if alpha < 1.0:
            best.evaluations += n * (n - 1) // 2  # greedy values are updated once per placed pair
        value = objective(matrix, perm)
        best.offer(perm, value)
        perm, value, st = local_search(matrix, perm, rng, neighbourhood, pivot, value, best)
        best.offer(perm, value)
        moves += st.moves
        it += 1
        if not st.interrupted:
            seen.add(perm.tobytes())
    diag = {"distinct_local_optima": len(seen) / max(it, 1), "mean_descent_moves": moves / max(it, 1)}
    return best.result(it, diag)


# --------------------------------------------------------------------------- ILS
def perturb_inserts(perm: np.ndarray, strength: int, rng: np.random.Generator) -> np.ndarray:
    """Apply ``strength`` random insertion moves (random element, random new position)."""
    n = len(perm)
    for _ in range(strength):
        i, j = rng.choice(n, size=2, replace=False)
        perm = apply_insert(perm, int(i), int(j))
    return perm


def ils(
    matrix: np.ndarray,
    rng: np.random.Generator,
    budget: Budget,
    strength: int = 5,
    acceptance: str = "better_equal",
    start: str = "becker",
    neighbourhood: str = "insert",
    pivot: str = "element",
    log_visits: bool = False,
) -> Result:
    """Iterated Local Search (Lourenço, Martin & Stützle, 2003).

    ``current <- LS(start)``, where ``start`` is ``"becker"`` (Becker's heuristic)
    or ``"random"``. Then repeat: kick ``current`` with ``strength`` random
    insertions, descend with local search, and accept the new local optimum
    according to ``acceptance``:

    * ``"better_equal"`` accepts if it is at least as good (a walk on plateaus of
      equal local optima);
    * ``"better"`` accepts only strict improvements;
    * ``"always"`` is a random walk over local optima.

    Diagnostics: ``return_rate`` (fraction of kicks after which the descent fell
    back into the very same local optimum, i.e. the kick was too weak to leave the
    basin) and ``acceptance_rate``.
    """
    n = matrix.shape[0]
    best = Incumbent(budget)
    current = becker(matrix) if start == "becker" else rng.permutation(n)
    best.offer(current, objective(matrix, current))
    current, current_value, _ = local_search(matrix, current, rng, neighbourhood, pivot, incumbent=best)
    visited = [current.copy()] if log_visits else None
    it = returns = accepted = 0
    while not best.done():
        candidate = perturb_inserts(current, strength, rng)
        candidate, value, st = local_search(matrix, candidate, rng, neighbourhood, pivot, incumbent=best)
        best.offer(candidate, value)
        if st.interrupted:
            break
        it += 1
        returns += np.array_equal(candidate, current)
        if (
            acceptance == "always"
            or (acceptance == "better_equal" and value >= current_value)
            or (acceptance == "better" and value > current_value)
        ):
            current, current_value = candidate, value
            accepted += 1
        if log_visits:
            visited.append(current.copy())
    diag = {"return_rate": returns / max(it, 1), "acceptance_rate": accepted / max(it, 1)}
    return best.result(it, diag, visited)


# --------------------------------------------------------------------------- Tabu search
def _crossing_counts(frozen: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """``X[i, j]`` = number of frozen pairs that inserting position ``i`` at ``j`` would reorder.

    Same prefix-sum structure as :func:`lop.objective.insert_gains`, applied to the
    0/1 matrix of frozen element pairs instead of the gain matrix.
    """
    p = frozen[np.ix_(perm, perm)].astype(np.int32)
    inclusive = np.cumsum(p, axis=1)
    exclusive = inclusive - p
    base = np.diagonal(exclusive)[:, None]
    upper = np.triu(np.ones_like(p, dtype=bool), 1)
    return np.where(upper, inclusive - base, base - exclusive)


def tabu_search(
    matrix: np.ndarray,
    rng: np.random.Generator,
    budget: Budget,
    tenure: int = 7,
    attribute: str = "pair",
    null_moves: bool = False,
    diversification: float = 0.0,
    restart_after: int | None = None,
    restart_strength: int = 10,
    start: str = "becker",
    log_visits: bool = False,
) -> Result:
    """Tabu Search over the insertion neighbourhood (Glover, 1989; Laguna et al., 1999).

    Each iteration applies the best *admissible* insertion move even when it
    worsens the solution, which lets the search climb out of local optima. After
    a move, part of its reversal is forbidden for ``tenure`` iterations:

    * ``attribute="element"``: the moved element may not be moved again (coarse).
    * ``attribute="position"``: the moved element may not return to the position it
      left (fine).
    * ``attribute="pair"``: the element pairs whose order the move changed are
      frozen, and any move that would reorder one of them is tabu. This is the only
      one of the three that forbids undoing a move by moving *another* element: an
      adjacent insertion can be reversed either by moving the element back or by
      moving its neighbour across it.

    Aspiration: a tabu move is allowed anyway if it would give a new best solution.
    ``tenure = 0`` removes the memory and leaves a steepest-ascent/mildest-descent
    walk, the control that shows what the tabu list contributes. Ties between
    equally good moves are broken at random.

    Three further components can be switched on and off for the ablation study:

    * ``null_moves=False`` (default) excludes zero-gain moves. LOLIB input-output
      matrices are sparse, so every solution has many insertions that leave the
      objective unchanged. With them allowed, the best admissible move is often one
      of these, and the search drifts across a plateau of equal value instead of
      taking the worsening step that leads out of it.
    * ``diversification`` adds long-term frequency memory in the spirit of Laguna
      et al. A non-improving move of element ``e`` is scored ``gain - diversification
      * s * freq(e)``, where ``freq(e)`` is the fraction of iterations since the last
      restart that moved ``e`` and ``s`` is the mean pairwise imbalance ``|m_ij - m_ji|``
      (which makes the weight unit-free). Improving moves are never penalised.
    * ``restart_after`` restarts from the *incumbent* (best-so-far) solution,
      kicked by ``restart_strength`` random insertions and with all memories
      cleared, after that many consecutive iterations without a new best. Unlike
      Laguna et al., there is no elite set and no path relinking. With restarts on,
      the method is a tabu walk nested inside an ILS-like outer loop, and the
      README reports it as such.

    Diagnostics (trajectory statistics): fractions of improving, zero-gain and
    worsening moves applied; ``distinct_solutions`` (distinct permutations visited
    divided by iterations; low values mean cycling); ``distinct_values`` (number of
    distinct objective values visited); ``restarts``.
    """
    n = matrix.shape[0]
    best = Incumbent(budget)
    perm = becker(matrix) if start == "becker" else rng.permutation(n)
    value = objective(matrix, perm)
    best.offer(perm, value)
    tabu_element = np.full(n, -1, dtype=np.int64)  # element -> last tabu iteration
    tabu_position = np.full((n, n), -1, dtype=np.int64)  # (element, position) -> last tabu iteration
    tabu_pair = np.full((n, n), -1, dtype=np.int64)  # (element, element) -> last tabu iteration
    moved = np.zeros(n, dtype=np.float64)  # element -> times moved since the last restart
    iu = np.triu_indices(n, 1)
    scale = float(np.abs(matrix - matrix.T)[iu].mean()) or 1.0
    visited = [perm.copy()] if log_visits else None
    seen_perms: set[int] = set()
    seen_values: set[int] = set()
    n_up = n_zero = n_down = restarts = 0
    it = last_improvement = phase_start = 0
    while not best.done():
        if restart_after is not None and it - last_improvement >= restart_after:
            perm = perturb_inserts(best.perm, restart_strength, rng)
            value = objective(matrix, perm)
            best.offer(perm, value)
            tabu_element[:] = tabu_position[:] = tabu_pair[:] = -1
            moved[:] = 0
            last_improvement = phase_start = it
            restarts += 1
        gains = insert_gains(matrix, perm).astype(np.float64)
        best.evaluations += n * (n - 1)
        np.fill_diagonal(gains, -np.inf)
        if attribute == "element":
            tabu = np.repeat((tabu_element[perm] >= it)[:, None], n, axis=1)
        elif attribute == "position":
            tabu = tabu_position[perm] >= it
        elif attribute == "pair":
            tabu = _crossing_counts(tabu_pair >= it, perm) > 0
        else:
            raise ValueError(f"unknown tabu attribute {attribute!r}")
        aspiration = value + gains > best.value
        gains[tabu & ~aspiration] = -np.inf
        if not null_moves:
            gains[gains == 0] = -np.inf
        if not np.isfinite(gains).any():  # no admissible move: lift the short-term memory
            tabu_element[:] = tabu_position[:] = tabu_pair[:] = -1
            it += 1
            continue
        score = gains
        if diversification > 0 and it > phase_start:
            penalty = (diversification * scale * moved[perm] / (it - phase_start))[:, None]
            score = np.where(gains > 0, gains, gains - penalty)
        i, j = divmod(_argmax_random_tie(score, rng), n)
        e = perm[i]
        gain = int(gains[i, j])
        crossed = perm[i + 1 : j + 1] if j > i else perm[j:i]
        value += gain
        perm = apply_insert(perm, i, j)
        it += 1
        until = it + tenure - 1
        tabu_element[e] = until
        tabu_position[e, i] = until
        tabu_pair[e, crossed] = until
        tabu_pair[crossed, e] = until
        moved[e] += 1
        n_up += gain > 0
        n_zero += gain == 0
        n_down += gain < 0
        seen_perms.add(hash(perm.tobytes()))
        seen_values.add(value)
        if best.offer(perm, value):
            last_improvement = it
        if log_visits:
            visited.append(perm.copy())
    moves = max(n_up + n_zero + n_down, 1)
    diag = {
        "improving_moves": n_up / moves,
        "zero_moves": n_zero / moves,
        "worsening_moves": n_down / moves,
        "distinct_solutions": len(seen_perms) / moves,
        "distinct_values": len(seen_values),
        "restarts": restarts,
    }
    return best.result(it, diag, visited)


ALGORITHMS = {"grasp": grasp, "ils": ils, "tabu": tabu_search}
