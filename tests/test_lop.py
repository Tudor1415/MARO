import itertools

import numpy as np
import pytest

from lop import (
    Budget, adjacent_swap_gains, apply_insert, becker, grasp, grasp_construct, ils,
    insert_gains, insert_gains_row, kendall_distance, load_instances, local_search,
    objective, perturb_inserts, tabu_search,
)


def random_matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 100, size=(n, n))
    np.fill_diagonal(m, 0)
    return m


def brute_force_optimum(m):
    n = m.shape[0]
    return max(objective(m, np.array(p)) for p in itertools.permutations(range(n)))


def is_permutation(p, n):
    return sorted(p.tolist()) == list(range(n))


def test_objective_matches_definition():
    m = random_matrix(9)
    p = np.random.default_rng(1).permutation(9)
    expected = sum(m[p[i], p[j]] for i in range(9) for j in range(i + 1, 9))
    assert objective(m, p) == expected


def test_objective_of_pair_is_complementary():
    # f(pi) + f(reverse(pi)) = total off-diagonal weight
    m = random_matrix(11, 3)
    p = np.random.default_rng(2).permutation(11)
    assert objective(m, p) + objective(m, p[::-1]) == m.sum() - np.trace(m)


@pytest.mark.parametrize("seed", range(5))
def test_insert_gains_match_brute_force(seed):
    n = 10
    m = random_matrix(n, seed)
    p = np.random.default_rng(seed + 100).permutation(n)
    g = insert_gains(m, p)
    base = objective(m, p)
    for i in range(n):
        for j in range(n):
            assert g[i, j] == objective(m, apply_insert(p, i, j)) - base
        np.testing.assert_array_equal(insert_gains_row(m, p, i), g[i])


def test_apply_insert():
    p = np.arange(6)
    assert apply_insert(p, 1, 4).tolist() == [0, 2, 3, 4, 1, 5]
    assert apply_insert(p, 4, 1).tolist() == [0, 4, 1, 2, 3, 5]
    assert apply_insert(p, 3, 3).tolist() == p.tolist()


def test_adjacent_swap_gains():
    m = random_matrix(8, 4)
    p = np.random.default_rng(5).permutation(8)
    g = adjacent_swap_gains(m, p)
    for i in range(7):
        q = p.copy()
        q[i], q[i + 1] = q[i + 1], q[i]
        assert g[i] == objective(m, q) - objective(m, p)


def test_kendall_distance():
    p = np.arange(7)
    assert kendall_distance(p, p) == 0
    assert kendall_distance(p, p[::-1]) == 21
    q = p.copy(); q[2], q[3] = q[3], q[2]
    assert kendall_distance(p, q) == 1


def test_constructors_return_permutations():
    m = random_matrix(15, 6)
    rng = np.random.default_rng(0)
    assert is_permutation(becker(m), 15)
    for a in (0.0, 0.3, 1.0):
        assert is_permutation(grasp_construct(m, rng, a), 15)
    assert is_permutation(perturb_inserts(np.arange(15), 7, rng), 15)


def test_becker_orders_a_transitive_tournament():
    # m[i, j] = 1 iff i < j: identity is the unique optimum and Becker finds it.
    n = 8
    m = np.triu(np.ones((n, n), dtype=np.int64), 1)
    assert becker(m).tolist() == list(range(n))
    assert grasp_construct(m, np.random.default_rng(0), 0.0).tolist() == list(range(n))


@pytest.mark.parametrize("nb", ["insert", "swap"])
@pytest.mark.parametrize("pivot", ["best", "first", "element"])
def test_local_search_reaches_local_optimum(nb, pivot):
    m = random_matrix(20, 7)
    rng = np.random.default_rng(1)
    start = rng.permutation(20)
    p, v, stats = local_search(m, start, rng, nb, pivot)
    assert is_permutation(p, 20)
    assert v == objective(m, p) >= objective(m, start)
    gains = insert_gains(m, p) if nb == "insert" else adjacent_swap_gains(m, p)
    assert gains.max() <= 0


def test_insert_local_optima_are_swap_local_optima():
    # an adjacent swap is an insertion move, so N_swap is a subset of N_insert
    m = random_matrix(25, 8)
    rng = np.random.default_rng(3)
    p, _, _ = local_search(m, rng.permutation(25), rng, "insert", "best")
    assert adjacent_swap_gains(m, p).max() <= 0


@pytest.mark.parametrize("algo,kwargs", [
    (grasp, {"alpha": 0.2}),
    (grasp, {"alpha": 1.0}),
    (ils, {"strength": 3}),
    (ils, {"strength": 3, "start": "random", "pivot": "first"}),
    (tabu_search, {"tenure": 2, "restart_after": 20}),
    (tabu_search, {"tenure": 2, "attribute": "element", "restart_after": 20}),
    (tabu_search, {"tenure": 2, "attribute": "position", "restart_after": 20, "diversification": 10.0}),
])
@pytest.mark.parametrize("seed", range(3))
def test_metaheuristics_find_small_optima(algo, kwargs, seed):
    m = random_matrix(7, seed + 20)
    opt = brute_force_optimum(m)
    res = algo(m, np.random.default_rng(seed), Budget(5.0, target=opt), **kwargs)
    assert res.value == opt == objective(m, res.perm)
    assert res.time_to_target is not None
    values = [v for _, v, _ in res.trace]
    assert values == sorted(values) and values[-1] == res.value
    assert res.time_to_target <= res.elapsed


@pytest.mark.parametrize("algo", [grasp, ils, tabu_search])
@pytest.mark.parametrize("seconds", [0.002, 0.2])
def test_budget_is_respected(algo, seconds):
    m = random_matrix(120, 9)
    res = algo(m, np.random.default_rng(0), Budget(seconds))
    assert res.elapsed < seconds + 0.02  # overrun bounded by one move evaluation, not a descent
    assert res.time_to_target is None
    assert res.value == objective(m, res.perm)


def test_pair_attribute_forbids_reversal_by_the_other_element():
    from lop.search import _crossing_counts
    n = 6
    perm = np.array([0, 1, 2, 3, 4, 5])
    frozen = np.zeros((n, n), dtype=bool)
    frozen[2, 3] = frozen[3, 2] = True  # the last move put 2 before 3
    x = _crossing_counts(frozen, perm)
    assert x[2, 3] > 0 and x[3, 2] > 0  # moving either element across the other is tabu
    assert x[2, 1] == 0 and x[3, 4] == 0 and x[0, 5] == 0 and x[5, 0] == 0
    for i in range(n):
        for j in range(n):
            q = apply_insert(perm, i, j)
            pos = np.empty(n, int); pos[q] = np.arange(n)
            assert (x[i, j] > 0) == (pos[3] < pos[2])


def test_tabu_diagnostics_and_restart_offer():
    m = random_matrix(30, 10)
    r = tabu_search(m, np.random.default_rng(0), Budget(0.3), tenure=5, restart_after=10)
    d = r.diagnostics
    assert abs(d["improving_moves"] + d["zero_moves"] + d["worsening_moves"] - 1) < 1e-9
    assert d["restarts"] > 0 and 0 < d["distinct_solutions"] <= 1
    assert r.value == objective(m, r.perm)


def test_restart_that_lands_on_optimum_is_recorded():
    # restart_strength=0 restarts exactly at the incumbent; the incumbent can never be lost
    m = random_matrix(7, 3)
    opt = brute_force_optimum(m)
    r = tabu_search(m, np.random.default_rng(1), Budget(0.5, target=opt), tenure=3, restart_after=1,
                    restart_strength=0)
    assert r.value == opt


def test_repository_instances_load():
    insts = load_instances("instances", "instances/optima.json")
    assert len(insts) == 49
    for inst in insts:
        assert inst.matrix.shape == (inst.n, inst.n)
        if inst.optimum is not None:
            assert inst.symmetric_lower_bound <= inst.optimum <= inst.diagonal_free_total
