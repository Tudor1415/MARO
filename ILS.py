import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from GRASP import objective_function


def becker_constructive_algorithm(matrix):
    """
    Implements Becker's constructive heuristic for the Linear Ordering Problem.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).

    Returns:
    - permutation: List of indices representing the order.
    """
    n = matrix.shape[0]  # Number of rows/columns in the matrix
    indices = list(range(n))  # List of all indices
    permutation = []  # The final permutation

    while indices:
        # Compute q_i values for all remaining indices
        q_values = [np.sum(matrix[i, :]) / np.sum(matrix[:, i]) for i in indices]

        # Find the index with the maximum q value
        max_index = indices[np.argmax(q_values)]

        # Append this index to the permutation
        permutation.append(max_index)

        # Remove the selected index and its row and column from consideration
        indices.remove(max_index)

    return permutation


def interchange(permutation, i, j):
    """
    Performs an interchange operation on a permutation.

    Parameters:
    - permutation: List of indices representing the current order.
    - i: Index of the first element to swap.
    - j: Index of the second element to swap.

    Returns:
    - new_permutation: A new list with the elements at i and j swapped.
    """
    new_permutation = permutation.copy()
    new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]
    return new_permutation


def insert(permutation, i, j):
    """
    Performs an insert operation on a permutation.

    Parameters:
    - permutation: List of indices representing the current order.
    - i: Index of the element to move.
    - j: Index where the element should be inserted.

    Returns:
    - new_permutation: A new list with the element at index i inserted at index j.
    """
    new_permutation = permutation.copy()
    element = new_permutation.pop(i)
    new_permutation.insert(j, element)
    return new_permutation


def delta_insert(matrix, permutation, i, j):
    """
    Computes the delta function for an insert operation on the permutation.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - permutation: Current permutation of indices.
    - i: Index of the element to move.
    - j: Index where the element is to be inserted.

    Returns:
    - delta: The change in the objective function value due to the insert operation.
    """
    n = len(permutation)
    if i < j:
        delta = sum(
            matrix[permutation[k], permutation[i]] for k in range(i + 1, j + 1)
        ) - sum(matrix[permutation[i], permutation[k]] for k in range(i + 1, j + 1))
    else:
        delta = sum(matrix[permutation[i], permutation[k]] for k in range(j, i)) - sum(
            matrix[permutation[k], permutation[i]] for k in range(j, i)
        )
    return delta


def delta_swap(matrix, permutation, i, j):
    """
    Computes the delta function for a swap operation on the permutation.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - permutation: Current permutation of indices.
    - i: Index of the first element to swap.
    - j: Index of the second element to swap.

    Returns:
    - delta: The change in the objective function value due to the swap operation.
    """
    if i == j:
        return 0

    if i == (j + 1):
        return (
            matrix[permutation[i], permutation[j]]
            - matrix[permutation[j], permutation[i]]
        )
    elif i == (j - 1):
        return (
            matrix[permutation[j], permutation[i]]
            - matrix[permutation[i], permutation[j]]
        )
    else:
        raise Exception("Invalid swap operation")


def visit_CK(matrix, permutation, objective_function):
    """
    Recursively sorts a permutation based on the defined criteria.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - permutation: List of indices representing the current order.

    Returns:
    - sorted_permutation: A new permutation that is sorted.
    """
    if len(permutation) == 1:
        return permutation

    last_element = permutation[-1]
    sorted_sublist = visit_CK(matrix, permutation[:-1], objective_function)

    # Determine the best position r that maximizes the delta function
    best_position = 0
    best_delta = float("-inf")
    for r in range(len(sorted_sublist) + 1):
        delta = sum(
            matrix[permutation[j], last_element] for j in sorted_sublist[:r]
        ) + sum(permutation[last_element, permutation[j]] for j in sorted_sublist[r:])
        if delta > best_delta:
            best_delta = delta
            best_position = r

    # Insert the last element at the best position
    sorted_permutation = (
        sorted_sublist[:best_position] + [last_element] + sorted_sublist[best_position:]
    )
    return sorted_permutation


def visit_NI(matrix, permutation, objective_function):
    """
    Implements the visit_NI visit neighbourhood function for the insertion neighborhood.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - permutation: List of indices representing the current order.
    - objective_function: A function that computes the objective value for a given permutation.

    Returns:
    - new_permutation: A modified permutation if an improvement is found, otherwise the original permutation.
    """
    n = len(permutation)
    current_value = objective_function(matrix, permutation)

    for i in range(n):
        delta_insert_values = [
            delta_insert(matrix, permutation, i, r) for r in range(n) if r != i
        ]

        best_index = np.argmax(delta_insert_values)

        new_permutation = insert(permutation, i, best_index)

        if objective_function(matrix, new_permutation) > current_value:
            return new_permutation

    return permutation


def visit_NS(matrix, permutation, objective_function):
    """
    Implements the visit_NS visit neighbourhood function for the swap neighborhood.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - permutation: List of indices representing the current order.
    - objective_function: A function that computes the objective value for a given permutation.

    Returns:
    - new_permutation: A modified permutation if an improvement is found, otherwise the original permutation.
    """
    n = len(permutation)
    current_value = objective_function(matrix, permutation)

    for i in range(n):
        delta_swap_values = [
            delta_swap(matrix, permutation, i, r) for r in [i - 1, i + 1] if 0 <= r < n
        ]

        best_index = np.argmax(delta_swap_values)

        new_permutation = interchange(permutation, i, best_index)

        if objective_function(matrix, new_permutation) > current_value:
            return new_permutation

    return permutation


def objective_function(matrix, permutation):
    """
    Computes the objective function value for a given permutation.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - permutation: List of indices representing the current order.

    Returns:
    - objective_value: The computed objective value for the permutation.
    """
    objective_value = 0
    n = len(permutation)

    if n != len(matrix):
        raise ValueError(
            f"Permutation length {n} does not match matrix size {len(matrix)}"
        )

    if max(permutation) >= len(matrix):
        raise ValueError(
            f"Permutations should have values between 0 and {len(matrix) - 1}"
        )

    for i in range(n):
        for j in range(i + 1, n):
            objective_value += matrix[permutation[i], permutation[j]]
    return objective_value


def LS(
    matrix,
    objective_function,
    visit_N,
    start_permutation,
    max_iter=40,
    log_visits=False,
    debug=False,
):
    """
    Implements the Local Search (LS) algorithm for the Linear Ordering Problem.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - objective_function: A function that computes the objective value for a given permutation.
    - visit_N: A function that generates a new permutation based on the neighbourhood.
    - start_permutation: The initial permutation to start the search from.
    - max_iter: Maximum number of iterations to perform.
    - log_visits: Boolean flag to log the visited permutations.
    - debug: Boolean flag to enable/disable debugging statements.

    Returns:
    - best_permutation: The best permutation found by the algorithm.
    - best_value: The objective value of the best permutation.
    """
    if debug:
        print("Initial permutation:", start_permutation)
        print("Initial objective value:", objective_function(matrix, start_permutation))

    visited = []
    current_permutation = start_permutation
    best_value = objective_function(matrix, current_permutation)
    for iteration in range(max_iter):
        # Perform a local search on the current solution
        new_permutation = visit_N(matrix, current_permutation, objective_function)
        if log_visits:
            visited.append(new_permutation)

        # Update the best solution if a better one is found
        new_value = objective_function(matrix, new_permutation)
        if new_value > best_value:
            current_permutation = new_permutation
            best_value = new_value
            if debug:
                print(
                    f"Iteration {iteration}: Found better permutation {new_permutation} with value {new_value}"
                )

    if log_visits:
        return current_permutation, best_value, visited
    else:
        return current_permutation, best_value


def perturb_random(permutation, percentage=1):
    """
    Perturbs a permutation by removing half of the permutation and replacing it with a random permutation.

    Parameters:
    - permutation: List of indices representing the current order.
    - percentage: Percentage of the permutation to shuffle.
    - k: Number of random swap operations to perform.

    Returns:
    - perturbed_permutation: A new permutation that has been perturbed.
    """
    n = len(permutation)
    to_shuffle = int(percentage * n)
    indices = list(range(n))
    np.random.shuffle(indices)
    perturbed_permutation = permutation[:to_shuffle] + indices[to_shuffle:]
    return perturbed_permutation


def ILS(
    matrix,
    objective_function,
    constructive_heuristic,
    perturbation,
    visit_N,
    max_iter=10,
    max_local_iter=40,
    log_visits=False,
    debug=False,
):
    """
    Implements the Iterated Local Search (ILS) algorithm for the Linear Ordering Problem.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - objective_function: A function that computes the objective value for a given permutation.
    - constructive_heuristic: A function that generates an initial solution.
    - perturbation: A function that perturbs a permutation.
    - max_iter: Maximum number of iterations to perform.
    - log_visits: Boolean flag to log the visited permutations.
    - debug: Boolean flag to enable/disable debugging statements.

    Returns:
    - best_permutation: The best permutation found by the algorithm.
    - best_value: The objective value of the best permutation.
    """
    n = matrix.shape[0]
    current_permutation = constructive_heuristic(matrix)
    best_permutation = current_permutation
    best_value = objective_function(matrix, current_permutation)
    visited = [current_permutation]

    for _ in range(max_iter):
        if log_visits:
            local_best_permutation, best_value, visited = LS(
                matrix=matrix,
                objective_function=objective_function,
                visit_N=visit_N,
                start_permutation=current_permutation,
                log_visits=True,
                max_iter=max_local_iter,
            )
        else:
            local_best_permutation, best_value = LS(
                matrix=matrix,
                objective_function=objective_function,
                visit_N=visit_N,
                start_permutation=current_permutation,
                max_iter=max_local_iter,
            )
        if best_value > objective_function(matrix, local_best_permutation):
            best_permutation = local_best_permutation

        current_permutation = perturbation(best_permutation)

    if log_visits:
        return best_permutation, best_value, visited
    else:
        return best_permutation, best_value
