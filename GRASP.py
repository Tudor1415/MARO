import copy
import random


def contribution(candidate, matrix):
    """
    Calculates the contribution of a candidate based on the given matrix.

    Parameters:
    - candidate: Index of the candidate.
    - matrix: 2D array representing the cost matrix.

    Returns:
    - sum: Total contribution of the candidate.
    """
    total = 0
    n = len(matrix)
    for i in range(n):
        if i != candidate:
            total += matrix[candidate][i]
    return total


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
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            objective_value += matrix[permutation[i], permutation[j]]
    return objective_value


def calculate_candidates(current_sol, matrix):
    """
    Calculates the list of candidates not in the current solution.

    Parameters:
    - current_sol: List of indices in the current solution.
    - matrix: 2D array representing the cost matrix.

    Returns:
    - candidates: List of tuples (candidate index, contribution).
    """
    candidates = []
    n = len(matrix)
    for candidate in range(n):
        if candidate not in current_sol:
            candidates.append((candidate, contribution(candidate, matrix)))
    return candidates


def calculate_rcl(candidates, alpha):
    """
    Calculates the Restricted Candidate List (RCL).

    Parameters:
    - candidates: List of tuples (candidate index, contribution).
    - alpha: Parameter to control the size of the RCL.

    Returns:
    - rcl: Restricted Candidate List.
    """
    min_cost = candidates[0][1]
    max_cost = candidates[-1][1]
    threshold = min_cost + alpha * (max_cost - min_cost)

    # If threshold equals max_cost, return all candidates
    if threshold == max_cost:
        return candidates

    # Filter candidates by threshold
    rcl = [candidate for candidate in candidates if candidate[1] >= threshold]
    return rcl


def construct_grasp(matrix, alpha=0.1):
    """
    Constructs a solution using the GRASP (Greedy Randomized Adaptive Search Procedure) heuristic.

    Parameters:
    - matrix: 2D array representing the cost matrix.
    - alpha: Parameter to control the size of the RCL (default 0.1).

    Returns:
    - permutation: Constructed solution as a list of indices.
    """
    permutation = []
    n = len(matrix)
    current_matrix = copy.deepcopy(matrix)

    # Construct the solution iteratively
    for _ in range(n):
        candidates = calculate_candidates(permutation, current_matrix)
        candidates.sort(
            key=lambda x: x[1], reverse=True
        )  # Sort by contribution (descending)
        rcl = calculate_rcl(candidates, alpha)

        # Choose a random candidate from the RCL
        candidate, _ = random.choice(rcl)
        permutation.append(candidate)

        # Update the matrix to remove the chosen candidate
        for index in range(n):
            current_matrix[index][candidate] = 0
            current_matrix[candidate][index] = 0

    return permutation


import copy
import random


def contribution(candidate, matrix):
    """
    Calculates the contribution of a candidate based on the given matrix.

    Parameters:
    - candidate: Index of the candidate.
    - matrix: 2D array representing the cost matrix.

    Returns:
    - sum: Total contribution of the candidate.
    """
    total = 0
    n = len(matrix)
    for i in range(n):
        if i != candidate:
            total += matrix[candidate][i]
    return total


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
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            objective_value += matrix[permutation[i], permutation[j]]
    return objective_value


def calculate_candidates(current_sol, matrix):
    """
    Calculates the list of candidates not in the current solution.

    Parameters:
    - current_sol: List of indices in the current solution.
    - matrix: 2D array representing the cost matrix.

    Returns:
    - candidates: List of tuples (candidate index, contribution).
    """
    candidates = []
    n = len(matrix)
    for candidate in range(n):
        if candidate not in current_sol:
            candidates.append((candidate, contribution(candidate, matrix)))
    return candidates


def calculate_rcl(candidates, alpha):
    """
    Calculates the Restricted Candidate List (RCL).

    Parameters:
    - candidates: List of tuples (candidate index, contribution).
    - alpha: Parameter to control the size of the RCL.

    Returns:
    - rcl: Restricted Candidate List.
    """
    min_cost = candidates[0][1]
    max_cost = candidates[-1][1]
    threshold = min_cost + alpha * (max_cost - min_cost)

    # If threshold equals max_cost, return all candidates
    if threshold == max_cost:
        return candidates

    # Filter candidates by threshold
    rcl = [candidate for candidate in candidates if candidate[1] >= threshold]
    return rcl


def construct_grasp(matrix, alpha=0.1):
    """
    Constructs a solution using the GRASP (Greedy Randomized Adaptive Search Procedure) heuristic.

    Parameters:
    - matrix: 2D array representing the cost matrix.
    - alpha: Parameter to control the size of the RCL (default 0.1).

    Returns:
    - permutation: Constructed solution as a list of indices.
    """
    permutation = []
    n = len(matrix)
    current_matrix = copy.deepcopy(matrix)

    # Construct the solution iteratively
    for _ in range(n):
        candidates = calculate_candidates(permutation, current_matrix)
        candidates.sort(
            key=lambda x: x[1], reverse=True
        )  # Sort by contribution (descending)
        rcl = calculate_rcl(candidates, alpha)

        # Choose a random candidate from the RCL
        candidate, _ = random.choice(rcl)
        permutation.append(candidate)

        # Update the matrix to remove the chosen candidate
        for index in range(n):
            current_matrix[index][candidate] = 0
            current_matrix[candidate][index] = 0

    return permutation
