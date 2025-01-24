def insert(pi, i, j):
    """
    Inserts the element at position i into position j in the permutation pi.

    Parameters:
    - pi: List representing the permutation.
    - i: Index of the element to move.
    - j: Target index for the element.

    Returns:
    - new_pi: Modified permutation with the element moved.
    """
    new_pi = pi[:]
    element = new_pi.pop(i)
    new_pi.insert(j, element)
    return new_pi


def delta(matrix, pi, i, j):
    """
    Computes the change in objective value for moving element i to position j.

    Parameters:
    - matrix: 2D list representing the cost matrix.
    - pi: List representing the current permutation.
    - i: Index of the element to move.
    - j: Target index for the element.

    Returns:
    - delta: The calculated change in objective value.
    """
    if i < j:
        return sum(
            matrix[pi[k]][pi[i]] - matrix[pi[i]][pi[k]] for k in range(i + 1, j + 1)
        )
    elif i > j:
        return sum(matrix[pi[i]][pi[k]] - matrix[pi[k]][pi[i]] for k in range(j, i))
    else:
        return 0


def get_obj_value(matrix, pi):
    """
    Computes the objective value of a given permutation.

    Parameters:
    - matrix: 2D list representing the cost matrix.
    - pi: List representing the permutation.

    Returns:
    - obj_value: The calculated objective value.
    """
    return sum(
        sum(matrix[pi[i]][pi[j]] for j in range(i + 1, len(pi))) for i in range(len(pi))
    )


def granularity_fine(i, j):
    """
    Fine-grained granularity representation for tabu list.

    Parameters:
    - i: Index of the element being moved.
    - j: Target index for the element.

    Returns:
    - Tuple (i, j): Fine granularity representation.
    """
    return (i, j)


def granularity_coarse(i, j):
    """
    Coarse-grained granularity representation for tabu list.

    Parameters:
    - i: Index of the element being moved.
    - j: Target index for the element.

    Returns:
    - i: Coarse granularity representation.
    """
    return i


def tabu_iteration(matrix, pi, tabu_list, tenure):
    """
    Performs one iteration of the tabu search to find the best neighbor.

    Parameters:
    - matrix: 2D list representing the cost matrix.
    - pi: List representing the current permutation.
    - tabu_list: List of tabu moves.
    - tenure: Maximum size of the tabu list.

    Returns:
    - Tuple (best_neighbor_value, best_neighbor): Best objective value and corresponding permutation.
    - None: If no valid neighbor is found.
    """
    best_neighbor_value = 0
    best_neighbor = None
    size = len(pi)
    # local search : parcourt le voisinage et choisit le meilleur
    for i in range(size):
        for j in range(size):
            if i!=j :
                element_at_i= pi[i]

                if element_at_i not in tabu_list and insert(pi,i,j)!=pi:
                    current_neighbor=insert(pi,i,j)
                    current_neighbor_value=get_obj_value(matrix,current_neighbor)

                    if current_neighbor_value>best_neighbor_value:
                        best_neighbor_value=current_neighbor_value
                        best_neighbor=current_neighbor
                        tabu_move=element_at_i

    if best_neighbor is not None and tabu_move not in tabu_list:
        tabu_list.append(tabu_move)
    if len(tabu_list)>tenure:
        tabu_list.pop(0)
    if best_neighbor is not None:
        return best_neighbor_value,best_neighbor
    return None


def tabu_search(matrix, pi_init=False, tenure=False, iter=50, log_visits=False):
    """
    Performs tabu search to optimize the objective function.

    Parameters:
    - matrix: 2D list representing the cost matrix.
    - pi_init: Initial permutation.
    - tenure: Maximum size of the tabu list.
    - iter: Number of iterations to perform.
    - log_visits: Boolean flag to log the visited permutations.

    Returns:
    - Tuple (best_obj_value, best_pi): Best objective value and corresponding permutation.
    """
    if not pi_init:
        pi_init = list(range(len(matrix)))
    if not tenure:
        tenure = int(len(matrix)/4)
    best_obj_value=get_obj_value(matrix,pi_init)
    best_pi=pi_init
    current_pi=best_pi
    tabu_list=[]
    visited = [best_pi]
    for k in range(iter):
        best_neighbor=tabu_iteration(matrix, current_pi, tabu_list, tenure)

        if best_neighbor is not None:
            current_value,current_pi=best_neighbor[0],best_neighbor[1]
            if log_visits:
                visited.append(current_pi)
        if current_value>best_obj_value:
            best_obj_value=current_value
            best_pi=current_pi
    if log_visits:
        return best_obj_value, best_pi, visited
    else:
        return best_obj_value, best_pi

# Example usage
# Replace `matrix` and `pi_init` with actual values.
# tabu_search(matrix=matrix, pi_init=pi_init, tenure=10, n=10, granularity_func=granularity_fine)
