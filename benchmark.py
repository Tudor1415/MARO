from ILS import *
from GRASP import construct_grasp
from vizualization import *
from scipy import stats
import timeit


def benchmark_neighbourhood_instance(matrix, max_iter=100, debug=False):
    """
    Runs the ILS algorithm and benchmarks its performance.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - max_iter: Maximum number of iterations for each ILS run.
    - debug: Boolean flag to enable/disable debugging statements.
    """

    _, best_value_NI = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NI,
        max_iter,
        debug,
    )
    _, best_value_NS = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NS,
        max_iter,
        debug,
    )

    if debug:
        print(f"Best value NI = {max(best_value_NI)}, NS = {max(best_value_NS)}")

    return (best_value_NI, best_value_NS)


def print_neighbourhood_benchmark_statistics(results):
    improvements = []
    for key, (n_i, n_s) in results.items():
        improvement = (n_i - n_s) / n_s  # Relative improvement
        improvements.append(improvement)

    # Compute mean and standard deviation
    mean_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)
    print(
        f"Overall mean relative improvement of N_I over N_S: {mean_improvement*100:.2f}% ± {std_improvement*100:.2f}%"
    )


def benchmark_starting_solution_instance(
    matrix, max_iter=100, nb_repetitions=10, debug=False
):
    """
    Runs the ILS algorithm with different starting solutions and benchmarks its performance.
    Starting solutions are generated using:
        - a random permutation
        - Becker's constructive heuristic
    """
    n = matrix.shape[0]

    random_score_values = []
    # Random permutation
    for _ in range(nb_repetitions):
        random_permutation = np.random.permutation(n).tolist()
        _, best_value_random = ILS(
            matrix,
            objective_function,
            lambda x: random_permutation,
            visit_NI,
            max_iter,
            debug,
        )
        random_score_values.append(best_value_random)

    mean_random_score = np.mean(random_score_values)

    # Becker's constructive heuristic
    _, best_value_becker = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NI,
        max_iter,
        debug,
    )

    if debug:
        print(f"Best value Random = {mean_random_score}")
        print(f"Best value Becker = {best_value_becker}")

    return (mean_random_score, best_value_becker)


def print_starting_solution_benchmark_statistics(results):
    """
    Prints the relative improvements of the different starting solutions.
    """

    improvements_becker_random, improvements_random_monotone = [], []
    for key, (monotone, random, becker) in results.items():
        improvement_becker_random = (becker - random) / random
        improvements_becker_random.append(improvement_becker_random)
        improvement_random_monotone = (random - monotone) / monotone
        improvements_random_monotone.append(improvement_random_monotone)

    mean_barker_random = np.mean(improvements_becker_random)
    std_barker_random = np.std(improvements_becker_random)

    mean_random_monotone = np.mean(improvements_random_monotone)
    std_random_monotone = np.std(improvements_random_monotone)

    print(
        f"Mean relative improvement of Becker over Random: {mean_barker_random*100:.2f}% ± {std_barker_random*100:.2f}%"
    )
    print(
        f"Mean relative improvement of Random over Monotone: {mean_random_monotone*100:.2f}% ± {std_random_monotone*100:.2f}%"
    )


def benchmark_visited_points(matrix, max_iter=100, debug=False):
    """
    Benchmarks the number of visited points using the Iterated Local Search (ILS) algorithm.
    Args:
        matrix (list of list of int): The input matrix representing the problem instance.
        max_iter (int, optional): The maximum number of iterations for the ILS algorithm. Defaults to 100.
        debug (bool, optional): If True, enables debug mode for additional output. Defaults to False.
    Returns:
        list: A list of visited points during the ILS algorithm execution.
    """
    _, best_value, visited = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NI,
        max_iter,
        True,
        False,
    )

    return visited


def plot_permutations_with_pca_benchmark(results):
    """
    Plots the visited points during the ILS algorithm execution.
    Args:
        results (dict): A dict of filename and visited points.
    """
    for key, visited_points in results.items():
        matrix = read_square_matrix_from_file(key, False)["matrix"]
        obj_func = lambda x: objective_function(matrix, x)
        plot_permutations_with_pca(visited_points, obj_func)
        break


def benchmark_score_evolution(matrix, max_iter=100, debug=False):
    """
    Benchmarks the score evolution using the Iterated Local Search (ILS) algorithm.
    Args:
        matrix (list of list of int): The input matrix representing the problem instance.
        max_iter (int, optional): The maximum number of iterations for the ILS algorithm. Defaults to 100.
        debug (bool, optional): If True, enables debug mode for additional output. Defaults to False.
    Returns:
        list: A list of visited points during the ILS algorithm execution.
    """
    _, best_value, visited = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NI,
        max_iter,
        True,
        False,
    )

    if debug:
        print(f"Best value = {best_value}")
        print(f"Visited points = {visited}")

    return [objective_function(matrix, permutation) for permutation in visited]


def benchmark_neighbourhood_diversity(matrix, max_iter=50, debug=False):
    """
    Runs the ILS algorithm and computes the pairwise kendall tau distance between the permutations.
    Parameters:
        matrix (np.array): The cost matrix.
        max_iter (int): The maximum number of iterations for the ILS algorithm.
        debug (bool): Flag to enable/disable debugging. Default is False.
    Returns:
        cummulative distribution of the pairwise kendall tau distance.
    """
    _, _, visited = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NI,
        max_iter,
        True,
        False,
    )
    if debug:
        print(f"Visited points: {visited}")

    n = len(visited)
    distances = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            distances[idx] = stats.spearmanr(visited[i], visited[j])[0]
            idx += 1

    return distances


def benchmark_execution_time(matrix, max_iter=100, debug=False):
    """
    Runs the ILS algorithm and benchmarks its performance in terms of computational time.
    """
    start_time_NI = timeit.default_timer()
    _, best_value_NI = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NI,
        max_iter,
        False,
        debug,
    )
    end_time_NI = timeit.default_timer()

    start_time_NS = timeit.default_timer()
    _, best_value_NS = ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        visit_NS,
        max_iter,
        False,
        debug,
    )
    end_time_NS = timeit.default_timer()

    execution_time_NI = (end_time_NI - start_time_NI) / 5
    execution_time_NS = (end_time_NS - start_time_NS) / 5

    if debug:
        print(f"Execution time NI: {execution_time_NI:.2f} seconds")
        print(f"Execution time NS: {execution_time_NS:.2f} seconds")

    return len(matrix), execution_time_NI, execution_time_NS


def print_execution_time_statistics(results):
    """
    Prints the execution time statistics.
    """
    size_time_ni, size_time_ns = {}, {}
    for key, values in results.items():
        size_time_ni.setdefault(values[0], []).append(values[1])
        size_time_ns.setdefault(values[0], []).append(values[2])

    sizes = sorted(size_time_ni.keys())
    ni_means = [np.mean(size_time_ni[size]) for size in sizes]
    ni_vars = [np.var(size_time_ni[size]) for size in sizes]

    ns_means = [np.mean(size_time_ns[size]) for size in sizes]
    ns_vars = [np.var(size_time_ns[size]) for size in sizes]

    for size in sizes:
        print(
            f"Size: {size}, NI: {np.mean(size_time_ni[size]):.2f} ± {np.var(size_time_ns[size]):.2f}, , NS: {np.mean(size_time_ns[size]):.2f} ± {np.var(size_time_ni[size]):.2f}"
        )


def benchmark_grasp_constructive(matrix, nb_repeats=10, debug=False):
    """
    Runs the ILS algorithm with GRASP constructive heuristic with different alphas and benchmarks its performance.
    """
    alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1]

    results = []
    for alpha in alpha_values:
        for _ in range(nb_repeats):
            permutation = construct_grasp(matrix, alpha)
            value = objective_function(matrix, permutation)
            results.append((alpha, value))

    return tuple(results)


def print_grasp_constructive_benchmark_statistics(results, debug=False):
    """
    Prints the results of the GRASP constructive heuristic benchmark.
    """
    scores = {}
    for key, values in results.items():
        for i in range(len(values)):
            alpha, value = values[i]
            if alpha not in scores:
                scores[alpha] = []
            scores[alpha].append(value)

    if debug:
        print("Scores by alpha:")
        for alpha, value in scores.items():
            print(f"Alpha = {alpha}: Values = {value}")

    print("Statistics:")
    for alpha, value in scores.items():
        print(f"Alpha = {alpha}: Value = {np.mean(value):.2f} ± {np.std(value):.2f}")


def read_square_matrix_from_file(file_path, debug=False):
    """
    Reads a square matrix from a file where the size is specified at the beginning.

    Parameters:
        file_path (str): Path to the file containing the matrix.
        debug (bool): Boolean flag to enable/disable debugging statements.

    Returns:
        dict: A dictionary containing:
            - 'header': The header line of the file.
            - 'size': The size of the square matrix (int).
            - 'matrix': A 2D list representing the matrix.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the header and size
    header = lines[0].strip()
    size = int(lines[1].strip())

    if debug:
        print(f"Header: {header}")
        print(f"Matrix size: {size}")

    # Process the matrix
    matrix = []
    nb_entries = 0
    for line in lines[2:]:
        row = [int(num) for num in line.split()]
        nb_entries += len(row)
        if row:  # Add only non-empty rows
            matrix.extend(row)

    if nb_entries % size != 0:
        raise ValueError(
            "The number of entries in the matrix does not match the specified size."
        )

    if debug:
        print("Number of entries:", nb_entries)
        print(f"Shape: {nb_entries // size} x {size}")
    # Reshape the matrix to the desired size
    matrix = np.array(matrix).reshape((size, size))

    return {"header": header, "size": size, "matrix": matrix}


def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def process_file(file_name, benchmark_instance, max_iter, debug):
    matrix = read_square_matrix_from_file(file_name, debug)["matrix"]
    results = benchmark_instance(matrix, max_iter, debug)
    return file_name, results


def benchmark(
    filename, benchmark_instance, print_statistics, max_iter=100, debug=False
):
    """
    Reads all the files in the current directory that have the .mat extension and runs the benchmark_instance function on them.

    Parameters:
        - filename (str): The name of the file to print the results to.
        - benchmark_instance (callable): The function to run for benchmarking each file.
        - print_statistics (callable): The function to print the statistics of the results.
        - max_iter (int, optional): Maximum number of iterations for each ILS run. Default is 100.
        - debug (bool, optional): Boolean flag to enable/disable debugging statements. Default is False.

    Returns:
        dict: A dictionary containing the results of the benchmarking:
            - key: The name of the file.
            - value: A tuple of lists containing the results for the NI and NS neighborhoods.
    """
    results = {}
    files = [
        "instances/" + file_name
        for file_name in os.listdir("instances")
        if file_name.endswith(".mat")
    ]

    for file_name in tqdm(files, desc="Processing files"):
        try:
            file_name, result = process_file(
                file_name, benchmark_instance, max_iter, debug
            )
            results[file_name] = result
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    os.makedirs("results", exist_ok=True)
    with open("results/" + filename, "w") as file:
        json.dump(results, file, indent=4, default=convert_to_native)

    if print_statistics:
        print_statistics(results)

    return results


if __name__ == "__main__":
    # benchmark(
    #     "results_neigh.json",
    #     benchmark_neighbourhood_instance,
    #     print_neighbourhood_benchmark_statistics,
    #     max_iter=100,
    #     debug=False,
    # )
    # benchmark(
    #     "results_start.json",
    #     benchmark_starting_solution_instance,
    #     print_starting_solution_benchmark_statistics,
    #     max_iter=100,
    #     debug=False,
    # )
    # benchmark(
    #     "results_grasp_constructive.json",
    #     benchmark_grasp_constructive,
    #     print_grasp_constructive_benchmark_statistics,
    #     max_iter=100,
    #     debug=False,
    # )
    benchmark(
        "results_visited_points.json",
        benchmark_visited_points,
        plot_permutations_with_pca_benchmark,
        max_iter=50,
        debug=False,
    )
    # benchmark(
    #     "results_score_evolution.json",
    #     benchmark_score_evolution,
    #     plot_score_evolution,
    #     max_iter=50,
    #     debug=False,
    # )
    # benchmark(
    #     "results_neigh_diversity.json",
    #     benchmark_neighbourhood_diversity,
    #     plot_pairwise_diversity_cdf,
    #     max_iter=50,
    #     debug=False,
    # )
    # benchmark(
    #     "results_execution_time.json",
    #     benchmark_execution_time,
    #     plot_execution_time_statistics,
    #     max_iter=50,
    # )
