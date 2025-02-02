import timeit
from ILS import *
from scipy import stats
from vizualization import *
from GRASP import construct_grasp


def benchmark_neighbourhood_instance(matrix, search_function, debug=False):
    """
    Runs the ILS algorithm and benchmarks its performance.

    Parameters:
    - matrix: 2D numpy array representing the cost matrix (n x n).
    - search_function: The search function to use.
    - debug: Boolean flag to enable/disable debugging statements.
    """

    _, best_value_NI = search_function(matrix, visit_NI)
    _, best_value_NS = search_function(matrix, visit_NS)

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
    matrix, search_function, nb_repetitions=10, debug=False
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
        search_function(matrix, lambda x: random_permutation)
        _, best_value_random = search_function(matrix, lambda x: random_permutation)
        random_score_values.append(best_value_random)

    mean_random_score = np.mean(random_score_values)

    # Becker's constructive heuristic
    _, best_value_becker = search_function(matrix, becker_constructive_algorithm)

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


def benchmark_visited_points(matrix, search_function, debug=False):
    """
    Benchmarks the number of visited points using the Iterated Local Search (ILS) algorithm.
    Args:
        matrix (list of list of int): The input matrix representing the problem instance.
        search_function (callable): The search function to use.
        debug (bool, optional): If True, enables debug mode for additional output. Defaults to False.
    Returns:
        list: A list of visited points during the ILS algorithm execution.
    """
    _, best_value, visited = search_function(matrix)

    return visited


def plot_permutations_with_pca_benchmark(results, folder="default"):
    """
    Plots the visited points during the ILS algorithm execution.
    Args:
        results (dict): A dict of filename and visited points.
    """
    for key, visited_points in results.items():
        matrix = read_square_matrix_from_file(key, False)["matrix"]
        obj_func = lambda x: objective_function(matrix, x)
        filename = key.split("/")[-1].split(".")[0]
        plot_permutations_with_pca(
            visited_points, obj_func, folder=folder, filename=filename
        )


def benchmark_score_evolution(matrix, search_function, debug=False):
    """
    Benchmarks the score evolution using the Iterated Local Search (ILS) algorithm.
    Args:
        matrix (list of list of int): The input matrix representing the problem instance.
        search_function (callable): The search function to use.
        debug (bool, optional): If True, enables debug mode for additional output. Defaults to False.
    Returns:
        list: A list of visited points during the ILS algorithm execution.
    """
    _, best_value, visited = search_function(matrix)

    if debug:
        print(f"Best value = {best_value}")
        print(f"Visited points = {visited}")

    return [objective_function(matrix, permutation) for permutation in visited]


def benchmark_neighbourhood_diversity(matrix, search_function, debug=False):
    """
    Runs the ILS algorithm and computes the pairwise spearman r distance between the permutations.
    Also computes the spearman r distance between the starting solution and the best solution found.
    Parameters:
        matrix (np.array): The cost matrix.
        search_function (callable): The search function to use.
        debug (bool): Flag to enable/disable debugging. Default is False.
    Returns:
        cummulative distribution of the pairwise spearman r distance.
    """
    _, _, visited = search_function(matrix)
    if debug:
        print(f"Visited points: {visited}")

    n = len(visited)
    pairwise_dist = np.zeros(n * (n - 1) // 2)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_dist[idx] = stats.spearmanr(visited[i], visited[j])[0]
            idx += 1

    return pairwise_dist, stats.spearmanr(visited[0], visited[-1])[0]


def benchmark_execution_time(matrix, search_functions, debug=False):
    """
    Runs the ILS algorithm and benchmarks its performance in terms of computational time.
    Arguments:
        matrix (np.array): The cost matrix.
        max_iter (int): The maximum number of iterations for the ILS algorithm.
        debug (bool): Flag to enable/disable debugging. Default is False.
    """
    start_times, end_times = [], []
    for search_function in search_functions:
        start_time = timeit.default_timer()
        _, best_value = search_function
        end_time = timeit.default_timer()
        start_times.append(start_time)
        end_times.append(end_time)

    execution_times = np.array(end_times) - np.array(start_times)

    if debug:
        for i, search_function in enumerate(search_functions):
            print(f"Execution time: {execution_times[i]:.2f} seconds")

    return len(matrix), execution_times


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


def process_file(file_name, search_function, benchmark_instance, debug):
    matrix = read_square_matrix_from_file(file_name, debug)["matrix"]
    results = benchmark_instance(matrix, search_function, debug)
    return file_name, results


def benchmark(
    filename,
    search_function,
    benchmark_instance,
    print_statistics,
    debug=False,
):
    """
    Reads all the files in the current directory that have the .mat extension and runs the benchmark_instance function on them.

    Parameters:
        - filename (str): The name of the file to print the results to.
        - search_function (callable): The search function to use.
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
                file_name, search_function, benchmark_instance, debug
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

    ILS_10 = lambda matrix, neigh: ILS(
        matrix,
        objective_function,
        becker_constructive_algorithm,
        perturb_random,
        neigh,
        10,
        log_visits=False,
        debug=False,
    )
    benchmark(
        "results_neigh.json",
        ILS_10,
        benchmark_neighbourhood_instance,
        print_neighbourhood_benchmark_statistics,
        debug=False,
    )
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
    # benchmark(
    #     "results_visited_points.json",
    #     benchmark_visited_points,
    #     lambda x: plot_permutations_with_pca_benchmark(x, "ILS"),
    #     max_iter=10,
    #     debug=False,
    # )
    # benchmark(
    #     "results_score_evolution.json",
    #     benchmark_score_evolution,
    #     plot_score_evolution,
    #     max_iter=50,
    #     debug=False,
    # )

    # ILS_NS_10_40 = lambda x: ILS(
    #     x,
    #     objective_function,
    #     becker_constructive_algorithm,
    #     perturb_random,
    #     visit_NS,
    #     10,
    #     log_visits=True,
    #     debug=False,
    # )
    # benchmark(
    #     "results_neigh_diversity.json",
    #     ILS_NS_10_40,
    #     benchmark_neighbourhood_diversity,
    #     plot_pairwise_diversity_cdf,
    #     debug=False,
    # )

    # ILS_NI_10 = lambda x: ILS(
    #     x,
    #     objective_function,
    #     becker_constructive_algorithm,
    #     perturb_random,
    #     visit_NI,
    #     10,
    #     log_visits=True,
    #     debug=False,
    # )
    # benchmark(
    #     "results_neigh_diversity.json",
    #     ILS_NI_10,
    #     benchmark_neighbourhood_diversity,
    #     lambda x: plot_pairwise_diversity_cdf(
    #         x, folder="ILS", filename="NI", title="ILS N_I 10 iterations"
    #     ),
    #     debug=False,
    # )
    # benchmark(
    #     "results_execution_time.json",
    #     benchmark_execution_time,
    #     plot_execution_time_statistics,
    #     max_iter=50,
    # )

    # _, best_value_NS = ILS(
    #     matrix,
    #     objective_function,
    #     becker_constructive_algorithm,
    #     perturb_random,
    #     visit_NS,
    #     max_iter,
    #     False,
    #     debug,
    # )
