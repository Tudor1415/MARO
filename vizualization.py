import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_permutations_with_pca(
    permutations_to_plot,
    objective_function,
    filename="default",
    folder="default",
):
    """
    Function to plot permutations in 2D space using PCA.
    The function:
      - Generates 100 random permutations (blue, alpha=0.5).
      - Projects the given permutations (red, alpha=1).
      - Highlights the first point in orange and the last in yellow.
      - Colors and sizes random permutations based on their objective function score.

    Args:
        permutations_to_plot (list of lists): List of permutations to plot in red.
        objective_function (function): Function to compute the score of a permutation.
    """
    # Parameters
    n = len(permutations_to_plot[0])

    # Generate 10,000 random permutations
    random_permutations = [np.random.permutation(n) for _ in range(100_000)]

    # Combine random permutations and the given permutations
    all_permutations = random_permutations + permutations_to_plot

    # Convert to NumPy array
    X = np.array(all_permutations)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    num_random = len(random_permutations)
    random_pca = X_pca[:num_random]
    specific_pca = X_pca[num_random:]

    # Compute scores for random permutations
    scores = np.array([objective_function(p) for p in random_permutations])
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    sizes = 50 + normalized_scores * 200

    # Plot the results
    plt.figure(figsize=(11, 9))

    # Plot random permutations with color and size based on score
    f = 150
    plt.scatter(
        random_pca[::f, 0],
        random_pca[::f, 1],
        c=plt.cm.Reds(normalized_scores[::f]),
        alpha=normalized_scores[::f],
        s=sizes[::f],
    )

    # Plot visited permutations in blue with size based on score
    plt.scatter(
        specific_pca[:, 0],
        specific_pca[:, 1],
        c="blue",
        alpha=1,
        s=50,
    )

    # Highlight the first point in orange and the last in yellow
    plt.scatter(
        specific_pca[0, 0],
        specific_pca[0, 1],
        c="orange",
        alpha=1,
        edgecolor="k",
        s=100,
    )
    plt.scatter(
        specific_pca[-1, 0],
        specific_pca[-1, 1],
        c="yellow",
        alpha=1,
        edgecolor="k",
        s=100,
    )

    # Save the plot without legend
    os.makedirs(f"plots/visited_permutations/{folder}", exist_ok=True)
    plt.savefig(f"plots/visited_permutations/{folder}/{filename}.pdf")
    plt.show()

    # Create a separate plot for the legend
    fig_legend = plt.figure(figsize=(12, 1))
    ax = fig_legend.add_subplot(111)
    ax.plot([], [], "o", color="red", label="Random Permutations")
    ax.plot([], [], "o", color="blue", label="Visited Permutations")
    ax.plot([], [], "o", color="orange", label="Starting point")
    ax.plot([], [], "o", color="yellow", label="End point")
    ax.legend(loc="center", ncol=4, fontsize=12)
    ax.axis("off")
    plt.savefig(f"plots/visited_permutations/{filename}_legend.pdf")
    plt.show()


def plot_score_evolution(results, folder="score_evolution", filename="ILS"):
    """
    Plots the evolution of the best score over iterations for each instance.

    For each key in the results dictionary (which maps instance names to a list of
    best scores over iterations), a separate plot is generated, saved to a PDF file,
    and displayed.

    Args:
        results (dict): Dictionary linking instance names to a list of best scores over iterations.
        folder (str): Subfolder (under "plots") where the figures will be saved.
    """
    # Create the folder if it does not exist.
    os.makedirs(f"plots/{folder}", exist_ok=True)

    for key, values in results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(values, marker="o", color="b", label="Best Score")
        plt.xlabel("Solution Swap", fontsize=15)
        plt.ylabel("Best Score", fontsize=15)
        plt.title(f"Evolution of Best Score for {filename}_{key}", fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save the plot to a PDF file
        save_path = f"plots/{folder}/{filename}_.pdf"
        plt.savefig(save_path)
        plt.show()


def plot_pairwise_diversity_cdf(
    results, folder="default", filename="default", title="default"
):
    """
    Function to plot the CDF of pairwise diversity over iterations.

    Args:
        results (dict of list of floats): Dict linking instance name to list of pairwise diversities over iterations.
    """
    all_pairwise_values = []
    all_start_end_values = []
    for key, values in results.items():
        pairwise_distances, start_end_distances = values
        all_start_end_values.append(start_end_distances)
        for value in pairwise_distances:
            all_pairwise_values.append(value)
    plt.figure(figsize=(10, 6))
    sorted_pairwise_values = np.sort(all_pairwise_values)
    cdf = np.arange(1, len(sorted_pairwise_values) + 1) / len(sorted_pairwise_values)
    plt.plot(sorted_pairwise_values, cdf, linestyle="-", color="b", alpha=0.7)
    plt.xlabel("Pairwise Diversity")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel("CDF")
    plt.title("Pairwise Diversity CDF for {}".format(title))
    plt.grid(True)
    os.makedirs(f"plots/diversity/{folder}", exist_ok=True)
    plt.savefig(f"plots/diversity/{folder}/{filename}.pdf")
    plt.show()

    plt.figure(figsize=(10, 6))
    sorted_start_end_values = np.sort(all_start_end_values)
    cdf = np.arange(1, len(sorted_start_end_values) + 1) / len(sorted_start_end_values)
    plt.plot(sorted_start_end_values, cdf, linestyle="-", color="orange", alpha=0.7)
    plt.xlabel("Start-End Distance")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel("CDF")
    plt.title("Start-End Diversity CDF for {}".format(title))
    plt.grid(True)
    os.makedirs(f"plots/start_end_distance/{folder}", exist_ok=True)
    plt.savefig(f"plots/start_end_distance/{folder}/{filename}.pdf")
    plt.show()


def plot_execution_time_statistics(results):
    """
    Function to plot the execution time statistics.

    Args:
        results (dict of list of floats): Dict linking instance name to list of execution times starting with the matrix size.
    """
    size_time_ni, size_time_ns = {}, {}
    for key, values in results.items():
        size_time_ni.setdefault(values[0], []).append(values[1][0])
        size_time_ns.setdefault(values[0], []).append(values[1][1])

    sizes = sorted(size_time_ni.keys())
    ni_means = [np.mean(size_time_ni[size]) for size in sizes]
    ns_means = [np.mean(size_time_ns[size]) for size in sizes]
    ni_stds = [np.std(size_time_ni[size]) for size in sizes]
    ns_stds = [np.std(size_time_ns[size]) for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sizes,
        ni_means,
        yerr=ni_stds,
        marker="o",
        color="b",
        label="NI Execution Time",
        capsize=5,
    )
    plt.errorbar(
        sizes,
        ns_means,
        yerr=ns_stds,
        marker="o",
        color="r",
        label="NS Execution Time",
        capsize=5,
    )
    plt.xlabel("Matrix Size", fontsize=25)
    plt.ylabel("Execution Time (s)", fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Execution Time", fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=22)

    # Create directory if it doesn't exist
    os.makedirs("plots/exec_times", exist_ok=True)
    plt.savefig("plots/exec_times/execution_time_statistics.pdf")
    plt.show()


def extract_optimal(filename):
    result = {}
    value = None
    linear_ordering = None

    with open(filename, "r") as f:
        for line in f:
            # Check for the line with "Value" (but not "Value + Diagonals")
            if line.lstrip().startswith("Value") and "Value +" not in line:
                # Split on ':' and convert the second part to an integer.
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        value = int(parts[1].strip())
                    except ValueError:
                        pass  # Handle conversion errors gracefully if needed.
            elif line.lstrip().startswith("Linear ordering"):
                parts = line.split(":")
                if len(parts) >= 2:
                    # Convert each number into an integer.
                    linear_ordering = [int(num) for num in parts[1].split()]

    # Create the dictionary with filename as key and a tuple (value, linear_ordering) as the item.
    result[filename] = (value, linear_ordering)
    return result


def extract_all_optimal(directory):
    """Iterates over all files in the specified directory and applies extract_optimal."""
    all_results = {}
    # Loop over every file in the directory.
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Only process if it's a file.
        if os.path.isfile(file_path):
            result = extract_optimal(file_path)
            all_results.update(result)
    return all_results


def get_base_name(filepath):
    """
    Extracts the base filename without its directory and extension.
    For example, 'optimal/t65i11xx.opt' becomes 't65i11xx'.
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def plot_starting_solution_benchmark_statistics(results):
    """
    Plots a bar plot of the relative improvement of the optimal solution with respect to
    each method (random and becker). Only keys present in both 'results' and in the optimal
    dictionary are considered.

    Relative improvement is defined as:

        improvement = ((method_score - optimal_score) / method_score) * 100

    (assuming a minimization problem).

    Additionally, prints the random and becker improvements for each instance and
    adds extra vertical space between the title and the top of the plot.
    """
    # Get the optimal results from the "optimal" directory.
    results_optimal = extract_all_optimal("optimal")
    results_keys = {get_base_name(key): key for key in results.keys()}
    results_optimal_keys = {get_base_name(key): key for key in results_optimal.keys()}

    # Determine common base names
    common_bases = sorted(
        set(results_keys.keys()).intersection(results_optimal_keys.keys())
    )

    # Build lists only for instances where we can compute an improvement.
    final_common_bases = []
    improvements_random = []
    improvements_becker = []

    for key in common_bases:
        # Build the full keys for accessing the dictionaries.
        opt_key = "optimal/" + key + ".opt"
        inst_key = "instances/" + key + ".mat"

        # Retrieve scores.
        optimal_score, _ = results_optimal[opt_key]
        random_score, becker_score = results[inst_key]

        # Check for an invalid optimal score.
        if optimal_score is None or optimal_score == 0:
            print(f"Optimal score is zero for {key}. Skipping this instance.")
            continue

        # Calculate relative improvement (only if the method score is non-zero).
        if random_score:
            imp_random = (random_score - optimal_score) / random_score * 100
        else:
            imp_random = 0

        if becker_score:
            imp_becker = (becker_score - optimal_score) / becker_score * 100
        else:
            imp_becker = 0

        improvements_random.append(imp_random)
        improvements_becker.append(imp_becker)
        final_common_bases.append(key)

    # Print improvements for each instance.
    for base, imp_r, imp_b in zip(
        final_common_bases, improvements_random, improvements_becker
    ):
        print(
            f"Instance {base}: Random improvement = {imp_r:.1f}%, Becker improvement = {imp_b:.1f}%"
        )

    # Create a grouped bar plot.
    x = np.arange(len(final_common_bases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_random = ax.bar(x - width / 2, improvements_random, width, label="Random")
    bars_becker = ax.bar(x + width / 2, improvements_becker, width, label="Becker")

    ax.set_ylabel("Relative Improvement (%)")
    ax.set_title("Relative Improvement of Optimal Solutions", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(final_common_bases, rotation=45, ha="right")
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # Offset above the bar
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(bars_random)
    autolabel(bars_becker)

    plt.tight_layout()
    os.makedirs("plots/start_instance", exist_ok=True)
    plt.savefig("plots/start_instance/relative_improvement.pdf")
    plt.show()
