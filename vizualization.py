import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


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


def plot_score_evolution(results):
    """
    Function to plot the evolution of the best score over iterations.

    Args:
        results (list of floats): List of best scores over iterations.
    """
    for key, values in results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(values, marker="o", color="b", label="Best Score")
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("Evolution of Best Score for {}".format(key))
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_pairwise_diversity_cdf(results, folder="default", filename="default"):
    """
    Function to plot the CDF of pairwise diversity over iterations.

    Args:
        results (dict of list of floats): Dict linking instance name to list of pairwise diversities over iterations.
    """
    all_values = set()
    for key, values in results.items():
        for value in values:
            all_values.add(value)
    plt.figure(figsize=(10, 6))
    sorted_values = np.sort(list(all_values))
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, cdf, linestyle="-", color="b", alpha=0.7)
    plt.xlabel("Pairwise Diversity")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel("CDF")
    plt.title("Pairwise Diversity CDF for {}".format(key))
    plt.grid(True)
    os.makedirs(f"plots/diversity/{folder}", exist_ok=True)
    plt.savefig(f"plots/diversity/{folder}/{filename}.pdf")
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
