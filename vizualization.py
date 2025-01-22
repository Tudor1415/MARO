import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_permutations_with_pca(permutations_to_plot, objective_function):
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
    random_permutations = [np.random.permutation(n) for _ in range(10_000)]

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
    sizes = 50 + normalized_scores * 200  # Scale sizes for better visualization

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot random permutations with color and size based on score
    f = 100
    plt.scatter(
        random_pca[::f, 0],
        random_pca[::f, 1],
        c=plt.cm.Reds(normalized_scores[::f]),
        alpha=normalized_scores[::f],
        s=sizes[::f],
        label="Random Permutations",
    )

    # Plot specific permutations in red
    plt.scatter(
        specific_pca[:, 0],
        specific_pca[:, 1],
        c="blue",
        alpha=1,
        label="Provided Permutations",
    )

    # Highlight the first point in orange and the last in yellow
    plt.scatter(
        specific_pca[0, 0],
        specific_pca[0, 1],
        c="orange",
        alpha=1,
        label="First Provided Permutation",
        edgecolor="k",
        s=100,
    )
    plt.scatter(
        specific_pca[-1, 0],
        specific_pca[-1, 1],
        c="yellow",
        alpha=1,
        label="Last Provided Permutation",
        edgecolor="k",
        s=100,
    )

    # Add labels and legend
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA of Permutations with Score-based Color and Size Mapping")
    plt.grid(True)
    plt.legend()
    plt.show()
