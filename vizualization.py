import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_permutations_with_pca(permutations_to_plot):
    """
    Function to plot permutations in 2D space using PCA.
    The function:
      - Generates 100 random permutations (blue, alpha=0.5).
      - Projects the given permutations (red, alpha=1).

    Args:
        permutations_to_plot (list of lists): List of permutations to plot in red.
    """
    # Parameters
    n = len(permutations_to_plot[0])  # Length of each permutation

    # Generate 100 random permutations
    random_permutations = [np.random.permutation(n) + 1 for _ in range(100)]

    # Combine random permutations and the given permutations
    all_permutations = random_permutations + permutations_to_plot

    # Convert to NumPy array
    X = np.array(all_permutations)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Identify indices for plotting
    num_random = len(random_permutations)
    random_pca = X_pca[:num_random]
    specific_pca = X_pca[num_random:]

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot random permutations in blue
    plt.scatter(
        random_pca[:, 0],
        random_pca[:, 1],
        c="blue",
        alpha=0.5,
        label="Random Permutations",
    )

    # Plot specific permutations in red
    plt.scatter(
        specific_pca[:, 0],
        specific_pca[:, 1],
        c="red",
        alpha=1,
        label="Provided Permutations",
    )

    # Add labels and legend
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA of Permutations with Highlighted Specific Permutations")
    plt.grid(True)
    plt.legend()
    plt.show()
