import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def find_optimal_k(X_customer_scaled):

    inertias = []
    silhouette_scores = []
    k_values = range(2, 9)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(
            X_customer_scaled
        )

        inertias.append(
            kmeans.inertia_
        )

        silhouette_scores.append(
            silhouette_score(
                X_customer_scaled,
                labels
            )
        )

    # --------------------------------------------------------
    # Elbow Method
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.title(
        "Elbow Method — Choosing the Number of Clusters"
    )

    plt.xlabel(
        "Number of Clusters"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.show()


    # --------------------------------------------------------
    # Silhouette Score
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o"
    )

    plt.title(
        "Silhouette Score by Number of Clusters"
    )

    plt.xlabel(
        "Number of Clusters"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.show()


    # --------------------------------------------------------
    # Best silhouette score
    # --------------------------------------------------------

    best_silhouette_index = np.argmax(
        silhouette_scores
    )

    best_silhouette_k = list(
        k_values
    )[best_silhouette_index]

    print(
        f"Highest silhouette score occurs at "
        f"K={best_silhouette_k}"
    )

    print(
        "The final K should be selected by "
        "considering both the Elbow Method and "
        "Silhouette Score, along with cluster "
        "interpretability."
    )


    return (
        list(k_values),
        inertias,
        silhouette_scores
    )