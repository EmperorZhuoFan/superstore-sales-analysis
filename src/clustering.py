import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ============================================================
# FIND OPTIMAL K
# ============================================================

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

    plt.figure(figsize=(10, 6))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.title(
        "Elbow Method — Choosing the Number of Clusters"
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o"
    )

    plt.title(
        "Silhouette Score by Number of Clusters"
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

    best_silhouette_index = np.argmax(
        silhouette_scores
    )

    best_silhouette_k = (
        list(k_values)[best_silhouette_index]
    )

    print(
        f"Highest silhouette score occurs at "
        f"K={best_silhouette_k}"
    )

    print(
        "The final K should be selected by considering "
        "both the Elbow Method and Silhouette Score, "
        "along with cluster interpretability."
    )

    return (
        list(k_values),
        inertias,
        silhouette_scores
    )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

def run_clustering(df, optimal_k=4):

    customer_data = (
        df.groupby("Customer ID")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Total_Profit=("Profit", "sum"),
            Average_Discount=("Discount", "mean"),
            Order_Count=("Order ID", "nunique")
        )
        .reset_index()
    )

    features = [
        "Total_Sales",
        "Total_Quantity",
        "Total_Profit",
        "Average_Discount",
        "Order_Count"
    ]

    X_customer = customer_data[features]

    scaler = StandardScaler()

    X_customer_scaled = scaler.fit_transform(
        X_customer
    )

    print("\n--- CUSTOMER CLUSTERING ---")

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(
        X_customer_scaled
    )

    kmeans_silhouette = silhouette_score(
        X_customer_scaled,
        kmeans_labels
    )

    hierarchical = AgglomerativeClustering(
        n_clusters=optimal_k
    )

    hierarchical_labels = hierarchical.fit_predict(
        X_customer_scaled
    )

    hierarchical_silhouette = silhouette_score(
        X_customer_scaled,
        hierarchical_labels
    )

    customer_data["KMeans Cluster"] = (
        kmeans_labels
    )

    customer_data["Hierarchical Cluster"] = (
        hierarchical_labels
    )

    cluster_sizes = (
        customer_data["KMeans Cluster"]
        .value_counts()
        .sort_index()
    )

    print(
        f"K-Means Silhouette Score: "
        f"{kmeans_silhouette:.3f}"
    )

    print(
        f"Hierarchical Silhouette Score: "
        f"{hierarchical_silhouette:.3f}"
    )

    print("\nCustomer Segment Sizes:")

    display(cluster_sizes)

    return {
        "customer_data": customer_data,
        "kmeans_silhouette": kmeans_silhouette,
        "hierarchical_silhouette": hierarchical_silhouette,
        "cluster_sizes": cluster_sizes,
        "kmeans_model": kmeans,
        "hierarchical_labels": hierarchical_labels
    }