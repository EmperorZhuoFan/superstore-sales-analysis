import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def create_customer_features(df):
    customer_features = (
        df.groupby("Customer ID")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Number_of_Orders=("Order ID", "nunique"),
            Average_Discount=("Discount", "mean")
        )
        .reset_index()
    )

    customer_features["Average_Order_Value"] = (
        customer_features["Total_Sales"] /
        customer_features["Number_of_Orders"]
    )

    display(customer_features.head())

    return customer_features


def prepare_clustering_data(customer_features):
    clustering_features = [
        "Total_Sales",
        "Total_Profit",
        "Total_Quantity",
        "Number_of_Orders",
        "Average_Discount",
        "Average_Order_Value"
    ]

    X_customer = customer_features[clustering_features].copy()

    scaler = StandardScaler()
    X_customer_scaled = scaler.fit_transform(X_customer)

    return clustering_features, X_customer, X_customer_scaled, scaler


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

        labels = kmeans.fit_predict(X_customer_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(
            silhouette_score(X_customer_scaled, labels)
        )

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("Elbow Method — Choosing the Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), silhouette_scores, marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

    return list(k_values), inertias, silhouette_scores


def run_kmeans(customer_features, clustering_features, X_customer_scaled, optimal_k=4):
    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    customer_features["KMeans_Cluster"] = kmeans.fit_predict(
        X_customer_scaled
    )

    kmeans_silhouette = silhouette_score(
        X_customer_scaled,
        customer_features["KMeans_Cluster"]
    )

    print(f"K-Means clusters: {optimal_k}")
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")

    return kmeans, kmeans_silhouette


def run_hierarchical(customer_features, X_customer_scaled, optimal_k=4):
    hierarchical = AgglomerativeClustering(
        n_clusters=optimal_k
    )

    customer_features["Hierarchical_Cluster"] = (
        hierarchical.fit_predict(X_customer_scaled)
    )

    hierarchical_silhouette = silhouette_score(
        X_customer_scaled,
        customer_features["Hierarchical_Cluster"]
    )

    print(
        f"Hierarchical Clustering Silhouette Score: "
        f"{hierarchical_silhouette:.3f}"
    )

    return hierarchical, hierarchical_silhouette


def create_customer_profiles(customer_features, clustering_features):
    kmeans_profile = (
        customer_features
        .groupby("KMeans_Cluster")[clustering_features]
        .mean()
        .round(2)
    )

    display(kmeans_profile)

    cluster_sizes = (
        customer_features["KMeans_Cluster"]
        .value_counts()
        .sort_index()
        .rename("Customer Count")
    )

    display(cluster_sizes)

    return kmeans_profile, cluster_sizes


def plot_customer_segments(customer_features):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=customer_features,
        x="Total_Sales",
        y="Total_Profit",
        hue="KMeans_Cluster",
        s=80,
        alpha=0.8
    )

    plt.title("Customer Segments by Sales and Profit")
    plt.xlabel("Total Sales")
    plt.ylabel("Total Profit")
    plt.show()


def run_clustering(df, optimal_k=4):
    customer_features = create_customer_features(df)

    (
        clustering_features,
        X_customer,
        X_customer_scaled,
        scaler
    ) = prepare_clustering_data(customer_features)

    k_values, inertias, silhouette_scores = find_optimal_k(
        X_customer_scaled
    )

    kmeans, kmeans_silhouette = run_kmeans(
        customer_features,
        clustering_features,
        X_customer_scaled,
        optimal_k
    )

    hierarchical, hierarchical_silhouette = run_hierarchical(
        customer_features,
        X_customer_scaled,
        optimal_k
    )

    kmeans_profile, cluster_sizes = create_customer_profiles(
        customer_features,
        clustering_features
    )

    plot_customer_segments(customer_features)

    return {
        "customer_features": customer_features,
        "clustering_features": clustering_features,
        "X_customer": X_customer,
        "X_customer_scaled": X_customer_scaled,
        "scaler": scaler,
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "kmeans": kmeans,
        "kmeans_silhouette": kmeans_silhouette,
        "hierarchical": hierarchical,
        "hierarchical_silhouette": hierarchical_silhouette,
        "kmeans_profile": kmeans_profile,
        "cluster_sizes": cluster_sizes
    }
