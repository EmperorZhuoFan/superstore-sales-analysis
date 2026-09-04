from src.data_preparation import (
    load_data,
    inspect_data,
    prepare_data
)

from src.eda import (
    setup_visualization,
    run_eda
)

from src.sup_unsup_vised import (
    run_supervised_learning
)

from src.evaluation import (
    show_confusion_matrix,
    show_classification_report,
    show_feature_importance
)

from src.clustering import (
    run_clustering
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    setup_visualization()

    print("=" * 70)
    print("SUPERSTORE SALES ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    file_path = "samplesuperstore.csv"

    df, df_original = load_data(
        file_path
    )

    # --------------------------------------------------------
    # DATA UNDERSTANDING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATA UNDERSTANDING")
    print("=" * 70)

    df = inspect_data(df)

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    df = prepare_data(df)

    # --------------------------------------------------------
    # EXPLORATORY DATA ANALYSIS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    profit_summary = run_eda(df)

    # --------------------------------------------------------
    # SUPERVISED LEARNING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SUPERVISED LEARNING")
    print("=" * 70)

    supervised_results = (
        run_supervised_learning(df)
    )

    # --------------------------------------------------------
    # BEST MODEL EVALUATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BEST MODEL EVALUATION")
    print("=" * 70)

    X_test = supervised_results["X_test"]
    y_test = supervised_results["y_test"]

    best_model = (
        supervised_results["best_model"]
    )

    best_model_name = (
        supervised_results["best_model_name"]
    )

    print(
        f"\nBest Model: {best_model_name}"
    )

    best_predictions = show_confusion_matrix(
        best_model,
        X_test,
        y_test
    )

    print(
        "\nClassification Report:"
    )

    show_classification_report(
        y_test,
        best_predictions
    )

    print(
        "\nFeature Importance:"
    )

    importance = show_feature_importance(
        best_model,
        best_model_name
    )

    # --------------------------------------------------------
    # UNSUPERVISED LEARNING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "UNSUPERVISED LEARNING — "
        "CUSTOMER SEGMENTATION"
    )
    print("=" * 70)

    clustering_results = run_clustering(
        df,
        optimal_k=4
    )

    # --------------------------------------------------------
    # FINAL PROJECT SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL PROJECT SUMMARY")
    print("=" * 70)

    print(
        f"\nOriginal Dataset Shape: "
        f"{df_original.shape}"
    )

    print(
        f"Prepared Dataset Shape: "
        f"{df.shape}"
    )

    print(
        f"\nBest Supervised Model: "
        f"{best_model_name}"
    )

    print(
        f"Tuned Random Forest F1: "
        f"{supervised_results['tuned_rf_results']['F1']:.4f}"
    )

    print(
        f"Random Forest Mean CV F1: "
        f"{supervised_results['cv_scores'].mean():.4f}"
    )

    print(
        f"\nK-Means Silhouette Score: "
        f"{clustering_results['kmeans_silhouette']:.3f}"
    )

    print(
        f"Hierarchical Silhouette Score: "
        f"{clustering_results['hierarchical_silhouette']:.3f}"
    )

    print(
        f"\nNumber of Customer Segments: "
        f"{len(clustering_results['cluster_sizes'])}"
    )

    print(
        "\nCustomer Segment Sizes:"
    )

    print(
        clustering_results["cluster_sizes"]
    )

    print("\n")
    print("=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# RUN PROJECT
# ============================================================

if __name__ == "__main__":
    main()