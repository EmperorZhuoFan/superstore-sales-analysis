import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }


def compare_models(results):

    model_results = pd.DataFrame(results)

    model_results = (
        model_results
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    print("\nModel Comparison:")

    display(model_results.round(3))

    return model_results


# ============================================================
# CONFUSION MATRIX
# ============================================================

def show_confusion_matrix(best_model, X_test, y_test):

    best_predictions = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Unprofitable",
            "Profitable"
        ],
        yticklabels=[
            "Unprofitable",
            "Profitable"
        ]
    )

    plt.title("Confusion Matrix — Final Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return best_predictions


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def show_classification_report(y_test, best_predictions):

    print(
        classification_report(
            y_test,
            best_predictions,
            target_names=[
                "Unprofitable",
                "Profitable"
            ]
        )
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(best_model, best_model_name):

    tree_models = [
        "Decision Tree",
        "Bagging",
        "Random Forest",
        "AdaBoost",
        "Gradient Boosting",
        "Tuned Decision Tree",
        "Tuned Random Forest"
    ]

    if best_model_name in tree_models:

        best_preprocessor = (
            best_model
            .named_steps["preprocessor"]
        )

        best_estimator = (
            best_model
            .named_steps["model"]
        )

        feature_names = (
            best_preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            best_estimator,
            "feature_importances_"
        ):

            importance = (
                pd.Series(
                    best_estimator.feature_importances_,
                    index=feature_names
                )
                .sort_values(ascending=False)
            )

            top_features = importance.head(15)

            plt.figure(figsize=(10, 7))

            top_features.sort_values().plot(
                kind="barh"
            )

            plt.title(
                "Most Important Features — Final Best Model"
            )
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")

            plt.show()

            display(
                top_features.to_frame(
                    "Importance"
                )
            )

            return importance

    return None


# ============================================================
# ERROR ANALYSIS
# ============================================================

def analyze_errors(best_model, X_test, y_test):

    predictions = best_model.predict(X_test)

    error_analysis = X_test.copy()

    error_analysis["Actual"] = y_test.values
    error_analysis["Predicted"] = predictions
    error_analysis["Error"] = (
        error_analysis["Actual"]
        != error_analysis["Predicted"]
    )

    misclassified = (
        error_analysis[
            error_analysis["Error"]
        ]
        .copy()
    )

    print(
        f"Misclassified transactions: "
        f"{len(misclassified)}"
    )

    print("\nSample of misclassified transactions:")

    display(
        misclassified.head()
    )

    print(
        "\nMisclassified transactions by Category:"
    )

    display(
        misclassified[
            "Category"
        ].value_counts()
    )

    print(
        "\nMisclassified transactions by Region:"
    )

    display(
        misclassified[
            "Region"
        ].value_counts()
    )

    print(
        "\nMisclassified transactions by Discount:"
    )

    display(
        misclassified[
            "Discount"
        ].describe()
    )

    return misclassified