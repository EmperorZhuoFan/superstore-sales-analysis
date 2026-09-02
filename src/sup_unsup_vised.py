import pandas as pd

from IPython.display import display

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score

from src.evaluation import (
    evaluate_model,
    compare_models
)


# ============================================================
# FEATURES AND TARGET
# ============================================================


def create_features_and_target(df):

    features = [
        "Sales",
        "Quantity",
        "Discount",
        "Category",
        "Sub-Category",
        "Segment",
        "Region",
        "Ship Mode",
        "Order Year",
        "Order Month",
        "Order Quarter",
        "Order Day",
        "Order Day of Week",
        "Sales Per Unit"
    ]

    # Profit is intentionally excluded.
    #
    # Profit Status is directly derived from Profit.
    # Including Profit as a feature would cause target leakage.

    X = df[features].copy()

    y = df["Profit Status"].copy()

    print(" --- Supervised Learning ---")

    print(
        "Feature matrix:",
        X.shape
    )

    print(
        "Target:",
        y.shape
    )

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================


def create_preprocessor():

    categorical_features = [
        "Category",
        "Sub-Category",
        "Segment",
        "Region",
        "Ship Mode"
    ]

    numerical_features = [
        "Sales",
        "Quantity",
        "Discount",
        "Order Year",
        "Order Month",
        "Order Quarter",
        "Order Day",
        "Order Day of Week",
        "Sales Per Unit"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================


def split_data(X, y):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# BASELINE
# ============================================================


def train_models(
    X_train,
    y_train,
    preprocessor
):

    baseline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent"
                )
            )
        ]
    )

    baseline.fit(
        X_train,
        y_train
    )

    return baseline


# ============================================================
# TRAIN ALL MODELS
# ============================================================


def train_all_models(
    X_train,
    y_train,
    preprocessor
):

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    logistic_model.fit(
        X_train,
        y_train
    )


    decision_tree = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=10,
                    random_state=42
                )
            )
        ]
    )

    decision_tree.fit(
        X_train,
        y_train
    )


    bagging_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                BaggingClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=6,
                        random_state=42
                    ),
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    bagging_model.fit(
        X_train,
        y_train
    )


    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    random_forest.fit(
        X_train,
        y_train
    )


    adaboost_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.5,
                    random_state=42
                )
            )
        ]
    )

    adaboost_model.fit(
        X_train,
        y_train
    )


    gradient_boosting = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )

    gradient_boosting.fit(
        X_train,
        y_train
    )


    models = {
        "Logistic Regression": logistic_model,
        "Decision Tree": decision_tree,
        "Bagging": bagging_model,
        "Random Forest": random_forest,
        "AdaBoost": adaboost_model,
        "Gradient Boosting": gradient_boosting
    }

    return models


# ============================================================
# EVALUATE ALL MODELS
# ============================================================


def evaluate_all_models(
    models,
    X_test,
    y_test
):

    results = []

    for model_name, model in models.items():

        results.append(
            evaluate_model(
                model,
                X_test,
                y_test,
                model_name
            )
        )

    model_results = compare_models(
        results
    )

    return model_results


# ============================================================
# CROSS-VALIDATE ALL MODELS
# ============================================================


def cross_validate_models(
    models,
    X_train,
    y_train
):

    cv_results = []

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        cv_results.append(
            {
                "Model": name,
                "Mean CV F1": scores.mean(),
                "Std CV F1": scores.std()
            }
        )

    cv_results = (
        pd.DataFrame(cv_results)
        .sort_values(
            "Mean CV F1",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "5-Fold Cross-Validation Results:"
    )

    display(
        cv_results.style.format({
            "Mean CV F1": "{:.3f}",
            "Std CV F1": "{:.3f}"
        })
    )

    return cv_results


# ============================================================
# TUNE DECISION TREE
# ============================================================


def tune_decision_tree(
    X_train,
    y_train,
    preprocessor
):

    tree_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]
    )

    tree_param_grid = {
        "model__max_depth": [
            3,
            5,
            7,
            10,
            None
        ],
        "model__min_samples_leaf": [
            1,
            5,
            10,
            20
        ],
        "model__criterion": [
            "gini",
            "entropy"
        ]
    }

    tree_grid = GridSearchCV(
        tree_pipeline,
        tree_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    tree_grid.fit(
        X_train,
        y_train
    )

    print(
        "Best Decision Tree parameters:"
    )

    print(
        tree_grid.best_params_
    )

    print(
        f"Best CV F1: "
        f"{tree_grid.best_score_:.4f}"
    )

    return tree_grid


# ============================================================
# TUNE RANDOM FOREST
# ============================================================


def tune_random_forest(
    X_train,
    y_train,
    preprocessor
):

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    rf_param_grid = {
        "model__n_estimators": [
            100,
            200,
            300
        ],
        "model__max_depth": [
            5,
            10,
            15,
            None
        ],
        "model__min_samples_leaf": [
            1,
            5,
            10
        ]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(
        X_train,
        y_train
    )

    print(
        "Best Random Forest parameters:"
    )

    print(
        rf_grid.best_params_
    )

    print(
        f"Best CV F1: "
        f"{rf_grid.best_score_:.4f}"
    )

    return rf_grid


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================


def create_final_model_comparison(
    models,
    tree_grid,
    rf_grid,
    X_test,
    y_test
):

    tuned_tree = (
        tree_grid.best_estimator_
    )

    tuned_random_forest = (
        rf_grid.best_estimator_
    )

    final_models = {
        **models,
        "Tuned Decision Tree": tuned_tree,
        "Tuned Random Forest": tuned_random_forest
    }

    final_results = []

    for model_name, model in final_models.items():

        final_results.append(
            evaluate_model(
                model,
                X_test,
                y_test,
                model_name
            )
        )

    final_model_results = (
        pd.DataFrame(final_results)
        .sort_values(
            "F1",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "Final Model Comparison:"
    )

    display(
        final_model_results.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        })
    )

    return (
        final_models,
        final_model_results
    )


# ============================================================
# FINAL CROSS-VALIDATION COMPARISON
# ============================================================


def create_final_cv_comparison(
    cv_results,
    tree_grid,
    rf_grid
):

    final_cv_results = cv_results.copy()

    tuned_results = pd.DataFrame([
        {
            "Model": "Tuned Decision Tree",
            "Mean CV F1": tree_grid.best_score_,
            "Std CV F1": None
        },
        {
            "Model": "Tuned Random Forest",
            "Mean CV F1": rf_grid.best_score_,
            "Std CV F1": None
        }
    ])

    final_cv_results = pd.concat(
        [
            final_cv_results,
            tuned_results
        ],
        ignore_index=True
    )

    final_cv_results = (
        final_cv_results
        .sort_values(
            "Mean CV F1",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "Final Cross-Validation Comparison:"
    )

    display(
        final_cv_results.style.format({
            "Mean CV F1": "{:.3f}",
            "Std CV F1": "{:.3f}"
        })
    )

    return final_cv_results


# ============================================================
# MAIN SUPERVISED LEARNING PIPELINE
# ============================================================


def run_supervised_learning(df):

    # --------------------------------------------------------
    # Create features and target
    # --------------------------------------------------------

    X, y = create_features_and_target(df)


    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = create_preprocessor()


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = train_models(
        X_train,
        y_train,
        preprocessor
    )

    baseline_pred = baseline.predict(
        X_test
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_pred
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Train original models
    # --------------------------------------------------------

    models = train_all_models(
        X_train,
        y_train,
        preprocessor
    )


    # --------------------------------------------------------
    # Test-set evaluation
    # --------------------------------------------------------

    model_results = evaluate_all_models(
        models,
        X_test,
        y_test
    )


    # --------------------------------------------------------
    # Cross-validation for ALL original models
    # --------------------------------------------------------

    cv_results = cross_validate_models(
        models,
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Keep Random Forest CV scores for compatibility
    # --------------------------------------------------------

    random_forest_cv_scores = cross_val_score(
        models["Random Forest"],
        X_train,
        y_train,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )


    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    tree_grid = tune_decision_tree(
        X_train,
        y_train,
        preprocessor
    )

    rf_grid = tune_random_forest(
        X_train,
        y_train,
        preprocessor
    )


    # --------------------------------------------------------
    # Final model comparison
    # --------------------------------------------------------

    (
        final_models,
        final_model_results
    ) = create_final_model_comparison(
        models,
        tree_grid,
        rf_grid,
        X_test,
        y_test
    )


    # --------------------------------------------------------
    # Final CV comparison
    # --------------------------------------------------------

    final_cv_results = (
        create_final_cv_comparison(
            cv_results,
            tree_grid,
            rf_grid
        )
    )


    # --------------------------------------------------------
    # Select final model
    #
    # Model selection is based on CV F1,
    # not the test set.
    # --------------------------------------------------------

    best_model_name = (
        final_cv_results.iloc[0]["Model"]
    )

    best_model = (
        final_models[best_model_name]
    )

    print(
        f"Final best model based on CV F1: "
        f"{best_model_name}"
    )


    # --------------------------------------------------------
    # Tuned Random Forest result
    # --------------------------------------------------------

    tuned_random_forest = (
        rf_grid.best_estimator_
    )

    tuned_rf_results = evaluate_model(
        tuned_random_forest,
        X_test,
        y_test,
        "Tuned Random Forest"
    )


    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {

        "X_test": X_test,

        "y_test": y_test,

        "models": models,

        "model_results": model_results,

        "cv_results": cv_results,

        "final_models": final_models,

        "final_model_results":
            final_model_results,

        "final_cv_results":
            final_cv_results,

        "best_model_name":
            best_model_name,

        "best_model":
            best_model,

        "tuned_random_forest":
            tuned_random_forest,

        "tuned_rf_results":
            tuned_rf_results,

        "tree_grid":
            tree_grid,

        "rf_grid":
            rf_grid,

        "cv_scores":
            random_forest_cv_scores
    }