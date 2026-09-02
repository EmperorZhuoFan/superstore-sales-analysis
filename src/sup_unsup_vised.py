import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

from src.evaluation import evaluate_model, compare_models


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

    X = df[features].copy()
    y = df["Profit Status"].copy()

    print(" --- Supervised Learning ---")
    print("Feature matrix:", X.shape)
    print("Target:", y.shape)

    return X, y


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
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train, preprocessor):
    baseline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])

    baseline.fit(X_train, y_train)
    return baseline


def train_all_models(X_train, y_train, preprocessor):
    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    logistic_model.fit(X_train, y_train)

    decision_tree = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=10,
                random_state=42
            )
        )
    ])
    decision_tree.fit(X_train, y_train)

    bagging_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
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
    ])
    bagging_model.fit(X_train, y_train)

    random_forest = Pipeline(steps=[
        ("preprocessor", preprocessor),
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
    ])
    random_forest.fit(X_train, y_train)

    adaboost_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=42
            )
        )
    ])
    adaboost_model.fit(X_train, y_train)

    gradient_boosting = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ])
    gradient_boosting.fit(X_train, y_train)

    models = {
        "Logistic Regression": logistic_model,
        "Decision Tree": decision_tree,
        "Bagging": bagging_model,
        "Random Forest": random_forest,
        "AdaBoost": adaboost_model,
        "Gradient Boosting": gradient_boosting
    }

    return models


def evaluate_all_models(models, X_test, y_test):
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

    model_results = compare_models(results)
    return model_results


def cross_validate_random_forest(random_forest, X_train, y_train):
    cv_scores = cross_val_score(
        random_forest,
        X_train,
        y_train,
        cv=5,
        scoring="f1"
    )

    print("Random Forest 5-Fold CV F1 Scores:")
    print(cv_scores)
    print(f"Mean CV F1: {cv_scores.mean():.4f}")

    return cv_scores


def tune_decision_tree(X_train, y_train, preprocessor):
    tree_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    tree_param_grid = {
        "model__max_depth": [3, 5, 7, 10, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__criterion": ["gini", "entropy"]
    }

    tree_grid = GridSearchCV(
        tree_pipeline,
        tree_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    tree_grid.fit(X_train, y_train)

    print("Best Decision Tree parameters:")
    print(tree_grid.best_params_)
    print(f"Best CV F1: {tree_grid.best_score_:.4f}")

    return tree_grid


def tune_random_forest(X_train, y_train, preprocessor):
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    rf_param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_leaf": [1, 5, 10]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    print("Best Random Forest parameters:")
    print(rf_grid.best_params_)
    print(f"Best CV F1: {rf_grid.best_score_:.4f}")

    return rf_grid


def run_supervised_learning(df):
    X, y = create_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = create_preprocessor()

    baseline = train_models(X_train, y_train, preprocessor)
    baseline_pred = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print("=" * 70)

    models = train_all_models(X_train, y_train, preprocessor)
    model_results = evaluate_all_models(models, X_test, y_test)

    cv_scores = cross_validate_random_forest(
        models["Random Forest"],
        X_train,
        y_train
    )

    best_model_name = model_results.iloc[0]["Model"]
    best_model = models[best_model_name]

    print(f"Best model based on F1: {best_model_name}")

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

    tuned_random_forest = rf_grid.best_estimator_

    tuned_rf_results = evaluate_model(
        tuned_random_forest,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    display(pd.DataFrame([tuned_rf_results]))

    return {
        "X_test": X_test,
        "y_test": y_test,
        "models": models,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "tuned_random_forest": tuned_random_forest,
        "tuned_rf_results": tuned_rf_results,
        "tree_grid": tree_grid,
        "rf_grid": rf_grid,
        "cv_scores": cv_scores
    }
