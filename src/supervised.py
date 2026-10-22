import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score
from src.mlflow_tracking import track_model_run
from src.mlflow_tracking import log_artifact
from src.evaluation import evaluate_model, compare_models

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

    X = df[features]
    y = df["Profit Status"]

    return X, y


# ============================================================
# PREPROCESSING
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

    numerical_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ( "encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


# ============================================================
# BASELINE MODEL
# ============================================================

def train_baseline(X_train, X_test, y_train, y_test):

    baseline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", DummyClassifier(strategy="most_frequent"))
            ])

    baseline.fit(X_train, y_train)

    predictions = baseline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Baseline Accuracy: {accuracy:.3f}")

    return baseline


# ============================================================
# TRAIN ALL MODELS
# ============================================================
def train_all_models(X_train, y_train):

    preprocessor = create_preprocessor()

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=10, random_state=42
            ))
        ]),

        "Bagging": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
                n_estimators=100, random_state=42, n_jobs=-1
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ))
        ]),

        "AdaBoost": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=42
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=3, random_state=42
            ))
        ])
    }

    trained_models = {}

    for name, model in models.items():

        estimator = model.named_steps["model"]

        parameters = {
            key: value
            for key, value in estimator.get_params(deep=False).items()
            if isinstance(value, (str, int, float, bool))
            or value is None
        }

        print(f"Training {name}...")

        with track_model_run(name, parameters=parameters):
            model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models

# ============================================================
# MODEL EVALUATION
# ============================================================
def evaluate_all_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

        with track_model_run(
            f"{name} Test Evaluation",
            tags={"evaluation_type": "test"}
        ):
            from src.mlflow_tracking import log_metrics

            log_metrics({
                key: value
                for key, value in result.items()
                if key != "Model"
            })

    comparison = compare_models(results)

    artifact_path = "model_comparison.csv"
    comparison.to_csv(artifact_path, index=False)

    with track_model_run("Model Comparison", tags={"artifact_type": "comparison"}):
        log_artifact(artifact_path)

    return comparison

# ============================================================
# CROSS-VALIDATION
# ============================================================
def cross_validate_models(models, X_train, y_train):

    cv_results = []

    for name, model in models.items():

        with track_model_run(
            model_name=f"{name} Cross-Validation",
            tags={"evaluation_type": "cross_validation"}
        ) as run:

            scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1"
            )

            mean_f1 = scores.mean()
            std_f1 = scores.std()

            from src.mlflow_tracking import log_metrics
            log_metrics({
                "cv_mean_f1": mean_f1,
                "cv_std_f1": std_f1
            })

            cv_results.append({
                "Model": name,
                "Mean F1": mean_f1,
                "Std": std_f1
            })

    cv_results = pd.DataFrame(cv_results).sort_values(
        "Mean F1", ascending=False
    )

    print("Cross-Validation Results:")
    display(cv_results.style.format({
        "Mean F1": "{:.3f}",
        "Std": "{:.3f}"
    }))

    return cv_results


# ============================================================
# DECISION TREE TUNING
# ============================================================
def tune_decision_tree(X_train, y_train):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "model__max_depth": [2, 4, 6],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__criterion": ["gini", "entropy"]
    }

    with track_model_run("Tuned Decision Tree") as run:

        grid = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)

        from src.mlflow_tracking import log_parameters, log_metrics

        log_parameters(grid.best_params_)
        log_metrics({"cv_best_f1": grid.best_score_})

    print("Best Decision Tree Parameters:")
    print(grid.best_params_)
    print(f"Best Decision Tree CV F1: {grid.best_score_:.4f}")

    return grid

# ============================================================
# RANDOM FOREST TUNING
# ============================================================
def tune_random_forest(X_train, y_train):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_leaf": [1, 5, 10]
    }

    with track_model_run("Tuned Random Forest") as run:

        grid = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)

        from src.mlflow_tracking import log_parameters, log_metrics

        log_parameters(grid.best_params_)
        log_metrics({"cv_best_f1": grid.best_score_})

    print("Best Random Forest Parameters:")
    print(grid.best_params_)
    print(f"Best Random Forest CV F1: {grid.best_score_:.4f}")

    return grid 


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

def create_final_model_comparison(model_results, tree_grid, rf_grid, X_test, y_test):

    final_results = model_results.copy()

    tuned_models = {
        "Tuned Decision Tree": tree_grid.best_estimator_,
        "Tuned Random Forest": rf_grid.best_estimator_
    }

    tuned_results = []

    for name, model in tuned_models.items():

        result = evaluate_model(model, X_test, y_test, name)
        tuned_results.append(result)

    tuned_results = pd.DataFrame(tuned_results)

    final_results = pd.concat([ final_results, tuned_results],ignore_index=True)
    final_results = final_results.sort_values("F1", ascending=False).reset_index(drop=True)


    print("Final Model Comparison:")
    display(
        final_results.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        })
    )

    return final_results, tuned_models


# ============================================================
# FINAL CROSS-VALIDATION COMPARISON
# ============================================================

def create_final_cv_comparison(cv_results, tree_grid, rf_grid):

    final_cv_results = cv_results.copy()

    tuned_cv_results = pd.DataFrame([
        {
            "Model": "Tuned Decision Tree",
            "Mean F1": tree_grid.best_score_,
            "Std": None
        },
        {
            "Model": "Tuned Random Forest",
            "Mean F1": rf_grid.best_score_,
            "Std": None
        }
    ])

    final_cv_results = pd.concat([ final_cv_results, tuned_cv_results], ignore_index=True)
    final_cv_results = final_cv_results.sort_values("Mean F1", ascending=False).reset_index(drop=True)

    print("Final Cross-Validation Comparison:" )
    display(final_cv_results)

    return final_cv_results


# ============================================================
# COMPLETE SUPERVISED LEARNING PIPELINE
# ============================================================

def run_supervised_learning(df):

    X, y = create_features_and_target( df )

    X_train, X_test, y_train, y_test = split_data( X, y )

    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")



    print("--- BASELINE MODEL ---")

    train_baseline(X_train, X_test, y_train, y_test)



    print("--- TRAINING MODELS ---")

    models = train_all_models( X_train, y_train)



    print("--- MODEL EVALUATION ---")

    model_results = evaluate_all_models(models, X_test, y_test)



    print("--- CROSS-VALIDATION ---")

    cv_results = cross_validate_models(models, X_train, y_train)



    print("--- DECISION TREE TUNING ---")

    tree_grid = tune_decision_tree(X_train, y_train)



    print("--- RANDOM FOREST TUNING ---"  )

    rf_grid = tune_random_forest(X_train,  y_train)



    print("--- FINAL MODEL COMPARISON ---")

    final_model_results,final_models = create_final_model_comparison(model_results, tree_grid, rf_grid, X_test, y_test )
    final_cv_results = create_final_cv_comparison(cv_results, tree_grid, rf_grid)
    best_model_name = final_model_results.iloc[0]["Model"]

    if best_model_name == "Tuned Decision Tree":
        best_model = tree_grid.best_estimator_

    elif best_model_name == "Tuned Random Forest":
        best_model = rf_grid.best_estimator_

    else:
        best_model = models[best_model_name]

    tuned_rf_results = evaluate_model( rf_grid.best_estimator_, X_test, y_test, "Tuned Random Forest" )

    cv_scores = cross_val_score( rf_grid.best_estimator_, X_train, y_train, cv=5, scoring="f1")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "models": models,
        "model_results": model_results,
        "cv_results": cv_results,
        "final_models": final_models,
        "final_model_results": final_model_results,
        "final_cv_results": final_cv_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "tuned_random_forest": rf_grid.best_estimator_,
        "tuned_rf_results": tuned_rf_results,
        "tree_grid": tree_grid,
        "rf_grid": rf_grid,
        "cv_scores": cv_scores
    }