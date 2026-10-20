import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from src.mlflow_tracking import setup_mlflow


class SuperstoreProfitModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow wrapper for the trained Superstore classifier."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, context, model_input, params=None):
        return self.estimator.predict(model_input)


def to_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()

    return pd.DataFrame(data)


def log_and_evaluate_model(
    estimator,
    X_train,
    X_test,
    y_test,
):
    print("DEBUG: log_and_evaluate_model() was called")

    setup_mlflow()

    X_train = to_dataframe(X_train)
    X_test = to_dataframe(X_test)
    y_test = pd.Series(np.asarray(y_test).ravel(), name="target")

    input_example = X_train.head(3)
    example_predictions = estimator.predict(input_example)

    signature = infer_signature(
        input_example,
        example_predictions,
    )

    # Log the native scikit-learn model and custom PyFunc model.
    with mlflow.start_run(run_name="Chapter 2 - MLflow Models") as run:
        mlflow.set_tag("chapter", "MLflow Models")
        mlflow.set_tag("model_name", "Gradient Boosting")

        sklearn_info = mlflow.sklearn.log_model(
        sk_model=estimator,
        name="profit_status_sklearn",
        signature=signature,
        input_example=input_example,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        mlflow.pyfunc.log_model(
            name="profit_status_custom",
            python_model=SuperstoreProfitModel(estimator),
            signature=signature,
            input_example=input_example,
        )

        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/profit_status_sklearn"
    custom_model_uri = f"runs:/{run_id}/profit_status_custom"

    print("\nLogged model URIs:")
    print("Scikit-learn:", model_uri)
    print("Custom model:", custom_model_uri)

    # Load the logged model and generate predictions.
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    predictions = loaded_model.predict(X_test)

    print("\nLoaded model predictions:")
    print(predictions[:5])

    # Evaluate the logged model in a separate MLflow run.
    evaluation_data = X_test.copy()
    evaluation_data["target"] = y_test.to_numpy()

    with mlflow.start_run(run_name="Chapter 2 - Model Evaluation"):
        evaluation = mlflow.models.evaluate(
            model=model_uri,
            data=evaluation_data,
            targets="target",
            model_type="classifier",
            evaluator_config={"log_explainer": False},
        )

        print("\nMLflow evaluation metrics:")
        for metric, value in evaluation.metrics.items():
            print(f"{metric}: {value}")

    # Save a request payload for the model-serving exercise.
    serving_input = {
        "dataframe_split": json.loads(
            input_example.to_json(orient="split")
        )
    }

    payload_path = Path("serving_payload.json")
    payload_path.write_text(
        json.dumps(serving_input, indent=2),
        encoding="utf-8",
    )

    print(f"\nServing payload saved to: {payload_path.resolve()}")

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "custom_model_uri": custom_model_uri,
        "sklearn_model_info": sklearn_info,
        "evaluation": evaluation,
    }