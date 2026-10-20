from contextlib import contextmanager
from pathlib import Path
import math

import mlflow
import pandas as pd


EXPERIMENT_NAME = "Superstore Profit Status"


def setup_mlflow():
    mlflow.set_tracking_uri(
        "sqlite:///F:/Working Plan/__Projects__/superstore/Superstore_vscode/mlflow.db"
    )
    return mlflow.set_experiment(EXPERIMENT_NAME)

@contextmanager
def track_model_run(model_name, parameters=None, tags=None):
    """Track one model training run."""
    setup_mlflow()

    with mlflow.start_run(run_name=model_name, nested=True) as run:
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("project", "Superstore Sales Analysis")

        if tags:
            mlflow.set_tags(tags)

        if parameters:
            log_parameters(parameters)

        yield run


def log_parameters(parameters):
    """Log model configuration values."""
    cleaned_parameters = {
        str(key): str(value)
        for key, value in parameters.items()
        if value is not None
    }

    if cleaned_parameters:
        mlflow.log_params(cleaned_parameters)


def log_metrics(metrics):
    """Log numerical evaluation or validation metrics."""
    cleaned_metrics = {}

    for name, value in metrics.items():
        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(value):
            cleaned_metrics[name] = value

    if cleaned_metrics:
        mlflow.log_metrics(cleaned_metrics)


def log_artifact(file_path, artifact_path=None):
    """Save a file as an artifact of the active run."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")

    mlflow.log_artifact(
        str(path),
        artifact_path=artifact_path
    )


def search_experiment_runs(
    max_results=100,
    filter_string=None,
    order_by=None
):
    """Retrieve and compare recorded experiment runs."""
    setup_mlflow()

    return mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=filter_string or "",
        max_results=max_results,
        order_by=order_by or ["start_time DESC"]
    )