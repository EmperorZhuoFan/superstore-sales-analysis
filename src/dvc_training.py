import pandas as pd

from src.data_preparation import load_data, prepare_data
from src.supervised import run_supervised_learning


def main():

    file_path = "data/processed/superstore_v1.parquet"

    df = pd.read_parquet(file_path)

    results = run_supervised_learning(df)

    final_results = results["final_model_results"]

    best_model = final_results.iloc[0]

    metrics = {
        "best_model": best_model["Model"],
        "accuracy": best_model["Accuracy"],
        "precision": best_model["Precision"],
        "recall": best_model["Recall"],
        "f1": best_model["F1"],
        "roc_auc": best_model["ROC-AUC"],
    }

    with open("metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=4)

    print("\nTraining metrics:")
    print(metrics)


if __name__ == "__main__":
    main()