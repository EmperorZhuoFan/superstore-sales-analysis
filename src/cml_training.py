
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data_preparation import load_data, prepare_data
from src.supervised import create_features_and_target, create_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def main():
    os.makedirs("reports", exist_ok=True)

    url = (
        "https://raw.githubusercontent.com/"
        "EmperorZhuoFan/superstore-sales-analysis/"
        "refs/heads/main/data/raw/samplesuperstore.csv"
    )

    df, _ = load_data(url)
    df = prepare_data(df)

    X, y = create_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Weighted F1"],
        "Score": [accuracy, f1],
    })
    metrics.to_csv("reports/metrics.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(metrics["Metric"], metrics["Score"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Superstore Random Forest Evaluation")
    plt.tight_layout()
    plt.savefig("reports/model_metrics.png")
    plt.close()

    with open("report.md", "w", encoding="utf-8") as report:
        report.write("# Superstore ML Training Report\n\n")
        report.write("## Evaluation Metrics\n\n")
        report.write(metrics.to_markdown(index=False))
        report.write("\n\n## Metrics Plot\n\n")
        report.write("![Model Metrics](reports/model_metrics.png)\n")

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
