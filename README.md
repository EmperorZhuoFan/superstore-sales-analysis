# Superstore Sales Analysis

An end-to-end exploratory data analysis of the classic **Sample Superstore**
retail dataset — sales, profit, and customer behavior across regions,
categories, and segments.

This is one of the case studies in my journey toward becoming a
**Machine Learning Engineer**: building a strong data foundation first
(pandas, statistics, EDA), then moving into modeling, MLOps, and deep
learning. See the [portfolio](https://emperorzhuofan.github.io) for the
full picture.

## What the analysis covers

- **Data understanding** — shape, dtypes, summary statistics for numerical
  and categorical columns, missing-value checks
- **Data cleaning** — date parsing, consistency fixes, and preparation of
  the dataset for analysis
- **Outlier detection** — IQR-based identification of unusual orders
- **Exploratory data analysis** — sales and profit trends by category,
  sub-category, region, segment, and time, visualized with matplotlib
  and seaborn

## Project structure

```
.
├── superstore_sales_analysis.py   # Main analysis script (runs end to end)
├── data/
│   └── samplesuperstore.csv       # Retail dataset (~10k orders)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python superstore_sales_analysis.py
```

## Next steps

- Profitability prediction (classification of profit/loss per order)
- Sales forecasting with time-series models
- Feature engineering pipeline for ML experiments

## Roadmap context

| Phase | Focus |
| ----- | ----- |
| Now | Data science foundations — Python, pandas, SQL, EDA, Power BI |
| Oct 2026 | ML engineering — MLOps, MLflow, Docker, DVC, CI/CD, monitoring |
| Nov 2026 | Deep learning (PyTorch) + NLP + Transformers |
| Dec 2026 | LLM foundations — fine-tuning, evaluation, Hugging Face |
