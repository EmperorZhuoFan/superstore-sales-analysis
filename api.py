from pathlib import Path
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "deployment" / "model"
model = mlflow.pyfunc.load_model(str(MODEL_PATH))
 

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_URI_PATH = Path(__file__).parent / "model_uri.txt"

if not MODEL_URI_PATH.exists():
    raise FileNotFoundError(
        "model_uri.txt not found. Run main.py to train and log the model first."
    )

model_uri = MODEL_URI_PATH.read_text(encoding="utf-8").strip()

model = mlflow.pyfunc.load_model(model_uri)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Superstore Profit Prediction API",
    version="0.1.0",
)


# ============================================================
# EXPECTED FEATURES
# ============================================================

FEATURES = [
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
    "Sales Per Unit",
]


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    features: dict[str, object]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Superstore Profit Prediction API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    missing = [
        column
        for column in FEATURES
        if column not in request.features
    ]

    extra = [
        column
        for column in request.features
        if column not in FEATURES
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    try:
        input_df = pd.DataFrame(
            [[request.features[column] for column in FEATURES]],
            columns=FEATURES,
        )

        integer_features = [
            "Order Year",
            "Order Month",
            "Order Quarter",
            "Order Day",
            "Order Day of Week",
        ]

        for column in integer_features:
            input_df[column] = input_df[column].astype("int32")

        prediction = model.predict(input_df)

        return {
            "prediction": prediction.tolist()
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc