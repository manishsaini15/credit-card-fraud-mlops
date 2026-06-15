
# =============================================================================
# FastAPI Inference Application
# =============================================================================

# Import FastAPI
from fastapi import FastAPI

# Import request validation model
from pydantic import BaseModel

# Import pandas
import pandas as pd

# Import joblib
import joblib

# Import configuration loader
from src.utils.config import load_schema


# =============================================================================
# Load configuration
# =============================================================================

schema = load_schema()

# Get feature columns in correct order
feature_columns = [
    column
    for column in schema["columns"]
    if column != schema["target_column"]
]

# Load model
model = joblib.load(
    schema["artifacts"]["best_model_path"]
)

# Load preprocessing pipeline
preprocessor = joblib.load(
    schema["artifacts"]["preprocessor_path"]
)

# Create FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0"
)


# =============================================================================
# Input Schema
# =============================================================================

class TransactionInput(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get("/")
def health():

    return {
        "status": "healthy",
        "message": "Credit Card Fraud Detection API is running."
    }


# =============================================================================
# Prediction Endpoint
# =============================================================================

@app.post("/predict")
def predict(data: TransactionInput):

    # Convert request into dictionary
    input_dict = data.model_dump()

    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_dict])

    # Reorder columns to match training data
    df = df[feature_columns]

    # Apply preprocessing pipeline
    transformed_data = preprocessor.transform(df)

    # Predict class
    prediction = model.predict(
        transformed_data
    )[0]

    # Predict fraud probability
    probability = model.predict_proba(
        transformed_data
    )[0][1]

    # Convert prediction to label
    label = (
        "Fraud"
        if prediction == 1
        else "Legitimate"
    )

    # Return response
    return {

        "prediction": label,

        "prediction_code": int(prediction),

        "fraud_probability": round(
            float(probability),
            6
        )
    }


