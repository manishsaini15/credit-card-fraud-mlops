import json
import os
import joblib
import pandas as pd

model = None
preprocessor = None


def init():
    global model, preprocessor

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    model = joblib.load(
        os.path.join(model_dir, "best_model.pkl")
    )

    preprocessor = joblib.load(
        os.path.join(model_dir, "preprocessor.pkl")
    )


def run(raw_data):

    data = json.loads(raw_data)

    df = pd.DataFrame([data])

    df[["Time", "Amount"]] = preprocessor.transform(
        df[["Time", "Amount"]]
    )

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Fraud" if prediction else "Legit",
        "fraud_probability": float(probability)
    }