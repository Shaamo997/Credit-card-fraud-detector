import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# load the saved model once when the server starts
model = joblib.load('../models/fraud_detector.pkl')

# initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detector",
    description="ML model that detects fraudulent credit card transactions",
    version="1.0.0"
)

# define what a transaction looks like
class Transaction(BaseModel):
    features: List[float]

# health check endpoint
@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud Detector API is running"}

# prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    # convert to numpy array and reshape for model
    data = np.array(transaction.features).reshape(1, -1)
    
    # get prediction and probability
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    return {
        "prediction": int(prediction),
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability": round(float(probability), 4)
    }

# batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    results = []
    for transaction in transactions:
        data = np.array(transaction.features).reshape(1, -1)
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        results.append({
            "prediction": int(prediction),
            "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": round(float(probability), 4)
        })
    return {"results": results, "total": len(results)}