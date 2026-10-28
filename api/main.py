from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Allow Python to find files inside the src folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from predict import predict_transaction


# -----------------------------
# Create FastAPI application
# -----------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting potentially suspicious credit card transactions",
    version="1.0.0"
)


# -----------------------------
# Transaction input format
# -----------------------------

class Transaction(BaseModel):

    features: list[float]


# -----------------------------
# Home endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------

from fastapi import HTTPException

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(
            status_code=400,
            detail="Transaction must contain exactly 30 features"
        )

    result = predict_transaction(transaction.features)
    return result