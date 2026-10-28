from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 features: Time, V1-V28, and Amount"
    )


# -----------------------------
# Home endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Credit Card Fraud Detection API is operational"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------

from fastapi import HTTPException

@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_transaction(transaction.features)
    return result