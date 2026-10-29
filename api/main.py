import math
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import sys
import os

# Allow Python to find files inside the src folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from predict import predict_transaction
from database.database import create_database, save_transaction


# -----------------------------
# Create FastAPI application
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_database()
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting potentially suspicious credit card transactions",
    version="1.0.0",
    lifespan=lifespan
)


# -----------------------------
# Create database on startup
# -----------------------------

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

    @field_validator("features")
    @classmethod
    def validate_features(cls, features):
        if not all(math.isfinite(value) for value in features):
            raise ValueError(
                "All features must be finite numbers; NaN and infinity are not allowed."
            )
        return features



# -----------------------------
# Home endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


# -----------------------------
# Health endpoint
# -----------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Credit Card Fraud Detection API is operational"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_transaction(transaction.features)

    # Time is the first feature and Amount is the last
    time_value = transaction.features[0]
    amount_value = transaction.features[-1]

    # Save the prediction result to SQLite
    save_transaction(time_value, amount_value, result)

    return result
