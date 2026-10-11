import joblib
import torch
import numpy as np

from anomaly_detection import (
    Autoencoder,
    get_isolation_forest_scores,
    calculate_reconstruction_errors
)


# -----------------------------
# Load saved models
# -----------------------------

scaler = joblib.load("models/scaler.pkl")
isolation_forest = joblib.load("models/isolation_forest.pkl")

with open("models/threshold.txt", "r") as f:
    isolation_threshold = float(f.read())

with open("models/autoencoder_threshold.txt", "r") as f:
    autoencoder_threshold = float(f.read())


# -----------------------------
# Load Autoencoder
# -----------------------------

autoencoder = Autoencoder()

autoencoder.load_state_dict(
    torch.load(
        "models/autoencoder.pth",
        map_location="cpu"
    )
)

autoencoder.eval()


# -----------------------------
# Prediction function
# -----------------------------

def predict_transaction(transaction):

    # Convert transaction into NumPy array
    transaction = np.array(transaction).reshape(1, -1)

    transaction = __import__("pandas").DataFrame(
        transaction,
        columns=scaler.feature_names_in_
    )

    # Scale transaction
    transaction_scaled = scaler.transform(transaction)

    # Isolation Forest score
    isolation_score = get_isolation_forest_scores(
        isolation_forest,
        transaction_scaled
    )[0]

    # Autoencoder reconstruction error
    autoencoder_error = calculate_reconstruction_errors(
        autoencoder,
        transaction_scaled
    )[0]

    # Individual predictions
    isolation_prediction = (
        isolation_score >= isolation_threshold
    )

    autoencoder_prediction = (
        autoencoder_error >= autoencoder_threshold
    )

    # Final status
    if isolation_prediction or autoencoder_prediction:
        status = "Potentially Suspicious"
    else:
        status = "Normal"

    return {
    "status": status,
    "isolation_forest_score": float(isolation_score),
    "autoencoder_error": float(autoencoder_error),
    "isolation_forest_flag": bool(isolation_prediction),
    "autoencoder_flag": bool(autoencoder_prediction)
}


# -----------------------------
# Example transaction
# -----------------------------

if __name__ == "__main__":

    example_transaction = [
        0.0,      # Time
        -1.35,    # V1
        -0.07,    # V2
        2.54,     # V3
        1.38,     # V4
        -0.34,    # V5
        0.46,     # V6
        0.24,     # V7
        0.10,     # V8
        0.36,     # V9
        -0.23,    # V10
        -0.16,    # V11
        -0.08,    # V12
        0.12,     # V13
        -0.21,    # V14
        -0.04,    # V15
        0.08,     # V16
        -0.10,    # V17
        0.05,     # V18
        -0.03,    # V19
        0.07,     # V20
        -0.02,    # V21
        0.04,     # V22
        -0.01,    # V23
        0.02,     # V24
        -0.01,    # V25
        0.01,     # V26
        -0.01,    # V27
        0.01,     # V28
        100.0     # Amount
    ]

    result = predict_transaction(example_transaction)

    print("\nTransaction Prediction")
    print("----------------------")
    print("Status:", result["status"])
    print("Isolation Forest Score:", result["isolation_forest_score"])
    print("Autoencoder Error:", result["autoencoder_error"])
    print("Isolation Forest Flag:", result["isolation_forest_flag"])
    print("Autoencoder Flag:", result["autoencoder_flag"])