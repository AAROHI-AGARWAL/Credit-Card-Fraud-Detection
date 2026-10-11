import joblib
import torch
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from anomaly_detection import (
    Autoencoder,
    get_isolation_forest_scores,
    calculate_reconstruction_errors
)


# -----------------------------
# Load test data
# -----------------------------

df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

if "Hour" in X.columns:
    X = X.drop("Hour", axis=1)


# -----------------------------
# Recreate the same data split
# -----------------------------

from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)


# -----------------------------
# Load scaler
# -----------------------------

scaler = joblib.load("models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Load Isolation Forest
# -----------------------------

isolation_forest = joblib.load(
    "models/isolation_forest.pkl"
)

with open("models/threshold.txt", "r") as f:
    isolation_threshold = float(f.read())


# -----------------------------
# Isolation Forest predictions
# -----------------------------

isolation_scores = get_isolation_forest_scores(
    isolation_forest,
    X_test_scaled
)

isolation_predictions = (
    isolation_scores >= isolation_threshold
).astype(int)


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

with open("models/autoencoder_threshold.txt", "r") as f:
    autoencoder_threshold = float(f.read())


# -----------------------------
# Autoencoder predictions
# -----------------------------

autoencoder_errors = calculate_reconstruction_errors(
    autoencoder,
    X_test_scaled
)

autoencoder_predictions = (
    autoencoder_errors >= autoencoder_threshold
).astype(int)


# -----------------------------
# Evaluation function
# -----------------------------

def evaluate_model(name, y_true, predictions, scores):

    cm = confusion_matrix(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("PR-AUC:", pr_auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return {
        "Model": name,
        "PR-AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "True Negatives": cm[0][0],
        "False Positives": cm[0][1],
        "False Negatives": cm[1][0],
        "True Positives": cm[1][1]
    }


# -----------------------------
# Evaluate both models
# -----------------------------

isolation_results = evaluate_model(
    "Isolation Forest",
    y_test,
    isolation_predictions,
    isolation_scores
)

autoencoder_results = evaluate_model(
    "Autoencoder",
    y_test,
    autoencoder_predictions,
    autoencoder_errors
)


# -----------------------------
# Save final comparison
# -----------------------------

comparison = pd.DataFrame([
    isolation_results,
    autoencoder_results
])

comparison.to_csv(
    "data/processed/final_model_comparison.csv",
    index=False
)


print("\n" + "=" * 50)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 50)

print("\nResults saved to:")
print("data/processed/final_model_comparison.csv")