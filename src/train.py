import os
import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import load_data, prepare_data
from anomaly_detection import (
    train_isolation_forest,
    get_isolation_forest_scores,
    Autoencoder
)
from evaluation import find_best_threshold


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = "data/processed/creditcard_cleaned.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

AE_EPOCHS = 20
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 0.001


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

df = load_data(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# PREPARE FEATURES
# =========================================================

X, y = prepare_data(df)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)


print("\nData split:")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)


# =========================================================
# SCALE DATA
# =========================================================

print("\nScaling data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)


# Save scaler

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

print("Scaler saved.")


# =========================================================
# SELECT NORMAL TRANSACTIONS
# =========================================================

X_train_normal = X_train_scaled[
    y_train.values == 0
]

print(
    "Normal training transactions:",
    X_train_normal.shape
)


# =========================================================
# ISOLATION FOREST
# =========================================================

print("\nTraining Isolation Forest...")

iso_forest = train_isolation_forest(
    X_train_normal
)

print("Isolation Forest training completed.")


# Calculate validation anomaly scores

validation_scores_if = get_isolation_forest_scores(
    iso_forest,
    X_validation_scaled
)


# Find threshold using validation data

(
    threshold_if,
    precision_if,
    recall_if,
    f1_if
) = find_best_threshold(
    y_validation,
    validation_scores_if
)


print("\nIsolation Forest validation results:")
print("Threshold:", threshold_if)
print("Precision:", precision_if)
print("Recall:", recall_if)
print("F1-Score:", f1_if)


# Save Isolation Forest

joblib.dump(
    iso_forest,
    os.path.join(
        MODEL_DIR,
        "isolation_forest.pkl"
    )
)


with open(
    os.path.join(
        MODEL_DIR,
        "threshold.txt"
    ),
    "w"
) as file:

    file.write(str(threshold_if))


print("Isolation Forest model saved.")


# =========================================================
# AUTOENCODER
# =========================================================

print("\nTraining Autoencoder...")


# Convert normal training data to tensors

X_train_tensor = torch.tensor(
    X_train_normal,
    dtype=torch.float32
)


train_dataset = TensorDataset(
    X_train_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=AE_BATCH_SIZE,
    shuffle=True
)


# Create model

autoencoder = Autoencoder()


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    autoencoder.parameters(),
    lr=AE_LEARNING_RATE
)


# Training loop

autoencoder.train()


for epoch in range(AE_EPOCHS):

    total_loss = 0.0

    for batch in train_loader:

        inputs = batch[0]

        optimizer.zero_grad()

        outputs = autoencoder(inputs)

        loss = criterion(
            outputs,
            inputs
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{AE_EPOCHS} "
        f"- Loss: {average_loss:.6f}"
    )


print("Autoencoder training completed.")


# =========================================================
# AUTOENCODER VALIDATION SCORES
# =========================================================

autoencoder.eval()


X_validation_tensor = torch.tensor(
    X_validation_scaled,
    dtype=torch.float32
)


with torch.no_grad():

    reconstructed_validation = autoencoder(
        X_validation_tensor
    )


validation_errors_ae = torch.mean(
    (
        X_validation_tensor
        - reconstructed_validation
    ) ** 2,
    dim=1
).numpy()


# Find threshold using validation data

(
    threshold_ae,
    precision_ae,
    recall_ae,
    f1_ae
) = find_best_threshold(
    y_validation,
    validation_errors_ae
)


print("\nAutoencoder validation results:")
print("Threshold:", threshold_ae)
print("Precision:", precision_ae)
print("Recall:", recall_ae)
print("F1-Score:", f1_ae)


# =========================================================
# SAVE AUTOENCODER
# =========================================================

torch.save(
    autoencoder.state_dict(),
    os.path.join(
        MODEL_DIR,
        "autoencoder.pth"
    )
)


with open(
    os.path.join(
        MODEL_DIR,
        "autoencoder_threshold.txt"
    ),
    "w"
) as file:

    file.write(str(threshold_ae))


print("Autoencoder model saved.")


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("\nSaved files:")

print("models/scaler.pkl")
print("models/isolation_forest.pkl")
print("models/threshold.txt")
print("models/autoencoder.pth")
print("models/autoencoder_threshold.txt")

print("\nThe test set has NOT been used for threshold selection.")
print("It will be used later for final model evaluation.")