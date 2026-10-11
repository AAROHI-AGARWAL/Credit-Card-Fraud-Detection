import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn


# =========================================================
# ISOLATION FOREST
# =========================================================

def train_isolation_forest(X_train_normal):
    """
    Train Isolation Forest using only legitimate transactions.
    """

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_normal)

    return model


def get_isolation_forest_scores(model, X_test_scaled):
    """
    Generate anomaly scores using Isolation Forest.

    Higher score = more anomalous.
    """

    scores = -model.decision_function(X_test_scaled)

    return scores


def predict_isolation_forest(scores, threshold):
    """
    Convert anomaly scores into predictions.

    0 = normal
    1 = potentially suspicious
    """

    predictions = (scores >= threshold).astype(int)

    return predictions


# =========================================================
# AUTOENCODER
# =========================================================

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 30)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def calculate_reconstruction_errors(model, X_test_scaled):

    model.eval()

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        reconstructed = model(X_test_tensor)

        errors = torch.mean(
            (X_test_tensor - reconstructed) ** 2,
            dim=1
        ).numpy()

    return errors


def predict_autoencoder(errors, threshold):
    """
    Convert reconstruction errors into predictions.

    0 = normal
    1 = potentially suspicious
    """

    predictions = (errors >= threshold).astype(int)

    return predictions