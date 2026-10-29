from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Credit Card Fraud Detection API is running"
    }


def test_prediction_endpoint():
    transaction = {
        "features": [0.0] * 30
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    result = response.json()

    assert result["status"] in [
        "Normal",
        "Potentially Suspicious"
    ]
    assert "isolation_forest_score" in result
    assert "autoencoder_error" in result


def test_prediction_with_wrong_number_of_features():
    transaction = {
        "features": [0.0] * 29
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Credit Card Fraud Detection API is operational"
    }



def test_prediction_is_saved_to_database(tmp_path, monkeypatch):
    import sqlite3
    from database import database
    from fastapi.testclient import TestClient

    test_db = tmp_path / "test_fraud_detection.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        str(test_db)
    )

    database.create_database()

    # Use a fresh client so the temporary database is initialized.
    with TestClient(app) as test_client:
        response = test_client.post(
            "/predict",
            json={"features": [0.0] * 29 + [250.0]}
        )

    assert response.status_code == 200

    connection = sqlite3.connect(test_db)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT amount, status FROM transactions "
        "ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    connection.close()

    assert row is not None
    assert row[0] == 250.0
    assert row[1] == response.json()["status"]



def test_prediction_rejects_nan():
    import json

    features = [0.0] * 30
    features[0] = "NaN"

    response = client.post(
        "/predict",
        content=json.dumps({"features": features}),
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_prediction_rejects_infinity():
    import json

    features = [0.0] * 30
    features[0] = "Infinity"

    response = client.post(
        "/predict",
        content=json.dumps({"features": features}),
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
