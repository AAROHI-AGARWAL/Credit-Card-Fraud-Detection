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

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Transaction must contain exactly 30 features"
    )