from src.predict import predict_transaction


def test_prediction_returns_expected_fields():
    # 30 features: Time, V1-V28, Amount
    transaction = [0.0] * 30

    result = predict_transaction(transaction)

    assert "status" in result
    assert "isolation_forest_score" in result
    assert "autoencoder_error" in result
    assert "isolation_forest_flag" in result
    assert "autoencoder_flag" in result

    assert result["status"] in [
        "Normal",
        "Potentially Suspicious"
    ]

    assert isinstance(result["isolation_forest_score"], float)
    assert isinstance(result["autoencoder_error"], float)
    assert isinstance(result["isolation_forest_flag"], bool)
    assert isinstance(result["autoencoder_flag"], bool)