import pandas as pd

from src.preprocessing import prepare_data


def test_prepare_data_removes_target_and_hour():
    df = pd.DataFrame({
        "Time": [100.0, 200.0],
        "V1": [0.1, 0.2],
        "Amount": [10.0, 20.0],
        "Hour": [0.0, 0.0],
        "Class": [0, 1]
    })

    X, y = prepare_data(df)

    assert "Class" not in X.columns
    assert "Hour" not in X.columns
    assert len(X) == 2
    assert y.tolist() == [0, 1]


def test_prepare_data_keeps_expected_features():
    df = pd.DataFrame({
        "Time": [100.0],
        "V1": [0.1],
        "Amount": [10.0],
        "Class": [0]
    })

    X, y = prepare_data(df)

    assert list(X.columns) == ["Time", "V1", "Amount"]
    assert len(y) == 1