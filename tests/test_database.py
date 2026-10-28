import sqlite3

from database.database import create_database, save_transaction


def test_database_creation(tmp_path, monkeypatch):
    # Use a temporary database so the real database is not changed.
    test_db = tmp_path / "test_fraud_detection.db"

    monkeypatch.setattr(
        "database.database.DATABASE_PATH",
        str(test_db)
    )

    create_database()

    connection = sqlite3.connect(test_db)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='transactions'"
    )

    assert cursor.fetchone() is not None
    connection.close()


def test_save_transaction(tmp_path, monkeypatch):
    test_db = tmp_path / "test_fraud_detection.db"

    monkeypatch.setattr(
        "database.database.DATABASE_PATH",
        str(test_db)
    )

    create_database()

    result = {
        "isolation_forest_score": 0.15,
        "autoencoder_error": 17.2,
        "isolation_forest_flag": True,
        "autoencoder_flag": False,
        "status": "Potentially Suspicious"
    }

    save_transaction(1000.0, 250.0, result)

    connection = sqlite3.connect(test_db)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT time, amount, status "
        "FROM transactions"
    )

    row = cursor.fetchone()
    connection.close()

    assert row == (
        1000.0,
        250.0,
        "Potentially Suspicious"
    )