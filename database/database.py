import sqlite3
import os


DATABASE_PATH = os.path.join(
    os.path.dirname(__file__),
    "fraud_detection.db"
)


def create_database():
    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL,
            amount REAL,
            isolation_forest_score REAL,
            autoencoder_error REAL,
            isolation_forest_flag INTEGER,
            autoencoder_flag INTEGER,
            status TEXT
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully!")


def save_transaction(time, amount, result):
    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO transactions (
            time,
            amount,
            isolation_forest_score,
            autoencoder_error,
            isolation_forest_flag,
            autoencoder_flag,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        float(time),
        float(amount),
        float(result["isolation_forest_score"]),
        float(result["autoencoder_error"]),
        int(result["isolation_forest_flag"]),
        int(result["autoencoder_flag"]),
        result["status"]
    ))

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_database()

