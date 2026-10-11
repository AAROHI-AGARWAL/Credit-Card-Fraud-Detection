import sqlite3
import os
import pandas as pd


DATABASE_PATH = os.path.join(
    os.path.dirname(__file__),
    "fraud_detection.db"
)


connection = sqlite3.connect(DATABASE_PATH)

query = """
SELECT *
FROM transactions
"""

df = pd.read_sql_query(query, connection)

connection.close()


print("\nSaved Transactions:")
print(df.to_string(index=False))