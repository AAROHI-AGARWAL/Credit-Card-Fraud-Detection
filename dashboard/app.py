import streamlit as st
import sys
import os

# Allow Python to find files inside the src folder
# Allow Python to find files inside the src folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

# Allow Python to find files inside the database folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "database")
    )
)

from predict import predict_transaction
from database import save_transaction

import sqlite3
import pandas as pd

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "An anomaly detection system that identifies "
    "potentially suspicious credit card transactions."
)

st.info(
    "This system uses two anomaly detection models: "
    "Isolation Forest and Autoencoder."
)


# -------------------------------------------------
# Transaction Input
# -------------------------------------------------

st.header("📝 Transaction Details")

st.write(
    "Enter the 30 numerical features required by the trained models."
)

feature_names = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

features = []

# Create columns for input fields
for i in range(0, len(feature_names), 3):

    cols = st.columns(3)

    for j, col in enumerate(cols):

        index = i + j

        if index < len(feature_names):

            feature = feature_names[index]

            with col:

                value = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=feature
                )

                features.append(value)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

st.divider()

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    result = predict_transaction(features)

    save_transaction(
        features[0],
        features[29],
        result
    )
    st.header("📊 Prediction Result")


    # -------------------------------------------------
    # Overall Status
    # -------------------------------------------------

    if result["status"] == "Normal":

        st.success(
            "✅ Transaction Status: Normal"
        )

        st.info(
            "The anomaly detection models did not identify "
            "this transaction as potentially suspicious."
        )

    else:

        st.warning(
            "⚠️ Transaction Status: Potentially Suspicious"
        )

        st.warning(
            "The transaction should be reviewed because "
            "at least one anomaly detection model flagged it."
        )


    # -------------------------------------------------
    # Risk Level
    # -------------------------------------------------

    if result["isolation_forest_flag"] or result["autoencoder_flag"]:

        risk_level = "HIGHER RISK"

    else:

        risk_level = "LOWER RISK"


    st.subheader("🚦 Risk Level")

    st.metric(
        "Current Risk Level",
        risk_level
    )


    # -------------------------------------------------
    # Model Results
    # -------------------------------------------------

    st.subheader("🤖 Model Analysis")

    col1, col2 = st.columns(2)


    with col1:

        st.write("### Isolation Forest")

        st.metric(
            "Anomaly Score",
            f"{result['isolation_forest_score']:.6f}"
        )

        if result["isolation_forest_flag"]:

            st.error(
                "⚠️ Isolation Forest: Anomaly Detected"
            )

        else:

            st.success(
                "✅ Isolation Forest: No Anomaly"
            )


    with col2:

        st.write("### Autoencoder")

        st.metric(
            "Reconstruction Error",
            f"{result['autoencoder_error']:.6f}"
        )

        if result["autoencoder_flag"]:

            st.error(
                "⚠️ Autoencoder: Anomaly Detected"
            )

        else:

            st.success(
                "✅ Autoencoder: No Anomaly"
            )


    # -------------------------------------------------
    # Decision Explanation
    # -------------------------------------------------

    st.subheader("ℹ️ Decision Explanation")

    if (
        result["isolation_forest_flag"]
        and result["autoencoder_flag"]
    ):

        st.write(
            "Both anomaly detection models flagged this "
            "transaction. It should be reviewed."
        )

    elif result["isolation_forest_flag"]:

        st.write(
            "Isolation Forest flagged this transaction, "
            "while the Autoencoder did not."
        )

    elif result["autoencoder_flag"]:

        st.write(
            "Autoencoder flagged this transaction, "
            "while Isolation Forest did not."
        )

    else:

        st.write(
            "Neither anomaly detection model flagged this "
            "transaction."
        )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection using Anomaly Detection Techniques"
)

st.caption(
    "Models: Isolation Forest + Autoencoder"
)

# -------------------------------------------------
# Transaction History
# -------------------------------------------------

st.divider()

st.header("📜 Transaction History")

database_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "database",
        "fraud_detection.db"
    )
)

connection = sqlite3.connect(database_path)

history_query = """
SELECT
    id,
    time,
    amount,
    isolation_forest_score,
    autoencoder_error,
    status
FROM transactions
ORDER BY id DESC
"""

history_df = pd.read_sql_query(
    history_query,
    connection
)

connection.close()

if history_df.empty:

    st.info("No transactions have been analyzed yet.")

else:

    # Dashboard statistics
    total_transactions = len(history_df)

    normal_transactions = (
        history_df["status"] == "Normal"
    ).sum()

    suspicious_transactions = (
        history_df["status"] == "Potentially Suspicious"
    ).sum()

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Transactions",
            total_transactions
        )

    with col2:
        st.metric(
            "Normal Transactions",
            normal_transactions
        )

    with col3:
        st.metric(
            "Potentially Suspicious",
            suspicious_transactions
        )
    # Risk distribution chart
    st.subheader("📈 Transaction Risk Distribution")

    chart_data = pd.DataFrame({
        "Status": [
            "Normal",
            "Potentially Suspicious"
        ],
        "Count": [
            normal_transactions,
            suspicious_transactions
        ]
    })

    st.bar_chart(
        chart_data.set_index("Status")
    )

    #Transaction table
    st.subheader("📋 Analyzed Transactions")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )