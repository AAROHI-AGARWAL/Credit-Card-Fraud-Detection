import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the cleaned credit card dataset.
    """
    df = pd.read_csv(file_path)
    return df


def prepare_data(df):
    """
    Separate features and target variable.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Hour was created only for EDA, so it is not used by the model
    if "Hour" in X.columns:
        X = X.drop("Hour", axis=1)

    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler.
    Scaler is fitted only on training data.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def get_normal_training_data(X_train_scaled, y_train):
    """
    Select only legitimate transactions for anomaly detection training.
    Class 0 = legitimate transaction.
    """
    X_train_normal = X_train_scaled[y_train.values == 0]

    return X_train_normal