import pandas as pd


def add_time_features(df):
    """
    Add time-based features to the transaction dataset.
    """

    df = df.copy()

    if "Time" in df.columns:
        # Convert elapsed seconds into hour of day
        df["Hour"] = (df["Time"] // 3600) % 24

    return df


def remove_unused_features(df):
    """
    Remove features that are not used by the ML models.
    """

    df = df.copy()

    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    if "Hour" in df.columns:
        df = df.drop("Hour", axis=1)

    return df


def prepare_features(df):
    """
    Complete feature-engineering pipeline.
    """

    df = df.copy()

    # Add time feature for possible analysis/future use
    df = add_time_features(df)

    # Remove target and EDA-only feature
    df = remove_unused_features(df)

    return df