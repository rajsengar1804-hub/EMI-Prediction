import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """
    Load the raw EMI dataset.
    """
    dataset = pd.read_csv(data_path)

    return dataset


def split_features_target(dataset):
    """
    Separate features (X) and target (y)
    using the same logic as the regression notebook.
    """

    X = dataset.drop(
        columns=[
            "emi_eligibility",
            "max_monthly_emi"
        ],
        errors="ignore"
    )

    y = dataset["max_monthly_emi"]

    return X, y


def train_test_split_data(X, y):
    """
    Reproduce the exact train/test split
    used in the regression notebook.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test