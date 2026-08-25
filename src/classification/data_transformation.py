import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder


# ---------------------------------------------------------
# Features used with Yeo-Johnson transformation
# ---------------------------------------------------------

YEO_JOHNSON_FEATURES = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure"
]


def transform_data(X_train, X_test, selected_features):
    """
    Reproduce the preprocessing used in the classification notebook.

    Important:
    All preprocessing objects are fitted ONLY on X_train
    and then applied to X_test.
    """

    # Make copies so original data is not modified
    X_train = X_train.copy()
    X_test = X_test.copy()

    # -----------------------------------------------------
    # 1. Numerical missing-value imputation
    # -----------------------------------------------------

    numerical_with_nan = [
        feature
        for feature in X_train.columns
        if X_train[feature].isnull().sum() > 0
        and X_train[feature].dtype != "object"
    ]

    num_imputer = SimpleImputer(strategy="median")

    if numerical_with_nan:
        X_train[numerical_with_nan] = num_imputer.fit_transform(
            X_train[numerical_with_nan]
        )

        X_test[numerical_with_nan] = num_imputer.transform(
            X_test[numerical_with_nan]
        )

    # -----------------------------------------------------
    # 2. Categorical missing-value imputation
    # -----------------------------------------------------

    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train[["education"]] = cat_imputer.fit_transform(
        X_train[["education"]]
    )

    X_test[["education"]] = cat_imputer.transform(
        X_test[["education"]]
    )

    # -----------------------------------------------------
    # 3. Square-root transformation of school_fees
    # -----------------------------------------------------

    X_train["school_fees"] = np.sqrt(
        X_train["school_fees"]
    )

    X_test["school_fees"] = np.sqrt(
        X_test["school_fees"]
    )

    # -----------------------------------------------------
    # 4. Yeo-Johnson transformation
    # -----------------------------------------------------

    pt = PowerTransformer(
        method="yeo-johnson",
        standardize=False
    )

    X_train[YEO_JOHNSON_FEATURES] = pt.fit_transform(
        X_train[YEO_JOHNSON_FEATURES]
    )

    X_test[YEO_JOHNSON_FEATURES] = pt.transform(
        X_test[YEO_JOHNSON_FEATURES]
    )

    # -----------------------------------------------------
    # 5. Identify categorical features
    # -----------------------------------------------------

    categorical_features = [
        feature
        for feature in X_train.columns
        if X_train[feature].dtype == "object"
    ]

    # -----------------------------------------------------
    # 6. One-Hot Encoding
    # -----------------------------------------------------

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded = encoder.fit_transform(
        X_train[categorical_features]
    )

    X_test_encoded = encoder.transform(
        X_test[categorical_features]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_features
    )

    X_train_encoded = pd.DataFrame(
        X_train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    # -----------------------------------------------------
    # 7. Remove original categorical columns
    # -----------------------------------------------------

    X_train = X_train.drop(
        columns=categorical_features
    )

    X_test = X_test.drop(
        columns=categorical_features
    )

    # -----------------------------------------------------
    # 8. Combine numerical + encoded features
    # -----------------------------------------------------

    X_train_final = pd.concat(
        [X_train, X_train_encoded],
        axis=1
    )

    X_test_final = pd.concat(
        [X_test, X_test_encoded],
        axis=1
    )

    # -----------------------------------------------------
    # 9. Keep the final 15 features
    # -----------------------------------------------------

    X_train_selected = X_train_final[
        selected_features
    ]

    X_test_selected = X_test_final[
        selected_features
    ]

    return (
        X_train_selected,
        X_test_selected,
        num_imputer,
        cat_imputer,
        pt,
        encoder
    )