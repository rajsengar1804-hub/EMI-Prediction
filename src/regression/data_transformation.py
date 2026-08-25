import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder


# =========================================================
# Features transformed using Yeo-Johnson
# =========================================================

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


# =========================================================
# Feature transformed using Square Root
# =========================================================

SQRT_FEATURES = [
    "school_fees"
]


# =========================================================
# Final 15 features selected in the notebook
# =========================================================

FINAL_SELECTED_FEATURES = [
    "monthly_salary",
    "current_emi_amount",
    "monthly_rent",
    "college_fees",
    "school_fees",
    "groceries_utilities",
    "credit_score",
    "bank_balance",
    "other_monthly_expenses",
    "travel_expenses",
    "years_of_employment",
    "house_type_Own",
    "requested_amount",
    "requested_tenure",
    "employment_type_Private"
]


def transform_data(X_train, X_test):
    """
    Reproduce the preprocessing used in the
    regression notebook.

    All preprocessing objects are fitted only
    on X_train and then applied to X_test.
    """

    # -----------------------------------------------------
    # Make copies
    # -----------------------------------------------------

    X_train = X_train.copy()
    X_test = X_test.copy()

    # -----------------------------------------------------
    # 1. Numerical missing-value imputation
    # -----------------------------------------------------

    numerical_with_nan = [
        feature
        for feature in X_train.columns
        if X_train[feature].isnull().sum() > 1
        and X_train[feature].dtype != "object"
    ]

    num_imputer = SimpleImputer(
        strategy="median"
    )

    if numerical_with_nan:

        X_train[numerical_with_nan] = (
            num_imputer.fit_transform(
                X_train[numerical_with_nan]
            )
        )

        X_test[numerical_with_nan] = (
            num_imputer.transform(
                X_test[numerical_with_nan]
            )
        )

    # -----------------------------------------------------
    # 2. Categorical missing-value imputation
    # -----------------------------------------------------

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    cat_imputer.fit(
        X_train[["education"]]
    )

    X_train[["education"]] = (
        cat_imputer.transform(
            X_train[["education"]]
        )
    )

    X_test[["education"]] = (
        cat_imputer.transform(
            X_test[["education"]]
        )
    )

    # -----------------------------------------------------
    # 3. Yeo-Johnson transformation
    # -----------------------------------------------------

    pt = PowerTransformer(
        method="yeo-johnson",
        standardize=False
    )

    X_train[YEO_JOHNSON_FEATURES] = (
        pt.fit_transform(
            X_train[YEO_JOHNSON_FEATURES]
        )
    )

    X_test[YEO_JOHNSON_FEATURES] = (
        pt.transform(
            X_test[YEO_JOHNSON_FEATURES]
        )
    )

    # -----------------------------------------------------
    # 4. Square-root transformation
    # -----------------------------------------------------

    for feature in SQRT_FEATURES:

        X_train[feature] = np.sqrt(
            X_train[feature]
        )

        X_test[feature] = np.sqrt(
            X_test[feature]
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

    encoded_columns = (
        encoder.get_feature_names_out(
            categorical_features
        )
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

    X_train_num = X_train.drop(
        columns=categorical_features
    )

    X_test_num = X_test.drop(
        columns=categorical_features
    )

    # -----------------------------------------------------
    # 8. Combine numerical + encoded features
    # -----------------------------------------------------

    X_train_final = pd.concat(
        [
            X_train_num,
            X_train_encoded
        ],
        axis=1
    )

    X_test_final = pd.concat(
        [
            X_test_num,
            X_test_encoded
        ],
        axis=1
    )

    # -----------------------------------------------------
    # 9. Remove constant features
    # -----------------------------------------------------

    constant_features = [
        column
        for column in X_train_final.columns
        if X_train_final[column].nunique() <= 1
    ]

    X_train_v1 = X_train_final.drop(
        columns=constant_features
    )

    X_test_v1 = X_test_final.drop(
        columns=constant_features
    )

    # -----------------------------------------------------
    # 10. Remove highly correlated features
    # -----------------------------------------------------

    corr_matrix = X_train_v1.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    CORR_THRESHOLD = 0.90

    high_corr_features = [
        column
        for column in upper_triangle.columns
        if any(
            upper_triangle[column] >
            CORR_THRESHOLD
        )
    ]

    X_train_v2 = X_train_v1.drop(
        columns=high_corr_features
    )

    X_test_v2 = X_test_v1.drop(
        columns=high_corr_features
    )

    # -----------------------------------------------------
    # 11. Select final 15 features
    # -----------------------------------------------------

    X_train_selected = X_train_v2[
        FINAL_SELECTED_FEATURES
    ]

    X_test_selected = X_test_v2[
        FINAL_SELECTED_FEATURES
    ]

    return (
    X_train_selected,
    X_test_selected,
    num_imputer,
    cat_imputer,
    pt,
    encoder,
    numerical_with_nan,
    categorical_features
)
def transform_new_data(
    X,
    num_imputer,
    cat_imputer,
    pt,
    encoder,
    numerical_with_nan,
    categorical_features
):
    """
    Transform new/unseen customer data using
    preprocessing objects fitted during training.
    """

    X = X.copy()

    # -----------------------------------------------------
    # 1. Numerical missing-value imputation
    # -----------------------------------------------------

    if numerical_with_nan:
        X[numerical_with_nan] = num_imputer.transform(
            X[numerical_with_nan]
        )

    # -----------------------------------------------------
    # 2. Categorical missing-value imputation
    # -----------------------------------------------------

    X[["education"]] = cat_imputer.transform(
        X[["education"]]
    )

    # -----------------------------------------------------
    # 3. Yeo-Johnson transformation
    # -----------------------------------------------------

    X[YEO_JOHNSON_FEATURES] = pt.transform(
        X[YEO_JOHNSON_FEATURES]
    )

    # -----------------------------------------------------
    # 4. Square-root transformation
    # -----------------------------------------------------

    for feature in SQRT_FEATURES:
        X[feature] = np.sqrt(
            X[feature]
        )

    # -----------------------------------------------------
    # 5. One-Hot Encoding
    # -----------------------------------------------------

    X_encoded = encoder.transform(
        X[categorical_features]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_features
    )

    X_encoded = pd.DataFrame(
        X_encoded,
        columns=encoded_columns,
        index=X.index
    )

    # -----------------------------------------------------
    # 6. Remove original categorical columns
    # -----------------------------------------------------

    X = X.drop(
        columns=categorical_features
    )

    # -----------------------------------------------------
    # 7. Combine numerical + encoded features
    # -----------------------------------------------------

    X_final = pd.concat(
        [X, X_encoded],
        axis=1
    )

    # -----------------------------------------------------
    # 8. Select final 15 features
    # -----------------------------------------------------

    X_selected = X_final[
        FINAL_SELECTED_FEATURES
    ]

    return X_selected