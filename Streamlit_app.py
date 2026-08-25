import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EMI Prediction System",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

CLASSIFICATION_DIR = (
    "final_model/emi_classification"
)

REGRESSION_DIR = (
    "final_model/emi_regression"
)


# =========================================================
# LOAD CLASSIFICATION FILES
# =========================================================

@st.cache_resource
def load_classification_model():

    model = joblib.load(
        f"{CLASSIFICATION_DIR}/emi_xgboost_balanced.pkl"
    )

    label_encoder = joblib.load(
        f"{CLASSIFICATION_DIR}/emi_label_encoder.pkl"
    )

    preprocessing = joblib.load(
        f"{CLASSIFICATION_DIR}/preprocessing.pkl"
    )

    return (
        model,
        label_encoder,
        preprocessing
    )


# =========================================================
# LOAD REGRESSION FILES
# =========================================================

@st.cache_resource
def load_regression_model():

    model = joblib.load(
        f"{REGRESSION_DIR}/final_lightgbm_model.joblib"
    )

    preprocessing = joblib.load(
        f"{REGRESSION_DIR}/preprocessing.pkl"
    )

    return (
        model,
        preprocessing
    )


# =========================================================
# CLASSIFICATION PREPROCESSING
# =========================================================

def preprocess_classification(
    input_data,
    preprocessing
):

    num_imputer = (
        preprocessing["num_imputer"]
    )

    cat_imputer = (
        preprocessing["cat_imputer"]
    )

    power_transformer = (
        preprocessing["power_transformer"]
    )

    encoder = (
        preprocessing["encoder"]
    )

    numerical_with_nan = (
        preprocessing["numerical_with_nan"]
    )

    categorical_features = (
        preprocessing["categorical_features"]
    )

    selected_features = (
        preprocessing["selected_features"]
    )

    X = input_data.copy()

    # -----------------------------------------------------
    # 1. Numerical missing-value imputation
    # -----------------------------------------------------

    if numerical_with_nan:

        X[numerical_with_nan] = (
            num_imputer.transform(
                X[numerical_with_nan]
            )
        )

    # -----------------------------------------------------
    # 2. Education missing-value imputation
    # -----------------------------------------------------

    X[["education"]] = (
        cat_imputer.transform(
            X[["education"]]
        )
    )

    # -----------------------------------------------------
    # 3. Square-root transformation
    # -----------------------------------------------------

    X["school_fees"] = np.sqrt(
        X["school_fees"]
    )

    # -----------------------------------------------------
    # 4. Yeo-Johnson transformation
    # -----------------------------------------------------

    yj_features = list(
        power_transformer.feature_names_in_
    )

    X[yj_features] = (
        power_transformer.transform(
            X[yj_features]
        )
    )

    # -----------------------------------------------------
    # 5. One-Hot Encoding
    # -----------------------------------------------------

    encoded = encoder.transform(
        X[categorical_features]
    )

    encoded_columns = (
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=X.index
    )

    # -----------------------------------------------------
    # 6. Remove categorical columns
    # -----------------------------------------------------

    X_numeric = X.drop(
        columns=categorical_features
    )

    # -----------------------------------------------------
    # 7. Combine
    # -----------------------------------------------------

    X_final = pd.concat(
        [
            X_numeric,
            encoded_df
        ],
        axis=1
    )

    # -----------------------------------------------------
    # 8. Select final 15 features
    # -----------------------------------------------------

    X_selected = X_final[
        selected_features
    ]

    return X_selected


# =========================================================
# REGRESSION PREPROCESSING
# =========================================================

def preprocess_regression(
    input_data,
    preprocessing
):

    num_imputer = (
        preprocessing["num_imputer"]
    )

    cat_imputer = (
        preprocessing["cat_imputer"]
    )

    power_transformer = (
        preprocessing["power_transformer"]
    )

    encoder = (
        preprocessing["encoder"]
    )

    numerical_with_nan = (
        preprocessing["numerical_with_nan"]
    )

    categorical_features = (
        preprocessing["categorical_features"]
    )

    selected_features = (
        preprocessing["selected_features"]
    )

    X = input_data.copy()

    # -----------------------------------------------------
    # 1. Numerical missing-value imputation
    # -----------------------------------------------------

    if numerical_with_nan:

        X[numerical_with_nan] = (
            num_imputer.transform(
                X[numerical_with_nan]
            )
        )

    # -----------------------------------------------------
    # 2. Education missing-value imputation
    # -----------------------------------------------------

    X[["education"]] = (
        cat_imputer.transform(
            X[["education"]]
        )
    )

    # -----------------------------------------------------
    # 3. Yeo-Johnson transformation
    # -----------------------------------------------------

    yj_features = list(
        power_transformer.feature_names_in_
    )

    X[yj_features] = (
        power_transformer.transform(
            X[yj_features]
        )
    )

    # -----------------------------------------------------
    # 4. Square-root transformation
    # -----------------------------------------------------

    X["school_fees"] = np.sqrt(
        X["school_fees"]
    )

    # -----------------------------------------------------
    # 5. One-Hot Encoding
    # -----------------------------------------------------

    encoded = encoder.transform(
        X[categorical_features]
    )

    encoded_columns = (
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=X.index
    )

    # -----------------------------------------------------
    # 6. Remove categorical columns
    # -----------------------------------------------------

    X_numeric = X.drop(
        columns=categorical_features
    )

    # -----------------------------------------------------
    # 7. Combine numerical + encoded
    # -----------------------------------------------------

    X_final = pd.concat(
        [
            X_numeric,
            encoded_df
        ],
        axis=1
    )

    # -----------------------------------------------------
    # 8. Select final 15 features
    # -----------------------------------------------------

    X_selected = X_final[
        selected_features
    ]

    return X_selected


# =========================================================
# INPUT FIELD HELPERS
# =========================================================

def number_input(
    label,
    min_value=0.0,
    max_value=None,
    value=0.0,
    step=1.0
):

    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step
    )


# =========================================================
# COMMON USER INPUTS
# =========================================================

def get_common_inputs():

    st.subheader(
        "Applicant Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=30,
            step=1
        )

        monthly_salary = number_input(
            "Monthly Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        years_of_employment = number_input(
            "Years of Employment",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        monthly_rent = number_input(
            "Monthly Rent",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        college_fees = number_input(
            "College Fees",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col2:

        school_fees = number_input(
            "School Fees",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

        travel_expenses = number_input(
            "Travel Expenses",
            min_value=0.0,
            value=3000.0,
            step=500.0
        )

        groceries_utilities = number_input(
            "Groceries & Utilities",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        other_monthly_expenses = number_input(
            "Other Monthly Expenses",
            min_value=0.0,
            value=3000.0,
            step=500.0
        )

        current_emi_amount = number_input(
            "Current EMI Amount",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col3:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700,
            step=1
        )

        bank_balance = number_input(
            "Bank Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        emergency_fund = number_input(
            "Emergency Fund",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        requested_amount = number_input(
            "Requested Loan Amount",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

        requested_tenure = st.number_input(
            "Requested Tenure (months)",
            min_value=1,
            max_value=120,
            value=60,
            step=1
        )

    st.subheader(
        "Categorical Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

    with col2:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married"
            ]
        )

    with col3:

        education = st.selectbox(
            "Education",
            [
                "High School",
                "Graduate",
                "Post Graduate",
                "Professional"
            ]
        )

    with col4:

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Self-employed",
                "Government"
            ]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        company_type = st.selectbox(
            "Company Type",
            [
                "MNC",
                "Mid-size",
                "Startup",
                "Small",
                "Other"
            ]
        )

    with col2:

        house_type = st.selectbox(
            "House Type",
            [
                "Own",
                "Rented",
                "Other"
            ]
        )

    with col3:

        existing_loans = st.selectbox(
            "Existing Loans",
            [
                "Yes",
                "No"
            ]
        )

    with col4:

        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "Personal Loan EMI",
                "Home Appliances EMI",
                "Education EMI",
                "Vehicle EMI",
                "Other"
            ]
        )

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    data = {

        "age": age,

        "monthly_salary":
            monthly_salary,

        "years_of_employment":
            years_of_employment,

        "monthly_rent":
            monthly_rent,

        "college_fees":
            college_fees,

        "school_fees":
            school_fees,

        "travel_expenses":
            travel_expenses,

        "groceries_utilities":
            groceries_utilities,

        "other_monthly_expenses":
            other_monthly_expenses,

        "current_emi_amount":
            current_emi_amount,

        "credit_score":
            credit_score,

        "bank_balance":
            bank_balance,

        "emergency_fund":
            emergency_fund,

        "requested_amount":
            requested_amount,

        "requested_tenure":
            requested_tenure,

        "gender":
            gender,

        "marital_status":
            marital_status,

        "education":
            education,

        "employment_type":
            employment_type,

        "company_type":
            company_type,

        "house_type":
            house_type,

        "existing_loans":
            existing_loans,

        "emi_scenario":
            emi_scenario,

        "family_size":
            family_size
    }

    return pd.DataFrame(
        [data]
    )


# =========================================================
# CLASSIFICATION PAGE
# =========================================================

def classification_page():

    st.header(
        "🎯 EMI Eligibility Prediction"
    )

    st.write(
        "Predict whether the applicant is "
        "Eligible, High Risk, or Not Eligible."
    )

    input_data = get_common_inputs()

    if st.button(
        "Predict EMI Eligibility",
        type="primary"
    ):

        try:

            model, label_encoder, preprocessing = (
                load_classification_model()
            )

            X = preprocess_classification(
                input_data,
                preprocessing
            )

            prediction_encoded = (
                model.predict(X)
            )

            prediction = (
                label_encoder.inverse_transform(
                    prediction_encoded.astype(int)
                )
            )[0]

            st.subheader(
                "Prediction Result"
            )

            if prediction == "Eligible":

                st.success(
                    "✅ EMI Status: Eligible"
                )

            elif prediction == "High_Risk":

                st.warning(
                    "⚠️ EMI Status: High Risk"
                )

            else:

                st.error(
                    "❌ EMI Status: Not Eligible"
                )

            # ---------------------------------------------
            # Prediction probabilities
            # ---------------------------------------------

            probabilities = (
                model.predict_proba(X)[0]
            )

            probability_df = pd.DataFrame(
                {
                    "Class":
                        label_encoder.classes_,
                    "Probability":
                        probabilities
                }
            )

            probability_df[
                "Probability"
            ] = (
                probability_df[
                    "Probability"
                ] * 100
            )

            probability_df[
                "Probability"
            ] = probability_df[
                "Probability"
            ].round(2)

            st.subheader(
                "Prediction Probabilities"
            )

            st.dataframe(
                probability_df,
                width="stretch"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# =========================================================
# REGRESSION PAGE
# =========================================================

def regression_page():

    st.header(
        "💰 Maximum Monthly EMI Prediction"
    )

    st.write(
        "Predict the maximum monthly EMI "
        "the applicant can afford."
    )

    input_data = get_common_inputs()

    if st.button(
        "Predict Maximum Monthly EMI",
        type="primary"
    ):

        try:

            model, preprocessing = (
                load_regression_model()
            )

            X = preprocess_regression(
                input_data,
                preprocessing
            )

            prediction = model.predict(X)[0]

            st.subheader(
                "Prediction Result"
            )

            st.success(
                f"Maximum Monthly EMI: "
                f"₹ {prediction:,.2f}"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# =========================================================
# MAIN APP
# =========================================================

def main():

    st.title(
        "💰 EMI Prediction System"
    )

    st.write(
        "Machine Learning based EMI "
        "Eligibility and EMI Amount Prediction"
    )

    st.divider()

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    st.sidebar.title(
        "Prediction Type"
    )

    prediction_type = st.sidebar.radio(
        "Choose a prediction:",
        [
            "EMI Eligibility Classification",
            "Maximum Monthly EMI Regression"
        ]
    )

    st.sidebar.divider()

    st.sidebar.info(
        """
        **Models used**

        Classification:
        XGBoost

        Regression:
        LightGBM
        """
    )

    # -----------------------------------------------------
    # Select page
    # -----------------------------------------------------

    if prediction_type == (
        "EMI Eligibility Classification"
    ):

        classification_page()

    else:

        regression_page()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()