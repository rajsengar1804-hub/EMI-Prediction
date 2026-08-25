import os
import joblib

import mlflow
import mlflow.lightgbm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.regression.data_ingestion import (
    load_data,
    split_features_target,
    train_test_split_data
)

from src.regression.data_transformation import (
    transform_data,
    FINAL_SELECTED_FEATURES
)

from src.regression.model_loader import (
    load_model,
    load_selected_features
)

from mlflow_config import setup_mlflow


# =========================================================
# Paths
# =========================================================

DATA_PATH = "dataset.csv"

MODEL_PATH = (
    "final_model/"
    "emi_regression/"
    "final_lightgbm_model.joblib"
)

FEATURES_PATH = (
    "final_model/"
    "emi_regression/"
    "selected_features.joblib"
)

PREPROCESSING_PATH = (
    "final_model/"
    "emi_regression/"
    "preprocessing.pkl"
)

ARTIFACT_DIR = "artifacts/regression"


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. Create artifact directory
    # -----------------------------------------------------

    os.makedirs(
        ARTIFACT_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 2. Setup MLflow + DagsHub
    # -----------------------------------------------------

    setup_mlflow()

    # -----------------------------------------------------
    # 3. Load dataset
    # -----------------------------------------------------

    dataset = load_data(DATA_PATH)

    print("Dataset loaded.")
    print("Dataset shape:", dataset.shape)

    # -----------------------------------------------------
    # 4. Separate X and y
    # -----------------------------------------------------

    X, y = split_features_target(dataset)

    # -----------------------------------------------------
    # 5. Train/Test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split_data(
            X,
            y
        )
    )

    print(
        "\nTrain/Test split completed."
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    # -----------------------------------------------------
    # 6. Load selected features
    # -----------------------------------------------------

    selected_features = (
        load_selected_features(
            FEATURES_PATH
        )
    )

    print(
        "\nNumber of selected features:",
        len(selected_features)
    )

    # -----------------------------------------------------
    # 7. Transform data
    # -----------------------------------------------------

    (
        X_train_transformed,
        X_test_transformed,
        num_imputer,
        cat_imputer,
        power_transformer,
        encoder,
        numerical_with_nan,
        categorical_features
    ) = transform_data(
        X_train,
        X_test
    )

    print(
        "\nData transformation completed."
    )

    print(
        "Transformed X_test:",
        X_test_transformed.shape
    )

    # -----------------------------------------------------
    # 8. Save preprocessing objects
    # -----------------------------------------------------

    preprocessing_bundle = {

        "num_imputer": num_imputer,

        "cat_imputer": cat_imputer,

        "power_transformer": power_transformer,

        "encoder": encoder,

        "numerical_with_nan": numerical_with_nan,

        "categorical_features": categorical_features,

        "selected_features": selected_features
    }

    joblib.dump(
        preprocessing_bundle,
        PREPROCESSING_PATH
    )

    print(
        "Regression preprocessing saved:",
        PREPROCESSING_PATH
    )

    # -----------------------------------------------------
    # 9. Load final LightGBM model
    # -----------------------------------------------------

    model = load_model(
        MODEL_PATH
    )

    print(
        "\nFinal LightGBM model loaded."
    )

    # -----------------------------------------------------
    # 10. Start MLflow run
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="final_lightgbm_regression"
    ):

        # =================================================
        # MODEL PARAMETERS
        # =================================================

        model_params = model.get_params()

        mlflow.log_param(
            "model_type",
            "LightGBM"
        )

        mlflow.log_param(
            "boosting_type",
            model_params.get(
                "boosting_type"
            )
        )

        mlflow.log_param(
            "n_estimators",
            model_params.get(
                "n_estimators"
            )
        )

        mlflow.log_param(
            "learning_rate",
            model_params.get(
                "learning_rate"
            )
        )

        mlflow.log_param(
            "max_depth",
            model_params.get(
                "max_depth"
            )
        )

        mlflow.log_param(
            "num_leaves",
            model_params.get(
                "num_leaves"
            )
        )

        mlflow.log_param(
            "min_child_samples",
            model_params.get(
                "min_child_samples"
            )
        )

        mlflow.log_param(
            "colsample_bytree",
            model_params.get(
                "colsample_bytree"
            )
        )

        mlflow.log_param(
            "reg_alpha",
            model_params.get(
                "reg_alpha"
            )
        )

        mlflow.log_param(
            "reg_lambda",
            model_params.get(
                "reg_lambda"
            )
        )

        mlflow.log_param(
            "random_state",
            model_params.get(
                "random_state"
            )
        )

        mlflow.log_param(
            "n_jobs",
            model_params.get(
                "n_jobs"
            )
        )

        # =================================================
        # DATA PARAMETERS
        # =================================================

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "split_random_state",
            42
        )

        mlflow.log_param(
            "number_of_selected_features",
            len(selected_features)
        )

        mlflow.log_param(
            "feature_selection_method",
            "Pre-selected features from Kaggle"
        )

        # =================================================
        # PREDICTION
        # =================================================

        y_pred = model.predict(
            X_test_transformed
        )

        # =================================================
        # REGRESSION METRICS
        # =================================================

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            y_pred
        )

        # =================================================
        # LOG METRICS
        # =================================================

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "mse",
            mse
        )

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "r2_score",
            r2
        )

        # =================================================
        # PRINT RESULTS
        # =================================================

        print(
            "\n=============================="
        )

        print(
            "REGRESSION RESULTS"
        )

        print(
            "=============================="
        )

        print(
            "MAE:",
            mae
        )

        print(
            "MSE:",
            mse
        )

        print(
            "RMSE:",
            rmse
        )

        print(
            "R² Score:",
            r2
        )

        # =================================================
        # SAVE METRICS ARTIFACT
        # =================================================

        metrics_path = os.path.join(
            ARTIFACT_DIR,
            "regression_metrics.txt"
        )

        with open(
            metrics_path,
            "w"
        ) as file:

            file.write(
                f"MAE: {mae}\n"
            )

            file.write(
                f"MSE: {mse}\n"
            )

            file.write(
                f"RMSE: {rmse}\n"
            )

            file.write(
                f"R2 Score: {r2}\n"
            )

        mlflow.log_artifact(
            metrics_path
        )

        # =================================================
        # SAVE SELECTED FEATURES
        # =================================================

        features_path = os.path.join(
            ARTIFACT_DIR,
            "selected_features.txt"
        )

        with open(
            features_path,
            "w"
        ) as file:

            for feature in selected_features:

                file.write(
                    feature + "\n"
                )

        mlflow.log_artifact(
            features_path
        )

        # =================================================
        # LOG FINAL LIGHTGBM MODEL
        # =================================================

        mlflow.lightgbm.log_model(
            model,
            name="final_lightgbm_model"
        )

        # =================================================
        # RUN INFORMATION
        # =================================================

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        print(
            "\n=============================="
        )

        print(
            "MLFLOW RUN"
        )

        print(
            "=============================="
        )

        print(
            "Run ID:",
            run_id
        )

        print(
            "Tracking URI:",
            mlflow.get_tracking_uri()
        )

        print(
            "\nMLflow regression run "
            "completed successfully!"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()