import os
import joblib

import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from src.classification.data_ingestion import (
    load_data,
    split_features_target,
    train_test_split_data
)

from src.classification.data_transformation import (
    transform_data
)

from src.classification.model_loader import (
    load_model,
    load_label_encoder,
    load_selected_features
)

from mlflow_config import setup_mlflow


# =========================================================
# Paths
# =========================================================

DATA_PATH = "dataset.csv"

MODEL_PATH = (
    "final_model/"
    "emi_classification/"
    "emi_xgboost_balanced.pkl"
)

LABEL_ENCODER_PATH = (
    "final_model/"
    "emi_classification/"
    "emi_label_encoder.pkl"
)

FEATURES_PATH = (
    "final_model/"
    "emi_classification/"
    "emi_selected_features.pkl"
)

PREPROCESSING_PATH = (
    "final_model/"
    "emi_classification/"
    "preprocessing.pkl"
)

ARTIFACT_DIR = "artifacts/classification"


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

    print("\nTrain/Test split completed.")

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
        X_test,
        selected_features
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
        "Classification preprocessing saved:",
        PREPROCESSING_PATH
    )

    # -----------------------------------------------------
    # 9. Load final XGBoost model
    # -----------------------------------------------------

    model = load_model(
        MODEL_PATH
    )

    print(
        "\nFinal XGBoost model loaded."
    )

    # -----------------------------------------------------
    # 10. Load label encoder
    # -----------------------------------------------------

    label_encoder = load_label_encoder(
        LABEL_ENCODER_PATH
    )

    print(
        "Label encoder loaded."
    )

    # -----------------------------------------------------
    # 11. Start MLflow run
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="final_xgboost_classification"
    ):

        # =================================================
        # MODEL PARAMETERS
        # =================================================

        model_params = model.get_params()

        mlflow.log_param(
            "model_type",
            "XGBoost"
        )

        mlflow.log_param(
            "objective",
            model_params.get(
                "objective"
            )
        )

        mlflow.log_param(
            "eval_metric",
            model_params.get(
                "eval_metric"
            )
        )

        mlflow.log_param(
            "n_estimators",
            model_params.get(
                "n_estimators"
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

        y_pred_encoded = model.predict(
            X_test_transformed
        )

        # -------------------------------------------------
        # Convert predictions to original labels
        # -------------------------------------------------

        y_pred = label_encoder.inverse_transform(
            y_pred_encoded.astype(int)
        )

        # =================================================
        # CLASSIFICATION METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        macro_precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )

        # =================================================
        # LOG METRICS
        # =================================================

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "macro_precision",
            macro_precision
        )

        mlflow.log_metric(
            "macro_recall",
            macro_recall
        )

        mlflow.log_metric(
            "macro_f1",
            macro_f1
        )

        mlflow.log_metric(
            "balanced_accuracy",
            balanced_accuracy
        )

        # =================================================
        # PRINT RESULTS
        # =================================================

        print(
            "\n=============================="
        )

        print(
            "CLASSIFICATION RESULTS"
        )

        print(
            "=============================="
        )

        print(
            "Accuracy:",
            accuracy
        )

        print(
            "Macro Precision:",
            macro_precision
        )

        print(
            "Macro Recall:",
            macro_recall
        )

        print(
            "Macro F1:",
            macro_f1
        )

        print(
            "Balanced Accuracy:",
            balanced_accuracy
        )

        print(
            "\nClassification Report:"
        )

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        print(report)

        print(
            "\nConfusion Matrix:"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print(cm)

        # =================================================
        # SAVE METRICS ARTIFACT
        # =================================================

        metrics_path = os.path.join(
            ARTIFACT_DIR,
            "classification_metrics.txt"
        )

        with open(
            metrics_path,
            "w"
        ) as file:

            file.write(
                f"Accuracy: {accuracy}\n"
            )

            file.write(
                f"Macro Precision: {macro_precision}\n"
            )

            file.write(
                f"Macro Recall: {macro_recall}\n"
            )

            file.write(
                f"Macro F1: {macro_f1}\n"
            )

            file.write(
                f"Balanced Accuracy: {balanced_accuracy}\n"
            )

            file.write(
                "\nClassification Report:\n"
            )

            file.write(report)

            file.write(
                "\nConfusion Matrix:\n"
            )

            file.write(
                str(cm)
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
        # LOG FINAL XGBOOST MODEL
        # =================================================

        mlflow.xgboost.log_model(
            model,
            name="final_xgboost_model"
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
            "\nMLflow classification run "
            "completed successfully!"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()