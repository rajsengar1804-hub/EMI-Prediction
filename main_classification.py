import os

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

ENCODER_PATH = (
    "final_model/"
    "emi_classification/"
    "emi_label_encoder.pkl"
)

FEATURES_PATH = (
    "final_model/"
    "emi_classification/"
    "emi_selected_features.pkl"
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
    # 4. Separate features and target
    # -----------------------------------------------------

    X, y = split_features_target(dataset)

    # -----------------------------------------------------
    # 5. Reproduce train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y
    )

    print("Train/Test split completed.")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # -----------------------------------------------------
    # 6. Load final selected features
    # -----------------------------------------------------

    selected_features = load_selected_features(
        FEATURES_PATH
    )

    print(
        "Number of selected features:",
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
        encoder
    ) = transform_data(
        X_train,
        X_test,
        selected_features
    )

    print("Data transformation completed.")

    print(
        "Transformed X_test:",
        X_test_transformed.shape
    )

    # -----------------------------------------------------
    # 8. Load final trained model
    # -----------------------------------------------------

    model = load_model(MODEL_PATH)

    # Load label encoder
    label_encoder = load_label_encoder(
        ENCODER_PATH
    )

    print("Final XGBoost model loaded.")

    # -----------------------------------------------------
    # 9. Start MLflow run
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
            "n_estimators",
            model_params.get("n_estimators")
        )

        mlflow.log_param(
            "random_state",
            model_params.get("random_state")
        )

        mlflow.log_param(
            "eval_metric",
            model_params.get("eval_metric")
        )

        mlflow.log_param(
            "objective",
            model_params.get("objective")
        )

        mlflow.log_param(
            "n_jobs",
            model_params.get("n_jobs")
        )

        # -------------------------------------------------
        # Data / preprocessing parameters
        # -------------------------------------------------

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "split_random_state",
            42
        )

        mlflow.log_param(
            "stratify",
            True
        )

        mlflow.log_param(
            "number_of_selected_features",
            len(selected_features)
        )

        mlflow.log_param(
            "feature_selection_method",
            "Pre-selected features from Kaggle"
        )

        mlflow.log_param(
            "imbalance_method",
            "sample_weight"
        )

        # =================================================
        # PREDICTION
        # =================================================

        y_pred_encoded = model.predict(
            X_test_transformed
        )

        # Convert predictions to original labels
        y_pred = label_encoder.inverse_transform(
            y_pred_encoded.astype(int)
        )

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
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

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred
        )

        # -------------------------------------------------
        # Log metrics
        # -------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "macro_precision",
            precision
        )

        mlflow.log_metric(
            "macro_recall",
            recall
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

        print("\n==============================")
        print("CLASSIFICATION RESULTS")
        print("==============================")

        print(
            "Accuracy:",
            accuracy
        )

        print(
            "Macro Precision:",
            precision
        )

        print(
            "Macro Recall:",
            recall
        )

        print(
            "Macro F1:",
            macro_f1
        )

        print(
            "Balanced Accuracy:",
            balanced_accuracy
        )

        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        print("\nClassification Report:")
        print(report)

        report_path = os.path.join(
            ARTIFACT_DIR,
            "classification_report.txt"
        )

        with open(
            report_path,
            "w"
        ) as file:

            file.write(report)

        mlflow.log_artifact(
            report_path
        )

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=label_encoder.classes_
        )

        print("\nConfusion Matrix:")
        print(cm)

        cm_path = os.path.join(
            ARTIFACT_DIR,
            "confusion_matrix.txt"
        )

        with open(
            cm_path,
            "w"
        ) as file:

            file.write(
                str(cm)
            )

        mlflow.log_artifact(
            cm_path
        )

        # =================================================
        # SELECTED FEATURES ARTIFACT
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

        run_id = mlflow.active_run().info.run_id

        print("\n==============================")
        print("MLFLOW RUN")
        print("==============================")

        print(
            "Run ID:",
            run_id
        )

        print(
            "Tracking URI:",
            mlflow.get_tracking_uri()
        )

        print(
            "MLflow run completed successfully!"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()