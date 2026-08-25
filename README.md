# EMI Prediction — Machine Learning & MLOps

A machine learning project for EMI (Equated Monthly Instalment) prediction.

The project contains two machine learning tasks:

1. **EMI Eligibility Classification**
2. **Maximum Monthly EMI Regression**

The models were trained separately and the final trained models are integrated into an MLflow-based tracking pipeline using DagsHub.

---

## 📌 Project Overview

The objective of this project is to use customer financial and demographic information to:

- Predict whether a customer is **Eligible**, **High_Risk**, or **Not_Eligible** for EMI.
- Predict the customer's **maximum monthly EMI amount**.

The final models were trained separately and saved locally. The Python pipeline loads these already-trained models, performs the required preprocessing, generates predictions, evaluates the models, and logs the results using **MLflow**.

---

## 🧠 Machine Learning Tasks

### 1. EMI Eligibility Classification

The classification problem predicts one of three classes:

- `Eligible`
- `High_Risk`
- `Not_Eligible`

### Final Model

**XGBoost Classifier**

The final XGBoost model was trained in Kaggle and saved locally. The model is loaded during the MLflow pipeline instead of being retrained on the local machine.

### Classification Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9155 |
| Macro Precision | 0.7560 |
| Macro Recall | 0.8763 |
| Macro F1 | 0.7815 |
| Balanced Accuracy | 0.8763 |

---

### 2. Maximum Monthly EMI Regression

The regression problem predicts:

`max_monthly_emi`

### Final Model

**LightGBM Regressor**

The final LightGBM model was trained in Kaggle and saved locally. The model is loaded during the MLflow pipeline.

### Regression Performance

| Metric | Score |
|---|---:|
| MAE | 592.68 |
| MSE | 1,127,265.11 |
| RMSE | 1,061.73 |
| R² Score | 0.9809 |

---

# 🔄 Machine Learning Workflow

The overall workflow is:

```text
Raw Dataset
     ↓
Data Ingestion
     ↓
Train/Test Split
     ↓
Data Transformation
     ↓
Feature Selection
     ↓
Load Final Trained Model
     ↓
Prediction
     ↓
Model Evaluation
     ↓
MLflow Tracking
     ↓
DagsHub


### 📂 Folder Description

EMI-Prediction/
│
├── Notebook/
│   ├── EDA.ipynb
│   ├── model_classification.ipynb
│   └── model_regression.ipynb
│
├── final_model/
│   ├── emi_classification/
│   │   ├── final_xgboost_model.pkl
│   │   ├── label_encoder.pkl
│   │   └── selected_features.pkl
│   │
│   └── emi_regression/
│       ├── final_lightgbm_model.joblib
│       └── selected_features.joblib
│
├── src/
│   ├── __init__.py
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_loader.py
│   │   └── model_training.py
│   │
│   └── regression/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_transformation.py
│       └── model_loader.py
│
├── .gitignore
├── README.md
├── main_classification.py
├── main_regression.py
├── mlflow_config.py
└── requirements.txt