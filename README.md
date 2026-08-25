# EMI Prediction — Machine Learning & MLOps

### 🔗 Project Links

🚀 **Live Streamlit Application:**  
https://emi-prediction-engxqceeljynyggeypcyqb.streamlit.app/

💻 **GitHub Repository:**  
https://github.com/rajsengar1804/emi-prediction

📊 **DagsHub MLflow Tracking:**  
https://dagshub.com/rajsengar1804/emi-prediction.mlflow

---

## 📌 Project Overview

A machine learning and MLOps project for predicting EMI (Equated Monthly Instalment) eligibility and maximum affordable monthly EMI.

The project contains two machine learning tasks:

1. **EMI Eligibility Classification**
2. **Maximum Monthly EMI Regression**

The final models were trained in **Kaggle** because of the computational requirements. The trained models were then saved and integrated into a local Python-based ML pipeline.

The project uses:

- **XGBoost** for classification
- **LightGBM** for regression
- **MLflow** for experiment tracking
- **DagsHub** for remote MLflow tracking
- **Streamlit** for deployment
- **GitHub** for version control

---

# 🎯 Objectives

The objective of this project is to use customer financial and demographic information to:

- Predict whether a customer is **Eligible**, **High_Risk**, or **Not_Eligible** for EMI.
- Predict the customer's **maximum monthly EMI amount**.

The complete project follows an end-to-end machine learning and MLOps workflow.

---

# 🧠 Machine Learning Tasks

## 1. EMI Eligibility Classification

The classification model predicts one of three classes:

- `Eligible`
- `High_Risk`
- `Not_Eligible`

### Final Model

**XGBoost Classifier**

The final XGBoost model was trained in Kaggle and saved locally.

The local pipeline loads the already-trained model instead of retraining it.

### Classification Selected Features

The final classification model uses 15 selected features:

```text
requested_amount
monthly_salary
requested_tenure
current_emi_amount
bank_balance
groceries_utilities
credit_score
travel_expenses
other_monthly_expenses
school_fees
monthly_rent
college_fees
years_of_employment
family_size
house_type_Own
```

### Classification Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9155 |
| Macro Precision | 0.7560 |
| Macro Recall | 0.8763 |
| Macro F1 | 0.7815 |
| Balanced Accuracy | 0.8763 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Eligible | 0.94 | 0.89 | 0.92 |
| High_Risk | 0.33 | 0.81 | 0.47 |
| Not_Eligible | 1.00 | 0.93 | 0.96 |

---

# 2. Maximum Monthly EMI Regression

The regression model predicts:

```text
max_monthly_emi
```

This represents the maximum monthly EMI amount that the customer can afford based on the available financial and demographic information.

### Final Model

**LightGBM Regressor**

The final LightGBM model was trained in Kaggle and saved locally.

The local pipeline loads the already-trained model instead of retraining it.

### Regression Selected Features

The final regression model uses 15 selected features:

```text
monthly_salary
current_emi_amount
monthly_rent
college_fees
school_fees
groceries_utilities
credit_score
bank_balance
other_monthly_expenses
travel_expenses
years_of_employment
house_type_Own
requested_amount
requested_tenure
employment_type_Private
```

### Regression Performance

| Metric | Score |
|---|---:|
| MAE | 592.68 |
| MSE | 1,127,265.11 |
| RMSE | 1,061.73 |
| R² Score | 0.9809 |

---

# 🔬 Data Preprocessing

The preprocessing pipeline was developed separately for classification and regression.

The main preprocessing steps include:

### 1. Missing Value Imputation

Numerical missing values are handled using:

```text
Median Imputation
```

Categorical missing values are handled using:

```text
Most Frequent Imputation
```

### 2. Yeo-Johnson Transformation

Several numerical features are transformed using the **Yeo-Johnson transformation**.

The transformation is fitted on the training data and then applied to the test data.

### 3. Square Root Transformation

The `school_fees` feature is transformed using a square-root transformation.

### 4. One-Hot Encoding

Categorical variables are converted into numerical features using:

```text
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

### 5. Feature Selection

After preprocessing and feature engineering, the final selected features are passed to the trained models.

The preprocessing objects are saved and reused during inference so that the deployed application uses the same transformations as the model pipeline.

---

# 🔄 Machine Learning Workflow

The overall workflow is:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Ingestion
     ↓
Train/Test Split
     ↓
Data Transformation
     ↓
Feature Selection
     ↓
Model Training in Kaggle
     ↓
Final Model Saved
     ↓
Final Model Loaded Locally
     ↓
Prediction
     ↓
Model Evaluation
     ↓
MLflow Tracking
     ↓
DagsHub
     ↓
Streamlit Application
     ↓
Cloud Deployment
```

---

# 🤖 Model Training

The final models were trained in **Kaggle** because the dataset contains approximately **404,800 records** and model training required more computational resources than available locally.

### Classification

```text
XGBoost Classifier
```

### Regression

```text
LightGBM Regressor
```

The local machine does not retrain these final models.

Instead, the workflow is:

```text
Kaggle
   ↓
Train Final Model
   ↓
Save Model
   ↓
Local Project
   ↓
Load Final Model
   ↓
Evaluate Model
   ↓
MLflow Tracking
   ↓
Streamlit Deployment
```

---

# 📊 Dataset

The dataset contains:

```text
404,800 records
27 columns
```

The data contains customer financial, employment, demographic, and EMI-related information.

The dataset is not included in the GitHub repository because it is excluded through `.gitignore`.

---

# 📈 MLflow Experiment Tracking

MLflow is used to track the model evaluation runs.

The pipeline tracks:

- Model parameters
- Evaluation metrics
- Model information
- Run information
- Experiment information

### Classification Run

```text
final_xgboost_classification
```

The classification pipeline logs:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Balanced Accuracy

### Regression Run

```text
final_lightgbm_regression
```

The regression pipeline logs:

- MAE
- MSE
- RMSE
- R² Score

---

# 🧪 DagsHub Integration

DagsHub is used as the remote MLflow tracking server.

The project is connected to:

```text
rajsengar1804/emi-prediction
```

MLflow Tracking URI:

```text
https://dagshub.com/rajsengar1804/emi-prediction.mlflow
```

This allows experiments and model evaluation results to be tracked remotely.

---

# 🌐 Streamlit Application

The trained models are integrated into a Streamlit application.

The application provides two prediction tasks:

### Classification

Predicts the EMI eligibility category:

```text
Eligible
High_Risk
Not_Eligible
```

### Regression

Predicts:

```text
Maximum Monthly EMI
```

The application uses the saved preprocessing objects and final trained models.

No model retraining is performed during prediction.

---

# 🚀 Deployment

The Streamlit application has been deployed using **Streamlit Community Cloud**.

### Live Application

👉 https://emi-prediction-engxqceeljynyggeypcyqb.streamlit.app/

The deployed application provides both:

- EMI Eligibility Classification
- Maximum Monthly EMI Regression

---

# 📂 Project Structure

```text
EMI-Prediction/
│
├── Notebook/
│   ├── EDA.ipynb
│   ├── model_classification.ipynb
│   └── model_regression.ipynb
│
├── final_model/
│   │
│   ├── emi_classification/
│   │   ├── emi_xgboost_balanced.pkl
│   │   ├── emi_label_encoder.pkl
│   │   ├── emi_selected_features.pkl
│   │   └── preprocessing.pkl
│   │
│   └── emi_regression/
│       ├── final_lightgbm_model.joblib
│       ├── selected_features.joblib
│       └── preprocessing.pkl
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
├── artifacts/
│   ├── classification/
│   └── regression/
│
├── Streamlit_app.py
├── main_classification.py
├── main_regression.py
├── mlflow_config.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📁 Folder & File Description

| Folder/File | Description |
|---|---|
| `Notebook/` | Contains EDA and model development notebooks |
| `final_model/` | Contains final trained models and preprocessing objects |
| `src/` | Contains reusable ML pipeline modules |
| `src/classification/` | Classification data ingestion, transformation, model loading and training |
| `src/regression/` | Regression data ingestion, transformation and model loading |
| `artifacts/` | Stores generated pipeline artifacts |
| `Streamlit_app.py` | Streamlit application for both prediction tasks |
| `main_classification.py` | Runs classification evaluation and MLflow tracking |
| `main_regression.py` | Runs regression evaluation and MLflow tracking |
| `mlflow_config.py` | MLflow and DagsHub configuration |
| `requirements.txt` | Project dependencies |
| `.gitignore` | Files excluded from Git |
| `README.md` | Project documentation |

---

# 🛠️ Technologies Used

## Programming Language

- Python 3.12

## Data Processing

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Imbalanced-learn

## MLOps

- MLflow
- DagsHub
- Joblib

## Deployment

- Streamlit
- Streamlit Community Cloud

## Version Control

- Git
- GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/rajsengar1804/emi-prediction.git
```

Navigate to the project directory:

```bash
cd emi-prediction
```

Create and activate a virtual environment.

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the ML Pipelines

## Classification

```bash
python main_classification.py
```

This pipeline:

1. Loads the dataset.
2. Performs the train/test split.
3. Applies preprocessing.
4. Loads the final XGBoost model.
5. Generates predictions.
6. Calculates evaluation metrics.
7. Logs results to MLflow/DagsHub.

---

## Regression

```bash
python main_regression.py
```

This pipeline:

1. Loads the dataset.
2. Performs the train/test split.
3. Applies preprocessing.
4. Loads the final LightGBM model.
5. Generates predictions.
6. Calculates evaluation metrics.
7. Logs results to MLflow/DagsHub.

---

# 🌐 Run Streamlit Locally

Run:

```bash
streamlit run Streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🔗 Important Project Links

| Resource | Link |
|---|---|
| 🚀 Live Streamlit App | https://emi-prediction-engxqceeljynyggeypcyqb.streamlit.app/ |
| 💻 GitHub Repository | https://github.com/rajsengar1804/emi-prediction |
| 📊 DagsHub MLflow | https://dagshub.com/rajsengar1804/emi-prediction.mlflow |

---

# 🔮 Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Improved handling of class imbalance
- SHAP-based model explainability
- Automated model retraining
- CI/CD pipeline
- Model versioning
- Automated data validation
- Model monitoring
- Data drift detection
- Improved Streamlit UI
- Cloud-based model serving

---

# 👨‍💻 Author

**Raj Pratap Singh Sengar**

Machine Learning & MLOps Project

---

# ⭐ Project Summary

This project demonstrates an end-to-end machine learning and MLOps workflow:

```text
EDA
 ↓
Feature Engineering
 ↓
Feature Selection
 ↓
Model Training in Kaggle
 ↓
Final Model Serialization
 ↓
Python ML Pipeline
 ↓
MLflow
 ↓
DagsHub
 ↓
GitHub
 ↓
Streamlit
 ↓
Cloud Deployment
```

The final deployed system provides:

**XGBoost Classification**  
→ EMI Eligibility Prediction

**LightGBM Regression**  
→ Maximum Monthly EMI Prediction

### 🚀 Try the deployed application:

https://emi-prediction-engxqceeljynyggeypcyqb.streamlit.app/