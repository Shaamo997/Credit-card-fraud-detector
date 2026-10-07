# Credit Card Fraud Detector

An end-to-end machine learning project that detects fraudulent credit card transactions. Built to address the core challenge of **extreme class imbalance** — only 0.17% of transactions in the dataset are fraud — and deployed as a REST API inside a Docker container.

---

## The Problem

A naive model trained on this dataset achieves 99.9% accuracy by simply predicting "legitimate" for every transaction. That model misses 31% of actual fraud. This project demonstrates why accuracy is the wrong metric for fraud detection and how to build a model that actually catches fraud.

---

## Results

### Model Comparison

| Model                          | Precision | Recall   | F1 Score | AUC-PR   |
| ------------------------------ | --------- | -------- | -------- | -------- |
| Logistic Regression (baseline) | 0.06      | 0.92     | 0.11     | 0.72     |
| Random Forest                  | 0.87      | 0.83     | 0.85     | 0.87     |
| **XGBoost (final)**            | **0.72**  | **0.85** | **0.79** | **0.84** |

XGBoost was selected as the final model for its superior recall (0.85) — in fraud detection, catching more fraud is the primary objective. Missing a fraudulent charge costs the customer directly; a false alarm is an inconvenience.

### Before vs After SMOTE

Without SMOTE, Logistic Regression catches 69% of fraud with high precision but misses 31% of actual fraud cases. After applying SMOTE to balance the training data, recall jumps to 92% — the model catches significantly more fraud, demonstrating the impact of handling class imbalance correctly.

---

## Dataset

**Kaggle — Credit Card Fraud Detection (ULB)**

- 284,807 transactions over two days
- 492 fraud cases (0.17% of total)
- 30 features: `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class`

The V1–V28 features are the result of PCA transformation applied by the dataset creators to protect cardholder privacy. The mathematical patterns between these components are what the model learns to identify as fraud.

---

## Project Architecture

Raw Data
│
▼
EDA & Class Imbalance Analysis
│
▼
StandardScaler → SMOTE → XGBoost (imblearn Pipeline)
│
▼
SHAP Explainability
│
▼
FastAPI /predict endpoint
│
▼
Docker Container

---

## Model Explainability (SHAP)

The model doesn't just predict fraud — it explains why. SHAP (SHapley Additive exPlanations) assigns each feature a value showing how much it pushed a prediction towards or away from fraud.

**Top 10 features by mean SHAP value:**

| Feature | Mean SHAP Value |
| ------- | --------------- |
| V4      | 3.63            |
| V14     | 3.56            |
| V1      | 2.25            |
| V12     | 2.11            |
| V3      | 2.04            |
| V8      | 1.73            |
| V11     | 1.51            |
| V18     | 1.32            |
| V22     | 1.23            |
| V10     | 1.13            |

V4 and V14 are the strongest fraud signals. This is auditable — every prediction can be traced back to which features drove it, which is critical in financial applications.

---

## Tech Stack

| Tool                 | Purpose                           |
| -------------------- | --------------------------------- |
| pandas               | Data loading and manipulation     |
| seaborn / matplotlib | Visualization                     |
| scikit-learn         | Model training, metrics, pipeline |
| imbalanced-learn     | SMOTE, imblearn Pipeline          |
| XGBoost              | Final fraud detection model       |
| SHAP                 | Model explainability              |
| MLflow               | Experiment tracking               |
| FastAPI + Uvicorn    | REST API serving                  |
| Docker               | Containerization                  |

---

## Project Structure

credit-card-fraud-detector/
├── data/ # Dataset (gitignored)
├── notebooks/
│ ├── eda.ipynb # Exploratory data analysis
│ ├── baseline_model.ipynb # Naive model — accuracy trap
│ ├── smote.ipynb # SMOTE before/after comparison
│ ├── pipeline.ipynb # Leak-free imblearn pipeline
│ ├── model_comparison.ipynb # Three model comparison
│ ├── tuning.ipynb # XGBoost hyperparameter tuning
│ ├── shap_explanation.ipynb # SHAP explainability
│ └── mlflow_tracking.ipynb # Experiment tracking
├── src/
│ └── main.py # FastAPI application
├── models/
│ └── fraud_detector.pkl # Saved final model
├── Dockerfile
├── requirements.txt
└── README.md

---

## Key Technical Decisions

**Why SMOTE over random oversampling?**
SMOTE generates synthetic fraud examples by interpolating between existing fraud cases rather than duplicating them. This produces more diverse training data and avoids overfitting to the exact same examples.

**Why imblearn Pipeline over sklearn Pipeline?**
sklearn's Pipeline cannot handle SMOTE because SMOTE changes the number of rows in the training data. imbalanced-learn's Pipeline wrapper handles this correctly and prevents data leakage by ensuring SMOTE is only applied to training folds during cross-validation.

**Why AUC-PR over ROC-AUC?**
ROC-AUC can look artificially strong on imbalanced datasets because it factors in true negatives, which are abundant when one class dominates. Precision-Recall AUC focuses only on the positive class (fraud) and gives a more honest picture of model performance.

**Why XGBoost over Random Forest?**
Random Forest achieved a slightly higher F1 score (0.85 vs 0.79), but XGBoost's superior recall (0.85 vs 0.83) means it catches more actual fraud. In production fraud detection, missing fraud is more costly than a false alarm, so recall is the priority metric.

---

## Running Locally

**Prerequisites:** Python 3.10+, pip

```bash
# clone the repo
git clone https://github.com/Shaamo997/Credit-card-fraud-detector.git
cd Credit-card-fraud-detector

# set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# download dataset from Kaggle and place in data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# run the API
cd src
uvicorn main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

---

## Running with Docker

```bash
# build the image
docker build -t fraud-detector .

# run the container
docker run -p 8000:8000 fraud-detector
```

API docs available at `http://127.0.0.1:8000/docs`

---

## API Usage

**Single prediction:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [<30 feature values>]}'
```

**Response:**

```json
{
  "prediction": 1,
  "label": "FRAUD",
  "fraud_probability": 0.9823
}
```

**Batch prediction:** `POST /predict/batch` accepts a list of transactions and returns predictions for all of them.

---

## Experiment Tracking

All model runs were tracked with MLflow — parameters, metrics, and model artifacts logged for every experiment. To view:

```bash
cd notebooks
mlflow ui
```

Open `http://127.0.0.1:5000` to browse all runs.
