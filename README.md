# AI for Finance: Credit Card Default Prediction

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/webforworld2512/AI-for-Finance/HEAD)

Predict whether a credit card customer will default on their next payment using XGBoost — trained locally and deployed at scale with AWS SageMaker AutoML.

---

## Problem Statement

Credit card default is a major risk for financial institutions. This project builds a binary classification model to predict whether a customer will default on their next monthly payment, enabling proactive risk management and targeted intervention.

**Dataset:** [UCI Credit Card Default Dataset](http://archive.ics.uci.edu/ml) — 30,000 customers, 24 features including credit limit, payment history, bill amounts, and demographic information.

---

## Project Workflow

| Step | Description |
|------|-------------|
| 1 | Understand the business case and problem statement |
| 2 | Import libraries and load dataset |
| 3 | Exploratory data analysis and visualization |
| 4 | Data cleaning, encoding, and train/test split |
| 5–6 | XGBoost theory and algorithm deep-dive |
| 7 | Train and evaluate XGBoost locally |
| 8 | Hyperparameter tuning with GridSearchCV |
| 9–10 | Train XGBoost on AWS SageMaker |
| 11 | Deploy model endpoint and run inference |

---

## Key Techniques

- **Exploratory Data Analysis** — correlation heatmaps, KDE plots, count plots by age, education, sex, and marriage status
- **Feature Engineering** — One-Hot Encoding for categorical variables, Min-Max Scaling for numeric features
- **XGBoost Classifier** — gradient boosted trees for tabular classification
- **Hyperparameter Tuning** — GridSearchCV to optimize learning rate, depth, and estimators
- **AWS SageMaker** — cloud-scale training and real-time inference endpoint deployment

---

## Results

- Trained an XGBoost classifier achieving competitive accuracy on an imbalanced dataset (~22% default rate)
- Optimized via grid search and deployed as a live SageMaker endpoint for real-time predictions

---

## Run Locally

> **Note:** Tasks 9–11 require an AWS account with SageMaker access.

```bash
git clone https://github.com/webforworld2512/AI-for-Finance.git
cd AI-for-Finance
pip install -r requirements.txt
jupyter notebook AI_in_Business_AutoML.ipynb
