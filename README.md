# 🏦 German Credit Risk Prediction with Flask Web App

This project is an end-to-end machine learning solution for predicting credit risk using the **German Credit Dataset**. It includes a Flask-based web application with both **individual** and **batch** prediction functionality.

---

## 📌 Overview

Credit risk prediction is crucial for financial institutions to assess the likelihood of a customer defaulting on a loan. This project uses various machine learning models, selects the best one through evaluation metrics, and deploys it using a user-friendly web interface built with Flask.

---

## 📊 Dataset

- **Source**: [Kaggle]([https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk))
- **Records**: 1000 applicants
- **Features**: 20 attributes (mix of numerical and categorical)
- **Target**: Credit Risk (`Good` / `Bad`)

---

## 🛠 Tech Stack

- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Model**: Gradient Boosting Classifier (best performing)
- **Deployment**: Flask Web Framework
- **UI Styling**: HTML, Bootstrap 4

---

## 🔍 Exploratory Data Analysis (EDA)

- **Credit Amount**: Right-skewed with outliers
- **Age vs Risk**: Younger applicants show higher risk
- **Categorical Features**: Distribution analyzed for features like job type, account status, and purpose
- **Correlation Matrix**: Reveals inter-feature relationships

---

## 🤖 Modeling

Models evaluated:
- Random Forest
- XGBoost
- **Gradient Boosting** (Final choice)

SMOTE was used to balance classes. GridSearchCV was employed for hyperparameter tuning.

---

## 📈 Model Performance

| Model             | Accuracy | Precision | Recall | F1-Score | AUC  |
|------------------|----------|-----------|--------|----------|------|
| XGBoost           | 69.5%    | 79.3%     | 76.4%  | 77.8%    | 0.74 |
| Random Forest     | 75.5%    | 76.9%     | 92.9%  | 84.1%    | 0.74 |
| **Gradient Boosting** | **77.0%** | **77.3%** | **95.0%** | **85.3%** | **0.77** |

---

## 🔍 Feature Importance

- **Top Features**: Checking account status, Credit per month, Duration, Age
- **Gradient Boosting** prioritized financial behavior more than demographics

---

## 🌐 Web Application (Flask)

The web app provides:
- ✅ **Individual Prediction**: Enter applicant details via form
- 📁 **Batch Prediction**: Upload CSV file to predict risk for multiple applicants

### 🧾 Input Requirements

For batch mode, your CSV should have:
- Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose

Missing values in `Saving accounts` or `Checking account` will be treated as `"unknown"` automatically.

---

## 💻 Installation & Running the App

### 🔧 Prerequisites

- Python 3.7+
- pip

### 🔄 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/your-username/german-credit-risk-prediction.git
cd german-credit-risk-prediction
```
Ensure your trained model (credit_risk_gradient_boosting.pkl) is in the root folder.
Run the Flask app:
```bash
python app.py
```
