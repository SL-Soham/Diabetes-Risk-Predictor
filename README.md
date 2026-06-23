# 🩺 Type 2 Diabetes Risk Predictor

An end-to-end machine learning diagnostic tool designed to predict the onset of Type 2 Diabetes based on clinical metrics. This project was built to address evaluation biases commonly found in small medical datasets by enforcing strict data hygiene, domain-specific feature engineering, and rigorous cross-validation.

## 📌 Project Overview
While many academic baselines on the PIMA Indians Diabetes Dataset report inflated accuracies due to data leakage and single-seed testing, this project establishes a mathematically rigorous **Logistic Regression** model. By applying custom physiological ratios and outlier-resistant scaling, the model achieved a stable, validated accuracy of **77.34%**, outperforming the standard literature baseline of 70.5% for this algorithmic class.

## 🚀 Key Features
* **Smart Data Imputation:** Handled biological impossibilities (e.g., Blood Pressure = 0) using K-Nearest Neighbors (KNN) imputation rather than destructive row deletion.
* **Domain-Specific Feature Engineering:** Engineered two novel physiological metrics:
  * `Glucose_BMI_Ratio`: Isolates obese, hyperglycemic profiles.
  * `Insulin_Glucose_Ratio`: Acts as a proxy metric for insulin resistance (HOMA-IR).
* **Outlier Immunity:** Utilized `RobustScaler` (Interquartile Range scaling) to mitigate the impact of extreme medical anomalies (e.g., severe hyperinsulinemia).
* **Strict Validation:** Model evaluated exclusively via Stratified 5-Fold Cross-Validation to preserve the native class imbalance (65/35) and prevent overfitted "lucky" data splits.
* **Interactive UI:** Fully deployed as a user-friendly Streamlit web application.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Serialization:** Joblib

## 📁 Project Structure
```text
├── data/
│   └── diabetes.csv             # PIMA Indians dataset
├── models/
│   ├── diabetes_rf_model.pkl    # Trained Logistic Regression model
│   └── scaler.pkl               # Fitted RobustScaler
├── app.py                       # Streamlit web application
├── train_model.py               # ML pipeline and cross-validation script
├── requirements.txt             # Python dependencies
└── README.md
