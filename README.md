# Clinical Heart Disease Risk Prediction

This project implements a machine learning–based clinical risk prediction system
that estimates the likelihood of heart disease using patient health attributes.
The goal is to demonstrate how predictive analytics can support early risk
assessment and clinical decision-making.



## Project Overview

The application takes patient-level clinical measurements as input and outputs:

- A probability score indicating heart disease risk
- A categorical risk level (Low / Medium / High)
- Model performance metrics (ROC–AUC, confusion matrix)
- Feature importance for interpretability

The system is built as an end-to-end pipeline, from data preprocessing and model
training to deployment through an interactive Streamlit dashboard.



## Dataset

- **Source:** UCI Heart Disease dataset
- **Target variable:** Presence of heart disease (binary)
- **Features include:**  
  Age, sex, chest pain type, resting blood pressure, cholesterol,
  ECG results, exercise-induced angina, and related clinical indicators.

---

## Methodology

1. **Preprocessing**
   - Train/test split with stratification
   - Feature scaling using StandardScaler
   - Consistent feature alignment during inference

2. **Models Evaluated**
   - Logistic Regression (baseline, interpretable)
   - Random Forest (final selected model)

3. **Model Selection**
   - Evaluation based on ROC–AUC score
   - Random Forest achieved the best performance (AUC ≈ 0.94)

---

## Application Features

- Interactive patient input form
- Real-time risk prediction
- Stable evaluation using a cached test split
- Model interpretability via feature importance
- Clean separation between training and inference code

---

## Project Structure

clinical-risk-prediction/
│
├── app.py # Streamlit application
├── data/
│ └── heart.csv # Dataset
├── models/
│ ├── model.pkl # Trained model
│ └── scaler.pkl # Fitted scaler
├── src/
│ ├── preprocess.py # Data preprocessing
│ └── train.py # Model training and selection
├── test_preprocess.py # Preprocessing validation
├── requirements.txt
└── README.md

Future Work:

Add patient-level explainability using SHAP

Generate downloadable clinical risk reports

Extend to longitudinal EHR data analysis

