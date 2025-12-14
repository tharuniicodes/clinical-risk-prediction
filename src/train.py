import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from src.preprocess import preprocess_data
import os

def train_and_select_model(csv_path):
    # Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(csv_path)

    results = {}

    # ---------------------------
    # 1. Logistic Regression
    # ---------------------------
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    results["logistic_regression"] = {
        "model": log_reg,
        "auc": auc_lr
    }

    print("\nLogistic Regression Results")
    print("AUC:", auc_lr)
    print(classification_report(y_test, log_reg.predict(X_test)))

    # ---------------------------
    # 2. Random Forest
    # ---------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    results["random_forest"] = {
        "model": rf,
        "auc": auc_rf
    }

    print("\nRandom Forest Results")
    print("AUC:", auc_rf)
    print(classification_report(y_test, rf.predict(X_test)))

    # ---------------------------
    # Select Best Model
    # ---------------------------
    best_model_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_model_name]["model"]
    best_auc = results[best_model_name]["auc"]

    print("\nBest Model Selected:", best_model_name)
    print("Best AUC:", best_auc)

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")

    return best_model_name, best_auc, feature_names


if __name__ == "__main__":
    train_and_select_model("data/heart.csv")
