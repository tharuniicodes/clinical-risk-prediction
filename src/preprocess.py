import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Detect target column name
    if "target" in df.columns:
        target_col = "target"
    elif "output" in df.columns:
        target_col = "output"
    else:
        raise ValueError("No target/output column found in dataset")

    # Separate features and label
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
