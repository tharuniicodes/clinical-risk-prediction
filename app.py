import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# -----------------------------
# Page + assets
# -----------------------------
st.set_page_config(page_title="Clinical Risk Prediction", layout="wide")

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


@st.cache_data
def load_test_split(csv_path: str):
    """Load and cache the test split using the fitted scaler (no refit)."""
    df = pd.read_csv(csv_path)

    if "target" in df.columns:
        target_col = "target"
    elif "output" in df.columns:
        target_col = "output"
    else:
        raise ValueError("No target/output column found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Align columns to scaler expectation, then scale with the saved scaler
    X_test = X_test[scaler.feature_names_in_]
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test.reset_index(drop=True)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
    :root {
        --bg: #0b132b;
        --panel: #111a30;
        --muted: #a9b3c7;
        --accent: #4fd1c5;
        --accent-2: #9f7aea;
        --border: rgba(255, 255, 255, 0.08);
    }
    .main {
        background: radial-gradient(120% 120% at 10% 20%, rgba(79, 209, 197, 0.12), transparent),
                    radial-gradient(120% 120% at 90% 10%, rgba(159, 122, 234, 0.12), transparent),
                    var(--bg);
        color: #e8edf5;
        font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .block-container {padding: 2.6rem 2.4rem 3rem 2.4rem;}
    .card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem 1.4rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.28);
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.78rem;
        border-radius: 999px;
        background: rgba(79, 209, 197, 0.14);
        color: #b8f3ec;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    .hero-title {font-size: 2.05rem; font-weight: 800; margin-bottom: 0.2rem;}
    .muted {color: var(--muted);}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Hero section
# -----------------------------
st.markdown(
    """
    <div class="card" style="margin-bottom: 1rem;">
        <div class="pill">Clinical ML • Heart Disease</div>
        <div class="hero-title">Clinical Heart Disease Risk</div>
        <p class="muted" style="max-width: 760px;">
            Enter patient vitals and labs to estimate heart disease risk. Model performance and feature importance
            are shown for transparency.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Input form
# -----------------------------
with st.form("patient_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", value=120)
        chol = st.number_input("Cholesterol (mg/dl)", value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    with c2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submit = st.form_submit_button("Predict Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if submit:
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        "age", "sex", "cp", "trtbps", "chol", "fbs",
        "restecg", "thalach", "exng", "oldpeak",
        "slp", "caa", "thall"
    ])

    # Align column names with training data
    input_data.columns = scaler.feature_names_in_

    input_scaled = scaler.transform(input_data)
    risk_prob = model.predict_proba(input_scaled)[0][1]

    # Risk category
    if risk_prob < 0.35:
        risk_level = "Low Risk"
        color = "green"
    elif risk_prob < 0.65:
        risk_level = "Medium Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"

    card_color = {"green": "#0ea573", "orange": "#e8a340", "red": "#e75a5a"}[color]
    st.markdown(
        f"""
        <div class="card" style="margin-top: 1rem; border: 1px solid rgba(255,255,255,0.08);">
            <div style="display:flex; align-items:center; gap: 0.9rem;">
                <div style="width:12px; height:12px; border-radius:50%; background:{card_color}; box-shadow:0 0 0 8px rgba(255,255,255,0.04);"></div>
                <div>
                    <div style="font-size:1.8rem; font-weight:800;">{risk_prob:.1%}</div>
                    <div style="text-transform:uppercase; letter-spacing:1px; color: #c7d3e6;">{risk_level}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Interpretation: Probability reflects model confidence; risk banding uses 35% / 65% thresholds.")

# -----------------------------
# Model performance on test set
# -----------------------------
st.divider()
st.subheader("📊 Model Performance (Test Split)")

X_test_scaled, y_test = load_test_split("data/heart.csv")

y_probs = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_probs)
st.markdown(f"**ROC–AUC Score:** {auc:.3f}")

cm = confusion_matrix(y_test, model.predict(X_test_scaled))
st.markdown("**Confusion Matrix:**")
st.write(cm)

# -----------------------------
# Feature importance
# -----------------------------
st.divider()
st.subheader("🔍 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": scaler.feature_names_in_,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feature_df.set_index("Feature"))
else:
    st.write("Feature importance not available for this model.")
