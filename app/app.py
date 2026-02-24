import shap
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Breast Cancer AI", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("../models/random_forest_model.pkl")

# Load dataset
data = pd.read_csv("../data/data.csv")

# Drop unwanted columns
data = data.drop(columns=["id", "Unnamed: 32"])

# Convert diagnosis to numeric
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

st.title("🧠 Breast Cancer Prediction System")
st.write("Explainable AI-Based Decision Support System")

st.header("Select a Patient Sample")

# Select sample
sample_index = st.slider("Select Patient Index", 0, len(data)-1, 0)

# Extract selected sample
sample = data.drop("diagnosis", axis=1).iloc[sample_index]
actual = data["diagnosis"].iloc[sample_index]

st.write("### Selected Patient Features")
st.write(sample)

if st.button("Predict"):

    input_array = np.array([sample.values])
    
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]

    st.subheader("📊 Prediction Result")

    malignant_prob = probability[1]
    benign_prob = probability[0]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Malignant Tumor Detected")
        else:
            st.success("✅ Benign Tumor")

    with col2:
        st.metric("Malignant Probability", f"{malignant_prob:.2%}")
        st.metric("Benign Probability", f"{benign_prob:.2%}")

    # Risk Interpretation
    st.subheader("🧠 Risk Interpretation")

    if malignant_prob > 0.9:
        st.error("Very High Risk – Immediate Medical Attention Recommended")
    elif malignant_prob > 0.7:
        st.warning("High Risk – Further Clinical Testing Recommended")
    elif malignant_prob > 0.4:
        st.warning("Moderate Risk – Monitor & Consider Further Tests")
    else:
        st.success("Low Risk – Likely Benign")

    st.write(f"Actual Diagnosis: {'Malignant' if actual == 1 else 'Benign'}")

    # ---------------- SHAP SECTION ----------------

    st.markdown("---")
    st.subheader("🔍 Explainability (SHAP Analysis)")
    st.write("This plot shows how each feature contributed to the final prediction.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_array)

    # Handle SHAP format safely
    if isinstance(shap_values, list):
        shap_val = shap_values[1][0]
        base_val = explainer.expected_value[1]
    elif len(shap_values.shape) == 3:
        shap_val = shap_values[0, :, 1]
        base_val = explainer.expected_value[1]
    else:
        shap_val = shap_values[0]
        base_val = explainer.expected_value

    explanation = shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=input_array[0],
        feature_names=sample.index.tolist()
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)