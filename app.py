import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Paths (robuste local + cloud)
HERE = Path(__file__).parent
model_path = HERE / "linear_regression_model.pkl"
scaler_path = HERE / "scaler.pkl"

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("MetaBrains Student Test Score Predictor")
st.write("Enter the number of hours studied to predict the test score")

hours = st.number_input("Hours studied:", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        X = np.array([[hours]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        st.success(f"Prediction Test Score: {pred[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
