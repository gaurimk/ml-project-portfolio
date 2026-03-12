import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# Load Model
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "house_price_model.pkl")

model = joblib.load(model_path)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Enter house details to estimate the price.")

# -------------------------------
# User Inputs
# -------------------------------

area = st.number_input(
    "Living Area (sq ft)",
    min_value=200,
    max_value=10000,
    value=1500
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict House Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")