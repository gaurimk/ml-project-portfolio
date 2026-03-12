import streamlit as st
import joblib
import pandas as pd

import os
import joblib

model_path = os.path.join(os.path.dirname(__file__), "models", "house_price_model.pkl")
model = joblib.load(model_path)

st.title("🏠 House Price Prediction")

area = st.number_input("Living Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

if st.button("Predict"):

    features = pd.DataFrame({
        "GrLivArea":[area],
        "BedroomAbvGr":[bedrooms],
        "FullBath":[bathrooms]
    })

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")