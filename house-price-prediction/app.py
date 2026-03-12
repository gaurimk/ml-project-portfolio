import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/house_price_model.pkl")

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