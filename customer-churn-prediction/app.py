import streamlit as st
import pandas as pd
import joblib

import os
model_path = os.path.join(os.path.dirname(__file__), "model/model.pkl")
model = joblib.load(model_path)

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure",0,72)
monthly = st.number_input("Monthly Charges",0.0,200.0)
total = st.number_input("Total Charges",0.0,10000.0)

if st.button("Predict Churn"):

    data = pd.DataFrame({
        "tenure":[tenure],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will stay")