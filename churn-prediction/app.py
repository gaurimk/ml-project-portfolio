import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure", 1, 72)
monthly = st.number_input("Monthly Charges")

# create empty dataframe with correct columns
input_data = pd.DataFrame(columns=model.feature_names_in_)

# fill values
input_data.loc[0] = 0

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly
input_data["TotalCharges"] = tenure * monthly

if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")