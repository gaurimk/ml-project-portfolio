import streamlit as st
import pandas as pd
from src.predict_pipeline import PredictPipeline

st.title("✈ Flight Price Predictor")

airline = st.selectbox("Airline", ["IndiGo","Air India","Jet Airways"])
source = st.selectbox("Source", ["Delhi","Kolkata","Mumbai","Chennai"])
destination = st.selectbox("Destination", ["Cochin","Delhi","Hyderabad","Kolkata"])
stops = st.selectbox("Stops",[0,1,2,3])

predict = st.button("Predict Price")

if predict:

    data = pd.DataFrame({
        "Airline":[airline],
        "Source":[source],
        "Destination":[destination],
        "Total_Stops":[stops],
        "Date_of_Journey":["24/03/2019"],
        "Dep_Time":["10:00"],
        "Arrival_Time":["12:00"],
        "Duration":["2h 0m"],
        "Route":["DEL → CCU"],
        "Additional_Info":["No info"]
    })

    pipeline = PredictPipeline()

    result = pipeline.predict(data)

    st.success(f"Estimated Flight Price: ₹{int(result[0])}")