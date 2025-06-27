# app.py

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("car_model.pkl", "rb"))

st.title("🚗 Car Price Prediction App")

# Input fields
year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
engine = st.number_input("Engine (cc)", min_value=500, max_value=5000, value=1197)
max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=500, value=80)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, value=18.0)
seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol"])
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[year, engine, max_power, mileage, seats, fuel_type_petrol]],
                              columns=["year", "engine", "max_power", "mileage", "seats", "fuel_type_Petrol"])
    
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Car Price: ₹{int(prediction[0])}")
