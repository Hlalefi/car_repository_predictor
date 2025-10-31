# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model
with open("models/best_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = pipeline["model"]
scaler = pipeline["scaler"]
selected_features = pipeline["selected_features"]

st.title("Car Price Predictor")

st.markdown("Provide the car details and the model will predict the selling price (approx).")

# --- Input widgets ---
vehicle_age = st.number_input("Vehicle age (years)", min_value=0, max_value=50, value=5)
km_driven = st.number_input("Kilometres driven", min_value=0, value=20000)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=18.0, format="%.2f")
engine = st.number_input("Engine (cc)", min_value=500, value=1200)
max_power = st.number_input("Max power (bhp)", min_value=10.0, value=80.0, format="%.2f")
seats = st.number_input("Seats", min_value=1, max_value=10, value=5)

brand = st.selectbox("Brand", ["Maruti", "Hyundai", "Toyota", "Mahindra", "Honda", "Ford", "Others"])
seller_type = st.selectbox("Seller type", ["Individual", "Dealer"])
fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])

# --- Build feature row ---
row = {
    "vehicle_age": vehicle_age,
    "km_driven": km_driven,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats,
}

# Add all features initialized to 0 (for dummy columns)
for feat in selected_features:
    if feat not in row:
        row[feat] = 0

# Map categorical inputs to dummy columns
if f"brand_{brand}" in row:
    row[f"brand_{brand}"] = 1
if f"seller_type_{seller_type}" in row:
    row[f"seller_type_{seller_type}"] = 1
if f"fuel_type_{fuel_type}" in row:
    row[f"fuel_type_{fuel_type}"] = 1
if f"transmission_type_{transmission_type}" in row:
    row[f"transmission_type_{transmission_type}"] = 1

# --- Convert to DataFrame and reorder ---
X = pd.DataFrame([row])
X = X.reindex(columns=selected_features, fill_value=0)

# --- Scale and predict automatically ---
try:
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    st.subheader(f"Predicted selling price (approx): R {int(pred[0]):,}")
except Exception as e:
    st.error(f"Prediction error: {e}")

# Optional: Show the actual input values being used (for debugging)
# st.write("Feature values being used for prediction:")
# st.dataframe(X)


