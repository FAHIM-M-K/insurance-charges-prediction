import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("Insurance Charges Prediction App")
st.write("Enter patient details to predict medical insurance charges.")

# Input fields for features
age = st.slider("Age", 18, 64, 30)
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input data for prediction
input_data = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_male': 1 if sex == "male" else 0,
    'smoker_yes': 1 if smoker == "yes" else 0,
    'region_northwest': 1 if region == "northwest" else 0,
    'region_southeast': 1 if region == "southeast" else 0,
    'region_southwest': 1 if region == "southwest" else 0,
    'bmi_smoker': bmi * (1 if smoker == "yes" else 0),
    'age_bmi': age * bmi,
    'age_group_middle_aged': 1 if 30 <= age <= 50 else 0,
    'age_group_senior': 1 if age > 50 else 0
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Scale numerical features
numerical_cols = ['age', 'bmi', 'children']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Predict log_charges and convert back to USD
log_pred = model.predict(input_df)
pred_charges = np.expm1(log_pred)[0]  # Inverse of log1p

# Display prediction
st.subheader("Predicted Insurance Charges")
st.write(f"${pred_charges:,.2f}")

# Display model performance (replace 0.85 with your actual R² from results_df)
st.write(f"Model: Random Forest (R²: 0.85)")