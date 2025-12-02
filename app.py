import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved tools
preprocessor = joblib.load('preprocessor.sav')
model = joblib.load('house_model.sav')
scores = joblib.load('model_scores.sav')

st.title("House Price Prediction App")
# Display the scores dynamically
col1, col2 = st.columns(2)
with col1:
    st.metric("Model Accuracy (R²)", f"{scores['r2']:.2%}")
with col2:
    st.metric("Average Error (RMSE)", f"${scores['rmse']:,.0f}")
    
st.write("Enter house details below to estimate the price.")

# Input Form
st.header("House Details")
sq_input = st.number_input("Square Feet", min_value=100, value=1500)
rooms_input = st.number_input("Number of Rooms", min_value=1, value=3)
age_input = st.number_input("Building Age", min_value=0, value=10)
dist_input = st.number_input("Distance to City (Km)", min_value=0.0, value=5.5)

# Prediction Button
if st.button("Estimate Price"):
    # 1. Create DataFrame matching the training format

    input_data = pd.DataFrame({
        'square_feet': [sq_input],
        'num_rooms': [rooms_input],
        'age': [age_input],
        'distance_to_city(km)': [dist_input]
    })

    # 2. Preprocess the input

    try:
        input_processed = preprocessor.transform(input_data)
        
        # 3. Predict

        price_pred = model.predict(input_processed)[0]

        # 4. Display Result

        st.success(f"Estimated House Price: ${price_pred:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.caption("Based on Decision Tree Regressor Model")