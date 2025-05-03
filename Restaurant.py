import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load encoder dan scaler
city_encoder = joblib.load("city_encoder.pkl")
cuisine_encoder = joblib.load("cuisine_encoder.pkl")
delivery_encoder = joblib.load("delivery_encoder.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")



# Load model
interpreter = tf.lite.Interpreter(model_path="Restaurant_recommendation.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI
st.title("Restaurant Success Prediction")
st.write("Predict whether a restaurant will be successful based on its characteristics")

with st.form("restaurant_form"):
    # Categorical features
    city = st.selectbox("City", options=city_encoder.classes_)
    cuisine_type = st.selectbox("Cuisine Type", options=cuisine_encoder.classes_)
    delivery_service = st.radio("Delivery Service", options=['Yes', 'No'])

    # Numerical features
    average_meal_price = st.number_input("Average Meal Price ($)", min_value=0, max_value=100, value=25)
    seating_capacity = st.number_input("Seating Capacity", min_value=0, max_value=200, value=50)
    years_in_business = st.number_input("Years in Business", min_value=0, max_value=50, value=5)
    google_rating = st.number_input("Google Rating (1-5)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    social_media_followers = st.number_input("Social Media Followers", min_value=0, value=10000)
    weekend_reservations = st.number_input("Weekend Reservations", min_value=0, value=100)
    staff_count = st.number_input("Staff Count", min_value=0, value=20)
    marketing_budget = st.number_input("Marketing Budget ($)", min_value=0, value=10000)
    health_inspection_score = st.number_input("Health Inspection Score (0-100)", min_value=0, max_value=100, value=80)
    annual_revenue = st.number_input("Annual Revenue ($)", min_value=0, value=500000)

    submitted = st.form_submit_button("Predict Success")

if submitted:
    input_dict = {
        'City': city_encoder.transform([city])[0],
        'Cuisine_Type': cuisine_encoder.transform([cuisine_type])[0],
        'Average_Meal_Price': average_meal_price,
        'Seating_Capacity': seating_capacity,
        'Years_in_Business': years_in_business,
        'Google_Rating': google_rating,
        'Social_Media_Followers': social_media_followers,
        'Weekend_Reservations': weekend_reservations,
        'Staff_Count': staff_count,
        'Delivery_Service': delivery_encoder.transform([delivery_service])[0],
        'Marketing_Budget': marketing_budget,
        'Health_Inspection_Score': health_inspection_score,
        'Annual_Revenue': annual_revenue  # <- tambahan ini
    }

    input_df = pd.DataFrame([input_dict])[columns]
    input_scaled = scaler.transform(input_df).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    prediction_class = np.argmax(prediction, axis=1)[0]
    probability = prediction[0][prediction_class]

    if prediction_class == 1:
        st.success(f"✅ This restaurant is likely to be SUCCESSFUL (confidence: {probability:.2%})")
    else:
        st.error(f"❌ This restaurant is likely to NOT be successful (confidence: {probability:.2%})")
