
import streamlit as st
import pandas as pd
import joblib

# === Load trained model and preprocessing tools ===
model = joblib.load('class.pkl')

# Load encoders for categorical features
le_location = joblib.load('Location_encoder.pkl')
le_soil = joblib.load('Soil type_encoder.pkl')
le_irrigation = joblib.load('Irrigation_encoder.pkl')
le_crops = joblib.load('Crops_encoder.pkl')

# Load scaler for numeric features
scaler = joblib.load('numeric_scaler.pkl')

# Load target encoder to decode predicted season
le_target = joblib.load('target_encoder.pkl')

# === Streamlit app ===
st.title('🌾 Season Classification')

form_values = {}

with st.form("season_form"):
    form_values["Location"] = st.selectbox("Location", le_location.classes_)
    form_values["Rainfall"] = st.number_input('Rainfall (mm)', min_value=0)
    form_values["Temperature"] = st.number_input('Temperature (°C)', min_value=0, max_value=100)
    form_values["Soil type"] = st.selectbox("Soil Type", le_soil.classes_)
    form_values["Irrigation"] = st.selectbox("Irrigation Method", le_irrigation.classes_)
    form_values["yeilds"] = st.number_input('Yields (kg/ha)', min_value=0)
    form_values["Humidity"] = st.number_input('Humidity (%)', min_value=0)
    form_values["Crops"] = st.selectbox("Crops", le_crops.classes_)

    submit_button = st.form_submit_button(label='Predict')

# === On submit ===
if submit_button:
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([form_values])

        # Encode categorical inputs
        input_df['Location'] = le_location.transform(input_df['Location'])
        input_df['Soil type'] = le_soil.transform(input_df['Soil type'])
        input_df['Irrigation'] = le_irrigation.transform(input_df['Irrigation'])
        input_df['Crops'] = le_crops.transform(input_df['Crops'])

        # Scale numeric columns
        numeric_cols = ['Rainfall', 'Temperature', 'yeilds', 'Humidity']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Reorder columns to match model training
        input_df = input_df[['Location', 'Soil type', 'Irrigation', 'Crops',
                     'Rainfall', 'Temperature', 'yeilds', 'Humidity']]

        # Make prediction
        prediction = model.predict(input_df)[0]
        predicted_season = le_target.inverse_transform([prediction])[0]

        st.success(f"✅ Predicted Season: **{predicted_season.upper()}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

