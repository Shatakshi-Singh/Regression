import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing objects
model = joblib.load("Model_objects/xgb_model.pkl")
scaler = joblib.load("Model_objects/scaler.pkl")
feature_names = joblib.load("Model_objects/feature_names.pkl")

st.title("Truck Price Prediction App 🚛")

# Load dataset to extract unique categories
df = pd.read_csv("data/Truck_price_prediction.csv")
original_categorical_columns = ['Starting_Day', 'Day_of_Week', 'Load_Count', 'Weather']

# User Input Form
st.header("Input Trip Details")
input_data = {}

# Numerical Inputs
numeric_columns = ['Trip_Distance_km', 'Palette_Count', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
for feature in numeric_columns:
    input_data[feature] = st.number_input(feature, value=float(df[feature].median()))

# Categorical Inputs
for feature in original_categorical_columns:
    options = df[feature].unique().tolist()
    input_data[feature] = st.selectbox(feature, options)

# Convert input data into DataFrame
input_df = pd.DataFrame([input_data])
input_df_encoded = pd.get_dummies(input_df, columns=original_categorical_columns, drop_first=True)

# Ensure missing columns from one-hot encoding are added
for col in feature_names:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Reorder columns to match model input
input_df_encoded = input_df_encoded[feature_names]
input_df_encoded[numeric_columns] = scaler.transform(input_df_encoded[numeric_columns])

if st.button("Predict Trip Price"):
    prediction = model.predict(input_df_encoded)[0]
    st.success(f"Predicted Trip Price: ${prediction:.2f}")
