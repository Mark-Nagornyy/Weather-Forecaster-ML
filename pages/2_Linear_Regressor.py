import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

with open("models/temperature_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/temperature_features.pkl", "rb") as f:
    feature_list = pickle.load(f)


st.set_page_config(layout="wide", page_title="Weather App")
st.markdown("# Linear Regressor", text_alignment="left")
st.markdown("""
This linear regressor predicts tommorows average temperature from 10 weather features.
""")

st.markdown("***")
st.markdown("## Data Input")
col1, col2 = st.columns(2)
with col1:
    max_temp = st.number_input("Max Temp °C", min_value= -100, max_value=100, value=0)
    min_temp = st.number_input("Min Temp °C", min_value= -100, max_value=100, value=0)
    cloud_cover = st.number_input("Cloud Cover %", min_value= 0, max_value=100, value=0)
    pressure = st.number_input("Pressure Mbar", min_value= 0, max_value=100000, value=1000)
    date = st.date_input("Date")
with col2:
    one_day_before = st.number_input("One Day Before Date °C", min_value= -100, max_value=100, value=0)
    two_day_before = st.number_input("Two Days Before Date °C", min_value= -100, max_value=100, value=0)
    three_day_before = st.number_input("Three Days Before Date °C", min_value= -100, max_value=100, value=0)
    four_day_before = st.number_input("Four Days Before Date °C", min_value= -100, max_value=100, value=0)
    five_day_before = st.number_input("Five Days Before Date °C", min_value= -100, max_value=100, value=0)
    
st.markdown("***")

if st.button("Predict  Tommorows Termperature", use_container_width=True, type="primary"):
        
    mean_temp = (max_temp + min_temp) / 2

    temp_last5 = np.mean([
        one_day_before,
        two_day_before,
        three_day_before,
        four_day_before,
        five_day_before
    ])

    date = date.year

    pressure_pa = (pressure) * 100

    cloud_oktas = round((cloud_cover) / 100 * 8)



    input_row = {
        "max_temp": max_temp,
        "mean_temp": mean_temp,
        "cloud_cover": cloud_oktas,
        "min_temp": min_temp,
        "temp_last5": temp_last5,
        "year": date,
        "pressure": pressure_pa
    }
    
    st.markdown("***")

    X_input = pd.DataFrame([input_row])

    prediction = model.predict(X_input)[0]

    prediction = round(prediction, 3)



    st.metric(label="Tommorow's Temperature", value=prediction, delta = round(prediction - mean_temp, 3))