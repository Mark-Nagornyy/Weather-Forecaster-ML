import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
from datetime import timedelta
with open("models/multi_temperature_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(layout="wide", page_title="Weather App")
st.markdown("# Multi-Output Regressor", text_alignment="left")
st.markdown("""
Using Multi-Output Linear Regression to forcast the next 5 days of temperature in london based on 5 key features
""")

st.markdown("***")
st.markdown("## Data Input")
col1, col2 = st.columns(2)
with col1:
    max_temp = st.number_input("Max Temp °C", min_value= -100, max_value=100, value=0)
with col2:
    date = st.date_input("Date")
year = date.year
month = date.month
season_summer = 1 if month in [6, 7, 8] else 0
season_winter = 1 if month in [12, 1, 2] else 0
season_name = (
    "Summer ☀️" if season_summer
    else "Winter ❄️" if season_winter
    else "Spring 🌱" if month in [3, 4, 5]
    else "Autumn 🍂"
)
st.markdown(f"**Season:** {season_name} &nbsp; | &nbsp; **Year:** {year}")

st.markdown("#### Last 5 Days Average Temperature")
col3, col4, col5, col6, col7  = st.columns(5)
with col3:
    one_day_before = st.number_input("One Day Before Date °C", value=0.00)
with col4:
    two_day_before = st.number_input("Two Days Before Date °C", value=0.00)
with col5:
    three_day_before = st.number_input("Three Days Before Date °C", value=0.00)
with col6:
    four_day_before = st.number_input("Four Days Before Date °C", value=0.00)
with col7:
    five_day_before = st.number_input("Five Days Before Date °C", value=0.00)
st.markdown("***")
if  st.button("Predict Next 5 Days", use_container_width=True, type="primary"):
    temp_last5 = np.mean([one_day_before, two_day_before, three_day_before, four_day_before, five_day_before])
    input_data = pd.DataFrame
    
    input_row = {
        "max_temp": max_temp,
        "season_summer": season_summer,
        "season_winter": season_winter,
        "temp_last5": temp_last5,
        "year": year
    }
    
    X_input = pd.DataFrame([input_row])
 
    predictions = model.predict(X_input)[0]
    
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label=(date + timedelta(days=1)).strftime("%a %d %b"),
            value=f"{predictions[0]:.1f} °C",
        )
    with col2:
        st.metric(
            label=(date + timedelta(days=2)).strftime("%a %d %b"),
            value=f"{predictions[1]:.1f} °C",
        )
    with col3:
        st.metric(
            label=(date + timedelta(days=3)).strftime("%a %d %b"),
            value=f"{predictions[2]:.1f} °C",
        )
    with col4:
        st.metric(
            label=(date + timedelta(days=4)).strftime("%a %d %b"),
            value=f"{predictions[3]:.1f} °C",
        )
    with col5:
        st.metric(
            label=(date + timedelta(days=5)).strftime("%a %d %b"),
            value=f"{predictions[4]:.1f} °C",
        )