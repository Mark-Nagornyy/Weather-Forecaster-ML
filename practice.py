# python -m streamlit run practice.py
# https://docs.streamlit.io/develop/api-reference/widgets
import streamlit as st
from utils import load_and_crop
import pandas as pd
st.markdown("## Streamlit Practice")
st.markdown("""
***
### Markdown supported
- Bullet points
- **Bold**
- _Italic_
- `code`
---
""")
col1, col2, col3 = st.columns(3)
with col1:
    st.header("A cat", text_alignment="center")
    st.image(load_and_crop("https://static.streamlit.io/examples/cat.jpg", aspect_ratio=(1,2)))
with col2:
    st.header("A dog", text_alignment="center")
    st.image(load_and_crop("https://static.streamlit.io/examples/dog.jpg", aspect_ratio=(1,2)))
with col3:
    st.header("An owl", text_alignment="center")
    st.image(load_and_crop("https://static.streamlit.io/examples/owl.jpg", aspect_ratio=(1,2)))
st.markdown("---")
name = " "
name = st.text_input("Take in small text")
password = st.text_input("Password", type="password")
st.number_input("Number", min_value=0, max_value=100, value=10)
st.text_area("Long text")
st.selectbox("Select Box", ["Banana", "Washing Machine", "Tower of Pisa"])
st.multiselect("Muli Select", ["Banana", "Washing Machine", "Tower of Pisa"])
s1 = st.slider("Slider 1", 0, 100, 50)
s2 = st.slider("Slider 2", 0, 100, 50)
st.markdown(f"{s1} * {s2} = {s1 * s2}")
st.checkbox("Enable feature")
st.radio("Radio Button", ["Banana", "Washing Machine", "Tower of Pisa"])
st.date_input("Date")
st.time_input("Time")
file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.dataframe(df) # show dataframe
st.metric("Temperature", "12.3 °C", delta="+0.8 °C")
st.video("https://www.youtube.com/watch?v=xvFZjo5PgG0")
# shares global variable
if "count" not in st.session_state:
    st.session_state.count = 0
if st.button("Button"):
    st.markdown("You pressed a button!")

st.sidebar.slider("Sidebar Slider", 0, 100)

col1, col2 = st.columns(2)
with col1:
    s1 = st.slider("Slider 11", )
    
# DISCLAMER AI
st.markdown("---")

# 1. Create top-level tabs
main_tab1, main_tab2 = st.tabs(["Data", "Settings"])

with main_tab1:
    st.write("Main Data Panel")
    # 2. Create sub-tabs inside the first tab
    sub_tab1, sub_tab2 = st.tabs(["Subtab A", "Subtab B"])
    
    with sub_tab1:
        st.write("Content of Subtab A")
    with sub_tab2:
        st.write("Content of Subtab B")

with main_tab2:
    st.write("Settings Panel")