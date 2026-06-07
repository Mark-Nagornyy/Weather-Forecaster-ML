import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np  #
import streamlit.components.v1 as components
from streamlit import markdown as md
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit import code as cd

st.set_page_config(page_title="Weather Forecaster", page_icon="⛅")

# ––––––––––––––– PROLOGUE: LOAD AND CLEAN THE RAW DATA –––––––––––––––

df = pd.read_csv("data/london_weather.csv")
df = df.dropna(how="all")
df["cloud_cover"] = df["cloud_cover"].fillna(0)
df["snow_depth"] = df["snow_depth"].fillna(0)
df["precipitation"] = df["precipitation"].fillna(0)
df = df.dropna(subset=["min_temp", "max_temp", "global_radiation", "pressure"])
df["mean_temp"] = df["mean_temp"].fillna((df["max_temp"] + df["min_temp"]) / 2)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day_of_year


def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


df["season"] = df["month"].apply(get_season)
features_df = pd.read_csv("data/features.csv")

# ––––––––––––––– CHAPTER 0: HEADER –––––––––––––––

st.title("London Weather Forecaster")
st.markdown("### An end-to-end Machine Learning project - By Mark")
st.markdown("""
   This Project uses 40+ years of London weather data (1979 - 2021, over 15,000 days)
    to train two Machine Learning models: one that predicts tommorow's temperature, 
    and one that forecasts the next 5 days. Below is the full story of how I built it, from raw data to a working app.
            """)

st.info("⬅️ Use the sidebar to try the models yourself after reading!")
st.divider()

# ––––––––––––––– CHAPTER 1: THE DATA –––––––––––––––
st.markdown("## 1. The Data")
md("""
I started with a dataset of daily London weather.
Every row is one day, and each day has readings (cols), like temperature, cloud cover,
sunshine, pressure, rainfall, etc.
      """)

cd("""
import pandas as pd
   
df = pd.read_csv('data/london_weather.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
""")

col1, col2 = st.columns(2)
col1.metric("Total days of weather", len(df))
col2.metric("Columns in the dataset", df.shape[1])
st.dataframe(df.head())
st.divider()

# ––––––––––––––– CHAPTER 2: Cleaning –––––––––––––––
md("## 2. Cleaning of the Data")
md("""
Real-world data is often messy: some days have missing values and others are irrational.

The following is what applies to them:
- For rain, snow, and clouds, a missing values probably mean that *none* so we fill with 0.
- For values that we can't guess (such as min/max temperature), we drop the rows.
- We also convey the values to their correct format, eg. dates (string -> datetime)
""")
cd("""
df["cloud_cover"] = df["cloud_cover"].fillna(0)
df["snow_depth"] = df["snow_depth"].fillna(0)
df["precipitation"] = df["precipitation"].fillna(0)
df = df.dropna(subset=["min_temp", "max_temp", "global_radiation", "pressure"])
df["mean_temp"] = df["mean_temp"].fillna((df["max_temp"] + df["min_temp"]) / 2)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
""")

st.divider()

# —————————————— CHAPTER 3: EDA ——————————————
md("### 3. Exploring the Data")
md("""
Before building any models, we need to look at the data and find patterns.
Here are the most interesting things we discovered.
""")

fig1 = px.scatter(df, x="date", y="mean_temp",
                  trendline="ols", opacity=0.2, trendline_color_override="black",
                  title="Mean Temperature in London (1979-2021)",
                  labels={"mean_temp": "Mean Temperature (°C)", "date": "Date"})

with st.status("#### Temperature over time"):
    st.plotly_chart(fig1, use_container_width=True)
    md("London is slowly getting warmer, the trend line shows climate change in action.")



fig2 = px.scatter(df, x="day", y="mean_temp", color="season",
                  opacity=0.4, animation_frame="year",
                  title="Mean Daily Temperature by Year",
                  labels={"mean_temp": "Mean Temperature (°C)", "day": "Day of Year"})
with st.status("#### Daily temperatures, year by year"):
    st.plotly_chart(fig2, use_container_width=True)



fig3 = px.line(df, x="day", y="precipitation", animation_frame="year",
               title="Daily Precipitation Curve by Year",
               labels={"precipitation": "Rainfall (mm)", "day": "Day of Year"})

with st.status("#### Precipitation"):
    st.plotly_chart(fig3, use_container_width=True)
    md("Rainfall is unpredictable throughout the year, no clear seasonal pattern.")



fig4 = px.scatter_3d(df, x="day", y="precipitation", z="snow_depth",
                     color="season", opacity=0.6,
                     title="Daily Rainfall & Snow Depth (all years, 3D)",
                     labels={"day": "Day of Year", "precipitation": "Rainfall (mm)", "snow_depth": "Snow Depth (mm)"})
fig4.update_traces(marker=dict(size=2))
fig4.update_layout(height=700, scene_camera=dict(eye=dict(x=2, y=2, z=1.5)))

with st.status("#### Rainfall & Snow (3D)"):
    st.plotly_chart(fig4, use_container_width=True)
    md("X = day of year, Y = rainfall, Z = snow depth. Drag to rotate.")

st.divider()

# ––––––––––––––– CHAPTER 4: Feature Engineering –––––––––––––––

md("## 4. Feature Engineering")
md("""
A model can't predict tomorriw's weather from just today.
It also needs to know what has been happening recently.
So we built new columns called **features** that capture the recent trends:
    """)

cd("""
# What we want to predict
df["temp_tomorrow"] = df["mean_temp"].shift(-1)
# What happened yesterday
df["temp_yesterday"] = df["mean_temp"].shift(1)
df["pressure_yesterday"] = df["pressure"].shift(1)
# Rolling averages over the last 3 and 5 days
df["temp_last3"] = df["mean_temp"].rolling(3).mean()
df["temp_last5"] = df["mean_temp"].rolling(5).mean()
df["rain_last5"] = df["precipitation"].rolling(5).mean()
# Season as a number the model can read (one-hot encoding)
df = pd.get_dummies(df, columns=["season"], drop_first=True)
df.to_csv("data/features.csv", index=False)
""")

md("""
This is similar to how weather forcasters think, where they look at recent trends, instead of just one day.
   """)

st.divider()

# ––––––––––––––– CHAPTER 5: Linear Regression –––––––––––––––

md("## 5. Model 1 - Predicting Tomorrow's Temperature")
md("""
For The first model we used **Linear Regression** - it finds the best straight line relatiosnhip between the features and tomorrows temperature.
""")

md("### Feature Selection")
md("""
We first trained a Decision Tree just to rank which features matter most. `max temp` dominated - today's maximum tamperature is by far the strongest signal.
""")

imp_df = pd.DataFrame({
    "Feature":    ["max_temp", "mean_temp", "cloud_cover", "min_temp",
                   "global_radiation", "sunshine", "temp_last5",
                   "year", "pressure", "temp_yesterday"],
    "Importance": [0.931, 0.028, 0.004, 0.003,
                   0.003, 0.003, 0.002,
                   0.002, 0.002, 0.002],
})
fig_imp = px.bar(imp_df, x="Feature", y="Importance",
                 title="Feature Importance (Decision Tree)", log_y=True)
st.plotly_chart(fig_imp, use_container_width=True)
md("#### Training")
cd("""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
features = ["max_temp", "mean_temp", "cloud_cover", "min_temp",
            "temp_last5", "year", "pressure"]
X = df[features]
y = df["temp_tomorrow"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
""")
col1, col2, col3 = st.columns(3)
col1.metric("MAE",  "0.85 °C", help="Average prediction error")
col2.metric("RMSE", "1.10 °C", help="Typical error size")
col3.metric("R²",   "0.961",   help="96% of variation explained")
md("On average the model is off by less than 1 °C - and it explains 96% of the variation.")

md("### Actual vs Predicted")
md("""
A perfect model would put every dot on the diogonal line. The closer the dots cluster to it the better the model.
   """)

features_single = ["max_temp", "mean_temp", "cloud_cover", "min_temp", "temp_last5", "year", "pressure"]
X = features_df[features_single]
y = features_df["temp_tomorrow"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_single = LinearRegression()
model_single.fit(X_train, y_train)
y_pred = model_single.predict(X_test)

eval_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).sample(1000, random_state=1)
fig_eval = px.scatter(eval_df, x="Actual", y="Predicted",
                      trendline="ols", opacity=0.5,
                      title="Actual vs Predicted Temperature (°C)")
st.plotly_chart(fig_eval, use_container_width=True)
md("Most dots sit very close to the line — the model works!")

md("### Saving the Model")
cd("""
import pickle

with open("models/temperature_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/temperature_features.pkl", "wb") as f:
    pickle.dump(features, f)
""")

md("We save the trained model to a file so the Streamlit app can load it instantly without retraining")
st.divider()

# ––––––––––––––– CHAPTER 6: Model 2: Multi-Output Regression –––––––––––––––

md("## 6. Model 2 - Forecasting the Next 5 Days")
md("""
Predicting tomorrow is useful, but what about the next 5 days?
We used **Multi-Output Regression**, five linear regressinos working at the same time
one for each future day. Each predicts a different target.   
""")

cd("""
# Create one target column per future day
df["temp_day1"] = df["mean_temp"].shift(-1)
df["temp_day2"] = df["mean_temp"].shift(-2)
df["temp_day3"] = df["mean_temp"].shift(-3)
df["temp_day4"] = df["mean_temp"].shift(-4)
df["temp_day5"] = df["mean_temp"].shift(-5)

df = df.dropna()
""")

md("### Feature Importance for Day 5")
md("""
We ran the decision tree again, this time to meassure the importance of the features for 5 days.
`max_temp` still leads, but the season also matters more with `season_summer` being in the lead, as during summer there are less fluctuations due to reduced precipitation.
""")

df_m = features_df.drop(columns=["date", "tomorrow_rain", "temp_tomorrow", "day", "month"])
df_m = df_m.copy()
df_m["temp_day5"] = df_m["mean_temp"].shift(-5)
df_m = df_m.dropna()
X_m = df_m.drop(columns=["temp_day5"])
y_m = df_m["temp_day5"]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)
tree_m = DecisionTreeRegressor(random_state=42)
tree_m.fit(X_train_m, y_train_m)
imp_m = pd.Series(tree_m.feature_importances_, index=X_m.columns).sort_values(ascending=False)
imp_m_df = imp_m.reset_index()
imp_m_df.columns = ["Feature", "Importance"]
fig_imp_m = px.bar(imp_m_df.head(10), x="Feature", y="Importance",
                   title="Top 10 Feature Importances for Day 5 (Decision Tree)")
st.plotly_chart(fig_imp_m, use_container_width=True)

md("#### Training")
cd("""
from sklearn.multioutput import MultiOutputRegressor

features = ["max_temp", "season_summer", "season_winter", "temp_last5", "year"]

X = df[features]
y = df[["temp_day1", "temp_day2", "temp_day3", "temp_day4", "temp_day5"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
""")

md("#### How accurate is the 5-day forecast?")
results_df = pd.DataFrame({
    "Day": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
    "MAE": [1.073,   1.495,   1.912,   2.169,   2.311],
    "R²":  [0.944,   0.883,   0.818,   0.769,   0.735],
})
fig_acc = px.line(results_df, x="Day", y=["MAE", "R²"], markers=True,
                  title="Multi-Output Model: Accuracy by Forecast Day")
st.plotly_chart(fig_acc, use_container_width=True)

md("""
* **MAE going up**: each extra day adds roughly +0.3 of error.
* **R2 going down**: the model explains less variation the further ahead it looks.
* The shape is a gentle slope - the model stays useful all the way to Day 5. 
""")

col1, col2 = st.columns(2)
col1.metric("Day 1 MAE", "1.07 °C", help="Most accurate — just one day ahead")
col2.metric("Day 5 MAE", "2.31 °C", help="Furthest ahead — still useful")

md("#### Saving the Model")
cd("""
import pickle

with open("models/multi_temperature_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/multi_temperature_features.pkl", "wb") as f:
    pickle.dump(features, f)
""")
st.divider()


# —————————————— CHAPTER 7: TRY IT YOURSELF ——————————————
md("### 7. Try it yourself")
md("Use the sidebar to open either model and enter today's weather values:")

col1, col2 = st.columns(2)
with col1:
    st.info("""
**Linear Regression**

Predicts **tomorrow's** temperature
from today's weather readings.
    """)
with col2:
    st.info("""
**Multi-Output Regression**

Forecasts the **next 5 days**
of temperatures.
""")