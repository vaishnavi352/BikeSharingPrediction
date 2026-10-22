
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Bike Demand Prediction",
    page_icon="🚲"
)

st.title("🚲 Bike Sharing Demand Prediction")
st.write(
    "Predict bike rental demand using weather "
    "and time-related information."
)

@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/"
        "selva86/datasets/master/BikeSharingDemand.csv"
    )
    return pd.read_csv(url)

@st.cache_resource
def train_model():
    data = load_data()

    data["datetime"] = pd.to_datetime(data["datetime"])
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday
    data["month"] = data["datetime"].dt.month

    features = [
        "season", "holiday", "workingday", "weather",
        "temp", "humidity", "windspeed",
        "hour", "weekday", "month"
    ]

    X = data[features]
    y = data["count"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    return model

try:
    model = train_model()

    st.subheader("Enter Prediction Details")

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox(
            "Season",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Spring",
                2: "Summer",
                3: "Autumn",
                4: "Winter"
            }[x]
        )

        weather = st.selectbox(
            "Weather",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear",
                2: "Mist / cloudy",
                3: "Light rain or snow",
                4: "Heavy rain or snow"
            }[x]
        )

        temp = st.slider(
            "Temperature (normalized)",
            0.0, 1.0, 0.5
        )

        humidity = st.slider(
            "Humidity (normalized)",
            0.0, 1.0, 0.5
        )

    with col2:
        hour = st.slider("Hour of day", 0, 23, 8)

        weekday = st.selectbox(
            "Day of week",
            list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x]
        )

        month = st.slider("Month", 1, 12, 6)

        workingday = st.selectbox(
            "Is it a working day?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        holiday = st.selectbox(
            "Is it a holiday?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    windspeed = st.slider(
        "Wind speed (normalized)",
        0.0, 1.0, 0.2
    )

    if st.button("Predict Bike Demand"):
        input_data = pd.DataFrame([{
            "season": season,
            "holiday": holiday,
            "workingday": workingday,
            "weather": weather,
            "temp": temp,
            "humidity": humidity,
            "windspeed": windspeed,
            "hour": hour,
            "weekday": weekday,
            "month": month
        }])

        prediction = model.predict(input_data)[0]

        st.success(
            f"Estimated bike demand: "
            f"{max(0, round(prediction))} bikes"
        )

        st.info(
            "Use this estimate to help plan bike availability."
        )

except Exception as e:
    st.error(
        "Could not load the training dataset. "
        "Check your internet connection and try again."
    )
    st.caption(f"Technical details: {e}")
