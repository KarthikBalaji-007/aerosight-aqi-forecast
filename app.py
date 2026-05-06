from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.predict_aqi import get_max_forecast_days, predict_future_aqi


st.set_page_config(page_title="AeroSight AQI Forecast", layout="wide")


CITIES = [
    "delhi",
    "mumbai",
    "bengaluru",
    "chennai",
    "kolkata",
    "hyderabad",
    "jaipur",
    "lucknow",
    "gwalior",
    "visakhapatnam",
]

AQI_COLORS = {
    "Good": "#2e7d32",
    "Satisfactory": "#7cb342",
    "Moderate": "#f9a825",
    "Poor": "#ef6c00",
    "Very Poor": "#c62828",
    "Severe": "#6a1b9a",
}


def categorize_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data(ttl=1800)
def get_max_days_cached(city: str) -> int:
    return get_max_forecast_days(city)


def get_cluster_info(city: str) -> tuple[str | None, str]:
    cluster_path = Path("outputs/reports/city_clusters.csv")
    if not cluster_path.exists():
        return None, "Cluster information unavailable"

    clusters = pd.read_csv(cluster_path)
    required = {"City", "Cluster", "Cluster_Name"}
    if not required.issubset(clusters.columns):
        return None, "Cluster information unavailable"

    match = clusters[clusters["City"].str.lower() == city.lower()]
    if match.empty:
        return None, "Cluster information unavailable"

    return str(match.iloc[0]["Cluster"]), str(match.iloc[0]["Cluster_Name"])


def style_category(value: str) -> str:
    color = AQI_COLORS.get(value, "#455a64")
    return f"background-color: {color}; color: white; font-weight: 600"


st.title("AeroSight — Air Quality Forecast System")
st.caption("Predict future AQI using machine learning and real-time weather data.")
st.write(
    "Forecasts are generated using historical AQI trends combined with live meteorological "
    "forecast data including wind speed, precipitation, temperature, and humidity."
)

left, right = st.columns([1, 2])

with left:
    selected_city = st.selectbox("Select City", CITIES)

    try:
        max_days = get_max_days_cached(selected_city)
        st.success(f"Weather forecast available for {max_days} days")
        st.caption("Weather data fetched just now from Open-Meteo (UTC+5:30).")
    except Exception as exc:
        st.error(f"Weather forecast is unavailable right now: {str(exc)}")
        st.stop()

    days = st.slider("Select Forecast Horizon", 1, max_days, min(5, max_days))
    cluster_id, cluster_label = get_cluster_info(selected_city)
    if cluster_id is None:
        st.info(cluster_label)
    else:
        st.info(f"Cluster {cluster_id}: {cluster_label}")

    predict_clicked = st.button("Predict AQI", type="primary", use_container_width=True)

with right:
    st.subheader("Weather-Aware Forecasting")
    st.write(
        "The v3 model combines recent AQI history with Open-Meteo forecast variables. "
        "Recursive steps feed predicted AQI values back into lag features while future "
        "weather values are injected from the live forecast."
    )
    st.divider()
    st.subheader("AQI Category Legend")
    legend_cols = st.columns(3)
    legend_items = [
        ("Good", "0-50"),
        ("Satisfactory", "51-100"),
        ("Moderate", "101-200"),
        ("Poor", "201-300"),
        ("Very Poor", "301-400"),
        ("Severe", "401-500"),
    ]
    for index, (label, range_text) in enumerate(legend_items):
        with legend_cols[index % 3]:
            st.markdown(
                f"<span style='display:inline-block;width:0.85rem;height:0.85rem;"
                f"border-radius:0.15rem;background:{AQI_COLORS[label]};margin-right:0.4rem;'></span>"
                f"<strong>{label}</strong> ({range_text})",
                unsafe_allow_html=True,
            )


if predict_clicked:
    try:
        with st.spinner("Generating forecast..."):
            forecast_result = predict_future_aqi(selected_city, days, include_metadata=True)
    except Exception as exc:
        st.error(f"Prediction could not be generated: {str(exc)}")
        st.stop()

    metadata = forecast_result["metadata"]
    predictions = forecast_result["predictions"]
    results_df = pd.DataFrame(predictions)
    results_df.insert(0, "Day", range(1, len(results_df) + 1))
    if "Lower_Bound" in results_df.columns:
        results_df["Lower_Bound"] = results_df["Lower_Bound"].clip(lower=0)

    if "Predicted_AQI" not in results_df.columns:
        st.error("Prediction output did not contain Predicted_AQI.")
        st.stop()

    results_df["Category"] = results_df["Predicted_AQI"].apply(categorize_aqi)
    display_df = results_df.rename(
        columns={
            "Timestamp": "Date",
            "Predicted_AQI": "Predicted AQI",
            "Lower_Bound": "Lower Bound",
            "Upper_Bound": "Upper Bound",
            "AQI_Category": "Model Category",
            "temperature_2m_max": "Max Temp (C)",
            "temperature_2m_min": "Min Temp (C)",
            "precipitation_sum": "Rain (mm)",
            "wind_speed_10m_max": "Wind Max (km/h)",
            "relative_humidity_2m_max": "Humidity Max (%)",
        }
    )
    preferred_columns = [
        "Day",
        "Date",
        "City",
        "Predicted AQI",
        "Lower Bound",
        "Upper Bound",
        "Category",
        "Model Category",
        "Max Temp (C)",
        "Min Temp (C)",
        "Rain (mm)",
        "Wind Max (km/h)",
        "Humidity Max (%)",
    ]
    display_df = display_df[[col for col in preferred_columns if col in display_df.columns]]

    st.subheader("Forecast Results")

    st.markdown("#### Forecast Summary")
    metric_cols = st.columns(4)
    peak_row = display_df.loc[display_df["Predicted AQI"].idxmax()]
    average_aqi = display_df["Predicted AQI"].mean()
    metric_cols[0].metric("Peak AQI", f"{peak_row['Predicted AQI']:.1f}")
    metric_cols[1].metric("Peak Category", peak_row["Category"])
    metric_cols[2].metric("Average AQI", f"{average_aqi:.1f}")
    metric_cols[3].metric("Forecast Days", len(display_df))

    st.markdown("#### Forecast Metadata")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Model", metadata["model_name"])
    meta_cols[1].metric("Forecast Type", metadata["forecast_type"])
    meta_cols[2].metric("Weather Source", metadata["weather_source"])
    meta_cols[3].metric("Available Weather Days", metadata["available_weather_days"])
    st.caption(f"Generated at {metadata['prediction_timestamp']}")

    st.divider()
    st.markdown("#### Daily Forecast Table")
    styled_df = display_df.style.applymap(style_category, subset=["Category"])
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("#### Forecast Trend")
    chart_df = display_df[["Day", "Predicted AQI", "Lower Bound", "Upper Bound"]].melt(
        "Day",
        var_name="Series",
        value_name="AQI",
    )
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Day:O", title="Forecast Day"),
            y=alt.Y("AQI:Q", title="AQI", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                title="Forecast Series",
                scale=alt.Scale(
                    domain=["Predicted AQI", "Lower Bound", "Upper Bound"],
                    range=["#4fc3f7", "#81c784", "#ffb74d"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Day:O", title="Day"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("AQI:Q", title="AQI", format=".1f"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    weather_columns = [
        "Day",
        "Date",
        "Max Temp (C)",
        "Min Temp (C)",
        "Rain (mm)",
        "Wind Max (km/h)",
        "Humidity Max (%)",
    ]
    with st.expander("Weather forecast used by the model"):
        st.dataframe(display_df[weather_columns], use_container_width=True)

    st.divider()
    st.download_button(
        "Download Forecast CSV",
        display_df.to_csv(index=False),
        file_name=f"{selected_city}_aqi_forecast.csv",
        mime="text/csv",
    )
else:
    st.subheader("Forecast Results")
    st.write("Select a city and forecast horizon, then run the AQI prediction.")
