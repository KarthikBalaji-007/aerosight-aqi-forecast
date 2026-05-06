from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests


FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
WEATHER_SOURCE = "Open-Meteo Forecast API"
MODEL_NAME = "Random Forest v3"
FORECAST_TYPE = "Recursive Multi-Step Forecast"

MODEL_PATH = Path("outputs/models/best_model_v3.pkl")
SCALER_PATH = Path("outputs/models/scaler_v3.pkl")
FEATURE_COLUMNS_PATH = Path("outputs/models/feature_columns_v3.json")
TEST_SPLIT_PATH = Path("data/splits/test_v3.csv")

FORECAST_DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_max",
]
FORECAST_DAYS = 16
PREDICTION_INTERVAL = 10.0

CITY_COORDINATES = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "gwalior": (26.2183, 78.1828),
    "visakhapatnam": (17.6868, 83.2185),
}


def _get_json_without_env_proxy(url: str, params: dict[str, Any]) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Weather API request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Weather API returned a non-JSON response") from exc

    if not isinstance(payload, dict):
        raise ValueError("Weather API returned an invalid JSON payload")
    return payload


def _validate_weather_payload(city: str, payload: dict[str, Any]) -> dict[str, Any]:
    if "daily" not in payload:
        raise ValueError("Invalid weather API response: missing 'daily' object")
    if not isinstance(payload["daily"], dict):
        raise ValueError("Invalid weather API response: 'daily' must be an object")

    daily = payload["daily"]
    required_daily_keys = ["time"] + FORECAST_DAILY_VARIABLES
    missing_keys = [key for key in required_daily_keys if key not in daily]
    if missing_keys:
        raise ValueError(f"Weather forecast for {city} is missing required daily fields: {missing_keys}")

    lengths: dict[str, int] = {}
    for key in required_daily_keys:
        values = daily[key]
        if not isinstance(values, list):
            raise ValueError(f"Weather forecast field '{key}' for {city} must be an array")
        lengths[key] = len(values)

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Weather forecast daily arrays have mismatched lengths for {city}: {lengths}")
    if lengths["time"] == 0:
        raise ValueError(f"Weather forecast for {city} returned no daily rows")

    return daily


def _validate_weather_frame(city: str, weather_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["Timestamp", "City"] + FORECAST_DAILY_VARIABLES
    missing_columns = [column for column in required_columns if column not in weather_df.columns]
    if missing_columns:
        raise ValueError(f"Weather forecast frame for {city} is missing columns: {missing_columns}")

    validated = weather_df[required_columns].copy()
    validated["Timestamp"] = pd.to_datetime(validated["Timestamp"], errors="coerce")
    validated = validated.dropna(subset=["Timestamp"]).reset_index(drop=True)
    if validated.empty:
        raise ValueError(f"Weather forecast for {city} has no valid forecast dates")

    for column in FORECAST_DAILY_VARIABLES:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    invalid_columns = [column for column in FORECAST_DAILY_VARIABLES if validated[column].isna().any()]
    if invalid_columns:
        raise ValueError(f"Weather forecast for {city} contains invalid numeric values: {invalid_columns}")

    return validated


def load_artifacts() -> tuple[Any, Any, list[str]]:
    for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    return model, scaler, feature_columns


def fetch_weather_forecast(city: str) -> pd.DataFrame:
    city = city.lower()
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city '{city}'. Available cities: {sorted(CITY_COORDINATES)}")

    latitude, longitude = CITY_COORDINATES[city]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(FORECAST_DAILY_VARIABLES),
        "timezone": TIMEZONE,
        "forecast_days": FORECAST_DAYS,
    }

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            payload = _get_json_without_env_proxy(FORECAST_API_URL, params)
            daily = _validate_weather_payload(city, payload)
            weather_df = pd.DataFrame(
                {
                    "Timestamp": daily["time"],
                    "temperature_2m_max": daily["temperature_2m_max"],
                    "temperature_2m_min": daily["temperature_2m_min"],
                    "precipitation_sum": daily["precipitation_sum"],
                    "wind_speed_10m_max": daily["wind_speed_10m_max"],
                    "relative_humidity_2m_max": daily["relative_humidity_2m_max"],
                }
            )
            weather_df["City"] = city
            return _validate_weather_frame(city, weather_df)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                print(f"{city}: forecast attempt {attempt} failed ({exc}); retrying in 2s")
                time.sleep(2)

    raise RuntimeError(
        f"Unable to fetch a complete weather forecast for {city} from {WEATHER_SOURCE}"
    ) from last_error


def get_latest_row(city: str) -> pd.Series:
    city = city.lower()
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing split file: {TEST_SPLIT_PATH}")

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    if "Timestamp" not in test_df.columns or "City" not in test_df.columns:
        raise ValueError(f"{TEST_SPLIT_PATH} must contain Timestamp and City columns")

    city_df = test_df[test_df["City"].str.lower() == city].copy()
    if city_df.empty:
        raise ValueError(f"No rows found for city '{city}' in {TEST_SPLIT_PATH}")

    city_df["Timestamp"] = pd.to_datetime(city_df["Timestamp"], errors="coerce")
    city_df = city_df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    if city_df.empty:
        raise ValueError(f"No valid Timestamp rows found for city '{city}'")

    return city_df.iloc[-1]


def _latest_row_to_raw_features(latest_row: pd.Series, scaler: Any, feature_columns: list[str]) -> pd.Series:
    missing_features = [col for col in feature_columns if col not in latest_row.index]
    if missing_features:
        raise ValueError(f"Latest row missing model features: {missing_features}")

    scaled_values = pd.DataFrame([latest_row[feature_columns].astype(float).to_dict()], columns=feature_columns)
    raw_values = scaler.inverse_transform(scaled_values)[0]
    return pd.Series(raw_values, index=feature_columns, dtype=float)


def _month_to_season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _predict_one(model: Any, scaled_features: pd.DataFrame, raw_features: pd.Series) -> float:
    if isinstance(model, dict) and model.get("type") == "naive_baseline":
        return float(raw_features["AQI_lag1"])
    return float(model.predict(scaled_features)[0])


def build_forecast_metadata(city: str, requested_days: int, available_weather_days: int) -> dict[str, Any]:
    return {
        "prediction_timestamp": pd.Timestamp.now(tz=TIMEZONE).isoformat(),
        "weather_source": WEATHER_SOURCE,
        "model_name": MODEL_NAME,
        "forecast_type": FORECAST_TYPE,
        "city": city.lower(),
        "requested_days": int(requested_days),
        "available_weather_days": int(available_weather_days),
    }


def get_aqi_category(aqi: float) -> str:
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


def _build_aqi_history(city: str, latest_row: pd.Series) -> list[float]:
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    test_df["Timestamp"] = pd.to_datetime(test_df["Timestamp"], errors="coerce")
    latest_timestamp = pd.Timestamp(latest_row["Timestamp"])

    history = (
        test_df[
            (test_df["City"].str.lower() == city.lower())
            & (test_df["Timestamp"] <= latest_timestamp)
        ]
        .dropna(subset=["Timestamp", "AQI"])
        .sort_values("Timestamp")["AQI"]
        .astype(float)
        .tail(7)
        .tolist()
    )

    if not history:
        raise ValueError(f"Cannot build AQI history for city '{city}'")

    while len(history) < 7:
        history.insert(0, history[0])
    return history[-7:]


def _validate_prediction_output(y_pred: float, lower_bound: float, upper_bound: float) -> None:
    if not np.isfinite(y_pred):
        raise ValueError(f"Model produced a non-finite AQI prediction: {y_pred}")
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError(
            f"Model produced non-finite confidence bounds: lower={lower_bound}, upper={upper_bound}"
        )
    if lower_bound < 0:
        raise ValueError(f"Lower confidence bound cannot be negative: {lower_bound}")
    if upper_bound < lower_bound:
        raise ValueError(f"Upper confidence bound cannot be below lower bound: {upper_bound} < {lower_bound}")


def predict_future_aqi(city: str, days: int, include_metadata: bool = False) -> list[dict[str, Any]] | dict[str, Any]:
    if days <= 0:
        raise ValueError("days must be a positive integer")

    model, scaler, feature_columns = load_artifacts()
    weather_df = fetch_weather_forecast(city)
    max_days = len(weather_df)
    if days > max_days:
        raise ValueError(
            f"Requested {days} forecast days, but only {max_days} days are available from {WEATHER_SOURCE}"
        )
    weather_df = _validate_weather_frame(city, weather_df)

    latest_row = get_latest_row(city)
    current = _latest_row_to_raw_features(latest_row, scaler, feature_columns)

    predictions: list[dict[str, Any]] = []
    # Future pollutant rolling statistics are recursively approximated using previously predicted states because future pollutant observations are unavailable during inference. Meteorological variables are injected using live Open-Meteo forecast data.
    # AQI_lag1 is updated after every forecast step using the latest predicted AQI.
    # AQI_lag7 is updated from the rolling seven-day AQI history, which includes
    # model predictions once the forecast moves beyond the latest observed row.
    # PM2.5 rolling statistics remain in the carried-forward model state as a
    # deployment-time approximation because future pollutant observations do not exist.
    aqi_history = _build_aqi_history(city, latest_row)
    last_weather = {
        "wind_speed_10m_max": float(current.get("wind_speed_10m_max", np.nan)),
        "precipitation_sum": float(current.get("precipitation_sum", np.nan)),
        "temperature_2m_max": float(current.get("temperature_2m_max", np.nan)),
        "relative_humidity_2m_max": float(current.get("relative_humidity_2m_max", np.nan)),
    }

    for step in range(days):
        weather = weather_df.iloc[step]
        forecast_date = pd.Timestamp(weather["Timestamp"])

        if "wind_speed_lag1" in current:
            current["wind_speed_lag1"] = last_weather["wind_speed_10m_max"]
        if "precip_lag1" in current:
            current["precip_lag1"] = last_weather["precipitation_sum"]
        if "temp_max_lag1" in current:
            current["temp_max_lag1"] = last_weather["temperature_2m_max"]
        if "humidity_lag1" in current:
            current["humidity_lag1"] = last_weather["relative_humidity_2m_max"]

        for col in FORECAST_DAILY_VARIABLES:
            current[col] = float(weather[col])

        if "day_of_week" in current:
            current["day_of_week"] = forecast_date.dayofweek
        if "month" in current:
            current["month"] = forecast_date.month
        if "season" in current:
            current["season"] = _month_to_season(forecast_date.month)
        if "is_weekend" in current:
            current["is_weekend"] = int(forecast_date.dayofweek >= 5)
        if "year" in current:
            current["year"] = forecast_date.year

        ordered = current.reindex(feature_columns).astype(float)
        if ordered.isna().any():
            missing = ordered[ordered.isna()].index.tolist()
            raise ValueError(f"Forecast feature vector contains missing values: {missing}")
        ordered_df = pd.DataFrame([ordered.to_dict()], columns=feature_columns)
        scaled = pd.DataFrame(scaler.transform(ordered_df), columns=feature_columns)
        y_pred = _predict_one(model, scaled, ordered)
        lower_bound = max(0.0, y_pred - PREDICTION_INTERVAL)
        upper_bound = min(500.0, y_pred + PREDICTION_INTERVAL)
        _validate_prediction_output(y_pred, lower_bound, upper_bound)

        predictions.append(
            {
                "Timestamp": forecast_date.date().isoformat(),
                "City": city.lower(),
                "Predicted_AQI": y_pred,
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
                "AQI_Category": get_aqi_category(y_pred),
                "temperature_2m_max": float(weather["temperature_2m_max"]),
                "temperature_2m_min": float(weather["temperature_2m_min"]),
                "precipitation_sum": float(weather["precipitation_sum"]),
                "wind_speed_10m_max": float(weather["wind_speed_10m_max"]),
                "relative_humidity_2m_max": float(weather["relative_humidity_2m_max"]),
            }
        )

        aqi_history.append(float(y_pred))
        aqi_history = aqi_history[-7:]

        if "AQI_lag1" in current:
            current["AQI_lag1"] = aqi_history[-1]
        if "AQI_lag7" in current:
            current["AQI_lag7"] = aqi_history[0]

        last_weather = {
            "wind_speed_10m_max": float(weather["wind_speed_10m_max"]),
            "precipitation_sum": float(weather["precipitation_sum"]),
            "temperature_2m_max": float(weather["temperature_2m_max"]),
            "relative_humidity_2m_max": float(weather["relative_humidity_2m_max"]),
        }

    if include_metadata:
        return {
            "metadata": build_forecast_metadata(city, days, max_days),
            "predictions": predictions,
        }
    return predictions


def get_max_forecast_days(city: str) -> int:
    return len(fetch_weather_forecast(city))


if __name__ == "__main__":
    print(get_max_forecast_days("delhi"))
    print(predict_future_aqi("delhi", 5))
