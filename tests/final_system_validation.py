from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_aqi import (  # noqa: E402
    FORECAST_DAILY_VARIABLES,
    WEATHER_SOURCE,
    fetch_weather_forecast,
    get_aqi_category,
    get_max_forecast_days,
    predict_future_aqi,
)


Validation = Callable[[], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_weather_api_fetch() -> None:
    weather = fetch_weather_forecast("delhi")
    required_columns = ["Timestamp", "City"] + FORECAST_DAILY_VARIABLES
    _assert(not weather.empty, "weather forecast should not be empty")
    _assert(all(column in weather.columns for column in required_columns), "weather columns are incomplete")
    _assert(len(weather) == get_max_forecast_days("delhi"), "max forecast days should match fetched rows")


def validate_forecast_generation() -> None:
    result = predict_future_aqi("delhi", 3, include_metadata=True)
    _assert("metadata" in result and "predictions" in result, "metadata forecast response is malformed")
    _assert(result["metadata"]["weather_source"] == WEATHER_SOURCE, "weather source metadata mismatch")
    _assert(len(result["predictions"]) == 3, "forecast should return requested number of rows")


def validate_multi_city_forecasting() -> None:
    for city in ["delhi", "mumbai", "bengaluru"]:
        rows = predict_future_aqi(city, 2)
        _assert(len(rows) == 2, f"{city} forecast should return two rows")
        _assert(all(row["City"] == city for row in rows), f"{city} forecast city labels mismatch")


def validate_streamlit_backend_assumptions() -> None:
    result = predict_future_aqi("delhi", 2, include_metadata=True)
    metadata = result["metadata"]
    row = result["predictions"][0]
    expected_prediction_keys = {
        "Timestamp",
        "City",
        "Predicted_AQI",
        "Lower_Bound",
        "Upper_Bound",
        "AQI_Category",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "relative_humidity_2m_max",
    }
    expected_metadata_keys = {
        "prediction_timestamp",
        "weather_source",
        "model_name",
        "forecast_type",
        "city",
        "requested_days",
        "available_weather_days",
    }
    _assert(expected_prediction_keys.issubset(row), "prediction keys expected by Streamlit are missing")
    _assert(expected_metadata_keys.issubset(metadata), "metadata keys expected by Streamlit are missing")


def validate_recursive_prediction_sanity() -> None:
    rows = predict_future_aqi("delhi", 5)
    predictions = [row["Predicted_AQI"] for row in rows]
    _assert(len(predictions) == 5, "recursive forecast should preserve requested horizon")
    _assert(len({round(value, 3) for value in predictions}) > 1, "recursive forecast should not be constant")
    _assert(all(0 <= row["Lower_Bound"] <= row["Upper_Bound"] <= 500 for row in rows), "bounds are invalid")


def validate_prediction_finiteness() -> None:
    rows = predict_future_aqi("delhi", 5)
    numeric_fields = [
        "Predicted_AQI",
        "Lower_Bound",
        "Upper_Bound",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "relative_humidity_2m_max",
    ]
    for row in rows:
        for field in numeric_fields:
            _assert(np.isfinite(row[field]), f"{field} should be finite")


def validate_aqi_category_correctness() -> None:
    cases = {
        50: "Good",
        100: "Satisfactory",
        200: "Moderate",
        300: "Poor",
        400: "Very Poor",
        401: "Severe",
    }
    for value, expected in cases.items():
        _assert(get_aqi_category(value) == expected, f"AQI category mismatch for {value}")


def run_validation(name: str, validation: Validation) -> bool:
    try:
        validation()
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        return False

    print(f"PASS {name}")
    return True


def main() -> int:
    validations: list[tuple[str, Validation]] = [
        ("Weather API fetch", validate_weather_api_fetch),
        ("Forecast generation", validate_forecast_generation),
        ("Multi-city forecasting", validate_multi_city_forecasting),
        ("Streamlit/backend assumptions", validate_streamlit_backend_assumptions),
        ("Recursive prediction sanity", validate_recursive_prediction_sanity),
        ("Prediction finiteness", validate_prediction_finiteness),
        ("AQI category correctness", validate_aqi_category_correctness),
    ]

    results = [run_validation(name, validation) for name, validation in validations]
    passed = sum(results)
    total = len(results)
    print(f"\nSummary: {passed}/{total} validations passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
