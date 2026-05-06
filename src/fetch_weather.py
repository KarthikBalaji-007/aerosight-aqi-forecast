from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TIMEZONE = "Asia/Kolkata"
OUTPUT_DIR = Path("data/raw")

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_max",
]

CITIES = {
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


def _get_json_without_env_proxy(url: str, params: dict) -> dict:
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_city_weather(city: str, latitude: float, longitude: float) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": TIMEZONE,
    }

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            payload = _get_json_without_env_proxy(API_URL, params)
            if "daily" not in payload:
                raise ValueError("Invalid API response: 'daily' missing")

            daily = payload["daily"]
            required_daily_keys = ["time"] + DAILY_VARIABLES
            missing_keys = [key for key in required_daily_keys if key not in daily]
            if missing_keys:
                raise ValueError(f"Weather response missing daily keys for {city}: {missing_keys}")

            lengths = {key: len(daily[key]) for key in required_daily_keys}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Weather daily arrays have mismatched lengths for {city}: {lengths}")

            df = pd.DataFrame(
                {
                    "Timestamp": daily["time"],
                    "temperature_2m_max": daily["temperature_2m_max"],
                    "temperature_2m_min": daily["temperature_2m_min"],
                    "precipitation_sum": daily["precipitation_sum"],
                    "wind_speed_10m_max": daily["wind_speed_10m_max"],
                    "relative_humidity_2m_max": daily["relative_humidity_2m_max"],
                }
            )
            df["City"] = city
            return df
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                print(f"{city}: attempt {attempt} failed ({exc}); retrying in 2s")
                time.sleep(2)

    raise RuntimeError(f"Failed to fetch weather data for {city} after 3 attempts") from last_error


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    city_frames: list[pd.DataFrame] = []
    rows_per_city: dict[str, int] = {}

    for city, (latitude, longitude) in CITIES.items():
        print(f"Fetching {city}...")
        city_df = fetch_city_weather(city, latitude, longitude)

        output_path = OUTPUT_DIR / f"weather_{city}.csv"
        city_df.to_csv(output_path, index=False)

        rows_per_city[city] = len(city_df)
        city_frames.append(city_df)
        print(f"{city}: {len(city_df)} rows saved to {output_path}")

    combined = pd.concat(city_frames, ignore_index=True)
    combined_path = OUTPUT_DIR / "weather_all_cities.csv"
    combined.to_csv(combined_path, index=False)

    print("\nRows per city:")
    for city, row_count in rows_per_city.items():
        print(f"- {city}: {row_count}")

    print(f"\nFinal shape: {combined.shape}")
    print("\nNull counts:")
    print(combined.isna().sum())
    print(f"\nCombined weather data saved to {combined_path}")


if __name__ == "__main__":
    main()
