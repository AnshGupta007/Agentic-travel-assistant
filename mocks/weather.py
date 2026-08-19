"""Mock weather data provider simulating realistic weather service responses."""

import time
from datetime import date, timedelta
from typing import Optional
from interfaces.weather import WeatherServiceInterface
from models.weather import WeatherDataPoint, WeatherResult


# Preset mock weather profiles for key cities
_MOCK_WEATHER_DATA = {
    "paris": {
        "country": "France",
        "base_temp": 20.0,
        "conditions": ["Partly Cloudy", "Sunny", "Mild Rain", "Sunny", "Clear", "Cloudy", "Sunny"],
        "temp_variations": [0.0, 2.5, -1.0, 1.5, 3.0, 0.5, -2.0],
    },
    "tokyo": {
        "country": "Japan",
        "base_temp": 24.0,
        "conditions": ["Sunny", "Clear", "Partly Cloudy", "Rain", "Clear", "Sunny", "Clear"],
        "temp_variations": [0.0, 1.0, 2.0, -3.0, 0.5, 1.5, 2.0],
    },
    "new york": {
        "country": "United States",
        "base_temp": 23.0,
        "conditions": ["Sunny", "Humid", "Thunderstorm", "Clear", "Sunny", "Partly Cloudy", "Sunny"],
        "temp_variations": [0.0, 2.0, -4.0, 1.0, 3.0, 1.5, 0.0],
    },
    "kyoto": {
        "country": "Japan",
        "base_temp": 22.0,
        "conditions": ["Clear", "Sunny", "Light Rain", "Clear", "Sunny", "Partly Cloudy", "Clear"],
        "temp_variations": [0.0, 1.5, -2.0, 0.5, 2.0, 1.0, -1.0],
    },
    "snohomish": {
        "country": "United States",
        "base_temp": 16.0,
        "conditions": ["Light Rain", "Overcast", "Cloudy", "Drizzle", "Partly Cloudy", "Rain", "Overcast"],
        "temp_variations": [0.0, -1.0, 1.0, -0.5, 2.5, -2.0, 0.0],
    },
}

_DEFAULT_PROFILE = {
    "country": "Unknown",
    "base_temp": 21.0,
    "conditions": ["Sunny", "Partly Cloudy", "Clear", "Cloudy", "Sunny", "Clear", "Partly Cloudy"],
    "temp_variations": [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 1.5],
}


class MockWeatherProvider(WeatherServiceInterface):
    """Mock weather service provider with configurable delay and predefined city datasets."""

    def __init__(self, latency: float = 0.05):
        """Initialize mock weather provider.

        Args:
            latency: Simulated delay in seconds (default 0.05s).
        """
        self.latency = latency

    def get_weather(self, city: str, days: int = 7) -> WeatherResult:
        """Fetch simulated weather forecast for a city.

        Args:
            city: Target city name.
            days: Number of forecast days (1 to 14, default 7).

        Returns:
            WeatherResult instance with generated daily forecast.
        """
        if self.latency > 0:
            time.sleep(self.latency)

        city_clean = city.strip()
        city_key = city_clean.lower()

        profile = _MOCK_WEATHER_DATA.get(city_key, _DEFAULT_PROFILE)
        today = date.today()

        forecast_points: list[WeatherDataPoint] = []
        num_days = max(1, min(days, 14))

        for i in range(num_days):
            forecast_date = (today + timedelta(days=i)).isoformat()
            cond_idx = i % len(profile["conditions"])
            var_idx = i % len(profile["temp_variations"])
            temp = round(profile["base_temp"] + profile["temp_variations"][var_idx], 1)
            cond = profile["conditions"][cond_idx]

            forecast_points.append(
                WeatherDataPoint(
                    date=forecast_date,
                    temperature=temp,
                    condition=cond,
                )
            )

        current_temp = forecast_points[0].temperature if forecast_points else profile["base_temp"]
        current_condition = forecast_points[0].condition if forecast_points else "Clear"

        return WeatherResult(
            city=city_clean,
            forecast=forecast_points,
            current_temp=current_temp,
            condition=current_condition,
            error=None,
        )


# Alias for backward compatibility / flexibility
MockWeatherService = MockWeatherProvider
