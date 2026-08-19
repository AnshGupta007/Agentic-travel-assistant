"""Live weather service provider using OpenWeatherMap API."""

import logging
from typing import Optional
import httpx
from config.settings import settings
from interfaces.weather import WeatherServiceInterface
from models.weather import WeatherDataPoint, WeatherResult

logger = logging.getLogger(__name__)


class LiveWeatherProvider(WeatherServiceInterface):
    """Live weather service provider fetching real-time data from OpenWeatherMap."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        """Initialize live weather provider.

        Args:
            api_key: OpenWeatherMap API key (defaults to settings.openweather_api_key).
            timeout: HTTP timeout in seconds (default 10.0s).
        """
        self.api_key = api_key or settings.openweather_api_key
        self.timeout = timeout

    def get_weather(self, city: str, days: int = 7) -> WeatherResult:
        """Fetch live multi-day weather forecast for a target city.

        Args:
            city: Target city name.
            days: Number of forecast days (1 to 14, default 7).

        Returns:
            WeatherResult object containing structured forecast or error message.
        """
        city_clean = city.strip()
        if not city_clean:
            return WeatherResult(
                city=city,
                forecast=[],
                error="Invalid empty city name provided.",
            )

        if not self.api_key:
            logger.warning("OpenWeather API key is not configured for LiveWeatherProvider.")
            return WeatherResult(
                city=city_clean,
                forecast=[],
                error="OpenWeather API key is not configured.",
            )

        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city_clean,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=params)

            if response.status_code == 404:
                return WeatherResult(
                    city=city_clean,
                    forecast=[],
                    error=f"City '{city_clean}' not found in OpenWeather service.",
                )

            if response.status_code != 200:
                return WeatherResult(
                    city=city_clean,
                    forecast=[],
                    error=f"OpenWeather service returned HTTP {response.status_code}: {response.text}",
                )

            data = response.json()
            raw_list = data.get("list", [])

            # Extract daily data points (selecting one measurement per unique date up to requested days)
            seen_dates = set()
            forecast_points: list[WeatherDataPoint] = []
            num_days = max(1, min(days, 14))

            for item in raw_list:
                dt_txt = item.get("dt_txt", "")  # e.g., "2026-08-20 12:00:00"
                date_str = dt_txt.split(" ")[0] if " " in dt_txt else dt_txt

                if date_str and date_str not in seen_dates:
                    seen_dates.add(date_str)
                    main_info = item.get("main", {})
                    weather_list = item.get("weather", [{}])
                    condition_str = weather_list[0].get("main", "Clear") if weather_list else "Clear"
                    temp_val = round(float(main_info.get("temp", 0.0)), 1)

                    forecast_points.append(
                        WeatherDataPoint(
                            date=date_str,
                            temperature=temp_val,
                            condition=condition_str,
                        )
                    )

                    if len(forecast_points) >= num_days:
                        break

            current_temp = forecast_points[0].temperature if forecast_points else None
            current_condition = forecast_points[0].condition if forecast_points else None

            return WeatherResult(
                city=city_clean,
                forecast=forecast_points,
                current_temp=current_temp,
                condition=current_condition,
                error=None,
            )

        except httpx.RequestError as exc:
            logger.error("Network error accessing OpenWeather API: %s", exc)
            return WeatherResult(
                city=city_clean,
                forecast=[],
                error=f"Weather service network error: {str(exc)}",
            )
        except Exception as exc:
            logger.error("Unexpected error in LiveWeatherProvider: %s", exc)
            return WeatherResult(
                city=city_clean,
                forecast=[],
                error=f"Unexpected weather service failure: {str(exc)}",
            )


# Alias for flexibility
LiveWeatherService = LiveWeatherProvider
