"""Weather service interface contract."""

from abc import ABC, abstractmethod
from models.weather import WeatherResult


class WeatherServiceInterface(ABC):
    """Abstract base class for weather providers (live and mock)."""

    @abstractmethod
    def get_weather(self, city: str, days: int = 7) -> WeatherResult:
        """Fetch weather metrics and multi-day forecast for a city.

        Args:
            city: City name to look up.
            days: Number of forecast days (5-7).

        Returns:
            WeatherResult object containing city name, forecast points, current conditions, or error message.
        """
        pass
