"""Unit tests for LiveWeatherProvider."""

from unittest.mock import MagicMock, patch
import httpx
from models.weather import WeatherResult
from providers.weather_live import LiveWeatherProvider


def test_weather_live_missing_key():
    provider = LiveWeatherProvider(api_key=None)
    result = provider.get_weather("Paris")
    assert isinstance(result, WeatherResult)
    assert result.city == "Paris"
    assert result.error == "OpenWeather API key is not configured."
    assert len(result.forecast) == 0


def test_weather_live_empty_city():
    provider = LiveWeatherProvider(api_key="test_key")
    result = provider.get_weather("   ")
    assert isinstance(result, WeatherResult)
    assert "Invalid empty city name" in result.error


@patch("httpx.Client.get")
def test_weather_live_success_200(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "list": [
            {
                "dt_txt": "2026-08-20 12:00:00",
                "main": {"temp": 24.5},
                "weather": [{"main": "Sunny", "description": "clear sky"}],
            },
            {
                "dt_txt": "2026-08-21 12:00:00",
                "main": {"temp": 22.0},
                "weather": [{"main": "Cloudy", "description": "few clouds"}],
            },
        ]
    }
    mock_get.return_value = mock_response

    provider = LiveWeatherProvider(api_key="dummy_key")
    result = provider.get_weather("Tokyo", days=5)

    assert result.city == "Tokyo"
    assert result.error is None
    assert len(result.forecast) == 2
    assert result.forecast[0].date == "2026-08-20"
    assert result.forecast[0].temperature == 24.5
    assert result.forecast[0].condition == "Sunny"
    assert result.current_temp == 24.5
    assert result.condition == "Sunny"


@patch("httpx.Client.get")
def test_weather_live_not_found_404(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    provider = LiveWeatherProvider(api_key="dummy_key")
    result = provider.get_weather("NonexistentCity")

    assert result.city == "NonexistentCity"
    assert result.error is not None
    assert "not found" in result.error.lower()


@patch("httpx.Client.get")
def test_weather_live_network_error(mock_get):
    mock_get.side_effect = httpx.RequestError("Connection timeout", request=MagicMock())

    provider = LiveWeatherProvider(api_key="dummy_key")
    result = provider.get_weather("Tokyo")

    assert result.city == "Tokyo"
    assert result.error is not None
    assert "network error" in result.error.lower()
