"""Provider factory for resolving live or mock service implementations based on application configuration."""

from typing import Optional
from config.settings import Settings, settings
from interfaces.images import ImageServiceInterface
from interfaces.search import SearchServiceInterface
from interfaces.weather import WeatherServiceInterface
from mocks.images import MockImageProvider
from mocks.search import MockSearchProvider
from mocks.weather import MockWeatherProvider
from providers.images_live import LiveImageProvider
from providers.search_live import LiveSearchProvider
from providers.weather_live import LiveWeatherProvider


def get_weather_service(override_settings: Optional[Settings] = None) -> WeatherServiceInterface:
    """Resolve weather service instance based on provider_mode setting.

    Args:
        override_settings: Optional Settings instance override.

    Returns:
        WeatherServiceInterface implementation (MockWeatherProvider or LiveWeatherProvider).
    """
    app_settings = override_settings or settings
    if app_settings.is_mock():
        return MockWeatherProvider()
    return LiveWeatherProvider(api_key=app_settings.openweather_api_key)


def get_image_service(override_settings: Optional[Settings] = None) -> ImageServiceInterface:
    """Resolve image service instance based on provider_mode setting.

    Args:
        override_settings: Optional Settings instance override.

    Returns:
        ImageServiceInterface implementation (MockImageProvider or LiveImageProvider).
    """
    app_settings = override_settings or settings
    if app_settings.is_mock():
        return MockImageProvider()
    return LiveImageProvider(api_key=app_settings.unsplash_api_key)


def get_search_service(override_settings: Optional[Settings] = None) -> SearchServiceInterface:
    """Resolve web search service instance based on provider_mode setting.

    Args:
        override_settings: Optional Settings instance override.

    Returns:
        SearchServiceInterface implementation (MockSearchProvider or LiveSearchProvider).
    """
    app_settings = override_settings or settings
    if app_settings.is_mock():
        return MockSearchProvider()
    return LiveSearchProvider(api_key=app_settings.tavily_api_key)
