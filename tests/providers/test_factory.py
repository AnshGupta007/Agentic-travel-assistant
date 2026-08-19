"""Unit tests for Provider Factory layer."""

from config.settings import ProviderMode, Settings
from mocks.images import MockImageProvider
from mocks.search import MockSearchProvider
from mocks.weather import MockWeatherProvider
from providers.factory import get_image_service, get_search_service, get_weather_service
from providers.images_live import LiveImageProvider
from providers.search_live import LiveSearchProvider
from providers.weather_live import LiveWeatherProvider


def test_factory_mock_mode():
    mock_settings = Settings(provider_mode=ProviderMode.MOCK)

    weather_svc = get_weather_service(mock_settings)
    image_svc = get_image_service(mock_settings)
    search_svc = get_search_service(mock_settings)

    assert isinstance(weather_svc, MockWeatherProvider)
    assert isinstance(image_svc, MockImageProvider)
    assert isinstance(search_svc, MockSearchProvider)


def test_factory_live_mode():
    live_settings = Settings(
        provider_mode=ProviderMode.LIVE,
        openweather_api_key="ow_key",
        unsplash_api_key="us_key",
        tavily_api_key="tv_key",
    )

    weather_svc = get_weather_service(live_settings)
    image_svc = get_image_service(live_settings)
    search_svc = get_search_service(live_settings)

    assert isinstance(weather_svc, LiveWeatherProvider)
    assert isinstance(image_svc, LiveImageProvider)
    assert isinstance(search_svc, LiveSearchProvider)

    # Check contracts are fulfilled without raising
    weather_res = weather_svc.get_weather("Tokyo")
    assert weather_res.city == "Tokyo"

    image_urls = image_svc.search_images("Tokyo", limit=2)
    assert len(image_urls) == 2

    search_res = search_svc.search_city("Tokyo")
    assert search_res is not None
    assert search_res.city == "Tokyo"
