"""Unit tests for Phase 1 Mock Data Providers."""

import pytest
from interfaces.weather import WeatherServiceInterface
from interfaces.images import ImageServiceInterface
from interfaces.search import SearchServiceInterface
from models.weather import WeatherResult, WeatherDataPoint
from models.city import CityKnowledge
from mocks.weather import MockWeatherProvider, MockWeatherService
from mocks.images import MockImageProvider, MockImageService
from mocks.search import MockSearchProvider, MockSearchService


# List of mandatory cities required by Phase 1 specification
MANDATORY_CITIES = ["Paris", "Tokyo", "New York", "Kyoto", "Snohomish"]


# --- Weather Mock Tests ---

def test_mock_weather_interface_compliance():
    provider = MockWeatherProvider(latency=0.0)
    assert isinstance(provider, WeatherServiceInterface)


@pytest.mark.parametrize("city", MANDATORY_CITIES + ["Unknown City"])
def test_mock_weather_returns_valid_weather_result(city):
    provider = MockWeatherProvider(latency=0.0)
    result = provider.get_weather(city, days=7)

    assert isinstance(result, WeatherResult)
    assert result.city == city.strip()
    assert result.error is None
    assert result.current_temp is not None
    assert isinstance(result.current_temp, float)
    assert result.condition is not None
    assert isinstance(result.condition, str)
    assert len(result.forecast) == 7

    for point in result.forecast:
        assert isinstance(point, WeatherDataPoint)
        assert isinstance(point.date, str)
        assert isinstance(point.temperature, float)
        assert isinstance(point.condition, str)


def test_mock_weather_days_parameter():
    provider = MockWeatherProvider(latency=0.0)

    res5 = provider.get_weather("Tokyo", days=5)
    assert len(res5.forecast) == 5

    res10 = provider.get_weather("Paris", days=10)
    assert len(res10.forecast) == 10


# --- Image Mock Tests ---

def test_mock_image_interface_compliance():
    provider = MockImageProvider(latency=0.0)
    assert isinstance(provider, ImageServiceInterface)


@pytest.mark.parametrize("city", MANDATORY_CITIES + ["Unknown City"])
def test_mock_image_returns_urls(city):
    provider = MockImageProvider(latency=0.0)
    urls = provider.search_images(city, limit=5)

    assert isinstance(urls, list)
    assert len(urls) == 5
    for url in urls:
        assert isinstance(url, str)
        assert url.startswith("http://") or url.startswith("https://")


def test_mock_image_limit_parameter():
    provider = MockImageProvider(latency=0.0)

    urls3 = provider.search_images("Tokyo", limit=3)
    assert len(urls3) == 3

    urls1 = provider.search_images("Paris", limit=1)
    assert len(urls1) == 1


# --- Search Mock Tests ---

def test_mock_search_interface_compliance():
    provider = MockSearchProvider(latency=0.0)
    assert isinstance(provider, SearchServiceInterface)


@pytest.mark.parametrize("city", MANDATORY_CITIES)
def test_mock_search_known_cities(city):
    provider = MockSearchProvider(latency=0.0)
    knowledge = provider.search_city(city)

    assert knowledge is not None
    assert isinstance(knowledge, CityKnowledge)
    assert knowledge.city == city
    assert knowledge.country != "Unknown"
    assert len(knowledge.description) > 10
    assert len(knowledge.highlights) > 0
    assert len(knowledge.culture_tips) > 0
    assert knowledge.source == "web_search"


def test_mock_search_fallback_unknown_city():
    provider = MockSearchProvider(latency=0.0)
    knowledge = provider.search_city("Atlantis City")

    assert knowledge is not None
    assert isinstance(knowledge, CityKnowledge)
    assert knowledge.city == "Atlantis City"
    assert knowledge.source == "web_search"
    assert len(knowledge.description) > 0


# --- Alias Tests ---

def test_mock_provider_aliases():
    ws = MockWeatherService(latency=0.0)
    assert isinstance(ws, WeatherServiceInterface)

    ims = MockImageService(latency=0.0)
    assert isinstance(ims, ImageServiceInterface)

    ss = MockSearchService(latency=0.0)
    assert isinstance(ss, SearchServiceInterface)
