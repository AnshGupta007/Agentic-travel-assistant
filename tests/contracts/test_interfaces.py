"""Contract tests for service interfaces."""

import pytest
from interfaces.weather import WeatherServiceInterface
from interfaces.images import ImageServiceInterface
from interfaces.search import SearchServiceInterface
from interfaces.vector_store import VectorStoreServiceInterface
from models.weather import WeatherResult, WeatherDataPoint
from models.city import CityKnowledge


def test_cannot_instantiate_abstract_interfaces():
    with pytest.raises(TypeError):
        WeatherServiceInterface()

    with pytest.raises(TypeError):
        ImageServiceInterface()

    with pytest.raises(TypeError):
        SearchServiceInterface()

    with pytest.raises(TypeError):
        VectorStoreServiceInterface()


class DummyWeatherService(WeatherServiceInterface):
    def get_weather(self, city: str, days: int = 7) -> WeatherResult:
        return WeatherResult(
            city=city,
            forecast=[
                WeatherDataPoint(date="2026-08-20", temperature=20.0, condition="Sunny")
            ],
        )


class DummyImageService(ImageServiceInterface):
    def search_images(self, city: str, limit: int = 5) -> list[str]:
        return [f"https://example.com/{city}_1.jpg"]


class DummySearchService(SearchServiceInterface):
    def search_city(self, city: str) -> CityKnowledge | None:
        return CityKnowledge(city=city, country="Test", description="Search result", source="web_search")


class DummyVectorStoreService(VectorStoreServiceInterface):
    def search_city(self, city: str) -> CityKnowledge | None:
        return CityKnowledge(city=city, country="Test", description="Vector store result", source="vector_store")


def test_dummy_implementations():
    ws = DummyWeatherService()
    weather_res = ws.get_weather("Tokyo")
    assert weather_res.city == "Tokyo"

    ims = DummyImageService()
    imgs = ims.search_images("Tokyo")
    assert len(imgs) == 1

    ss = DummySearchService()
    search_res = ss.search_city("Tokyo")
    assert search_res is not None
    assert search_res.source == "web_search"

    vss = DummyVectorStoreService()
    vec_res = vss.search_city("Tokyo")
    assert vec_res is not None
    assert vec_res.source == "vector_store"
