"""Contract tests for data models."""

import pytest
from pydantic import ValidationError
from models.weather import WeatherDataPoint, WeatherForecastRequest, WeatherResult
from models.city import CityKnowledge
from models.travel import TravelResponse
from models.state import TravelAgentState


def test_weather_data_point_valid():
    wp = WeatherDataPoint(date="2026-08-20", temperature=25.5, condition="Sunny")
    assert wp.date == "2026-08-20"
    assert wp.temperature == 25.5
    assert wp.condition == "Sunny"


def test_weather_data_point_invalid():
    with pytest.raises(ValidationError):
        WeatherDataPoint(date="2026-08-20", temperature="invalid_temp", condition="Sunny")


def test_weather_forecast_request():
    req = WeatherForecastRequest(city="Tokyo")
    assert req.city == "Tokyo"
    assert req.days == 7

    req5 = WeatherForecastRequest(city="Paris", days=5)
    assert req5.days == 5

    with pytest.raises(ValidationError):
        WeatherForecastRequest(city="Tokyo", days=20)


def test_weather_result():
    res = WeatherResult(
        city="Tokyo",
        forecast=[
            WeatherDataPoint(date="2026-08-20", temperature=25.0, condition="Clear"),
            WeatherDataPoint(date="2026-08-21", temperature=22.0, condition="Rain"),
        ],
        current_temp=24.0,
        condition="Clear",
    )
    assert res.city == "Tokyo"
    assert len(res.forecast) == 2
    assert res.error is None


def test_city_knowledge():
    ck = CityKnowledge(
        city="Tokyo",
        country="Japan",
        description="Capital of Japan",
        highlights=["Tokyo Tower", "Shinjuku"],
        culture_tips=["Bow when greeting"],
    )
    assert ck.city == "Tokyo"
    assert ck.country == "Japan"
    assert ck.source == "vector_store"
    assert len(ck.highlights) == 2


def test_travel_response_mandatory_fields():
    resp = TravelResponse(
        city_summary="Tokyo is a bustling metropolis.",
        weather_forecast=[
            WeatherDataPoint(date="2026-08-20", temperature=25.0, condition="Sunny")
        ],
        image_urls=["https://example.com/tokyo.jpg"],
    )
    assert resp.city_summary == "Tokyo is a bustling metropolis."
    assert len(resp.weather_forecast) == 1
    assert len(resp.image_urls) == 1
    assert resp.weather_error is None


def test_travel_response_missing_mandatory():
    with pytest.raises(ValidationError):
        TravelResponse()


def test_travel_agent_state():
    state: TravelAgentState = {
        "query": "Tell me about Tokyo",
        "city": "Tokyo",
        "routed_to": "vector_store",
        "city_summary": "Summary here",
        "weather_forecast": [],
        "image_urls": [],
        "messages": [],
    }
    assert state["city"] == "Tokyo"
    assert state["routed_to"] == "vector_store"
