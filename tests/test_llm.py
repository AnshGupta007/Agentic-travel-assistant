"""Unit tests for LLM layer abstractions, prompts, structured output, and LLMClient."""

import pytest
from pydantic import ValidationError
from config.settings import Settings, ProviderMode
from models.weather import WeatherDataPoint
from models.travel import TravelResponse
from llm.prompts import (
    build_city_extraction_prompt,
    build_query_classification_prompt,
    build_tool_call_prompt,
    build_synthesis_prompt,
)
from llm.structured import (
    CityExtractionResult,
    QueryClassificationResult,
    ToolCallSpec,
    clean_json_markdown,
    parse_json_completion,
)
from llm.client import LLMClient


def test_prompts_building():
    p1 = build_city_extraction_prompt("Tell me about Tokyo")
    assert "Tokyo" in p1

    p1_ctx = build_city_extraction_prompt(
        "What about next week?",
        messages=[{"role": "user", "content": "Tell me about Paris"}]
    )
    assert "Paris" in p1_ctx
    assert "What about next week?" in p1_ctx

    p2 = build_query_classification_prompt("Show me images of Kyoto")
    assert "Kyoto" in p2

    p3 = build_tool_call_prompt("Get weather for Snohomish", city="Snohomish")
    assert "Snohomish" in p3

    weather_points = [WeatherDataPoint(date="2026-08-20", temperature=25.0, condition="Sunny")]
    p4 = build_synthesis_prompt(
        query="Tokyo info",
        city="Tokyo",
        city_knowledge="Tokyo is the capital of Japan.",
        weather_forecast=weather_points,
        image_urls=["https://example.com/tokyo.jpg"],
    )
    assert "Tokyo is the capital of Japan." in p4
    assert "2026-08-20" in p4
    assert "https://example.com/tokyo.jpg" in p4


def test_clean_json_markdown():
    raw_markdown = "```json\n{\"city\": \"Tokyo\", \"is_follow_up\": false, \"confidence\": 0.95}\n```"
    cleaned = clean_json_markdown(raw_markdown)
    assert cleaned == '{"city": "Tokyo", "is_follow_up": false, "confidence": 0.95}'

    raw_plain = "{\"city\": \"Paris\"}"
    assert clean_json_markdown(raw_plain) == "{\"city\": \"Paris\"}"


def test_parse_json_completion_valid():
    text = "```json\n{\"city\": \"Tokyo\", \"is_follow_up\": true, \"confidence\": 0.9}\n```"
    res = parse_json_completion(text, CityExtractionResult)
    assert res.city == "Tokyo"
    assert res.is_follow_up is True
    assert res.confidence == 0.9


def test_parse_json_completion_invalid_json():
    with pytest.raises(ValueError, match="Failed to parse LLM completion as JSON"):
        parse_json_completion("not valid json", CityExtractionResult)


def test_parse_json_completion_invalid_schema():
    with pytest.raises(ValueError, match="Failed to validate LLM JSON output"):
        parse_json_completion("{\"confidence\": \"invalid_confidence\"}", CityExtractionResult)


def test_llm_client_mock_city_extraction():
    mock_settings = Settings(provider_mode=ProviderMode.MOCK)
    client = LLMClient(settings=mock_settings)

    assert client.extract_city("Tell me about Tokyo") == "Tokyo"
    assert client.extract_city("What is the weather in Paris?") == "Paris"
    assert client.extract_city("Is there anything to see here?") is None

    # Follow-up extraction with conversation history
    history = [{"role": "user", "content": "Tell me about Kyoto"}]
    assert client.extract_city("What about next week?", messages=history) == "Kyoto"


def test_llm_client_mock_query_classification():
    mock_settings = Settings(provider_mode=ProviderMode.MOCK)
    client = LLMClient(settings=mock_settings)

    res1 = client.classify_query("Tell me about Tokyo")
    assert res1.target_city == "Tokyo"
    assert res1.needs_search is True

    res2 = client.classify_query("Show me photos of Paris")
    assert res2.intent == "image_request"
    assert res2.target_city == "Paris"

    res3 = client.classify_query("What is the forecast for next week in New York?")
    assert res3.intent == "weather_inquiry"
    assert res3.target_city == "New York"


def test_llm_client_mock_tool_calls():
    mock_settings = Settings(provider_mode=ProviderMode.MOCK)
    client = LLMClient(settings=mock_settings)

    tools = client.generate_tool_calls("Tell me about Tokyo", city="Tokyo")
    assert len(tools) == 3
    tool_names = [t.tool_name for t in tools]
    assert "search_city" in tool_names
    assert "get_weather" in tool_names
    assert "search_images" in tool_names
    assert tools[0].arguments.get("city") == "Tokyo"


def test_llm_client_mock_structured_response():
    mock_settings = Settings(provider_mode=ProviderMode.MOCK)
    client = LLMClient(settings=mock_settings)

    weather_points = [
        WeatherDataPoint(date="2026-08-20", temperature=22.0, condition="Clear"),
        WeatherDataPoint(date="2026-08-21", temperature=24.0, condition="Sunny"),
    ]
    imgs = ["https://example.com/tokyo1.jpg", "https://example.com/tokyo2.jpg"]

    resp = client.generate_structured_response(
        query="Tell me about Tokyo",
        city="Tokyo",
        city_knowledge="Tokyo is a bustling city.",
        weather_forecast=weather_points,
        image_urls=imgs,
    )

    assert isinstance(resp, TravelResponse)
    assert "Tokyo" in resp.city_summary
    assert len(resp.weather_forecast) == 2
    assert len(resp.image_urls) == 2
    assert resp.weather_error is None
    assert resp.image_error is None
