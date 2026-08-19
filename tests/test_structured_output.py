"""Unit and integration tests for Phase 7 - Structured Output."""

import pytest
from pydantic import ValidationError
from models.travel import TravelResponse
from models.weather import WeatherDataPoint
from models.state import TravelAgentState
from llm.structured import parse_json_completion
from llm.client import LLMClient
from graph.nodes import synthesize_response_node
from graph.workflow import create_travel_graph


def test_synthesize_response_node_returns_travel_response():
    """Verify that synthesize_response_node returns TravelResponse Pydantic instance."""
    state: TravelAgentState = {
        "query": "What is Tokyo like?",
        "city": "Tokyo",
        "city_summary": "Tokyo is the capital of Japan.",
        "weather_forecast": [
            WeatherDataPoint(date="2026-08-20", temperature=28.0, condition="Clear")
        ],
        "image_urls": ["https://example.com/tokyo.jpg"],
    }
    result = synthesize_response_node(state)
    assert "final_response" in result
    resp = result["final_response"]

    assert isinstance(resp, TravelResponse)
    assert "Tokyo" in resp.city_summary
    assert len(resp.weather_forecast) == 1
    assert resp.weather_forecast[0].temperature == 28.0
    assert len(resp.image_urls) == 1
    assert resp.image_urls[0] == "https://example.com/tokyo.jpg"
    assert resp.weather_error is None
    assert resp.image_error is None


def test_parse_json_completion_trailing_comma_repair():
    """Test parse_json_completion repairs trailing commas in LLM JSON outputs."""
    raw_with_trailing_comma = """```json
    {
      "city_summary": "Paris is known for its architecture.",
      "weather_forecast": [
        {"date": "2026-08-20", "temperature": 22.0, "condition": "Sunny"},
      ],
      "image_urls": ["https://example.com/paris.jpg"],
    }
    ```"""
    resp = parse_json_completion(raw_with_trailing_comma, TravelResponse)
    assert isinstance(resp, TravelResponse)
    assert "Paris" in resp.city_summary
    assert len(resp.weather_forecast) == 1


def test_malformed_llm_output_fallback_repair():
    """Test that malformed raw output gracefully falls back to a valid TravelResponse."""
    class MalformedLLMClient(LLMClient):
        def _call_raw_llm(self, system_prompt: str, user_prompt: str) -> str:
            # Return completely broken non-JSON
            return "This is totally broken output and not JSON!"

    # Force live mode with mock API key to test broken LLM output fallback
    client = MalformedLLMClient()
    client.provider = "openai"
    client.settings.openai_api_key = "fake-key-for-test"
    client.settings.provider_mode = "LIVE"

    resp = client.generate_structured_response(
        query="Tell me about Paris",
        city="Paris",
        city_knowledge="Paris is a city.",
    )

    assert isinstance(resp, TravelResponse)
    assert resp.city_summary is not None
    assert isinstance(resp.weather_forecast, list)
    assert isinstance(resp.image_urls, list)


def test_error_propagation_to_travel_response():
    """Test that service errors in state are correctly captured in TravelResponse."""
    state: TravelAgentState = {
        "query": "Tell me about Kyoto",
        "city": "Kyoto",
        "city_summary": "Kyoto is famous for temples.",
        "weather_error": "Weather service timeout",
        "image_error": "Image API rate limited",
        "search_error": None,
    }
    result = synthesize_response_node(state)
    resp = result["final_response"]

    assert isinstance(resp, TravelResponse)
    assert resp.weather_error == "Weather service timeout"
    assert resp.image_error == "Image API rate limited"


def test_graph_workflow_produces_structured_travel_response():
    """Test end-to-end graph execution returns structured TravelResponse in state."""
    app = create_travel_graph()
    output = app.invoke({"query": "Tell me about Tokyo"})

    assert "final_response" in output
    resp = output["final_response"]
    assert isinstance(resp, TravelResponse)
    assert isinstance(resp.city_summary, str)
    assert isinstance(resp.weather_forecast, list)
    assert isinstance(resp.image_urls, list)
