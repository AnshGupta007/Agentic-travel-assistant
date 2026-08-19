"""End-to-End Integration tests for Multi-Modal Agentic Travel Assistant.

This test suite satisfies Phase 14 of the Multi-Modal Agentic Travel Assistant challenge by
verifying end-to-end integration across all 5 mandatory scenarios:
1. Scenario 1: Known city retrieval ("Tell me about Tokyo.")
2. Scenario 2: Unknown city fallback ("Tell me about Snohomish.")
3. Scenario 3: Multi-turn memory & selective re-execution ("Tell me about Tokyo." -> "What about next week?")
4. Scenario 4: Graceful handling of weather service failure
5. Scenario 5: Graceful handling of image service failure
"""

from unittest.mock import MagicMock, patch
import pytest

from graph.checkpoint import TravelAssistantSession
from models.travel import TravelResponse
from models.weather import WeatherDataPoint, WeatherResult
from mocks.weather import MockWeatherProvider
from mocks.images import MockImageProvider
from mocks.search import MockSearchProvider


class FailingWeatherProvider:
    """Mock weather provider that simulates an API service failure."""

    def get_weather(self, city: str, days: int = 7) -> WeatherResult:
        return WeatherResult(
            city=city,
            forecast=[],
            error=f"Simulated OpenWeather API service outage for city '{city}'.",
        )


class FailingImageProvider:
    """Mock image provider that raises an unexpected exception."""

    def search_images(self, city: str, limit: int = 5) -> list[str]:
        raise RuntimeError(f"Unsplash API connection failed for '{city}'.")


def test_scenario_1_known_city_tokyo():
    """Scenario 1: Known city ("Tell me about Tokyo.")
    
    Expected Behavior:
    - City extraction identifies 'Tokyo'
    - Knowledge router routes to local vector store ('vector_store')
    - Parallel weather and image retrieval execute successfully
    - TravelResponse structured object contains city summary, weather forecast (5-7 items), and image URLs
    - No service errors reported
    """
    session = TravelAssistantSession(thread_id="e2e_test_scenario_1")
    output_state = session.invoke("Tell me about Tokyo.")

    assert output_state.get("city") == "Tokyo"
    assert output_state.get("routed_to") == "vector_store"
    assert output_state.get("search_error") is None
    assert output_state.get("weather_error") is None
    assert output_state.get("image_error") is None

    final_resp: TravelResponse = output_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert "Tokyo" in final_resp.city_summary or "Japan" in final_resp.city_summary
    assert len(final_resp.weather_forecast) >= 5
    assert len(final_resp.image_urls) > 0
    assert final_resp.weather_error is None
    assert final_resp.image_error is None


def test_scenario_2_unknown_city_snohomish():
    """Scenario 2: Unknown city ("Tell me about Snohomish.")
    
    Expected Behavior:
    - City extraction identifies 'Snohomish'
    - Knowledge router routes to Web Search fallback ('web_search')
    - Parallel weather and image retrieval execute
    - TravelResponse structured object contains web search summary, weather forecast, and image URLs
    """
    session = TravelAssistantSession(thread_id="e2e_test_scenario_2")
    output_state = session.invoke("Tell me about Snohomish.")

    assert output_state.get("city") == "Snohomish"
    assert output_state.get("routed_to") == "web_search"
    assert output_state.get("search_error") is None
    assert output_state.get("weather_error") is None
    assert output_state.get("image_error") is None

    final_resp: TravelResponse = output_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert "Snohomish" in final_resp.city_summary
    assert len(final_resp.weather_forecast) >= 5
    assert len(final_resp.image_urls) > 0


def test_scenario_3_multiturn_selective_reexecution():
    """Scenario 3: Multi-turn conversation ("Tell me about Tokyo." -> "What about next week?")
    
    Expected Behavior:
    - Turn 1: Retrieves full Tokyo data (summary, weather, images)
    - Turn 2: Retains city context ('Tokyo'), reuses city_summary and image_urls, refreshes weather forecast
    - Reused components list tracks 'city_summary' and 'image_urls'
    - Final TravelResponse retains summary & images while weather forecast is present
    """
    session = TravelAssistantSession(thread_id="e2e_test_scenario_3")

    # Turn 1: Initial query
    turn1_state = session.invoke("Tell me about Tokyo.")
    assert turn1_state.get("city") == "Tokyo"
    turn1_summary = turn1_state.get("city_summary")
    turn1_images = turn1_state.get("image_urls")

    # Turn 2: Follow-up query asking about weather next week
    turn2_state = session.invoke("What about next week?")

    assert turn2_state.get("city") == "Tokyo"
    assert turn2_state.get("city_changed") is False

    reused = turn2_state.get("reused_components") or []
    assert "city_summary" in reused
    assert "image_urls" in reused

    # Ensure summary and images remain identical while weather was refreshed
    assert turn2_state.get("city_summary") == turn1_summary
    assert turn2_state.get("image_urls") == turn1_images

    final_resp: TravelResponse = turn2_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert len(final_resp.weather_forecast) >= 5


def test_scenario_4_weather_failure_resilience():
    """Scenario 4: Graceful handling of weather service failure.
    
    Expected Behavior:
    - Workflow completes without throwing unhandled runtime exceptions
    - Weather error is captured and stored in state / TravelResponse.weather_error
    - City summary and image URLs are still fully populated and accessible
    """
    session = TravelAssistantSession(
        thread_id="e2e_test_scenario_4",
        weather_service=FailingWeatherProvider(),
    )
    output_state = session.invoke("Tell me about Paris.")

    assert output_state.get("city") == "Paris"
    assert output_state.get("weather_error") is not None
    assert "service outage" in output_state.get("weather_error")
    assert len(output_state.get("weather_forecast")) == 0

    # City summary and image gallery remain fully functional
    final_resp: TravelResponse = output_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert "Paris" in final_resp.city_summary
    assert len(final_resp.image_urls) > 0
    assert final_resp.weather_error is not None


def test_scenario_5_image_failure_resilience():
    """Scenario 5: Graceful handling of image service failure.
    
    Expected Behavior:
    - Workflow completes without throwing unhandled runtime exceptions
    - Image service exception is caught in image_node and saved as image_error
    - City summary and weather forecast are still fully populated
    """
    session = TravelAssistantSession(
        thread_id="e2e_test_scenario_5",
        image_service=FailingImageProvider(),
    )
    output_state = session.invoke("Tell me about New York.")

    assert output_state.get("city") == "New York"
    assert output_state.get("image_error") is not None
    assert "failed" in output_state.get("image_error").lower()
    assert len(output_state.get("image_urls")) == 0

    # Summary and weather forecast remain intact
    final_resp: TravelResponse = output_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert "New York" in final_resp.city_summary
    assert len(final_resp.weather_forecast) >= 5
    assert final_resp.image_error is not None
