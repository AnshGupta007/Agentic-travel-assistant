"""Unit and integration tests for UI error states (Phase 13).

Validates resilient handling and rendering of:
- Weather service failures
- Image service failures
- Search service failures
- LLM / graph execution failures
- Invalid structured output format
"""

from unittest.mock import MagicMock, patch
import pytest

from models.travel import TravelResponse
from models.weather import WeatherDataPoint
from ui.components import (
    render_city_summary,
    render_error_states,
    render_image_gallery,
    render_invalid_response_error,
    render_llm_error,
    render_weather_chart,
)
from ui.rendering import render_chat_history, render_travel_response


def test_weather_unavailable_error_state():
    """Verify UI error state when weather service fails."""
    response = TravelResponse(
        city_summary="Tokyo summary",
        weather_forecast=[],
        image_urls=["http://example.com/tokyo.jpg"],
        weather_error="Weather API timeout (504)",
    )

    with patch("streamlit.subheader"), \
         patch("streamlit.error") as mock_error, \
         patch("streamlit.info") as mock_info:

        render_weather_chart(response.weather_forecast, weather_error=response.weather_error)

        # Should render st.error and st.info placeholder for weather error
        mock_error.assert_called_once_with("⚠️ **Weather Service Error:** Weather API timeout (504)")
        mock_info.assert_called_once_with("Unable to retrieve weather forecast for this destination.")


def test_images_unavailable_error_state():
    """Verify UI error state when image service fails."""
    response = TravelResponse(
        city_summary="Paris summary",
        weather_forecast=[WeatherDataPoint(date="2026-08-20", temperature=20.0, condition="Clear")],
        image_urls=[],
        image_error="Unsplash service unavailable",
    )

    with patch("streamlit.subheader"), \
         patch("streamlit.error") as mock_error, \
         patch("streamlit.info") as mock_info:

        render_image_gallery(response.image_urls, image_error=response.image_error)

        mock_error.assert_called_once_with("⚠️ **Image Service Error:** Unsplash service unavailable")
        mock_info.assert_called_once_with("Unable to load destination gallery images.")


def test_search_unavailable_error_state():
    """Verify UI error state when search/vector service fails."""
    response = TravelResponse(
        city_summary="",
        weather_forecast=[],
        image_urls=[],
        search_error="Vector DB query connection error",
    )

    with patch("streamlit.subheader"), \
         patch("streamlit.error") as mock_error:

        render_city_summary(response.city_summary, city="Snohomish", search_error=response.search_error)

        mock_error.assert_called_once_with("⚠️ **Knowledge Search Unavailable:** Vector DB query connection error")


def test_llm_execution_failure_rendering():
    """Verify render_llm_error component rendering."""
    with patch("streamlit.error") as mock_error, patch("streamlit.caption") as mock_caption:
        render_llm_error("API key expired or invalid model specified")

        mock_error.assert_called_once_with("🤖 **LLM / Workflow Failure:** API key expired or invalid model specified")
        mock_caption.assert_called_once_with("Please check system configuration, API keys, or network connectivity.")


def test_invalid_structured_response_rendering():
    """Verify render_invalid_response_error component rendering."""
    with patch("streamlit.error") as mock_error, patch("streamlit.caption") as mock_caption:
        render_invalid_response_error("Field 'city_summary' missing in response dict")

        mock_error.assert_called_once_with(
            "⚠️ **Invalid Structured Response:** Application output did not match expected TravelResponse schema."
        )
        mock_caption.assert_called_once_with("Details: Field 'city_summary' missing in response dict")


def test_render_travel_response_with_invalid_type():
    """Verify render_travel_response handles non-TravelResponse gracefully."""
    with patch("ui.rendering.render_invalid_response_error") as mock_invalid_err:
        render_travel_response({"raw_text": "Invalid dictionary"}, city="Tokyo")

        mock_invalid_err.assert_called_once_with("Received type 'dict' instead of TravelResponse.")


def test_chat_history_with_error_messages():
    """Verify chat history correctly renders error entries."""
    chat_history = [
        {"role": "user", "content": "Tell me about Tokyo."},
        {"role": "assistant", "content": "An unexpected error occurred during execution: ConnectionRefusedError", "is_error": True},
        {"role": "user", "content": "Tell me about Paris."},
        {"role": "assistant", "response": "InvalidStringResponse"},
    ]

    with patch("streamlit.chat_message") as mock_chat_msg, \
         patch("ui.rendering.render_llm_error") as mock_llm_err, \
         patch("ui.rendering.render_invalid_response_error") as mock_invalid_err, \
         patch("streamlit.markdown"):

        mock_chat_msg.return_value.__enter__ = MagicMock()
        mock_chat_msg.return_value.__exit__ = MagicMock()

        render_chat_history(chat_history)

        mock_llm_err.assert_called_once_with("An unexpected error occurred during execution: ConnectionRefusedError")
        mock_invalid_err.assert_called_once_with("Received invalid type 'str' in chat history.")
