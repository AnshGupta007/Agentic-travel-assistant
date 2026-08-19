"""Unit and component integration tests for Streamlit UI components and rendering helpers."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from models.travel import TravelResponse
from models.weather import WeatherDataPoint
from ui.components import (
    render_city_summary,
    render_error_states,
    render_header,
    render_image_gallery,
    render_sidebar,
    render_weather_chart,
)
from ui.rendering import render_chat_history, render_travel_response


@pytest.fixture
def sample_travel_response():
    """Fixture providing a valid TravelResponse object."""
    return TravelResponse(
        city_summary="Tokyo is the bustling capital of Japan, blending ultramodern and traditional elements.",
        weather_forecast=[
            WeatherDataPoint(date="2026-08-20", temperature=25.5, condition="Sunny"),
            WeatherDataPoint(date="2026-08-21", temperature=23.0, condition="Partly Cloudy"),
            WeatherDataPoint(date="2026-08-22", temperature=21.0, condition="Rain"),
        ],
        image_urls=[
            "https://images.unsplash.com/photo-tokyo-1",
            "https://images.unsplash.com/photo-tokyo-2",
        ],
    )


@pytest.fixture
def sample_response_with_errors():
    """Fixture providing a TravelResponse with error messages."""
    return TravelResponse(
        city_summary="Paris summary knowledge.",
        weather_forecast=[],
        image_urls=[],
        weather_error="OpenWeather API timeout",
        image_error="Unsplash service unavailable",
        search_error="Vector store query degraded",
    )


def test_render_header():
    """Verify render_header calls Streamlit title and caption."""
    with patch("streamlit.title") as mock_title, patch("streamlit.caption") as mock_caption, patch("streamlit.divider"):
        render_header()
        mock_title.assert_called_once_with("🌍 Multi-Modal Agentic Travel Assistant")
        mock_caption.assert_called_once()


def test_render_city_summary():
    """Verify city summary rendering."""
    with patch("streamlit.subheader") as mock_sub, patch("streamlit.markdown") as mock_md:
        render_city_summary("Beautiful city overview", city="Tokyo")
        mock_sub.assert_called_once_with("📖 Overview - Tokyo")
        mock_md.assert_called_once_with("Beautiful city overview")


def test_render_weather_chart(sample_travel_response):
    """Verify weather forecast chart and metrics rendering."""
    with patch("streamlit.subheader"), \
         patch("streamlit.columns") as mock_cols, \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.markdown"), \
         patch("streamlit.line_chart") as mock_chart, \
         patch("streamlit.caption"):

        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col, mock_col]

        render_weather_chart(sample_travel_response.weather_forecast)

        # Ensure metrics were displayed
        assert mock_metric.call_count >= 3
        # Ensure line_chart was called
        mock_chart.assert_called_once()


def test_render_weather_chart_empty():
    """Verify weather chart handles empty forecast gracefully."""
    with patch("streamlit.subheader"), patch("streamlit.info") as mock_info:
        render_weather_chart([])
        mock_info.assert_called_once_with("Weather forecast data is currently unavailable.")


def test_render_image_gallery(sample_travel_response):
    """Verify image gallery grid layout rendering."""
    with patch("streamlit.subheader"), patch("streamlit.columns") as mock_cols, patch("streamlit.image") as mock_img:
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col, mock_col]

        render_image_gallery(sample_travel_response.image_urls)
        assert mock_img.call_count == 2


def test_render_image_gallery_empty():
    """Verify image gallery handles empty URL list gracefully."""
    with patch("streamlit.subheader"), patch("streamlit.info") as mock_info:
        render_image_gallery([])
        mock_info.assert_called_once_with("No destination images available.")


def test_render_error_states(sample_response_with_errors):
    """Verify warning badges are rendered for non-null error fields."""
    with patch("streamlit.warning") as mock_warn:
        render_error_states(sample_response_with_errors)
        assert mock_warn.call_count == 3


def test_render_travel_response(sample_travel_response):
    """Verify full TravelResponse orchestration calls component renderers."""
    with patch("ui.rendering.render_error_states") as mock_errors, \
         patch("ui.rendering.render_city_summary") as mock_summary, \
         patch("ui.rendering.render_weather_chart") as mock_weather, \
         patch("ui.rendering.render_image_gallery") as mock_images, \
         patch("streamlit.divider"):

        render_travel_response(sample_travel_response, city="Tokyo")

        mock_errors.assert_called_once_with(sample_travel_response)
        mock_summary.assert_called_once_with(
            sample_travel_response.city_summary,
            city="Tokyo",
            search_error=sample_travel_response.search_error,
        )
        mock_weather.assert_called_once_with(
            sample_travel_response.weather_forecast,
            weather_error=sample_travel_response.weather_error,
        )
        mock_images.assert_called_once_with(
            sample_travel_response.image_urls,
            image_error=sample_travel_response.image_error,
        )



def test_render_chat_history(sample_travel_response):
    """Verify chat history correctly renders user and assistant messages."""
    history = [
        {"role": "user", "content": "Tell me about Tokyo."},
        {"role": "assistant", "response": sample_travel_response, "city": "Tokyo"},
    ]

    with patch("streamlit.chat_message") as mock_chat_msg, \
         patch("ui.rendering.render_travel_response") as mock_render_resp, \
         patch("streamlit.markdown"):

        mock_chat_msg.return_value.__enter__ = MagicMock()
        mock_chat_msg.return_value.__exit__ = MagicMock()

        render_chat_history(history)
        mock_render_resp.assert_called_once_with(sample_travel_response, city="Tokyo")
