"""Unit tests for Phase 8 manual tool execution module (graph/tool_executor.py)."""

import pytest
from unittest.mock import MagicMock
from graph.tool_executor import (
    ToolExecutor,
    ToolResult,
    manual_tool_execution_node,
)
from llm.structured import ToolCallSpec
from models.city import CityKnowledge
from models.state import TravelAgentState
from models.weather import WeatherDataPoint, WeatherResult
from mocks.weather import MockWeatherProvider
from mocks.images import MockImageProvider
from mocks.search import MockSearchProvider


def test_tool_executor_weather_execution():
    """Test manual execution of 'get_weather' tool."""
    mock_weather = MockWeatherProvider(latency=0.0)
    executor = ToolExecutor(weather_service=mock_weather)

    res = executor.execute_tool("get_weather", {"city": "Tokyo", "days": 5})

    assert res.success is True
    assert res.tool_name == "get_weather"
    assert isinstance(res.result, WeatherResult)
    assert res.result.city == "Tokyo"
    assert len(res.result.forecast) == 5


def test_tool_executor_images_execution():
    """Test manual execution of 'search_images' tool."""
    mock_images = MockImageProvider(latency=0.0)
    executor = ToolExecutor(image_service=mock_images)

    res = executor.execute_tool("search_images", {"city": "Paris", "limit": 3})

    assert res.success is True
    assert res.tool_name == "search_images"
    assert isinstance(res.result, list)
    assert len(res.result) == 3


def test_tool_executor_search_city_execution():
    """Test manual execution of 'search_city' tool."""
    mock_search = MockSearchProvider(latency=0.0)
    executor = ToolExecutor(search_service=mock_search)

    res = executor.execute_tool("search_city", {"city": "New York"})

    assert res.success is True
    assert res.tool_name == "search_city"
    assert isinstance(res.result, CityKnowledge)
    assert res.result.city == "New York"


def test_tool_executor_unregistered_tool():
    """Test handling of an unregistered tool name."""
    executor = ToolExecutor()
    res = executor.execute_tool("nonexistent_tool", {"param": "val"})

    assert res.success is False
    assert "not registered" in res.error


def test_tool_executor_handler_exception_handling():
    """Test that tool exceptions are caught as error data without crashing."""
    failing_weather = MagicMock()
    failing_weather.get_weather.side_effect = RuntimeError("Weather service API down")

    executor = ToolExecutor(weather_service=failing_weather)
    res = executor.execute_tool("get_weather", {"city": "Tokyo"})

    assert res.success is False
    assert "Weather service API down" in res.error


def test_tool_result_to_tool_message():
    """Test conversion of ToolResult to ToolMessage."""
    tr_success = ToolResult(
        tool_name="get_weather",
        tool_call_id="call_123",
        success=True,
        result={"city": "Tokyo", "temp": 25},
    )
    msg_success = tr_success.to_tool_message()
    assert msg_success.tool_call_id == "call_123"
    assert "Tokyo" in msg_success.content

    tr_fail = ToolResult(
        tool_name="search_images",
        tool_call_id="call_456",
        success=False,
        error="Network error",
    )
    msg_fail = tr_fail.to_tool_message()
    assert msg_fail.tool_call_id == "call_456"
    assert "Network error" in msg_fail.content


def test_manual_tool_execution_node_with_prepopulated_tool_calls():
    """Test manual_tool_execution_node with pre-populated tool_calls list in state."""
    mock_weather = MockWeatherProvider(latency=0.0)
    mock_images = MockImageProvider(latency=0.0)
    mock_search = MockSearchProvider(latency=0.0)
    executor = ToolExecutor(
        weather_service=mock_weather,
        image_service=mock_images,
        search_service=mock_search,
    )

    state: TravelAgentState = {
        "query": "Tell me about Tokyo weather and photos",
        "city": "Tokyo",
        "tool_calls": [
            ToolCallSpec(tool_name="get_weather", arguments={"city": "Tokyo", "days": 5}),
            ToolCallSpec(tool_name="search_images", arguments={"city": "Tokyo", "limit": 4}),
            ToolCallSpec(tool_name="search_city", arguments={"city": "Tokyo"}),
        ],
        "messages": [],
    }

    updates = manual_tool_execution_node(state=state, tool_executor=executor)

    assert "weather_forecast" in updates
    assert len(updates["weather_forecast"]) == 5
    assert "image_urls" in updates
    assert len(updates["image_urls"]) == 4
    assert "city_summary" in updates
    assert "Tokyo" in updates["city_summary"]
    assert len(updates["messages"]) == 3
    assert updates["messages"][0]["role"] == "tool"


def test_manual_tool_execution_node_with_fallback_generation():
    """Test manual_tool_execution_node when tool_calls is missing in state."""
    mock_weather = MockWeatherProvider(latency=0.0)
    mock_images = MockImageProvider(latency=0.0)
    mock_search = MockSearchProvider(latency=0.0)
    executor = ToolExecutor(
        weather_service=mock_weather,
        image_service=mock_images,
        search_service=mock_search,
    )

    state: TravelAgentState = {
        "query": "Tell me about Paris",
        "city": "Paris",
        "messages": [],
    }

    updates = manual_tool_execution_node(state=state, tool_executor=executor)

    assert "weather_forecast" in updates
    assert "image_urls" in updates
    assert "city_summary" in updates
    assert len(updates["messages"]) >= 1
