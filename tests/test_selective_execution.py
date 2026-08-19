"""Unit and integration tests for Phase 11: Selective Re-execution."""

import pytest
from graph.checkpoint import TravelAssistantSession
from models.travel import TravelResponse


def test_multi_turn_selective_re_execution():
    """Verify selective re-execution: city_summary and image_urls reused, weather_forecast refreshed."""
    session = TravelAssistantSession(thread_id="test-selective-execution-thread")

    # Turn 1: Initial request for Tokyo
    state_turn1 = session.invoke("Tell me about Tokyo.")

    assert state_turn1["city"] == "Tokyo"
    assert state_turn1["city_summary"] is not None
    assert state_turn1.get("weather_forecast") is not None
    assert len(state_turn1.get("weather_forecast", [])) > 0
    assert state_turn1.get("image_urls") is not None
    assert len(state_turn1.get("image_urls", [])) > 0
    assert isinstance(state_turn1["final_response"], TravelResponse)

    summary_turn1 = state_turn1["city_summary"]
    images_turn1 = state_turn1["image_urls"]

    # Turn 2: Follow-up query asking about weather next week
    state_turn2 = session.invoke("What about next week?")

    assert state_turn2["city"] == "Tokyo"
    assert state_turn2.get("city_changed") is False
    assert state_turn2["city_summary"] == summary_turn1
    assert state_turn2["image_urls"] == images_turn1

    # Check selective re-execution tracking
    reused = state_turn2.get("reused_components") or []
    assert "city_summary" in reused
    assert "image_urls" in reused
    assert "weather_forecast" not in reused  # Weather refreshed for "next week"

    # Turn 3: Switch destination city to Paris
    state_turn3 = session.invoke("Tell me about Paris.")

    assert state_turn3["city"] == "Paris"
    assert state_turn3.get("city_changed") is True
    assert "Paris" in state_turn3["city_summary"]
    assert state_turn3["city_summary"] != summary_turn1


def test_non_weather_followup_reuses_all_components():
    """Verify non-weather follow-up query reuses existing summary, weather, and images."""
    session = TravelAssistantSession(thread_id="test-non-weather-followup")

    state_turn1 = session.invoke("Tell me about Tokyo.")
    reused_turn1 = state_turn1.get("reused_components") or []
    assert len(reused_turn1) == 0

    # Follow-up asking about culture/tips (no weather or image keywords)
    state_turn2 = session.invoke("Tell me more about local culture.")

    assert state_turn2["city"] == "Tokyo"
    assert state_turn2.get("city_changed") is False
    reused_turn2 = state_turn2.get("reused_components") or []
    assert "city_summary" in reused_turn2
    assert "weather_forecast" in reused_turn2
    assert "image_urls" in reused_turn2


def test_explicit_image_query_refreshes_images():
    """Verify query explicitly requesting photos triggers image fetch while reusing city summary."""
    session = TravelAssistantSession(thread_id="test-image-refresh-thread")

    session.invoke("Tell me about Tokyo.")
    state_turn2 = session.invoke("Show me photos of Tokyo.")

    assert state_turn2["city"] == "Tokyo"
    reused = state_turn2.get("reused_components") or []
    assert "city_summary" in reused
    assert "image_urls" not in reused  # Images explicitly requested & refreshed
