"""Unit and integration tests for Phase 5 LangGraph workflow."""

import pytest
from graph.state import TravelAgentState
from graph.routing import route_knowledge, is_in_vector_store
from graph.nodes import (
    extract_city_node,
    vector_search_node,
    web_search_node,
    weather_node,
    image_node,
    synthesize_response_node,
)
from graph.workflow import create_travel_graph
from models.travel import TravelResponse
from llm.client import LLMClient


# --- Routing Unit Tests ---

def test_routing_known_cities_vector_store():
    """Verify that known vector store cities route to 'vector_store'."""
    assert route_knowledge({"city": "Tokyo"}) == "vector_store"
    assert route_knowledge({"city": "Paris"}) == "vector_store"
    assert route_knowledge({"city": "New York"}) == "vector_store"
    assert route_knowledge({"city": "tokyo"}) == "vector_store"
    assert route_knowledge({"city": "  Paris  "}) == "vector_store"


def test_routing_unknown_cities_web_search():
    """Verify that unknown cities route to 'web_search'."""
    assert route_knowledge({"city": "Snohomish"}) == "web_search"
    assert route_knowledge({"city": "Kyoto"}) == "web_search"
    assert route_knowledge({"city": "Atlantis"}) == "web_search"
    assert route_knowledge({"city": None}) == "web_search"


def test_is_in_vector_store():
    """Test is_in_vector_store utility directly."""
    assert is_in_vector_store("Tokyo") is True
    assert is_in_vector_store("Paris") is True
    assert is_in_vector_store("New York") is True
    assert is_in_vector_store("Snohomish") is False
    assert is_in_vector_store(None) is False


# --- Node Unit Tests ---

def test_extract_city_node():
    state: TravelAgentState = {"query": "What can I do in Tokyo?"}
    res = extract_city_node(state)
    assert res["city"] == "Tokyo"


def test_vector_search_node_known_city():
    state: TravelAgentState = {"city": "Tokyo"}
    res = vector_search_node(state)
    assert res["routed_to"] == "vector_store"
    assert "Tokyo" in res["city_summary"]
    assert res["search_error"] is None


def test_web_search_node_fallback():
    state: TravelAgentState = {"city": "Snohomish"}
    res = web_search_node(state)
    assert res["routed_to"] == "web_search"
    assert "Snohomish" in res["city_summary"]
    assert res["search_error"] is None


def test_web_search_node_non_city_query():
    """BUG-03 regression: web_search_node must not use raw query as a city name when city is None.
    
    Non-city conversational queries (e.g. 'Tell me a joke') should receive a
    friendly redirect rather than triggering a bogus travel guide search.
    """
    for non_city_query in ["Tell me a joke", "What is this app?", "Hello!", "What is the meaning of life?"]:
        state: TravelAgentState = {"query": non_city_query, "city": None, "city_changed": True}
        result = web_search_node(state)
        # Must NOT search for the raw query as a city
        assert non_city_query not in result["city_summary"], (
            f"BUG-03: web_search_node searched '{non_city_query}' as a city name"
        )
        # Must return a helpful travel redirect message
        assert result["search_error"] is None
        assert result["routed_to"] == "web_search"
        assert "travel" in result["city_summary"].lower() or "destination" in result["city_summary"].lower() or "city" in result["city_summary"].lower()


def test_weather_node():
    state: TravelAgentState = {"city": "Tokyo"}
    res = weather_node(state)
    assert "weather_forecast" in res
    assert len(res["weather_forecast"]) >= 5
    assert res["weather_error"] is None


def test_weather_node_exception_handling():
    class FailingWeatherService:
        def get_weather(self, city, days=7):
            raise RuntimeError("API timeout connection failure")

    state: TravelAgentState = {"city": "Tokyo"}
    res = weather_node(state, weather_service=FailingWeatherService())
    assert res["weather_forecast"] == []
    assert "API timeout connection failure" in res["weather_error"]


def test_image_node():
    state: TravelAgentState = {"city": "Tokyo"}
    res = image_node(state)
    assert "image_urls" in res
    assert len(res["image_urls"]) > 0
    assert res["image_error"] is None


def test_image_node_exception_handling():
    class FailingImageService:
        def search_images(self, city, limit=5):
            raise RuntimeError("Image API unavailable")

    state: TravelAgentState = {"city": "Tokyo"}
    res = image_node(state, image_service=FailingImageService())
    assert res["image_urls"] == []
    assert "Image API unavailable" in res["image_error"]


def test_synthesize_response_node():
    state: TravelAgentState = {
        "query": "Tell me about Tokyo",
        "city": "Tokyo",
        "city_summary": "Tokyo is the capital of Japan.",
    }
    res = synthesize_response_node(state)
    assert "final_response" in res
    assert isinstance(res["final_response"], TravelResponse)
    assert "Tokyo" in res["final_response"].city_summary


# --- Graph Integration Tests ---

def test_workflow_execution_known_city_tokyo():
    """End-to-end workflow execution test for known city (Tokyo)."""
    app = create_travel_graph()
    input_state: TravelAgentState = {"query": "Tell me about Tokyo"}

    output = app.invoke(input_state)

    assert output["city"] == "Tokyo"
    assert output["routed_to"] == "vector_store"
    assert output["city_summary"] is not None
    assert output["weather_forecast"] is not None
    assert len(output["weather_forecast"]) >= 5
    assert output["image_urls"] is not None
    assert len(output["image_urls"]) > 0
    assert isinstance(output["final_response"], TravelResponse)
    assert "Tokyo" in output["final_response"].city_summary
    assert len(output["final_response"].weather_forecast) >= 5
    assert len(output["final_response"].image_urls) > 0


def test_workflow_execution_known_city_paris():
    """End-to-end workflow execution test for known city (Paris)."""
    app = create_travel_graph()
    input_state: TravelAgentState = {"query": "I want to visit Paris."}

    output = app.invoke(input_state)

    assert output["city"] == "Paris"
    assert output["routed_to"] == "vector_store"
    assert output["city_summary"] is not None
    assert output["weather_forecast"] is not None
    assert output["image_urls"] is not None
    assert isinstance(output["final_response"], TravelResponse)


def test_workflow_execution_unknown_city_snohomish():
    """End-to-end workflow execution test for unknown city (Snohomish -> web_search fallback)."""
    app = create_travel_graph()
    input_state: TravelAgentState = {"query": "Tell me about Snohomish."}

    output = app.invoke(input_state)

    assert output["city"] == "Snohomish"
    assert output["routed_to"] == "web_search"
    assert output["city_summary"] is not None
    assert output["weather_forecast"] is not None
    assert output["image_urls"] is not None
    assert isinstance(output["final_response"], TravelResponse)
    assert "Snohomish" in output["final_response"].city_summary


def test_workflow_execution_with_partial_service_failures():
    """End-to-end test verifying graph resilience when weather and image services fail."""
    class FailingWeatherService:
        def get_weather(self, city, days=7):
            raise RuntimeError("Weather service down")

    class FailingImageService:
        def search_images(self, city, limit=5):
            raise RuntimeError("Image service down")

    app = create_travel_graph(
        weather_service=FailingWeatherService(),
        image_service=FailingImageService(),
    )
    input_state: TravelAgentState = {"query": "Tell me about Tokyo"}

    output = app.invoke(input_state)

    assert output["city"] == "Tokyo"
    assert "Weather service down" in output["weather_error"]
    assert "Image service down" in output["image_error"]
    assert output["weather_forecast"] == []
    assert output["image_urls"] == []
    assert isinstance(output["final_response"], TravelResponse)
    assert output["final_response"].weather_error is not None
    assert output["final_response"].image_error is not None


def test_workflow_parallel_fanout_structure():
    """Verify graph nodes and parallel fan-out topology structure."""
    app = create_travel_graph()
    graph_obj = app.get_graph()
    nodes = set(graph_obj.nodes.keys())

    assert "extract_city" in nodes
    assert "vector_search" in nodes
    assert "web_search" in nodes
    assert "weather" in nodes
    assert "images" in nodes
    assert "synthesize_response" in nodes

    # Check edges to verify vector_search and web_search fan out to both weather and images
    edges = [(edge.source, edge.target) for edge in graph_obj.edges]
    assert ("vector_search", "weather") in edges
    assert ("vector_search", "images") in edges
    assert ("web_search", "weather") in edges
    assert ("web_search", "images") in edges

    # Check fan-in edges from weather and images to synthesize_response
    assert ("weather", "synthesize_response") in edges
    assert ("images", "synthesize_response") in edges


