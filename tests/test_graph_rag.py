"""Unit and integration test suite for Graph RAG and Knowledge Graph Service (Phase 15)."""

import pytest
from services.knowledge_graph import KnowledgeGraphService
from graph.routing import route_knowledge, is_graph_query
from graph.nodes import graph_rag_node
from graph.checkpoint import TravelAssistantSession
from models.travel import TravelResponse


def test_knowledge_graph_service_lookup():
    """Verify KnowledgeGraphService retrieves entities and subgraphs for known cities."""
    service = KnowledgeGraphService()

    # Test Tokyo lookup
    tokyo_subgraph = service.get_subgraph("Tokyo")
    assert tokyo_subgraph["city"] == "Tokyo"
    assert tokyo_subgraph["entity_count"] > 0
    assert any(n["label"] == "Senso-ji Temple" for n in tokyo_subgraph["nodes"])

    # Test Graph RAG context query
    res = service.query_graph("Tokyo", "What are the main POIs?")
    assert res is not None
    assert "Senso-ji Temple" in res["graph_context"]
    assert "Tokyo Tower" in res["entities"]
    assert len(res["nodes"]) > 0
    assert len(res["relationships"]) > 0


def test_knowledge_graph_unseen_city():
    """Verify KnowledgeGraphService fallback for unseen cities."""
    service = KnowledgeGraphService()
    subgraph = service.get_subgraph("Atlantis")

    assert subgraph["city"] == "Atlantis"
    assert subgraph["entity_count"] >= 1
    assert subgraph["nodes"][0]["label"] == "Atlantis"


def test_is_graph_query_detection():
    """Verify is_graph_query routing helper function."""
    assert is_graph_query("What is the entity graph for Tokyo?") is True
    assert is_graph_query("Show me connected POIs in Paris") is True
    assert is_graph_query("Tell me about Tokyo weather") is False


def test_route_knowledge_graph_rag():
    """Verify route_knowledge directs to graph_rag node when appropriate."""
    state_graph = {"query": "Tell me about Tokyo entity relationships", "city": "Tokyo"}
    assert route_knowledge(state_graph) == "graph_rag"

    state_mode = {"query": "Tell me about Paris", "city": "Paris", "retrieval_mode": "graph_rag"}
    assert route_knowledge(state_mode) == "graph_rag"


def test_graph_rag_node_execution():
    """Verify graph_rag_node retrieves graph state and updates state dict."""
    state = {"query": "Tell me about Tokyo POIs", "city": "Tokyo", "city_changed": True}
    res = graph_rag_node(state)

    assert res["routed_to"] == "graph_rag"
    assert res["search_error"] is None
    assert "Tokyo" in res["city_summary"]
    assert len(res["graph_entities"]) > 0
    assert len(res["graph_nodes"]) > 0


def test_end_to_end_graph_rag_workflow():
    """Verify end-to-end execution of LangGraph workflow via Graph RAG mode."""
    session = TravelAssistantSession(thread_id="test_graph_rag_e2e")
    output_state = session.invoke("Tell me about Paris entity topology graph", retrieval_mode="graph_rag")

    assert output_state.get("city") == "Paris"
    assert output_state.get("routed_to") == "graph_rag"
    assert output_state.get("search_error") is None

    final_resp: TravelResponse = output_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert len(final_resp.graph_entities) > 0
    assert len(final_resp.graph_nodes) > 0
    assert len(final_resp.weather_forecast) >= 5
    assert len(final_resp.image_urls) > 0
