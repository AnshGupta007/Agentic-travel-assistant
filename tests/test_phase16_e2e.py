"""End-to-End integration test for Phase 16 Graph RAG workflow execution."""

import pytest
from graph.checkpoint import TravelAssistantSession
from models.travel import TravelResponse


def test_phase16_graph_rag_workflow_e2e():
    """Test end-to-end Graph RAG execution in TravelAssistantSession."""
    session = TravelAssistantSession(thread_id="phase16_e2e_test")
    
    # Query asking for entity relationships / graph
    output_state = session.invoke("Tell me about the entity graph and relationships for Tokyo.")

    assert output_state.get("city") == "Tokyo"
    assert output_state.get("routed_to") == "graph_rag"
    assert output_state.get("graph_triples") is not None
    assert len(output_state.get("graph_triples")) > 0

    final_resp: TravelResponse = output_state.get("final_response")
    assert isinstance(final_resp, TravelResponse)
    assert final_resp.graph_triples is not None
    assert final_resp.retrieval_mode == "graph_rag"
    assert len(final_resp.weather_forecast) >= 5
    assert len(final_resp.image_urls) > 0


def test_phase16_hybrid_rag_workflow_e2e():
    """Test end-to-end Hybrid RAG execution."""
    session = TravelAssistantSession(thread_id="phase16_hybrid_test")
    
    # Force hybrid_rag in input state or verify node execution
    output_state = session.invoke("Tell me about Paris.")
    assert output_state.get("city") == "Paris"
    assert output_state.get("final_response") is not None
