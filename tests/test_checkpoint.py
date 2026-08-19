"""Unit and integration tests for Phase 10: LangGraph Memory and Checkpointing."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import create_travel_graph
from graph.checkpoint import get_memory_checkpointer, TravelAssistantSession
from models.travel import TravelResponse


def test_get_memory_checkpointer():
    """Verify checkpointer factory returns a MemorySaver instance."""
    checkpointer = get_memory_checkpointer()
    assert isinstance(checkpointer, MemorySaver)


def test_create_travel_graph_with_checkpointer():
    """Verify graph compiles successfully with a checkpointer attached."""
    checkpointer = MemorySaver()
    app = create_travel_graph(checkpointer=checkpointer)
    assert app is not None
    assert hasattr(app, "checkpointer")


def test_single_turn_checkpoint_invocation():
    """Verify single-turn execution using thread_id configuration."""
    checkpointer = MemorySaver()
    app = create_travel_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-single-1"}}

    state_input = {"query": "Tell me about Tokyo"}
    output = app.invoke(state_input, config=config)

    assert output["city"] == "Tokyo"
    assert output["routed_to"] == "vector_store"
    assert isinstance(output["final_response"], TravelResponse)
    assert len(output["messages"]) >= 2


def test_multi_turn_city_context_retention():
    """Verify city context is retained across turns in the same thread (e.g. 'What about next week?')."""
    session = TravelAssistantSession(thread_id="thread-multi-turn-test")

    # Turn 1: Specify target city
    out1 = session.invoke("Tell me about Tokyo.")
    assert out1["city"] == "Tokyo"
    assert "Tokyo" in out1["final_response"].city_summary

    # Turn 2: Follow-up query without explicitly repeating the city
    out2 = session.invoke("What about next week?")
    assert out2["city"] == "Tokyo"
    assert out2["final_response"] is not None
    assert len(out2["messages"]) >= 4

    # Verify session state snapshot
    saved_state = session.get_state()
    assert saved_state is not None
    assert saved_state.get("city") == "Tokyo"


def test_thread_state_isolation():
    """Verify independent thread IDs maintain isolated state."""
    session_a = TravelAssistantSession(thread_id="thread-paris")
    session_b = TravelAssistantSession(thread_id="thread-new-york")

    out_a = session_a.invoke("Tell me about Paris.")
    out_b = session_b.invoke("Tell me about New York.")

    assert out_a["city"] == "Paris"
    assert out_b["city"] == "New York"

    # Verify thread A retains Paris, thread B retains New York
    followup_a = session_a.invoke("How is the food?")
    followup_b = session_b.invoke("Where should I stay?")

    assert followup_a["city"] == "Paris"
    assert followup_b["city"] == "New York"
