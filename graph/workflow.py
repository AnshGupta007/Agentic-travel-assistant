"""LangGraph StateGraph workflow construction for Multi-Modal Travel Assistant."""

import logging
from typing import Optional
from langgraph.graph import StateGraph, END
from models.state import TravelAgentState
from graph.nodes import (
    extract_city_node,
    vector_search_node,
    web_search_node,
    graph_rag_node,
    hybrid_rag_node,
    weather_node,
    image_node,
    synthesize_response_node,
)
from graph.routing import route_knowledge

logger = logging.getLogger(__name__)


def create_travel_graph(
    llm_client=None,
    vector_store=None,
    search_service=None,
    weather_service=None,
    image_service=None,
    graph_service=None,
    checkpointer=None,
):
    """Construct and compile the LangGraph workflow.

    Workflow topology:
        Query
          ↓
        City Extraction Node
          ↓
        Knowledge Router (Conditional Edge)
          ├── "vector_store" → Vector Search Node ──┐
          ├── "graph_rag"    → Graph RAG Node     ──┤
          ├── "hybrid_rag"   → Hybrid RAG Node    ──┼─→ Weather Node → Image Node → Response Synthesis Node → END
          └── "web_search"   → Web Search Node    ──┘

    Args:
        llm_client: Optional custom LLMClient instance.
        vector_store: Optional vector store service instance.
        search_service: Optional web search service instance.
        weather_service: Optional weather service instance.
        image_service: Optional image service instance.
        graph_service: Optional graph RAG service instance.
        checkpointer: Optional LangGraph checkpointer instance (e.g. MemorySaver).

    Returns:
        Compiled LangGraph StateGraph application ready for .invoke().
    """
    builder = StateGraph(TravelAgentState)

    # Custom node wrappers allowing dependency injection
    def _extract_city(state: TravelAgentState):
        return extract_city_node(state, llm_client=llm_client)

    def _vector_search(state: TravelAgentState):
        return vector_search_node(state, vector_store=vector_store)

    def _web_search(state: TravelAgentState):
        return web_search_node(state, search_service=search_service)

    def _graph_rag(state: TravelAgentState):
        return graph_rag_node(state, graph_service=graph_service)

    def _hybrid_rag(state: TravelAgentState):
        return hybrid_rag_node(state, vector_store=vector_store, graph_service=graph_service)

    def _weather(state: TravelAgentState):
        return weather_node(state, weather_service=weather_service)

    def _images(state: TravelAgentState):
        return image_node(state, image_service=image_service)

    def _synthesize_response(state: TravelAgentState):
        return synthesize_response_node(state, llm_client=llm_client)

    # 1. Register Nodes
    builder.add_node("extract_city", _extract_city)
    builder.add_node("vector_search", _vector_search)
    builder.add_node("web_search", _web_search)
    builder.add_node("graph_rag", _graph_rag)
    builder.add_node("hybrid_rag", _hybrid_rag)
    builder.add_node("weather", _weather)
    builder.add_node("images", _images)
    builder.add_node("synthesize_response", _synthesize_response)

    # 2. Set Entry Point
    builder.set_entry_point("extract_city")

    # 3. Add Conditional Edge for Knowledge Routing
    builder.add_conditional_edges(
        "extract_city",
        route_knowledge,
        {
            "vector_store": "vector_search",
            "web_search": "web_search",
            "graph_rag": "graph_rag",
            "hybrid_rag": "hybrid_rag",
        },
    )

    # 4. Parallel Fan-Out: All Retrieval Nodes fan out to Weather and Images concurrently
    builder.add_edge("vector_search", "weather")
    builder.add_edge("vector_search", "images")
    builder.add_edge("web_search", "weather")
    builder.add_edge("web_search", "images")
    builder.add_edge("graph_rag", "weather")
    builder.add_edge("graph_rag", "images")
    builder.add_edge("hybrid_rag", "weather")
    builder.add_edge("hybrid_rag", "images")

    # 5. Fan-In: Weather and Images both join into Response Synthesis
    builder.add_edge("weather", "synthesize_response")
    builder.add_edge("images", "synthesize_response")

    # 6. Direct Response Synthesis to END
    builder.add_edge("synthesize_response", END)

    # 8. Compile Workflow Graph
    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()



