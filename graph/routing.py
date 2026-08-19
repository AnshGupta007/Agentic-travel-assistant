"""Routing logic for conditional edges in LangGraph workflow."""

import logging
from typing import Optional, Set
from models.state import TravelAgentState

logger = logging.getLogger(__name__)

# Known cities residing in local vector store (minimum requirement: Paris, Tokyo, New York)
DEFAULT_VECTOR_STORE_CITIES: Set[str] = {"tokyo", "paris", "new york"}


def is_in_vector_store(city: Optional[str], vector_store_service=None) -> bool:
    """Check if a city is available in local vector store knowledge base.
    
    Args:
        city: City name string.
        vector_store_service: Optional service instance implementing VectorStoreServiceInterface.
        
    Returns:
        True if city is in vector store, False otherwise.
    """
    if not city:
        return False

    city_clean = city.strip().lower()

    if vector_store_service is not None:
        try:
            res = vector_store_service.search_city(city)
            if res is not None:
                return True
        except Exception as e:
            logger.warning(f"Vector store lookup error during routing: {e}")

    return city_clean in DEFAULT_VECTOR_STORE_CITIES


def is_graph_query(query: Optional[str]) -> bool:
    """Check if query is specifically asking for graph relations, POI connections, or entity graphs."""
    if not query:
        return False
    query_lower = query.lower()
    graph_keywords = ["graph", "relation", "relationship", "connected", "poi", "entity", "topology", "network", "subgraph", "itinerary"]
    return any(k in query_lower for k in graph_keywords)


def route_knowledge(state: TravelAgentState) -> str:
    """Determine routing path: 'hybrid_rag', 'graph_rag', 'vector_store', or 'web_search'.

    Args:
        state: Current TravelAgentState dictionary.

    Returns:
        String node key: 'hybrid_rag', 'graph_rag', 'vector_store', or 'web_search'.
    """
    query = state.get("query", "")
    city = state.get("city")
    retrieval_mode = state.get("retrieval_mode")

    if retrieval_mode == "hybrid_rag":
        logger.info(f"Routing query for city '{city}' to hybrid_rag.")
        return "hybrid_rag"

    if retrieval_mode == "graph_rag" or is_graph_query(query):
        logger.info(f"Routing query for city '{city}' to graph_rag.")
        return "graph_rag"

    if is_in_vector_store(city):
        logger.info(f"Routing city '{city}' to vector_store.")
        return "vector_store"

    logger.info(f"Routing city '{city}' to web_search.")
    return "web_search"


