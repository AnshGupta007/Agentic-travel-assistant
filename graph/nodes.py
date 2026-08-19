"""Node functions for LangGraph travel assistant workflow."""

import logging
from typing import Any, Optional
from models.state import TravelAgentState
from models.city import CityKnowledge
from llm.client import LLMClient
from mocks.search import MockSearchProvider

logger = logging.getLogger(__name__)

# Preset vector store data for Phase 5 baseline execution (matching Phase 2 specifications)
_PRESET_VECTOR_KNOWLEDGE: dict[str, dict] = {
    "tokyo": {
        "country": "Japan",
        "description": "Tokyo is Japan's iconic capital, blending futuristic skyscrapers with traditional temples and world-class gastronomy.",
        "highlights": ["Sensō-ji Temple", "Tokyo Tower", "Shibuya Crossing"],
        "culture_tips": ["Bow when greeting", "Keep noise low on trains"],
    },
    "paris": {
        "country": "France",
        "description": "Paris is a global center for art, fashion, gastronomy, and culture, known for wide boulevards and historic architecture.",
        "highlights": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
        "culture_tips": ["Greet shopkeepers with 'Bonjour'", "Tipping is service-included"],
    },
    "new york": {
        "country": "United States",
        "description": "New York City is a global hub of finance, culture, and entertainment spanning 5 distinct boroughs.",
        "highlights": ["Statue of Liberty", "Central Park", "Times Square"],
        "culture_tips": ["Walk on the right side of sidewalks", "Tipping 18-20% is standard"],
    },
}


def _format_city_summary(knowledge: CityKnowledge) -> str:
    """Format CityKnowledge model into a readable summary string."""
    highlights_str = ", ".join(knowledge.highlights) if knowledge.highlights else "N/A"
    tips_str = " ".join(knowledge.culture_tips) if knowledge.culture_tips else ""
    return f"{knowledge.description}\n\nKey Highlights: {highlights_str}.\nLocal Culture & Travel Tips: {tips_str}"


def extract_city_node(state: TravelAgentState, llm_client: Optional[LLMClient] = None) -> dict[str, Any]:
    """Node: Extract target city from user query and chat history."""
    query = state.get("query", "")
    messages = list(state.get("messages") or [])
    previous_city = state.get("city")
    client = llm_client or LLMClient()

    logger.info(f"[extract_city_node] Extracting city from query: '{query}'")
    extracted_city = client.extract_city(query, messages)
    
    # Fallback to existing thread city if query doesn't specify a new city
    city = extracted_city or previous_city
    logger.info(f"[extract_city_node] Resolved city: '{city}' (extracted: '{extracted_city}', previous: '{previous_city}')")

    # Determine if city context has changed
    city_changed = True
    if previous_city and city:
        city_changed = previous_city.strip().lower() != city.strip().lower()
    elif not previous_city and city:
        city_changed = True

    # Append user query to conversation messages if not already present as the latest message
    if query and not (messages and messages[-1].get("role") == "user" and messages[-1].get("content") == query):
        messages.append({"role": "user", "content": query})

    reused_components = [] if city_changed else list(state.get("reused_components") or [])

    return {
        "city": city,
        "previous_city": previous_city,
        "city_changed": city_changed,
        "reused_components": reused_components,
        "messages": messages,
    }


def vector_search_node(state: TravelAgentState, vector_store=None) -> dict[str, Any]:
    """Node: Retrieve city knowledge from local vector store, with selective re-execution."""
    city = state.get("city") or "Tokyo"
    city_key = city.strip().lower()
    city_changed = state.get("city_changed", True)
    existing_summary = state.get("city_summary")

    # Selective Re-execution: Reuse city summary if city unchanged and summary already present
    if not city_changed and existing_summary:
        logger.info(f"[vector_search_node] Selective re-execution: Reusing existing city summary for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "city_summary" not in reused:
            reused.append("city_summary")
        return {
            "routed_to": "vector_store",
            "search_error": None,
            "reused_components": reused,
        }

    logger.info(f"[vector_search_node] Performing vector store retrieval for '{city}'")

    if vector_store is not None:
        try:
            knowledge = vector_store.search_city(city)
            if knowledge is not None:
                return {
                    "city_summary": _format_city_summary(knowledge),
                    "routed_to": "vector_store",
                    "search_error": None,
                }
        except Exception as e:
            logger.warning(f"[vector_search_node] Vector store service failed: {e}")

    # Fallback to preset vector knowledge for standard vector store cities
    if city_key in _PRESET_VECTOR_KNOWLEDGE:
        data = _PRESET_VECTOR_KNOWLEDGE[city_key]
        knowledge = CityKnowledge(
            city=city,
            country=data["country"],
            description=data["description"],
            highlights=data["highlights"],
            culture_tips=data["culture_tips"],
            source="vector_store",
        )
        return {
            "city_summary": _format_city_summary(knowledge),
            "routed_to": "vector_store",
            "search_error": None,
        }

    error_msg = f"City '{city}' not found in vector store."
    logger.warning(f"[vector_search_node] {error_msg}")
    return {
        "city_summary": f"Vector store lookup yielded no results for {city}.",
        "routed_to": "vector_store",
        "search_error": error_msg,
    }


def web_search_node(state: TravelAgentState, search_service=None) -> dict[str, Any]:
    """Node: Retrieve city knowledge via Web Search fallback, with selective re-execution."""
    city = state.get("city") or state.get("query") or "Unknown City"
    city_changed = state.get("city_changed", True)
    existing_summary = state.get("city_summary")

    # Selective Re-execution: Reuse city summary if city unchanged and summary already present
    if not city_changed and existing_summary:
        logger.info(f"[web_search_node] Selective re-execution: Reusing existing city summary for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "city_summary" not in reused:
            reused.append("city_summary")
        return {
            "routed_to": "web_search",
            "search_error": None,
            "reused_components": reused,
        }

    logger.info(f"[web_search_node] Performing web search fallback retrieval for '{city}'")

    provider = search_service or MockSearchProvider(latency=0.0)
    try:
        knowledge = provider.search_city(city)
        if knowledge is not None:
            return {
                "city_summary": _format_city_summary(knowledge),
                "routed_to": "web_search",
                "search_error": None,
            }
    except Exception as e:
        logger.error(f"[web_search_node] Search service failed: {e}")

    error_msg = f"Web search lookup failed for '{city}'."
    return {
        "city_summary": f"Unable to retrieve web search results for {city}.",
        "routed_to": "web_search",
        "search_error": error_msg,
    }


def graph_rag_node(state: TravelAgentState, graph_service=None) -> dict[str, Any]:
    """Node: Perform Knowledge Graph traversal and Graph RAG context retrieval."""
    city = state.get("city") or "Tokyo"
    query = state.get("query") or f"Tell me about {city}"
    city_changed = state.get("city_changed", True)
    existing_summary = state.get("city_summary")

    if not city_changed and existing_summary:
        logger.info(f"[graph_rag_node] Selective re-execution: Reusing existing city graph summary for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "city_summary" not in reused:
            reused.append("city_summary")
        return {
            "routed_to": "graph_rag",
            "search_error": None,
            "reused_components": reused,
        }

    logger.info(f"[graph_rag_node] Executing Graph RAG retrieval for city '{city}'")

    provider = graph_service
    if provider is None:
        from services.knowledge_graph import KnowledgeGraphService
        provider = KnowledgeGraphService()

    try:
        res = provider.query_graph(city, query)
        if res:
            return {
                "city_summary": res.get("graph_context", f"Graph RAG summary for {city}"),
                "graph_context": res.get("graph_context"),
                "graph_entities": res.get("entities", []),
                "graph_nodes": res.get("nodes", []),
                "routed_to": "graph_rag",
                "search_error": None,
            }
    except Exception as e:
        logger.error(f"[graph_rag_node] Graph RAG service failed for '{city}': {e}")

    error_msg = f"Graph RAG lookup failed for '{city}'."
    return {
        "city_summary": f"Unable to retrieve Knowledge Graph facts for {city}.",
        "routed_to": "graph_rag",
        "search_error": error_msg,
        "graph_entities": [],
        "graph_nodes": [],
    }



def weather_node(state: TravelAgentState, weather_service=None) -> dict[str, Any]:
    """Node: Retrieve weather forecast for target city with selective re-execution."""
    city = state.get("city") or "Tokyo"
    city_changed = state.get("city_changed", True)
    existing_forecast = state.get("weather_forecast")
    query = (state.get("query") or "").lower()

    # Weather/forecast/time intent keywords
    weather_keywords = [
        "weather", "forecast", "temperature", "rain", "sun", "snow",
        "next week", "tomorrow", "weekend", "today", "climate", "degree",
        "celsius", "fahrenheit", "hot", "cold", "warm"
    ]
    is_weather_query = any(kw in query for kw in weather_keywords)

    # Selective Re-execution: Reuse forecast if city unchanged, forecast present, and not a weather query
    if not city_changed and existing_forecast is not None and len(existing_forecast) > 0 and not is_weather_query:
        logger.info(f"[weather_node] Selective re-execution: Reusing existing weather forecast for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "weather_forecast" not in reused:
            reused.append("weather_forecast")
        return {
            "weather_error": None,
            "reused_components": reused,
        }

    logger.info(f"[weather_node] Fetching/Refreshing weather forecast for '{city}'")

    provider = weather_service
    if provider is None:
        from mocks.weather import MockWeatherProvider
        provider = MockWeatherProvider(latency=0.0)

    try:
        res = provider.get_weather(city=city, days=7)
        if res.error:
            logger.warning(f"[weather_node] Weather provider returned error for '{city}': {res.error}")
            return {
                "weather_forecast": res.forecast or [],
                "weather_error": res.error,
            }
        return {
            "weather_forecast": res.forecast,
            "weather_error": None,
        }
    except Exception as e:
        error_msg = f"Weather service failed for '{city}': {e}"
        logger.error(f"[weather_node] {error_msg}")
        return {
            "weather_forecast": [],
            "weather_error": error_msg,
        }


def image_node(state: TravelAgentState, image_service=None) -> dict[str, Any]:
    """Node: Retrieve image URLs for target city with selective re-execution."""
    city = state.get("city") or "Tokyo"
    city_changed = state.get("city_changed", True)
    existing_images = state.get("image_urls")
    query = (state.get("query") or "").lower()

    image_keywords = ["image", "images", "photo", "photos", "picture", "pictures", "look like", "gallery"]
    is_image_query = any(kw in query for kw in image_keywords)

    # Selective Re-execution: Reuse images if city unchanged, images present, and not an explicit image query
    if not city_changed and existing_images is not None and len(existing_images) > 0 and not is_image_query:
        logger.info(f"[image_node] Selective re-execution: Reusing existing image URLs for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "image_urls" not in reused:
            reused.append("image_urls")
        return {
            "image_error": None,
            "reused_components": reused,
        }

    logger.info(f"[image_node] Fetching image URLs for '{city}'")

    provider = image_service
    if provider is None:
        from mocks.images import MockImageProvider
        provider = MockImageProvider(latency=0.0)

    try:
        urls = provider.search_images(city=city, limit=5)
        return {
            "image_urls": urls,
            "image_error": None,
        }
    except Exception as e:
        error_msg = f"Image service failed for '{city}': {e}"
        logger.error(f"[image_node] {error_msg}")
        return {
            "image_urls": [],
            "image_error": error_msg,
        }


def graph_rag_node(state: TravelAgentState, graph_service=None) -> dict[str, Any]:
    """Node: Retrieve city knowledge using Knowledge Graph RAG entity traversal."""
    city = state.get("city") or "Tokyo"
    city_changed = state.get("city_changed", True)
    existing_summary = state.get("city_summary")

    if not city_changed and existing_summary:
        logger.info(f"[graph_rag_node] Selective re-execution: Reusing existing city summary for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "city_summary" not in reused:
            reused.append("city_summary")
        return {
            "routed_to": "graph_rag",
            "search_error": None,
            "reused_components": reused,
        }

    logger.info(f"[graph_rag_node] Performing Knowledge Graph traversal for '{city}'")
    service = graph_service
    if service is None:
        from services.graph_rag import GraphRAGService
        service = GraphRAGService()

    subgraph = service.get_subgraph(city)
    triples_str_list = [t.to_fact_string() for t in subgraph.triples]
    entities_str_list = [e.name for e in subgraph.entities]
    nodes_list = [{"id": e.name, "label": e.name, "type": e.entity_type, "attributes": e.attributes} for e in subgraph.entities]
    facts_summary = "; ".join(triples_str_list) if triples_str_list else f"No entity triples found for {city}."
    summary = f"Knowledge Graph analysis for {city}:\n{facts_summary}"

    return {
        "city_summary": summary,
        "graph_triples": triples_str_list,
        "graph_entities": entities_str_list,
        "graph_nodes": nodes_list,
        "routed_to": "graph_rag",
        "retrieval_mode": "graph_rag",
        "search_error": None,
    }



def hybrid_rag_node(state: TravelAgentState, vector_store=None, graph_service=None) -> dict[str, Any]:
    """Node: Retrieve city knowledge using Hybrid Vector RAG + Graph RAG traversal."""
    city = state.get("city") or "Tokyo"
    city_changed = state.get("city_changed", True)
    existing_summary = state.get("city_summary")

    if not city_changed and existing_summary:
        logger.info(f"[hybrid_rag_node] Selective re-execution: Reusing existing city summary for '{city}'")
        reused = list(state.get("reused_components") or [])
        if "city_summary" not in reused:
            reused.append("city_summary")
        return {
            "routed_to": "hybrid_rag",
            "search_error": None,
            "reused_components": reused,
        }

    logger.info(f"[hybrid_rag_node] Performing Hybrid Vector + Graph RAG retrieval for '{city}'")
    service = graph_service
    if service is None:
        from services.graph_rag import GraphRAGService
        service = GraphRAGService()

    vstore = vector_store
    if vstore is None:
        from services.vector_store import VectorStoreService
        vstore = VectorStoreService()

    res = service.hybrid_retrieval(city=city, vector_store=vstore, query=state.get("query", ""))
    return {
        "city_summary": res["summary"],
        "graph_triples": res["graph_triples"],
        "routed_to": "hybrid_rag",
        "retrieval_mode": "hybrid_rag",
        "search_error": None,
    }


def synthesize_response_node(
    state: TravelAgentState, llm_client: Optional[LLMClient] = None
) -> dict[str, Any]:
    """Node: Synthesize gathered knowledge into structured TravelResponse."""
    query = state.get("query", "")
    city = state.get("city") or "Target Destination"
    city_summary = state.get("city_summary") or "No detailed city information available."
    weather_forecast = state.get("weather_forecast")
    image_urls = state.get("image_urls")
    weather_error = state.get("weather_error")
    image_error = state.get("image_error")
    search_error = state.get("search_error")
    graph_triples = state.get("graph_triples") or []
    retrieval_mode = state.get("retrieval_mode") or state.get("routed_to") or "vector_store"
    messages = list(state.get("messages") or [])

    logger.info(f"[synthesize_response_node] Synthesizing response for '{city}' (mode: {retrieval_mode}, reused: {state.get('reused_components', [])})")
    client = llm_client or LLMClient()
    final_resp = client.generate_structured_response(
        query=query,
        city=city,
        city_knowledge=city_summary,
        weather_forecast=weather_forecast,
        image_urls=image_urls,
        weather_error=weather_error,
        image_error=image_error,
        search_error=search_error,
    )

    if final_resp:
        final_resp.graph_triples = graph_triples
        final_resp.retrieval_mode = retrieval_mode
        final_resp.graph_entities = state.get("graph_entities") or []
        final_resp.graph_nodes = state.get("graph_nodes") or []

    # Append assistant response to chat messages
    if final_resp and final_resp.city_summary:
        messages.append({"role": "assistant", "content": final_resp.city_summary})

    return {"final_response": final_resp, "messages": messages}




