"""Rendering orchestration layer for Travel Assistant Streamlit UI."""

from typing import Dict, List, Optional
import streamlit as st
from models.travel import TravelResponse
from ui.components import (
    render_city_summary,
    render_error_states,
    render_image_gallery,
    render_invalid_response_error,
    render_llm_error,
    render_weather_chart,
    render_graph_triples,
    render_graph_rag_visualization,
)


def render_travel_response(response: TravelResponse, city: Optional[str] = None) -> None:
    """Render a full TravelResponse output including summary, forecast, gallery, and warnings.

    Args:
        response: TravelResponse object returned by application layer.
        city: Target city name if identified.
    """
    if not isinstance(response, TravelResponse):
        render_invalid_response_error(f"Received type '{type(response).__name__}' instead of TravelResponse.")
        return

    # 1. Display partial error/warning alerts if services experienced issues
    render_error_states(response)

    # 2. Display City Overview
    render_city_summary(
        response.city_summary,
        city=city,
        search_error=response.search_error,
    )
    st.divider()

    # Display Graph RAG visualization if entities or graph nodes present
    graph_entities = getattr(response, "graph_entities", None)
    graph_nodes = getattr(response, "graph_nodes", None)
    if graph_entities or graph_nodes:
        render_graph_rag_visualization(graph_nodes=graph_nodes, graph_entities=graph_entities)
        st.divider()

    # Display Graph RAG triples if available
    if hasattr(response, "graph_triples") and response.graph_triples:
        render_graph_triples(response.graph_triples)
        st.divider()


    # 3. Display Weather Chart & Metrics
    render_weather_chart(
        response.weather_forecast,
        weather_error=response.weather_error,
    )
    st.divider()

    # 4. Display Destination Image Gallery
    render_image_gallery(
        response.image_urls,
        image_error=response.image_error,
    )



def render_chat_history(chat_history: List[Dict[str, str]]) -> None:
    """Render full chat history in Streamlit message style.

    Args:
        chat_history: List of dict objects containing 'role' and 'content' or 'response'.
    """
    for msg in chat_history:
        role = msg.get("role", "user")
        with st.chat_message(role):
            if role == "user":
                st.markdown(msg.get("content", ""))
            elif role == "assistant":
                content = msg.get("content")
                response_obj = msg.get("response")
                city = msg.get("city")
                is_error = msg.get("is_error", False)
                
                if is_error or (content and content.startswith("❌")):
                    render_llm_error(content or "An unexpected execution error occurred.")
                elif response_obj and isinstance(response_obj, TravelResponse):
                    render_travel_response(response_obj, city=city)
                elif response_obj is not None:
                    render_invalid_response_error(f"Received invalid type '{type(response_obj).__name__}' in chat history.")
                elif content:
                    st.markdown(content)

