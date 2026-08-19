"""Typed state dictionary for LangGraph workflow state management."""

from typing import TypedDict, Optional, Annotated
from models.weather import WeatherDataPoint
from models.travel import TravelResponse

def reduce_reused_components(left: Optional[list[str]], right: Optional[list[str]]) -> list[str]:
    """Reducer function for merging reused_components list state updates in LangGraph."""
    if right is None:
        return left or []
    if right == []:  # Reset signal for new turn / city change
        return []
    res = list(left or [])
    for item in right:
        if item not in res:
            res.append(item)
    return res


class TravelAgentState(TypedDict, total=False):
    """State schema propagated through the LangGraph workflow."""
    query: str
    city: Optional[str]
    previous_city: Optional[str]
    city_changed: Optional[bool]
    reused_components: Annotated[list[str], reduce_reused_components]
    routed_to: Optional[str]  # "vector_store" | "web_search" | "graph_rag" | "hybrid_rag"
    city_summary: Optional[str]
    weather_forecast: Optional[list[WeatherDataPoint]]
    image_urls: Optional[list[str]]
    weather_error: Optional[str]
    image_error: Optional[str]
    search_error: Optional[str]
    graph_context: Optional[str]
    graph_entities: Optional[list[str]]
    graph_nodes: Optional[list[dict]]
    graph_triples: Optional[list[str]]
    retrieval_mode: Optional[str]
    benchmark_report: Optional[dict]
    final_response: Optional[TravelResponse]
    messages: list[dict]



