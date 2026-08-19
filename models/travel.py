"""Travel response model matching the mandatory application output contract."""

from typing import Optional
from pydantic import BaseModel, Field
from models.weather import WeatherDataPoint


class TravelResponse(BaseModel):
    """Final structured output returned by the LangGraph workflow synthesis node."""
    city_summary: str = Field(..., description="Comprehensive summary of the city knowledge")
    weather_forecast: list[WeatherDataPoint] = Field(default_factory=list, description="5-7 day weather forecast")
    image_urls: list[str] = Field(default_factory=list, description="List of image URLs for the city")
    weather_error: Optional[str] = Field(default=None, description="Optional weather service error message")
    image_error: Optional[str] = Field(default=None, description="Optional image service error message")
    search_error: Optional[str] = Field(default=None, description="Optional knowledge search error message")
    graph_entities: Optional[list[str]] = Field(default_factory=list, description="Extracted entity names from Knowledge Graph")
    graph_nodes: Optional[list[dict]] = Field(default_factory=list, description="Structured graph nodes and relationships for UI rendering")

    graph_triples: list[str] = Field(default_factory=list, description="Optional list of Knowledge Graph fact triples")
    retrieval_mode: Optional[str] = Field(default="vector_store", description="Retrieval paradigm used ('vector_store', 'web_search', 'graph_rag', 'hybrid_rag')")

