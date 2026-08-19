"""Prompt templates and builders for LLM layer operations."""

from typing import Optional
from models.weather import WeatherDataPoint

CITY_EXTRACTION_SYSTEM_PROMPT = """You are a travel assistant's entity extraction component.
Your task is to identify the target city mentioned in a user query or conversation context.
Return JSON with the following fields:
- "city": string name of the city (e.g. "Tokyo", "Paris"), or null if no city is mentioned or inferred.
- "is_follow_up": boolean indicating if this relies on a previously mentioned city.
- "confidence": float between 0.0 and 1.0 indicating extraction confidence.
"""

QUERY_CLASSIFICATION_SYSTEM_PROMPT = """You are a travel assistant's query classification router.
Classify the user's intent and determine required components.
Return JSON with the following fields:
- "intent": string (e.g. "city_overview", "weather_inquiry", "image_request", "general_travel")
- "target_city": string name of target city if identified, else null
- "needs_search": boolean, true if vector store or web search is needed
- "needs_weather": boolean, true if weather or forecast data is requested
- "needs_images": boolean, true if city images or photos are requested
"""

TOOL_CALL_GENERATION_SYSTEM_PROMPT = """You are an AI tool planner for a travel assistant.
Identify which tools should be executed for the user query.
Available tools:
1. `get_weather(city: str, days: int = 7)`
2. `search_images(city: str, limit: int = 5)`
3. `search_city(city: str)`

Return a JSON list of tool call objects, each having:
- "tool_name": string matching one of the available tool names
- "arguments": dictionary of arguments to pass to the tool
"""

STRUCTURED_SYNTHESIS_SYSTEM_PROMPT = """You are a travel assistant response synthesizer.
Given city background knowledge, weather forecast, and image URLs, generate a comprehensive structured response.
Return JSON matching:
{
  "city_summary": "Comprehensive summary of the city knowledge...",
  "weather_forecast": [{"date": "YYYY-MM-DD", "temperature": 25.0, "condition": "Sunny"}],
  "image_urls": ["url1", "url2"],
  "weather_error": null,
  "image_error": null,
  "search_error": null
}
"""


def build_city_extraction_prompt(query: str, messages: Optional[list[dict]] = None) -> str:
    """Build prompt string for city extraction."""
    context_str = ""
    if messages:
        formatted_msgs = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-4:]])
        context_str = f"Recent conversation context:\n{formatted_msgs}\n\n"
    return f"{context_str}Current query: '{query}'\n\nExtract the target city and return JSON."


def build_query_classification_prompt(query: str, messages: Optional[list[dict]] = None) -> str:
    """Build prompt string for query classification."""
    context_str = ""
    if messages:
        formatted_msgs = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-4:]])
        context_str = f"Recent conversation context:\n{formatted_msgs}\n\n"
    return f"{context_str}Classify this query: '{query}' and return JSON."


def build_tool_call_prompt(query: str, city: Optional[str] = None) -> str:
    """Build prompt string for tool call generation."""
    city_str = f" Target city identified: '{city}'." if city else ""
    return f"User Query: '{query}'.{city_str}\nGenerate a list of required tool calls in JSON."


def build_synthesis_prompt(
    query: str,
    city: str,
    city_knowledge: str,
    weather_forecast: Optional[list[WeatherDataPoint]] = None,
    image_urls: Optional[list[str]] = None,
) -> str:
    """Build prompt string for synthesizing travel response."""
    weather_str = "None"
    if weather_forecast:
        weather_str = ", ".join([f"{w.date}: {w.temperature}°C ({w.condition})" for w in weather_forecast])
    images_str = ", ".join(image_urls) if image_urls else "None"

    return (
        f"Query: {query}\n"
        f"City: {city}\n"
        f"City Knowledge:\n{city_knowledge}\n\n"
        f"Weather Forecast Data: {weather_str}\n"
        f"Image URLs: {images_str}\n\n"
        "Synthesize a cohesive travel response structured as JSON."
    )
