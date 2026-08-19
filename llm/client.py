"""LLM Client wrapper providing multi-provider and mock support."""

import logging
import re
from typing import Optional
from config.settings import Settings, settings as global_settings
from models.weather import WeatherDataPoint
from models.travel import TravelResponse
from llm.prompts import (
    CITY_EXTRACTION_SYSTEM_PROMPT,
    QUERY_CLASSIFICATION_SYSTEM_PROMPT,
    TOOL_CALL_GENERATION_SYSTEM_PROMPT,
    STRUCTURED_SYNTHESIS_SYSTEM_PROMPT,
    build_city_extraction_prompt,
    build_query_classification_prompt,
    build_tool_call_prompt,
    build_synthesis_prompt,
)
from llm.structured import (
    CityExtractionResult,
    QueryClassificationResult,
    ToolCallSpec,
    parse_json_completion,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM Client abstraction supporting OpenAI, Anthropic, and Mock providers."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or global_settings
        self.provider = self.settings.llm_provider.lower()
        self.model = self.settings.llm_model

    def is_mock_active(self) -> bool:
        """Check if client should operate in mock mode."""
        if self.settings.is_mock():
            return True
        if self.provider == "openai" and not self.settings.openai_api_key:
            return True
        if self.provider == "anthropic" and not self.settings.anthropic_api_key:
            return True
        return False

    def _call_raw_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Execute raw LLM call via OpenAI or Anthropic SDKs."""
        if self.provider == "anthropic" and self.settings.anthropic_api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
                response = client.messages.create(
                    model=self.model if "claude" in self.model else "claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic LLM call failed: {e}. Falling back to mock logic.")
                raise e

        # Default: OpenAI
        if self.settings.openai_api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.settings.openai_api_key)
                response = client.chat.completions.create(
                    model=self.model if "gpt" in self.model else "gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"OpenAI LLM call failed: {e}. Falling back to mock logic.")
                raise e

        raise RuntimeError("No API key available for live LLM provider.")

    def extract_city(self, query: str, messages: Optional[list[dict]] = None) -> Optional[str]:
        """Extract city name from query or conversation history."""
        if self.is_mock_active():
            return self._mock_extract_city(query, messages)

        prompt = build_city_extraction_prompt(query, messages)
        try:
            raw_output = self._call_raw_llm(CITY_EXTRACTION_SYSTEM_PROMPT, prompt)
            result = parse_json_completion(raw_output, CityExtractionResult)
            return result.city
        except Exception as e:
            logger.warning(f"Live LLM city extraction failed ({e}), using mock fallback.")
            return self._mock_extract_city(query, messages)

    def classify_query(
        self, query: str, messages: Optional[list[dict]] = None
    ) -> QueryClassificationResult:
        """Classify user query intent and required execution steps."""
        if self.is_mock_active():
            return self._mock_classify_query(query, messages)

        prompt = build_query_classification_prompt(query, messages)
        try:
            raw_output = self._call_raw_llm(QUERY_CLASSIFICATION_SYSTEM_PROMPT, prompt)
            return parse_json_completion(raw_output, QueryClassificationResult)
        except Exception as e:
            logger.warning(f"Live LLM query classification failed ({e}), using mock fallback.")
            return self._mock_classify_query(query, messages)

    def generate_tool_calls(
        self, query: str, city: Optional[str] = None
    ) -> list[ToolCallSpec]:
        """Generate tool execution plan for user query."""
        if self.is_mock_active():
            return self._mock_generate_tool_calls(query, city)

        prompt = build_tool_call_prompt(query, city)
        try:
            raw_output = self._call_raw_llm(TOOL_CALL_GENERATION_SYSTEM_PROMPT, prompt)
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", cleaned).strip()
            import json
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [ToolCallSpec.model_validate(item) for item in data]
            elif isinstance(data, dict) and "tools" in data:
                return [ToolCallSpec.model_validate(item) for item in data["tools"]]
            return [ToolCallSpec.model_validate(data)]
        except Exception as e:
            logger.warning(f"Live LLM tool call generation failed ({e}), using mock fallback.")
            return self._mock_generate_tool_calls(query, city)

    def generate_structured_response(
        self,
        query: str,
        city: str,
        city_knowledge: str,
        weather_forecast: Optional[list[WeatherDataPoint]] = None,
        image_urls: Optional[list[str]] = None,
        weather_error: Optional[str] = None,
        image_error: Optional[str] = None,
        search_error: Optional[str] = None,
    ) -> TravelResponse:
        """Synthesize gathered data into TravelResponse."""
        if self.is_mock_active():
            return self._mock_generate_structured_response(
                query,
                city,
                city_knowledge,
                weather_forecast,
                image_urls,
                weather_error=weather_error,
                image_error=image_error,
                search_error=search_error,
            )

        prompt = build_synthesis_prompt(
            query, city, city_knowledge, weather_forecast, image_urls
        )
        try:
            raw_output = self._call_raw_llm(STRUCTURED_SYNTHESIS_SYSTEM_PROMPT, prompt)
            response = parse_json_completion(raw_output, TravelResponse)
            if weather_error and not response.weather_error:
                response.weather_error = weather_error
            if image_error and not response.image_error:
                response.image_error = image_error
            if search_error and not response.search_error:
                response.search_error = search_error
            return response
        except Exception as e:
            logger.warning(f"Live LLM response synthesis failed ({e}), using mock fallback.")
            return self._mock_generate_structured_response(
                query,
                city,
                city_knowledge,
                weather_forecast,
                image_urls,
                weather_error=weather_error,
                image_error=image_error,
                search_error=search_error,
            )

    # --- MOCK FALLBACK IMPLEMENTATIONS ---

    def _mock_extract_city(self, query: str, messages: Optional[list[dict]] = None) -> Optional[str]:
        known_cities = [
            "Tokyo", "Paris", "New York", "Kyoto", "Snohomish",
            "London", "Rome", "Sydney", "Berlin", "Barcelona"
        ]
        # First check current query
        for city in known_cities:
            if re.search(rf"\b{city}\b", query, re.IGNORECASE):
                return city

        # Check past messages if present
        if messages:
            for msg in reversed(messages):
                content = msg.get("content", "")
                for city in known_cities:
                    if re.search(rf"\b{city}\b", content, re.IGNORECASE):
                        return city

        # Basic fallback word match
        words = [w.strip("?,!.").capitalize() for w in query.split()]
        for w in words:
            if w in known_cities:
                return w
        return None

    def _mock_classify_query(
        self, query: str, messages: Optional[list[dict]] = None
    ) -> QueryClassificationResult:
        city = self._mock_extract_city(query, messages)
        q_lower = query.lower()

        intent = "general_travel"
        if "weather" in q_lower or "forecast" in q_lower or "temperature" in q_lower:
            intent = "weather_inquiry"
        elif "image" in q_lower or "photo" in q_lower or "picture" in q_lower:
            intent = "image_request"
        elif city:
            intent = "city_overview"

        return QueryClassificationResult(
            intent=intent,
            target_city=city,
            needs_search=True,
            needs_weather=True,
            needs_images=True,
        )

    def _mock_generate_tool_calls(
        self, query: str, city: Optional[str] = None
    ) -> list[ToolCallSpec]:
        target_city = city or self._mock_extract_city(query) or "Tokyo"
        return [
            ToolCallSpec(tool_name="search_city", arguments={"city": target_city}),
            ToolCallSpec(tool_name="get_weather", arguments={"city": target_city, "days": 7}),
            ToolCallSpec(tool_name="search_images", arguments={"city": target_city, "limit": 5}),
        ]

    def _mock_generate_structured_response(
        self,
        query: str,
        city: str,
        city_knowledge: str,
        weather_forecast: Optional[list[WeatherDataPoint]] = None,
        image_urls: Optional[list[str]] = None,
        weather_error: Optional[str] = None,
        image_error: Optional[str] = None,
        search_error: Optional[str] = None,
    ) -> TravelResponse:
        forecast = weather_forecast or []
        imgs = image_urls or []
        summary = (
            f"{city.capitalize()} Travel Guide:\n\n{city_knowledge}\n\n"
            f"It's a fantastic destination for your trip!"
        )
        w_err = weather_error if weather_error is not None else (None if forecast else "No weather data retrieved")
        img_err = image_error if image_error is not None else (None if imgs else "No images retrieved")
        s_err = search_error if search_error is not None else (None if city_knowledge else "No city knowledge retrieved")

        return TravelResponse(
            city_summary=summary,
            weather_forecast=forecast,
            image_urls=imgs,
            weather_error=w_err,
            image_error=img_err,
            search_error=s_err,
        )

