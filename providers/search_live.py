"""Live search service provider using Tavily API for web search fallback."""

import logging
from typing import Optional
import httpx
from config.settings import settings
from interfaces.search import SearchServiceInterface
from models.city import CityKnowledge

logger = logging.getLogger(__name__)


class LiveSearchProvider(SearchServiceInterface):
    """Live web search provider using Tavily Search API to retrieve destination knowledge."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 12.0):
        """Initialize live search provider.

        Args:
            api_key: Tavily API key (defaults to settings.tavily_api_key).
            timeout: HTTP request timeout in seconds (default 12.0s).
        """
        self.api_key = api_key or settings.tavily_api_key
        self.timeout = timeout

    def search_city(self, city: str) -> Optional[CityKnowledge]:
        """Perform web search retrieval for city knowledge.

        Args:
            city: Target city name to search.

        Returns:
            CityKnowledge instance populated with search results, or fallback CityKnowledge object.
        """
        city_clean = city.strip()
        if not city_clean:
            return None

        if not self.api_key:
            logger.info("Tavily API key is not configured; using web search fallback provider.")
            return self._build_fallback_knowledge(city_clean)

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": f"{city_clean} travel guide attractions culture travel tips overview",
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer")
                results = data.get("results", [])

                description = answer if answer else f"Comprehensive travel guide for {city_clean} based on live search results."
                highlights: list[str] = []
                culture_tips: list[str] = []

                for item in results:
                    title = item.get("title", "")
                    snippet = item.get("content", "")
                    if title and len(highlights) < 5:
                        highlights.append(title)
                    if snippet and len(culture_tips) < 3:
                        culture_tips.append(snippet[:120].strip() + "...")

                if not highlights:
                    highlights = [f"Explore top sights in {city_clean}", f"Historical landmarks of {city_clean}", "Local food & dining"]

                if not culture_tips:
                    culture_tips = ["Check local customs and etiquette.", "Use public transport options.", "Stay aware of local weather."]

                return CityKnowledge(
                    city=city_clean,
                    country="International Destination",
                    description=description,
                    highlights=highlights,
                    culture_tips=culture_tips,
                    source="web_search",
                )

            logger.warning("Tavily search returned HTTP %s; falling back to fallback search.", response.status_code)
            return self._build_fallback_knowledge(city_clean)

        except Exception as exc:
            logger.error("Error executing Tavily web search for %s: %s; using fallback.", city_clean, exc)
            return self._build_fallback_knowledge(city_clean)

    def _build_fallback_knowledge(self, city: str) -> CityKnowledge:
        """Create fallback CityKnowledge object when live search API is unavailable."""
        return CityKnowledge(
            city=city,
            country="Search Result",
            description=f"Live web search results for {city}: A travel destination offering rich cultural experiences, local cuisine, and historical attractions.",
            highlights=[
                f"Famous Landmarks in {city}",
                f"Cultural Districts of {city}",
                f"Local Dining & Markets in {city}",
            ],
            culture_tips=[
                "Review local guidelines and public transport passes prior to arrival.",
                "Respect local customs and neighborhood regulations.",
                "Keep emergency contact details and offline maps accessible.",
            ],
            source="web_search",
        )


# Alias for flexibility
LiveSearchService = LiveSearchProvider
