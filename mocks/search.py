"""Mock web search provider simulating external search engine retrieval."""

import time
from typing import Optional
from interfaces.search import SearchServiceInterface
from models.city import CityKnowledge

# Preset city knowledge data for web search fallback simulation
_MOCK_CITY_KNOWLEDGE: dict[str, dict] = {
    "paris": {
        "country": "France",
        "description": "Paris, the capital of France, is a global center for art, fashion, gastronomy, and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.",
        "highlights": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "Arc de Triomphe", "Montmartre"],
        "culture_tips": [
            "Always greet shopkeepers with 'Bonjour' upon entering.",
            "Tipping is included in restaurants, but small extra change is appreciated.",
            "Use the Metro system for convenient navigation across arrondissements.",
        ],
    },
    "tokyo": {
        "country": "Japan",
        "description": "Tokyo, Japan's bustling capital, mixes ultra-modern skyscrapers with historic temples. The city is famous for its vibrant street life, world-class culinary scene, and reliable public transit.",
        "highlights": ["Sensō-ji Temple", "Tokyo Tower", "Shibuya Crossing", "Meiji Shrine", "Akihabara"],
        "culture_tips": [
            "Bow slightly when greeting people and expressing gratitude.",
            "Avoid eating while walking on public streets.",
            "Keep voice volume low on public trains and subways.",
        ],
    },
    "new york": {
        "country": "United States",
        "description": "New York City comprises 5 boroughs sitting where the Hudson River meets the Atlantic Ocean. At its core is Manhattan, a densely populated borough that’s among the world’s major commercial, financial, and cultural centers.",
        "highlights": ["Statue of Liberty", "Central Park", "Times Square", "Empire State Building", "Metropolitan Museum of Art"],
        "culture_tips": [
            "Walk on the right side of sidewalks and don't block pedestrian traffic flow.",
            "Tipping 18-20% is customary at sit-down restaurants and taxi rides.",
            "Subway navigation is fastest using contactless payment (OMNY).",
        ],
    },
    "kyoto": {
        "country": "Japan",
        "description": "Kyoto, once the capital of Japan, is a city on the island of Honshu. It’s famous for its numerous classical Buddhist temples, gardens, imperial palaces, Shinto shrines, and traditional wooden houses.",
        "highlights": ["Fushimi Inari Taisha", "Kinkaku-ji (Golden Pavilion)", "Arashiyama Bamboo Grove", "Gion District", "Kiyomizu-dera"],
        "culture_tips": [
            "Remove shoes when entering traditional ryokans, temples, or homes.",
            "Respect geishas in Gion and do not take photos without permission.",
            "Maintain quiet demeanor near sacred temple grounds.",
        ],
    },
    "snohomish": {
        "country": "United States",
        "description": "Snohomish is a charming city in Snohomish County, Washington, known for its historic downtown district, antique shops, scenic river views, and agricultural festivals.",
        "highlights": ["Historic Downtown Snohomish", "Centennial Trail", "Snohomish Riverfront", "Stocker Farms", "Blackman House Museum"],
        "culture_tips": [
            "Dress in layers as Pacific Northwest weather can change rapidly.",
            "Enjoy local farm-to-table markets and antique hunting along First Street.",
            "Friendly casual greetings are common among locals.",
        ],
    },
}


class MockSearchProvider(SearchServiceInterface):
    """Mock web search provider returning realistic city information for fallback queries."""

    def __init__(self, latency: float = 0.05):
        """Initialize mock search provider.

        Args:
            latency: Simulated delay in seconds (default 0.05s).
        """
        self.latency = latency

    def search_city(self, city: str) -> Optional[CityKnowledge]:
        """Perform web search retrieval for city knowledge.

        Args:
            city: Target city name to search.

        Returns:
            CityKnowledge instance if city information is found, or generic fallback CityKnowledge.
        """
        if self.latency > 0:
            time.sleep(self.latency)

        city_clean = city.strip()
        city_key = city_clean.lower()

        if city_key in _MOCK_CITY_KNOWLEDGE:
            data = _MOCK_CITY_KNOWLEDGE[city_key]
            return CityKnowledge(
                city=city_clean,
                country=data["country"],
                description=data["description"],
                highlights=data["highlights"],
                culture_tips=data["culture_tips"],
                source="web_search",
            )

        # Fallback search result for unknown cities
        return CityKnowledge(
            city=city_clean,
            country="International Destination",
            description=f"Web search results for {city_clean}: A fascinating travel destination featuring local culture, historical landmarks, and unique local experiences.",
            highlights=[
                f"Historic Center of {city_clean}",
                f"Main City Square of {city_clean}",
                f"Local Cultural Museum",
            ],
            culture_tips=[
                "Check local weather forecasts and seasonal recommendations before travel.",
                "Verify passport and visa entry requirements ahead of departure.",
                "Explore local neighborhood cuisine and public transit options.",
            ],
            source="web_search",
        )


# Alias for backward compatibility / flexibility
MockSearchService = MockSearchProvider
