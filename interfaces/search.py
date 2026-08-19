"""Search service interface contract."""

from abc import ABC, abstractmethod
from typing import Optional
from models.city import CityKnowledge


class SearchServiceInterface(ABC):
    """Abstract base class for web search fallback providers (live and mock)."""

    @abstractmethod
    def search_city(self, city: str) -> Optional[CityKnowledge]:
        """Perform web search retrieval for city knowledge.

        Args:
            city: City name to look up.

        Returns:
            CityKnowledge instance if found, or None.
        """
        pass
