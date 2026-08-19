"""Vector store service interface contract."""

from abc import ABC, abstractmethod
from typing import Optional
from models.city import CityKnowledge


class VectorStoreServiceInterface(ABC):
    """Abstract base class for vector store knowledge retrieval."""

    @abstractmethod
    def search_city(self, city: str) -> Optional[CityKnowledge]:
        """Search local vector store for city knowledge.

        Args:
            city: City name to search in local database.

        Returns:
            CityKnowledge instance if city exists in vector store, or None.
        """
        pass
