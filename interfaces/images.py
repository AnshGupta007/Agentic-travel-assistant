"""Image service interface contract."""

from abc import ABC, abstractmethod


class ImageServiceInterface(ABC):
    """Abstract base class for image providers (live and mock)."""

    @abstractmethod
    def search_images(self, city: str, limit: int = 5) -> list[str]:
        """Search for image URLs representing a given city.

        Args:
            city: City name to search images for.
            limit: Maximum number of image URLs to return.

        Returns:
            List of valid image URL strings.
        """
        pass
