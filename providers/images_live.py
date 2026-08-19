"""Live image service provider using Unsplash API with fallback mechanisms."""

import logging
from typing import Optional
import httpx
from config.settings import settings
from interfaces.images import ImageServiceInterface

logger = logging.getLogger(__name__)

# Fallback image URLs per city if live API is unconfigured or unreachable
_FALLBACK_IMAGES: dict[str, list[str]] = {
    "paris": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?auto=format&fit=crop&w=800&q=80",
    ],
    "tokyo": [
        "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=800&q=80",
    ],
    "new york": [
        "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?auto=format&fit=crop&w=800&q=80",
    ],
}

_DEFAULT_FALLBACK: list[str] = [
    "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1476514525535-ce74f45814d3?auto=format&fit=crop&w=800&q=80",
]


class LiveImageProvider(ImageServiceInterface):
    """Live image service provider fetching destination images via Unsplash API."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        """Initialize live image provider.

        Args:
            api_key: Unsplash API Access Key (defaults to settings.unsplash_api_key).
            timeout: HTTP request timeout in seconds (default 10.0s).
        """
        self.api_key = api_key or settings.unsplash_api_key
        self.timeout = timeout

    def search_images(self, city: str, limit: int = 5) -> list[str]:
        """Search for image URLs representing a given city.

        Args:
            city: Target city name to find photos for.
            limit: Maximum number of image URLs to return (default 5).

        Returns:
            List of valid image URL strings.
        """
        city_clean = city.strip()
        max_limit = max(1, limit)

        if not city_clean:
            return self._get_fallback(city_clean, max_limit)

        if not self.api_key:
            logger.info("Unsplash API key not provided; returning public image search fallback.")
            return self._get_fallback(city_clean, max_limit)

        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": f"{city_clean} city travel landmark",
            "per_page": max_limit,
            "orientation": "landscape",
        }
        headers = {
            "Authorization": f"Client-ID {self.api_key}",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                urls: list[str] = []
                for item in results:
                    img_url = item.get("urls", {}).get("regular") or item.get("urls", {}).get("small")
                    if img_url:
                        urls.append(img_url)

                if urls:
                    return urls[:max_limit]

            logger.warning("Unsplash returned HTTP %s or empty results; using fallback.", response.status_code)
            return self._get_fallback(city_clean, max_limit)

        except Exception as exc:
            logger.error("Error calling Unsplash API for %s: %s; using fallback.", city_clean, exc)
            return self._get_fallback(city_clean, max_limit)

    def _get_fallback(self, city: str, limit: int) -> list[str]:
        """Retrieve fallback images if live search fails or key is missing."""
        city_key = city.lower()
        images = _FALLBACK_IMAGES.get(city_key, _DEFAULT_FALLBACK)
        return images[:limit]


# Alias for flexibility
LiveImageService = LiveImageProvider
