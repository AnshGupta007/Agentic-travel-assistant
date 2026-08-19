"""Unit tests for LiveImageProvider."""

from unittest.mock import MagicMock, patch
from providers.images_live import LiveImageProvider


def test_images_live_missing_key_fallback():
    provider = LiveImageProvider(api_key=None)
    urls = provider.search_images("Paris", limit=3)
    assert isinstance(urls, list)
    assert len(urls) == 3
    assert all(url.startswith("http") for url in urls)


def test_images_live_empty_city():
    provider = LiveImageProvider(api_key="test_key")
    urls = provider.search_images("", limit=2)
    assert isinstance(urls, list)
    assert len(urls) == 2


@patch("httpx.Client.get")
def test_images_live_success_200(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"urls": {"regular": "https://example.com/tokyo1.jpg"}},
            {"urls": {"regular": "https://example.com/tokyo2.jpg"}},
        ]
    }
    mock_get.return_value = mock_response

    provider = LiveImageProvider(api_key="dummy_key")
    urls = provider.search_images("Tokyo", limit=2)

    assert len(urls) == 2
    assert urls[0] == "https://example.com/tokyo1.jpg"
    assert urls[1] == "https://example.com/tokyo2.jpg"


@patch("httpx.Client.get")
def test_images_live_api_error_uses_fallback(mock_get):
    mock_get.side_effect = Exception("Unsplash API rate limit")

    provider = LiveImageProvider(api_key="dummy_key")
    urls = provider.search_images("Kyoto", limit=3)

    assert isinstance(urls, list)
    assert len(urls) == 3
    assert all(url.startswith("http") for url in urls)
