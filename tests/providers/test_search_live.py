"""Unit tests for LiveSearchProvider."""

from unittest.mock import MagicMock, patch
from models.city import CityKnowledge
from providers.search_live import LiveSearchProvider


def test_search_live_missing_key_fallback():
    provider = LiveSearchProvider(api_key=None)
    result = provider.search_city("Snohomish")
    assert isinstance(result, CityKnowledge)
    assert result.city == "Snohomish"
    assert result.source == "web_search"
    assert len(result.highlights) > 0


def test_search_live_empty_city():
    provider = LiveSearchProvider(api_key="test_key")
    result = provider.search_city("   ")
    assert result is None


@patch("httpx.Client.post")
def test_search_live_success_200(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answer": "Tokyo is the capital city of Japan known for its futuristic technology and culture.",
        "results": [
            {"title": "Sensō-ji Temple", "content": "Famous ancient Buddhist temple located in Asakusa."},
            {"title": "Shibuya Crossing", "content": "Scramble crossing in front of Shibuya Station."},
        ],
    }
    mock_post.return_value = mock_response

    provider = LiveSearchProvider(api_key="dummy_key")
    result = provider.search_city("Tokyo")

    assert isinstance(result, CityKnowledge)
    assert result.city == "Tokyo"
    assert result.source == "web_search"
    assert "Tokyo is the capital city" in result.description
    assert "Sensō-ji Temple" in result.highlights
    assert len(result.culture_tips) > 0


@patch("httpx.Client.post")
def test_search_live_api_error_fallback(mock_post):
    mock_post.side_effect = Exception("API connection error")

    provider = LiveSearchProvider(api_key="dummy_key")
    result = provider.search_city("London")

    assert isinstance(result, CityKnowledge)
    assert result.city == "London"
    assert result.source == "web_search"
