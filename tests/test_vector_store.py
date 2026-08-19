"""Tests for VectorStoreService and seeding script."""

import os
import shutil
import pytest
from interfaces.vector_store import VectorStoreServiceInterface
from models.city import CityKnowledge
from services.vector_store import VectorStoreService
from scripts.seed_vector_store import seed_vector_store


@pytest.fixture(scope="module")
def seeded_vector_store(tmp_path_factory):
    """Fixture providing a seeded VectorStoreService using a temporary database path."""
    test_db_dir = str(tmp_path_factory.mktemp("test_vector_db"))

    # Seed data into temporary test_db_dir
    seed_vector_store(data_dir="data/cities", reset=True, db_path=test_db_dir)

    # Return VectorStoreService initialized with test_db_dir
    service = VectorStoreService(db_path=test_db_dir)
    return service


def test_vector_store_implements_interface(seeded_vector_store):
    """Verify VectorStoreService implements VectorStoreServiceInterface."""
    assert isinstance(seeded_vector_store, VectorStoreServiceInterface)


def test_search_known_city_exact(seeded_vector_store):
    """Test searching for known cities returns expected CityKnowledge."""
    cities_to_test = ["Tokyo", "Paris", "New York", "Kyoto"]
    for city_name in cities_to_test:
        result = seeded_vector_store.search_city(city_name)
        assert result is not None
        assert isinstance(result, CityKnowledge)
        assert result.city.lower() == city_name.lower()
        assert result.source == "vector_store"
        assert len(result.highlights) > 0
        assert len(result.description) > 0


def test_search_known_city_case_insensitive(seeded_vector_store):
    """Test case-insensitive queries for known cities."""
    res_tokyo = seeded_vector_store.search_city("tokyo")
    assert res_tokyo is not None
    assert res_tokyo.city == "Tokyo"

    res_paris = seeded_vector_store.search_city("PARIS")
    assert res_paris is not None
    assert res_paris.city == "Paris"


def test_search_unknown_city_returns_none(seeded_vector_store):
    """Test searching for unknown cities (e.g. Snohomish) returns None as required."""
    unknown_cities = ["Snohomish", "Atlantis", "Springfield", "NonExistentCity123"]
    for unknown in unknown_cities:
        result = seeded_vector_store.search_city(unknown)
        assert result is None, f"Expected None for unknown city '{unknown}', but got {result}"


def test_empty_vector_store(tmp_path):
    """Test searching in an unseeded/empty vector store returns None gracefully."""
    empty_db_dir = str(tmp_path / "empty_db")
    service = VectorStoreService(db_path=empty_db_dir)
    assert service.search_city("Tokyo") is None
    assert service.search_city("Snohomish") is None


def test_empty_query(seeded_vector_store):
    """Test searching with empty string or whitespace returns None."""
    assert seeded_vector_store.search_city("") is None
    assert seeded_vector_store.search_city("   ") is None
