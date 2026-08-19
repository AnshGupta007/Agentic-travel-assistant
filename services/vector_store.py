"""Vector store service implementation for city knowledge retrieval."""

import json
import logging
import os
from typing import Optional
import chromadb

from config.settings import settings
from interfaces.vector_store import VectorStoreServiceInterface
from models.city import CityKnowledge

logger = logging.getLogger(__name__)


class VectorStoreService(VectorStoreServiceInterface):
    """ChromaDB-backed implementation of VectorStoreServiceInterface."""

    COLLECTION_NAME = "city_knowledge"
    # Distance threshold for cosine / L2 distance to consider a vector match relevant
    DISTANCE_THRESHOLD = 0.95

    def __init__(self, db_path: Optional[str] = None):
        """Initialize ChromaDB persistent client and collection.

        Args:
            db_path: Path to ChromaDB storage directory. Defaults to settings.vector_store_path.
        """
        self.db_path = db_path or settings.vector_store_path
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def search_city(self, city: str) -> Optional[CityKnowledge]:
        """Search local vector store for city knowledge.

        Args:
            city: City name to search in local database.

        Returns:
            CityKnowledge instance if city exists in vector store, or None.
        """
        clean_city = city.strip().lower()
        if not clean_city or self.collection.count() == 0:
            return None

        # Step 1: Direct metadata match check (case-insensitive)
        try:
            exact_results = self.collection.get(
                where={"city_lower": clean_city}
            )
            if exact_results and exact_results.get("metadatas") and len(exact_results["metadatas"]) > 0:
                meta = exact_results["metadatas"][0]
                if "json_data" in meta:
                    return CityKnowledge.model_validate_json(meta["json_data"])
        except Exception as e:
            logger.warning(f"Error during metadata search for '{city}': {e}")

        # Step 2: Vector similarity search fallback
        try:
            results = self.collection.query(
                query_texts=[f"City guide and travel information for {city}"],
                n_results=1
            )

            if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                return None

            distance = results["distances"][0][0] if results.get("distances") else 1.0
            meta = results["metadatas"][0][0] if results.get("metadatas") else {}

            # Check if distance is within threshold and the retrieved city matches
            result_city_lower = meta.get("city_lower", "")
            if distance <= self.DISTANCE_THRESHOLD:
                # If city names share root or distance is very low
                if clean_city in result_city_lower or result_city_lower in clean_city or distance < 0.4:
                    if "json_data" in meta:
                        return CityKnowledge.model_validate_json(meta["json_data"])

        except Exception as e:
            logger.error(f"Error during vector search for '{city}': {e}")

        return None
