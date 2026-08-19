"""Script to seed ChromaDB vector store with city knowledge data."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from models.city import CityKnowledge
from services.vector_store import VectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed_vector_store(data_dir: str = "data/cities", reset: bool = False, db_path: Optional[str] = None) -> int:
    """Load JSON files from data_dir and seed ChromaDB vector store.

    Args:
        data_dir: Path to directory containing city JSON files.
        reset: If True, delete and recreate the collection before seeding.
        db_path: Optional override path for vector store database directory.

    Returns:
        Number of cities successfully ingested.
    """
    service = VectorStoreService(db_path=db_path)

    if reset:
        logger.info(f"Resetting collection '{VectorStoreService.COLLECTION_NAME}'...")
        try:
            service.client.delete_collection(VectorStoreService.COLLECTION_NAME)
        except Exception as e:
            logger.info(f"Collection reset note: {e}")
        service.collection = service.client.get_or_create_collection(
            name=VectorStoreService.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory '{data_dir}' does not exist.")
        return 0

    json_files = list(data_path.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in '{data_dir}'.")
        return 0

    count = 0
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            city_obj = CityKnowledge.model_validate(raw_data)

            # Construct document text for embedding
            doc_text = (
                f"{city_obj.city}, {city_obj.country}: {city_obj.description}\n"
                f"Top Highlights: {', '.join(city_obj.highlights)}\n"
                f"Cultural Tips: {', '.join(city_obj.culture_tips)}"
            )

            # Prepare metadata (primitive types only)
            metadata = {
                "city": city_obj.city,
                "city_lower": city_obj.city.lower(),
                "country": city_obj.country,
                "source": city_obj.source,
                "json_data": city_obj.model_dump_json()
            }

            doc_id = f"city_{city_obj.city.lower().replace(' ', '_')}"

            service.collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata]
            )
            logger.info(f"Successfully seeded city: {city_obj.city} ({doc_id})")
            count += 1

        except Exception as e:
            logger.error(f"Failed to process '{file_path}': {e}")

    logger.info(f"Seeding completed. Total cities in store: {service.collection.count()}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed vector store with city knowledge.")
    parser.add_argument("--data-dir", default="data/cities", help="Directory containing city JSON files")
    parser.add_argument("--reset", action="store_true", help="Reset collection before seeding")

    args = parser.parse_args()
    seeded = seed_vector_store(data_dir=args.data_dir, reset=args.reset)
    print(f"Seeded {seeded} cities successfully.")
