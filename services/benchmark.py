"""RAG Benchmarking & Experimentation Suite.

Measures and compares performance across:
1. Standard Text RAG (Vector Store)
2. Graph RAG (Knowledge Graph Entity Traversal)
3. Web Search Fallback

Metrics evaluated:
- Execution Latency (ms)
- Entity Recall Rate (%)
- Context Depth (Count of structural facts)
- Multi-modal Grounding Score (0.0 - 1.0)
"""

import time
import logging
from typing import Dict, Any, List
from graph.checkpoint import TravelAssistantSession

logger = logging.getLogger(__name__)


class RAGBenchmarkSuite:
    """Benchmark runner for RAG performance comparison."""

    def __init__(self, sample_cities: List[str] = None):
        self.sample_cities = sample_cities or ["Tokyo", "Paris", "New York", "Snohomish"]

    def evaluate_retrieval_mode(self, city: str, retrieval_mode: str) -> Dict[str, Any]:
        """Execute query in specific retrieval mode and calculate metrics.

        Args:
            city: Target city name.
            retrieval_mode: 'vector_store' | 'graph_rag' | 'web_search'

        Returns:
            Metrics dict.
        """
        thread_id = f"bench_{retrieval_mode}_{city.lower().replace(' ', '_')}_{int(time.time()*1000)}"
        session = TravelAssistantSession(thread_id=thread_id)

        query = f"Tell me about {city} entity graph and travel recommendations"

        start_time = time.perf_counter()
        
        # Invoke graph passing explicit retrieval mode override if applicable
        if retrieval_mode == "graph_rag":
            output_state = session.invoke(f"Tell me about {city} graph relations POI", retrieval_mode="graph_rag")
        else:
            output_state = session.invoke(query)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        summary = output_state.get("city_summary", "")
        forecast = output_state.get("weather_forecast", [])
        images = output_state.get("image_urls", [])
        entities = output_state.get("graph_entities", [])

        # 1. Entity Recall: Check presence of key terms
        city_lower = city.lower()
        key_entities_map = {
            "tokyo": ["sensoji", "shibuya", "tokyo tower", "japan"],
            "paris": ["eiffel", "louvre", "notre", "france"],
            "new york": ["statue", "central park", "empire", "york"],
            "snohomish": ["downtown", "antique", "washington", "trail"]
        }
        target_entities = key_entities_map.get(city_lower, [city_lower])
        summary_lower = summary.lower()
        matches = sum(1 for e in target_entities if e in summary_lower)
        entity_recall = round((matches / len(target_entities)) * 100, 1)

        # 2. Context Depth
        lines = [line for line in summary.split("\n") if line.strip()]
        context_depth = len(lines) + len(entities)

        # 3. Multi-modal Grounding Score
        has_summary = 0.4 if len(summary) > 50 else 0.0
        has_weather = 0.3 if len(forecast) >= 5 else 0.0
        has_images = 0.3 if len(images) > 0 else 0.0
        grounding_score = round(has_summary + has_weather + has_images, 2)

        return {
            "city": city,
            "retrieval_mode": retrieval_mode,
            "latency_ms": latency_ms,
            "entity_recall": entity_recall,
            "context_depth": context_depth,
            "grounding_score": grounding_score,
            "weather_count": len(forecast),
            "image_count": len(images),
        }

    def run_comparative_benchmark(self, city: str = "Tokyo") -> Dict[str, Any]:
        """Run side-by-side benchmark comparing Vector RAG vs Graph RAG vs Web Search.

        Args:
            city: City name for test benchmark run.

        Returns:
            Benchmark report dictionary.
        """
        logger.info(f"Running comparative RAG benchmark for city '{city}'...")
        modes = ["vector_store", "graph_rag", "web_search"]
        results = {}

        for mode in modes:
            try:
                results[mode] = self.evaluate_retrieval_mode(city=city, retrieval_mode=mode)
            except Exception as e:
                logger.error(f"Benchmark error for mode '{mode}': {e}")
                results[mode] = {
                    "city": city,
                    "retrieval_mode": mode,
                    "latency_ms": 0.0,
                    "entity_recall": 0.0,
                    "context_depth": 0,
                    "grounding_score": 0.0,
                    "error": str(e)
                }

        # Calculate best performing mode
        best_mode = max(results.keys(), key=lambda m: (results[m]["entity_recall"], results[m]["grounding_score"]))

        return {
            "city": city,
            "comparative_results": results,
            "recommended_mode": best_mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
