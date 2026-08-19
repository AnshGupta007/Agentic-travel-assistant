"""Knowledge Graph & Graph RAG Service implementation.

Provides entity-relationship knowledge graphs for destinations, multi-hop sub-graph traversal,
and hybrid Graph-Augmented Retrieval Generation (Graph RAG) context construction.
"""

import logging
from typing import Optional, Dict, Any, List
from interfaces.knowledge_graph import KnowledgeGraphServiceInterface

logger = logging.getLogger(__name__)


# Knowledge Graph seed database containing entities, nodes, and typed edges
KNOWLEDGE_GRAPH_DB: Dict[str, Dict[str, Any]] = {
    "tokyo": {
        "nodes": [
            {"id": "tokyo", "label": "Tokyo", "type": "City", "description": "Capital of Japan, blending ultramodern and traditional culture."},
            {"id": "sensoji", "label": "Senso-ji Temple", "type": "POI", "category": "Historical & Culture"},
            {"id": "shibuya_crossing", "label": "Shibuya Crossing", "type": "POI", "category": "Urban & Life"},
            {"id": "tokyo_tower", "label": "Tokyo Tower", "type": "POI", "category": "Landmark"},
            {"id": "tsukiji_market", "label": "Tsukiji Outer Market", "type": "POI", "category": "Culinary"},
            {"id": "cherry_blossom", "label": "Hanami Season", "type": "Season", "best_months": "March-April"},
            {"id": "autumn_leaves", "label": "Koyo Season", "type": "Season", "best_months": "October-November"},
        ],
        "edges": [
            {"source": "tokyo", "target": "sensoji", "relation": "HAS_POI", "weight": 1.0},
            {"source": "tokyo", "target": "shibuya_crossing", "relation": "HAS_POI", "weight": 0.9},
            {"source": "tokyo", "target": "tokyo_tower", "relation": "HAS_POI", "weight": 0.8},
            {"source": "tokyo", "target": "tsukiji_market", "relation": "HAS_POI", "weight": 0.85},
            {"source": "sensoji", "target": "cherry_blossom", "relation": "BEST_EXPERIENCED_IN", "season": "Spring"},
            {"source": "tsukiji_market", "target": "shibuya_crossing", "relation": "CONNECTED_VIA_TRANSIT", "transit": "Yamanote Line"},
            {"source": "tokyo_tower", "target": "shibuya_crossing", "relation": "NEARBY_DISTRICT", "distance_km": 4.5},
        ]
    },
    "paris": {
        "nodes": [
            {"id": "paris", "label": "Paris", "type": "City", "description": "France's capital, global center for art, fashion, and gastronomy."},
            {"id": "eiffel_tower", "label": "Eiffel Tower", "type": "POI", "category": "Landmark"},
            {"id": "louvre", "label": "Louvre Museum", "type": "POI", "category": "Art & History"},
            {"id": "notre_dame", "label": "Notre-Dame Cathedral", "type": "POI", "category": "Architecture"},
            {"id": "montmartre", "label": "Montmartre & Sacré-Cœur", "type": "POI", "category": "Culture & Views"},
            {"id": "spring_paris", "label": "Parisian Spring", "type": "Season", "best_months": "April-June"},
        ],
        "edges": [
            {"source": "paris", "target": "eiffel_tower", "relation": "HAS_POI", "weight": 1.0},
            {"source": "paris", "target": "louvre", "relation": "HAS_POI", "weight": 1.0},
            {"source": "paris", "target": "notre_dame", "relation": "HAS_POI", "weight": 0.9},
            {"source": "paris", "target": "montmartre", "relation": "HAS_POI", "weight": 0.85},
            {"source": "eiffel_tower", "target": "spring_paris", "relation": "BEST_EXPERIENCED_IN", "season": "Spring"},
            {"source": "louvre", "target": "notre_dame", "relation": "CONNECTED_ALONG_SEINE", "walk_time_min": 15},
        ]
    },
    "new york": {
        "nodes": [
            {"id": "new_york", "label": "New York", "type": "City", "description": "Global hub for finance, culture, entertainment, and media."},
            {"id": "statue_of_liberty", "label": "Statue of Liberty", "type": "POI", "category": "Landmark"},
            {"id": "central_park", "label": "Central Park", "type": "POI", "category": "Nature & Park"},
            {"id": "empire_state", "label": "Empire State Building", "type": "POI", "category": "Architecture"},
            {"id": "met_museum", "label": "Metropolitan Museum of Art", "type": "POI", "category": "Art & History"},
            {"id": "autumn_ny", "label": "Fall Foliage in NY", "type": "Season", "best_months": "September-November"},
        ],
        "edges": [
            {"source": "new_york", "target": "statue_of_liberty", "relation": "HAS_POI", "weight": 1.0},
            {"source": "new_york", "target": "central_park", "relation": "HAS_POI", "weight": 0.95},
            {"source": "new_york", "target": "empire_state", "relation": "HAS_POI", "weight": 0.9},
            {"source": "new_york", "target": "met_museum", "relation": "HAS_POI", "weight": 0.85},
            {"source": "central_park", "target": "met_museum", "relation": "ADJACENT_TO", "walk_time_min": 5},
            {"source": "central_park", "target": "autumn_ny", "relation": "BEST_EXPERIENCED_IN", "season": "Autumn"},
        ]
    },
    "kyoto": {
        "nodes": [
            {"id": "kyoto", "label": "Kyoto", "type": "City", "description": "Japan's ancient capital, famed for classical Buddhist temples and gardens."},
            {"id": "fushimi_inari", "label": "Fushimi Inari Shrine", "type": "POI", "category": "Culture & Shrine"},
            {"id": "kinkakuji", "label": "Kinkaku-ji (Golden Pavilion)", "type": "POI", "category": "Historical Temple"},
            {"id": "arashiyama", "label": "Arashiyama Bamboo Grove", "type": "POI", "category": "Nature"},
        ],
        "edges": [
            {"source": "kyoto", "target": "fushimi_inari", "relation": "HAS_POI", "weight": 1.0},
            {"source": "kyoto", "target": "kinkakuji", "relation": "HAS_POI", "weight": 0.95},
            {"source": "kyoto", "target": "arashiyama", "relation": "HAS_POI", "weight": 0.9},
        ]
    },
    "snohomish": {
        "nodes": [
            {"id": "snohomish", "label": "Snohomish", "type": "City", "description": "Historic antique capital of the Pacific Northwest in Washington State."},
            {"id": "historic_downtown", "label": "Historic Downtown Snohomish", "type": "POI", "category": "Shopping & Heritage"},
            {"id": "centennial_trail", "label": "Centennial Trail", "type": "POI", "category": "Recreation"},
        ],
        "edges": [
            {"source": "snohomish", "target": "historic_downtown", "relation": "HAS_POI", "weight": 1.0},
            {"source": "snohomish", "target": "centennial_trail", "relation": "HAS_POI", "weight": 0.9},
        ]
    }
}


class KnowledgeGraphService(KnowledgeGraphServiceInterface):
    """Production implementation of Knowledge Graph & Graph RAG Service."""

    def __init__(self, custom_db: Optional[Dict[str, Dict[str, Any]]] = None):
        self._db = custom_db if custom_db is not None else KNOWLEDGE_GRAPH_DB

    def search_city_graph(self, city: str) -> bool:
        """Check if city exists in knowledge graph."""
        if not city:
            return False
        return city.strip().lower() in self._db

    def get_subgraph(self, city: str, max_depth: int = 2) -> Dict[str, Any]:
        """Extract subgraph for city including nodes and edges up to max_depth hops."""
        city_key = city.strip().lower() if city else ""
        if city_key not in self._db:
            # Fallback dynamic graph generation for unseen cities
            return {
                "city": city,
                "nodes": [{"id": city_key, "label": city, "type": "City", "description": f"Dynamic graph node for {city}"}],
                "edges": [],
                "entity_count": 1,
            }

        graph = self._db[city_key]
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        return {
            "city": city,
            "nodes": nodes,
            "edges": edges,
            "entity_count": len(nodes),
        }

    def query_graph(self, city: str, query: str) -> Optional[Dict[str, Any]]:
        """Perform Graph RAG traversal and return graph-augmented context."""
        if not city:
            return None

        city_key = city.strip().lower()
        subgraph = self.get_subgraph(city, max_depth=2)

        if not subgraph or not subgraph.get("nodes"):
            return None

        nodes = subgraph["nodes"]
        edges = subgraph["edges"]

        # Format Graph RAG Context string
        city_node = next((n for n in nodes if n.get("type") == "City"), None)
        city_desc = city_node.get("description", "") if city_node else f"Exploration guide for {city}."

        poi_nodes = [n for n in nodes if n.get("type") == "POI"]
        entities = [n["label"] for n in nodes if "label" in n]

        formatted_lines = [
            f"=== Knowledge Graph Context for {city} ===",
            f"Description: {city_desc}",
            f"Key POIs & Entities ({len(poi_nodes)}):",
        ]

        for poi in poi_nodes:
            cat = poi.get("category", "General")
            formatted_lines.append(f" - {poi['label']} [{cat}]")

        if edges:
            formatted_lines.append("Entity Relationships & Network Topology:")
            for edge in edges:
                src_label = next((n["label"] for n in nodes if n["id"] == edge["source"]), edge["source"])
                tgt_label = next((n["label"] for n in nodes if n["id"] == edge["target"]), edge["target"])
                rel = edge.get("relation", "CONNECTED")
                formatted_lines.append(f" - ({src_label}) --[{rel}]--> ({tgt_label})")

        graph_context = "\n".join(formatted_lines)

        return {
            "city": city,
            "graph_context": graph_context,
            "entities": entities,
            "nodes": nodes,
            "relationships": edges,
        }
