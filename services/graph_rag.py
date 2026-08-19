"""Knowledge Graph & Graph RAG Service implementation.

Provides entity extraction, relationship triple traversal, and hybrid retrieval
combining Vector RAG semantic search, Graph RAG triple reasoning, and Multi-Modal context.
"""

import logging
from typing import Optional, Any
from models.graph_rag import GraphEntity, GraphRelation, GraphTriple, GraphSubgraph

logger = logging.getLogger(__name__)


class GraphRAGService:
    """Graph RAG Service managing Knowledge Graph entities, triples, and graph retrieval."""

    def __init__(self):
        self._entities: dict[str, GraphEntity] = {}
        self._relations: list[GraphRelation] = []
        self._seed_knowledge_graph()

    def _seed_knowledge_graph(self):
        """Seed initial multi-hop Knowledge Graph for Paris, Tokyo, New York, Kyoto, and Snohomish."""
        # Tokyo Graph Data
        self._add_entity("tokyo", "Tokyo", "City", "Tokyo", {"country": "Japan", "population": "14M"})
        self._add_entity("tokyo_tower", "Tokyo Tower", "Attraction", "Tokyo", {"type": "Observation Tower", "height": "332.9m"})
        self._add_entity("sensoji", "Senso-ji Temple", "Attraction", "Tokyo", {"type": "Buddhist Temple", "district": "Asakusa"})
        self._add_entity("shibuya_crossing", "Shibuya Crossing", "Attraction", "Tokyo", {"type": "Landmark", "vibe": "Energetic"})
        self._add_entity("ramen", "Tonkotsu Ramen", "Cuisine", "Tokyo", {"flavor": "Savory Pork Broth"})
        self._add_entity("shinkansen", "Shinkansen Bullet Train", "Transport", "Tokyo", {"speed": "320 km/h"})

        self._add_relation("LOCATED_IN", "tokyo_tower", "tokyo")
        self._add_relation("LOCATED_IN", "sensoji", "tokyo")
        self._add_relation("LOCATED_IN", "shibuya_crossing", "tokyo")
        self._add_relation("FAMOUS_FOR", "tokyo", "ramen")
        self._add_relation("CONNECTED_BY", "tokyo", "shinkansen")

        # Paris Graph Data
        self._add_entity("paris", "Paris", "City", "Paris", {"country": "France", "nickname": "City of Light"})
        self._add_entity("eiffel_tower", "Eiffel Tower", "Attraction", "Paris", {"type": "Monument", "height": "330m"})
        self._add_entity("louvre", "Louvre Museum", "Attraction", "Paris", {"type": "Art Museum", "famous_art": "Mona Lisa"})
        self._add_entity("croissant", "Butter Croissant", "Cuisine", "Paris", {"category": "Pastry"})
        self._add_entity("metro", "Paris Métro", "Transport", "Paris", {"lines": 16})

        self._add_relation("LOCATED_IN", "eiffel_tower", "paris")
        self._add_relation("LOCATED_IN", "louvre", "paris")
        self._add_relation("FAMOUS_FOR", "paris", "croissant")
        self._add_relation("SERVICED_BY", "paris", "metro")

        # New York Graph Data
        self._add_entity("new_york", "New York", "City", "New York", {"country": "USA", "nickname": "The Big Apple"})
        self._add_entity("statue_liberty", "Statue of Liberty", "Attraction", "New York", {"type": "National Monument"})
        self._add_entity("central_park", "Central Park", "Attraction", "New York", {"type": "Urban Park"})
        self._add_entity("ny_pizza", "New York-Style Pizza", "Cuisine", "New York", {"crust": "Thin"})

        self._add_relation("LOCATED_IN", "statue_liberty", "new_york")
        self._add_relation("LOCATED_IN", "central_park", "new_york")
        self._add_relation("FAMOUS_FOR", "new_york", "ny_pizza")

        # Kyoto Graph Data
        self._add_entity("kyoto", "Kyoto", "City", "Kyoto", {"country": "Japan", "historic": True})
        self._add_entity("fushimi_inari", "Fushimi Inari Shrine", "Attraction", "Kyoto", {"type": "Shinto Shrine", "gates": "10000 Torii"})
        self._add_entity("matcha", "Uji Matcha", "Cuisine", "Kyoto", {"category": "Green Tea"})

        self._add_relation("LOCATED_IN", "fushimi_inari", "kyoto")
        self._add_relation("FAMOUS_FOR", "kyoto", "matcha")

        # Snohomish Graph Data
        self._add_entity("snohomish", "Snohomish", "City", "Snohomish", {"country": "USA", "state": "Washington", "vibe": "Historic Antique Capital"})
        self._add_entity("historic_downtown", "Historic Downtown Snohomish", "Attraction", "Snohomish", {"type": "District"})

        self._add_relation("LOCATED_IN", "historic_downtown", "snohomish")

    def _add_entity(self, entity_id: str, name: str, entity_type: str, city: str, attributes: dict):
        self._entities[entity_id] = GraphEntity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            city=city,
            attributes=attributes,
        )

    def _add_relation(self, relation_type: str, source_id: str, target_id: str):
        self._relations.append(GraphRelation(
            relation_type=relation_type,
            source_id=source_id,
            target_id=target_id,
        ))

    def get_subgraph(self, city: str) -> GraphSubgraph:
        """Extract entity nodes and relations for a given city context."""
        normalized_city = city.strip().lower()
        city_entities = [
            entity for entity in self._entities.values()
            if entity.city.lower() == normalized_city or entity.name.lower() == normalized_city
        ]
        entity_ids = {e.id for e in city_entities}

        city_relations = [
            rel for rel in self._relations
            if rel.source_id in entity_ids or rel.target_id in entity_ids
        ]

        triples = []
        for rel in city_relations:
            src = self._entities.get(rel.source_id)
            tgt = self._entities.get(rel.target_id)
            if src and tgt:
                triples.append(GraphTriple(
                    subject=src.name,
                    predicate=rel.relation_type,
                    object=tgt.name,
                ))

        # Add entity attribute triples
        for entity in city_entities:
            for k, v in entity.attributes.items():
                triples.append(GraphTriple(
                    subject=entity.name,
                    predicate=f"HAS_{k.upper()}",
                    object=str(v),
                ))

        return GraphSubgraph(
            city=city,
            entities=city_entities,
            relations=city_relations,
            triples=triples,
        )

    def search_graph_triples(self, city: str, query: Optional[str] = None) -> list[GraphTriple]:
        """Search and format triples into structured facts for graph RAG reasoning."""
        subgraph = self.get_subgraph(city)
        if not query:
            return subgraph.triples

        query_terms = query.lower().split()
        matched_triples = []
        for triple in subgraph.triples:
            fact = triple.to_fact_string().lower()
            if any(term in fact for term in query_terms):
                matched_triples.append(triple)

        return matched_triples if matched_triples else subgraph.triples

    def hybrid_retrieval(self, city: str, vector_store=None, query: str = "") -> dict[str, Any]:
        """Perform hybrid retrieval combining Vector Text RAG and Knowledge Graph RAG."""
        # 1. Graph RAG Subgraph extraction
        subgraph = self.get_subgraph(city)
        triples = [t.to_fact_string() for t in subgraph.triples]

        # 2. Vector Text RAG search if available
        vector_text = None
        if vector_store:
            vector_text = vector_store.search_city(city)

        # Construct combined summary
        graph_facts_summary = "; ".join(triples[:6]) if triples else f"No Knowledge Graph entries for {city}."
        if vector_text:
            combined_summary = f"{vector_text} [Knowledge Graph Triples: {graph_facts_summary}]"
        else:
            combined_summary = f"Overview of {city}: {graph_facts_summary}"

        return {
            "city": city,
            "summary": combined_summary,
            "vector_text": vector_text,
            "graph_triples": triples,
            "subgraph": subgraph,
            "retrieval_mode": "hybrid_rag" if vector_text else "graph_rag",
        }
