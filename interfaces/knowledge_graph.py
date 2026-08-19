"""Knowledge graph service interface contract for Graph RAG."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from models.city import CityKnowledge


class KnowledgeGraphServiceInterface(ABC):
    """Abstract base class for Knowledge Graph & Graph RAG retrieval operations."""

    @abstractmethod
    def query_graph(self, city: str, query: str) -> Optional[Dict[str, Any]]:
        """Query knowledge graph for entity sub-graphs, relations, and augmented RAG context.

        Args:
            city: Destination city name.
            query: Natural language user query.

        Returns:
            Dictionary containing:
            - 'city': City name
            - 'graph_context': Formatted text representation of nodes and relationships
            - 'entities': List of entity strings
            - 'nodes': List of node dicts with metadata
            - 'relationships': List of edge connection dicts
            or None if city/entities not present in graph.
        """
        pass

    @abstractmethod
    def get_subgraph(self, city: str, max_depth: int = 2) -> Dict[str, Any]:
        """Extract entity subgraph for a given city up to max_depth traversal hops.

        Args:
            city: Destination city name.
            max_depth: Maximum edge traversal depth.

        Returns:
            Subgraph dict containing nodes, edges, and entity counts.
        """
        pass
