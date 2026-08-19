"""Unit tests for Graph RAG Service, subgraphs, and triple traversal."""

import pytest
from services.graph_rag import GraphRAGService
from models.graph_rag import GraphSubgraph, GraphTriple


def test_graph_rag_subgraph_extraction():
    """Test extracting Knowledge Graph subgraph for Tokyo."""
    service = GraphRAGService()
    subgraph: GraphSubgraph = service.get_subgraph("Tokyo")

    assert subgraph.city == "Tokyo"
    assert len(subgraph.entities) >= 4
    entity_names = [e.name for e in subgraph.entities]
    assert "Tokyo Tower" in entity_names
    assert "Tonkotsu Ramen" in entity_names

    assert len(subgraph.triples) > 0
    fact_strings = [t.to_fact_string() for t in subgraph.triples]
    assert any("Tokyo Tower -> LOCATED_IN -> Tokyo" in f for f in fact_strings)


def test_graph_rag_paris_and_new_york():
    """Test extracting Knowledge Graph subgraphs for Paris and New York."""
    service = GraphRAGService()

    paris_subgraph = service.get_subgraph("Paris")
    assert any(e.name == "Eiffel Tower" for e in paris_subgraph.entities)

    ny_subgraph = service.get_subgraph("New York")
    assert any(e.name == "Statue of Liberty" for e in ny_subgraph.entities)


def test_hybrid_retrieval():
    """Test hybrid retrieval combining Vector Text and Graph RAG triples."""
    service = GraphRAGService()
    hybrid_res = service.hybrid_retrieval("Tokyo")

    assert hybrid_res["city"] == "Tokyo"
    assert "Knowledge Graph Triples" in hybrid_res["summary"] or "Overview of Tokyo" in hybrid_res["summary"]
    assert len(hybrid_res["graph_triples"]) > 0
    assert hybrid_res["retrieval_mode"] in ["hybrid_rag", "graph_rag"]
