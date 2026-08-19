"""Unit test suite for RAG Benchmark Suite (Phase 15)."""

import pytest
from services.benchmark import RAGBenchmarkSuite


def test_evaluate_retrieval_mode_vector():
    """Verify benchmark evaluation for vector_store mode."""
    suite = RAGBenchmarkSuite()
    res = suite.evaluate_retrieval_mode(city="Tokyo", retrieval_mode="vector_store")

    assert res["city"] == "Tokyo"
    assert res["retrieval_mode"] == "vector_store"
    assert res["latency_ms"] > 0.0
    assert res["entity_recall"] >= 0.0
    assert res["grounding_score"] > 0.0


def test_evaluate_retrieval_mode_graph():
    """Verify benchmark evaluation for graph_rag mode."""
    suite = RAGBenchmarkSuite()
    res = suite.evaluate_retrieval_mode(city="Paris", retrieval_mode="graph_rag")

    assert res["city"] == "Paris"
    assert res["retrieval_mode"] == "graph_rag"
    assert res["latency_ms"] > 0.0
    assert res["context_depth"] > 0
    assert res["grounding_score"] > 0.0


def test_run_comparative_benchmark():
    """Verify comparative benchmark across vector_store, graph_rag, and web_search."""
    suite = RAGBenchmarkSuite()
    report = suite.run_comparative_benchmark(city="Tokyo")

    assert report["city"] == "Tokyo"
    assert "vector_store" in report["comparative_results"]
    assert "graph_rag" in report["comparative_results"]
    assert "web_search" in report["comparative_results"]
    assert report["recommended_mode"] in ["vector_store", "graph_rag", "web_search"]
