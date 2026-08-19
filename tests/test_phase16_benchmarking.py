"""Unit tests for RAG Benchmarking Framework evaluation and report generation."""

import pytest
from services.benchmarking import RAGBenchmarker
from models.graph_rag import BenchmarkReport, RAGBenchmarkMetric


def test_rag_benchmarker_single_evaluations():
    """Test individual RAG mode evaluation metrics."""
    benchmarker = RAGBenchmarker()

    m_vec: RAGBenchmarkMetric = benchmarker.evaluate_vector_rag("Tokyo")
    assert m_vec.mode == "Vector RAG"
    assert m_vec.latency_ms >= 0
    assert m_vec.precision_score > 0

    m_graph: RAGBenchmarkMetric = benchmarker.evaluate_graph_rag("Tokyo")
    assert m_graph.mode == "Graph RAG"
    assert m_graph.entity_count >= 4
    assert m_graph.precision_score > 0.9

    m_hybrid: RAGBenchmarkMetric = benchmarker.evaluate_hybrid_multimodal_rag("Tokyo")
    assert m_hybrid.mode == "Hybrid Multi-Modal RAG"
    assert m_hybrid.fact_count >= m_graph.fact_count


def test_rag_benchmark_run_report():
    """Test full comparative RAG benchmark execution."""
    benchmarker = RAGBenchmarker()
    report: BenchmarkReport = benchmarker.run_benchmark(city="Tokyo")

    assert report.query == "Tell me about Tokyo"
    assert len(report.metrics) == 3
    assert report.winner in ["Vector RAG", "Graph RAG", "Hybrid Multi-Modal RAG"]
    assert "recall" in report.summary.lower() or "precision" in report.summary.lower()
