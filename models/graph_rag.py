"""Pydantic data models for Graph RAG and RAG Benchmarking Framework."""

from typing import Optional, Any
from pydantic import BaseModel, Field


class GraphEntity(BaseModel):
    """Entity node in the Travel Knowledge Graph."""
    id: str = Field(..., description="Unique identifier for the entity (e.g. 'tokyo_tower')")
    name: str = Field(..., description="Human-readable name of the entity")
    entity_type: str = Field(..., description="Category/Type of entity (e.g. 'Attraction', 'Cuisine', 'Transport', 'City')")
    city: str = Field(..., description="Associated city for the entity")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Metadata key-value properties")


class GraphRelation(BaseModel):
    """Directed edge relationship between two entities."""
    relation_type: str = Field(..., description="Relationship label (e.g. 'LOCATED_IN', 'FAMOUS_FOR', 'HAS_WEATHER_PROFILE')")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")


class GraphTriple(BaseModel):
    """Subject-Predicate-Object triple representation."""
    subject: str = Field(..., description="Subject entity name")
    predicate: str = Field(..., description="Relationship predicate")
    object: str = Field(..., description="Object entity name or attribute")

    def to_fact_string(self) -> str:
        """Format triple into a natural language fact string."""
        return f"{self.subject} -> {self.predicate} -> {self.object}"


class GraphSubgraph(BaseModel):
    """Extracted subgraph centered around a query/city context."""
    city: str = Field(..., description="Target city")
    entities: list[GraphEntity] = Field(default_factory=list, description="Extracted entities")
    relations: list[GraphRelation] = Field(default_factory=list, description="Extracted relations")
    triples: list[GraphTriple] = Field(default_factory=list, description="Flattened triple facts")


class RAGBenchmarkMetric(BaseModel):
    """Metric evaluation result for a single RAG retrieval execution."""
    mode: str = Field(..., description="RAG Mode evaluated ('Vector RAG', 'Graph RAG', 'Hybrid Multi-Modal RAG')")
    latency_ms: float = Field(..., description="Retrieval latency in milliseconds")
    entity_count: int = Field(default=0, description="Number of knowledge entities retrieved")
    fact_count: int = Field(default=0, description="Number of fact triples/units retrieved")
    precision_score: float = Field(..., description="Retrieval precision score (0.0 to 1.0)")
    recall_score: float = Field(..., description="Retrieval recall score (0.0 to 1.0)")
    structured_valid: bool = Field(default=True, description="Whether output satisfied structured output contract")


class BenchmarkReport(BaseModel):
    """Overall benchmark comparative report across paradigms."""
    query: str = Field(..., description="Evaluation benchmark query")
    metrics: list[RAGBenchmarkMetric] = Field(default_factory=list, description="Metrics per RAG mode")
    winner: str = Field(..., description="Best performing RAG mode for the query")
    summary: str = Field(..., description="Comparative analysis summary")
