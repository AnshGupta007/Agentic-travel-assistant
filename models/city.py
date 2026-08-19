"""City knowledge data model."""

from typing import Optional
from pydantic import BaseModel, Field


class CityKnowledge(BaseModel):
    """Represents knowledge retrieved for a specific city."""
    city: str = Field(..., description="Name of the city")
    country: str = Field(default="Unknown", description="Country where the city is located")
    description: str = Field(..., description="Overview and key details about the city")
    highlights: list[str] = Field(default_factory=list, description="Top attractions or landmarks")
    culture_tips: list[str] = Field(default_factory=list, description="Local customs and travel tips")
    source: str = Field(default="vector_store", description="Data provenance: 'vector_store' or 'web_search'")
