"""Structured output models and parsing utilities for LLM layer."""

import json
import re
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class CityExtractionResult(BaseModel):
    """Schema for LLM city extraction response."""
    city: Optional[str] = Field(default=None, description="Extracted city name")
    is_follow_up: bool = Field(default=False, description="Whether query relies on previous context")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")


class QueryClassificationResult(BaseModel):
    """Schema for LLM query classification response."""
    intent: str = Field(default="general_travel", description="Detected query intent")
    target_city: Optional[str] = Field(default=None, description="Target city if detected")
    needs_search: bool = Field(default=True, description="Whether vector or web search is needed")
    needs_weather: bool = Field(default=True, description="Whether weather data is needed")
    needs_images: bool = Field(default=True, description="Whether city images are needed")


class ToolCallSpec(BaseModel):
    """Schema for LLM-generated tool call specification."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")


def clean_json_markdown(text: str) -> str:
    """Remove markdown code blocks (e.g. ```json ... ```) from completion string."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def parse_json_completion(text: str, model_cls: Type[T]) -> T:
    """Parse a raw LLM text completion into a validated Pydantic model.
    
    Args:
        text: Raw text string from LLM output.
        model_cls: Target Pydantic model class.
        
    Returns:
        Validated instance of model_cls.
        
    Raises:
        ValueError: If JSON is invalid or fails Pydantic validation.
    """
    cleaned = clean_json_markdown(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt trailing comma repair
        repaired = re.sub(r",\s*([\}\]])", r"\1", cleaned)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM completion as JSON: {e}\nRaw output: {text}") from e

    try:
        return model_cls.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to validate LLM JSON output against {model_cls.__name__}: {e}\nData: {data}") from e

