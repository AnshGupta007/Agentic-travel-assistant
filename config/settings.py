"""Configuration settings for Multi-Modal Agentic Travel Assistant."""

import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ProviderMode(str, Enum):
    """Execution mode for external service providers."""
    MOCK = "MOCK"
    LIVE = "LIVE"


class Settings(BaseModel):
    """Application setting configuration abstraction."""
    provider_mode: ProviderMode = Field(
        default_factory=lambda: ProviderMode(os.getenv("PROVIDER_MODE", "MOCK").upper())
    )
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    vector_store_path: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    )
    openweather_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY"))
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    unsplash_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("UNSPLASH_API_KEY"))

    def is_mock(self) -> bool:
        """Helper method to check if the application is running in MOCK mode."""
        return self.provider_mode == ProviderMode.MOCK


settings = Settings()
