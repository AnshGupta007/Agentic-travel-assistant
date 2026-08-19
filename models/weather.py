"""Weather data models and contracts."""

from typing import Optional
from pydantic import BaseModel, Field


class WeatherDataPoint(BaseModel):
    """Represents weather metrics for a single date/day."""
    date: str = Field(..., description="Date string (e.g. '2026-08-20' or 'Monday')")
    temperature: float = Field(..., description="Temperature value in Celsius or Fahrenheit")
    condition: str = Field(..., description="Short text summary of weather (e.g. 'Sunny', 'Rain')")


class WeatherForecastRequest(BaseModel):
    """Parameters requested for weather information."""
    city: str = Field(..., description="Target city name")
    days: int = Field(default=7, ge=1, le=14, description="Number of forecast days (5-7 recommended)")


class WeatherResult(BaseModel):
    """Data object returned by weather providers."""
    city: str = Field(..., description="City name for which weather was retrieved")
    forecast: list[WeatherDataPoint] = Field(default_factory=list, description="List of forecast data points")
    current_temp: Optional[float] = Field(default=None, description="Optional current temperature")
    condition: Optional[str] = Field(default=None, description="Optional current condition summary")
    error: Optional[str] = Field(default=None, description="Error message if weather retrieval failed")
