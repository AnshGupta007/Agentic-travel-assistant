"""Manual tool execution module for LangGraph travel assistant workflow.

This module satisfies Phase 8 of the Multi-Modal Agentic Travel Assistant challenge by
providing explicit manual parsing, routing, resolution, execution, error capturing,
and ToolMessage creation for LLM tool calls.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from langchain_core.messages import ToolMessage
except ImportError:
    # Fallback ToolMessage class if langchain_core is not present
    class ToolMessage:  # type: ignore
        def __init__(self, content: str, tool_call_id: str, name: str = ""):
            self.content = content
            self.tool_call_id = tool_call_id
            self.name = name

        def to_dict(self) -> dict:
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "name": self.name,
                "content": self.content,
            }


from interfaces.images import ImageServiceInterface
from interfaces.search import SearchServiceInterface
from interfaces.weather import WeatherServiceInterface
from interfaces.vector_store import VectorStoreServiceInterface
from llm.client import LLMClient
from llm.structured import ToolCallSpec
from models.city import CityKnowledge
from models.state import TravelAgentState
from models.weather import WeatherResult
from providers.factory import (
    get_image_service,
    get_search_service,
    get_weather_service,
)
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Container holding the execution outcome of a single manual tool call."""

    tool_name: str
    tool_call_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None

    def to_tool_message(self) -> ToolMessage:
        """Convert result to a structured ToolMessage instance."""
        if self.success:
            if isinstance(self.result, (dict, list)):
                content_str = json.dumps(self.result, default=str)
            elif hasattr(self.result, "model_dump_json"):
                content_str = self.result.model_dump_json()
            else:
                content_str = str(self.result)
        else:
            content_str = json.dumps({"error": self.error or "Unknown tool execution error"})

        return ToolMessage(
            content=content_str,
            tool_call_id=self.tool_call_id,
            name=self.tool_name,
        )


class ToolExecutor:
    """Manual Tool Executor responsible for resolving, validating, and executing tools."""

    def __init__(
        self,
        weather_service: Optional[WeatherServiceInterface] = None,
        image_service: Optional[ImageServiceInterface] = None,
        search_service: Optional[SearchServiceInterface] = None,
        vector_store: Optional[VectorStoreServiceInterface] = None,
    ):
        """Initialize ToolExecutor with service providers.

        Args:
            weather_service: Weather service implementation.
            image_service: Image service implementation.
            search_service: Web search service implementation.
            vector_store: Vector store service implementation.
        """
        self.weather_service = weather_service or get_weather_service()
        self.image_service = image_service or get_image_service()
        self.search_service = search_service or get_search_service()
        self.vector_store = vector_store or VectorStoreService()

        # Tool registry mapping tool names to execution handlers
        self._registry: Dict[str, Callable[..., Any]] = {
            "get_weather": self._exec_weather,
            "search_images": self._exec_images,
            "search_city": self._exec_search_city,
            "vector_search": self._exec_vector_search,
        }

    def register_tool(self, tool_name: str, handler: Callable[..., Any]) -> None:
        """Register a custom tool handler function."""
        self._registry[tool_name] = handler
        logger.info(f"Registered tool handler: '{tool_name}'")

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> ToolResult:
        """Manually parse, resolve, and execute a tool call by name.

        Args:
            tool_name: Name of tool to execute.
            arguments: Dictionary of arguments for the tool.
            tool_call_id: Unique tool call identifier.

        Returns:
            ToolResult containing execution status, result output, or error message.
        """
        call_id = tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
        logger.info(f"[ToolExecutor] Resolving tool '{tool_name}' (call_id: {call_id}) with args: {arguments}")

        if tool_name not in self._registry:
            error_msg = f"Tool '{tool_name}' is not registered."
            logger.error(f"[ToolExecutor] {error_msg}")
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=call_id,
                success=False,
                error=error_msg,
            )

        handler = self._registry[tool_name]
        try:
            res = handler(**arguments)
            logger.info(f"[ToolExecutor] Tool '{tool_name}' executed successfully.")
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=call_id,
                success=True,
                result=res,
            )
        except Exception as e:
            error_msg = f"Exception executing tool '{tool_name}': {e}"
            logger.error(f"[ToolExecutor] {error_msg}", exc_info=True)
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=call_id,
                success=False,
                error=error_msg,
            )

    def execute_tool_calls(
        self, tool_calls: List[Union[Dict[str, Any], ToolCallSpec]]
    ) -> List[ToolResult]:
        """Batch execute multiple tool call specifications.

        Args:
            tool_calls: List of raw tool call dicts or ToolCallSpec instances.

        Returns:
            List of ToolResult objects.
        """
        results: List[ToolResult] = []
        for call in tool_calls:
            if isinstance(call, ToolCallSpec):
                name = call.tool_name
                args = call.arguments
                call_id = f"call_{uuid.uuid4().hex[:8]}"
            elif isinstance(call, dict):
                name = call.get("tool_name") or call.get("name") or ""
                args = call.get("arguments") or call.get("args") or {}
                call_id = call.get("tool_call_id") or call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            else:
                logger.warning(f"Unsupported tool call format: {call}")
                continue

            res = self.execute_tool(tool_name=name, arguments=args, tool_call_id=call_id)
            results.append(res)

        return results

    # --- Tool Execution Handlers ---

    def _exec_weather(self, city: str, days: int = 7) -> WeatherResult:
        """Handler for 'get_weather' tool."""
        return self.weather_service.get_weather(city=city, days=days)

    def _exec_images(self, city: str, limit: int = 5) -> List[str]:
        """Handler for 'search_images' tool."""
        return self.image_service.search_images(city=city, limit=limit)

    def _exec_search_city(self, city: str) -> Optional[CityKnowledge]:
        """Handler for 'search_city' (web search) tool."""
        return self.search_service.search_city(city=city)

    def _exec_vector_search(self, city: str) -> Optional[CityKnowledge]:
        """Handler for 'vector_search' (vector store lookup) tool."""
        return self.vector_store.search_city(city=city)


def _format_city_knowledge(knowledge: CityKnowledge) -> str:
    """Format CityKnowledge model into readable summary string."""
    highlights_str = ", ".join(knowledge.highlights) if knowledge.highlights else "N/A"
    tips_str = " ".join(knowledge.culture_tips) if knowledge.culture_tips else ""
    return f"{knowledge.description}\n\nKey Highlights: {highlights_str}.\nLocal Culture & Travel Tips: {tips_str}"


def manual_tool_execution_node(
    state: TravelAgentState,
    tool_executor: Optional[ToolExecutor] = None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """LangGraph node function executing tool calls manually and updating state.

    Args:
        state: Current TravelAgentState.
        tool_executor: Optional custom ToolExecutor instance.
        llm_client: Optional custom LLMClient instance.

    Returns:
        State update dictionary containing populated tool results, errors, and messages.
    """
    executor = tool_executor or ToolExecutor()
    client = llm_client or LLMClient()

    query = state.get("query", "")
    city = state.get("city")
    tool_calls_input = state.get("tool_calls")

    # Step 1: Read raw LLM tool calls from state or generate via LLMClient fallback
    if not tool_calls_input:
        logger.info("[manual_tool_execution_node] No pre-populated tool_calls found. Generating tool call specs...")
        tool_call_specs = client.generate_tool_calls(query=query, city=city)
    else:
        tool_call_specs = tool_calls_input

    # Step 2 & 3 & 4 & 5 & 6: Parse, resolve, execute tool calls and capture errors
    tool_results = executor.execute_tool_calls(tool_call_specs)

    # Step 7 & 8: Create ToolMessages and prepare state updates
    new_messages: List[Dict[str, Any]] = list(state.get("messages") or [])
    state_updates: Dict[str, Any] = {}

    for tr in tool_results:
        # Create ToolMessage and append
        tool_msg = tr.to_tool_message()
        if hasattr(tool_msg, "to_dict"):
            new_messages.append(tool_msg.to_dict())
        else:
            new_messages.append({
                "role": "tool",
                "tool_call_id": getattr(tool_msg, "tool_call_id", ""),
                "name": getattr(tool_msg, "name", tr.tool_name),
                "content": getattr(tool_msg, "content", ""),
            })

        # Process and map results into state fields
        if tr.tool_name == "get_weather":
            if tr.success and isinstance(tr.result, WeatherResult):
                if tr.result.error:
                    state_updates["weather_error"] = tr.result.error
                    state_updates["weather_forecast"] = []
                else:
                    state_updates["weather_forecast"] = tr.result.forecast
                    state_updates["weather_error"] = None
            else:
                state_updates["weather_error"] = tr.error or "Failed to retrieve weather"
                state_updates["weather_forecast"] = []

        elif tr.tool_name == "search_images":
            if tr.success and isinstance(tr.result, list):
                state_updates["image_urls"] = tr.result
                state_updates["image_error"] = None
            else:
                state_updates["image_error"] = tr.error or "Failed to retrieve images"
                state_updates["image_urls"] = []

        elif tr.tool_name in ("search_city", "vector_search"):
            if tr.success and isinstance(tr.result, CityKnowledge):
                state_updates["city_summary"] = _format_city_knowledge(tr.result)
                state_updates["search_error"] = None
                state_updates["routed_to"] = "vector_store" if tr.tool_name == "vector_search" else "web_search"
            elif tr.success and tr.result is None:
                state_updates["search_error"] = f"No city knowledge found for '{city or query}'."
            else:
                state_updates["search_error"] = tr.error or "Failed to retrieve city knowledge"

    state_updates["messages"] = new_messages
    return state_updates
