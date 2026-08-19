"""LangGraph Checkpointing and Memory Management module.

This module provides persistent memory and state management for multi-turn conversations
using LangGraph checkpointers (e.g. MemorySaver) and thread-scoped execution management.
"""

import logging
import uuid
from typing import Any, Dict, Optional
from langgraph.checkpoint.memory import MemorySaver
from models.state import TravelAgentState
from graph.workflow import create_travel_graph

logger = logging.getLogger(__name__)


def get_memory_checkpointer() -> MemorySaver:
    """Instantiate and return an in-memory LangGraph checkpointer.

    Returns:
        MemorySaver instance.
    """
    return MemorySaver()


class TravelAssistantSession:
    """Thread-scoped session manager for checkpointed travel graph execution."""

    def __init__(
        self,
        thread_id: Optional[str] = None,
        checkpointer: Optional[Any] = None,
        llm_client=None,
        vector_store=None,
        search_service=None,
        weather_service=None,
        image_service=None,
    ):
        """Initialize a persistent session.

        Args:
            thread_id: Unique thread identifier for conversation memory.
            checkpointer: Custom checkpointer (defaults to MemorySaver()).
            llm_client: Optional custom LLMClient instance.
            vector_store: Optional vector store instance.
            search_service: Optional web search service instance.
            weather_service: Optional weather service instance.
            image_service: Optional image service instance.
        """
        self.thread_id = thread_id or f"thread_{uuid.uuid4().hex[:8]}"
        self.checkpointer = checkpointer or get_memory_checkpointer()
        self.graph = create_travel_graph(
            llm_client=llm_client,
            vector_store=vector_store,
            search_service=search_service,
            weather_service=weather_service,
            image_service=image_service,
            checkpointer=self.checkpointer,
        )

    def invoke(
        self,
        query: str,
        state_overrides: Optional[Dict[str, Any]] = None,
        retrieval_mode: Optional[str] = None,
    ) -> TravelAgentState:
        """Invoke the graph for a conversation turn using persistent state.

        Args:
            query: User prompt/query string.
            state_overrides: Additional fields to override in input state.
            retrieval_mode: Optional RAG mode ('vector_store', 'graph_rag', 'web_search').

        Returns:
            Updated TravelAgentState after workflow execution.
        """
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Prepare input state payload
        payload: Dict[str, Any] = {"query": query}
        if retrieval_mode:
            payload["retrieval_mode"] = retrieval_mode
        if state_overrides:
            payload.update(state_overrides)


        logger.info(f"[TravelAssistantSession] Invoking session '{self.thread_id}' with query: '{query}'")
        output_state = self.graph.invoke(payload, config=config)
        return output_state

    def get_state(self) -> Optional[TravelAgentState]:
        """Retrieve current checkpoint state for this thread.

        Returns:
            TravelAgentState dict if available, else None.
        """
        config = {"configurable": {"thread_id": self.thread_id}}
        state_snapshot = self.graph.get_state(config)
        if state_snapshot and hasattr(state_snapshot, "values"):
            return state_snapshot.values
        return None
