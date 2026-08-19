# Starlette compatibility patch for Streamlit
try:
    import starlette.middleware.gzip
    if not hasattr(starlette.middleware.gzip, "DEFAULT_EXCLUDED_CONTENT_TYPES"):
        starlette.middleware.gzip.DEFAULT_EXCLUDED_CONTENT_TYPES = (
            "text/html", "text/css", "text/plain", "application/javascript", "application/json"
        )
except ImportError:
    pass

import logging
import uuid
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_assistant_ui")

# Import UI components and application session layer
from config.settings import settings
from graph.checkpoint import TravelAssistantSession
from models.travel import TravelResponse
from models.weather import WeatherDataPoint
from ui.components import (
    render_header,
    render_sidebar,
    render_llm_error,
    render_benchmark_dashboard,
    render_graph_triples,
)
from ui.rendering import render_chat_history, render_travel_response
from services.benchmarking import RAGBenchmarker
from services.graph_rag import GraphRAGService

# Page layout configuration
st.set_page_config(
    page_title="Multi-Modal Agentic Travel Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_or_create_session() -> TravelAssistantSession:
    """Retrieve existing session from st.session_state or initialize a new one."""
    if "thread_id" not in st.session_state or not st.session_state.thread_id:
        st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"

    if "session" not in st.session_state or st.session_state.session is None:
        logger.info(f"Initializing TravelAssistantSession for thread: {st.session_state.thread_id}")
        st.session_state.session = TravelAssistantSession(
            thread_id=st.session_state.thread_id
        )

    return st.session_state.session


def clear_session() -> None:
    """Reset session thread ID and chat history for a new conversation thread."""
    st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    st.session_state.session = TravelAssistantSession(
        thread_id=st.session_state.thread_id
    )
    st.session_state.chat_history = []
    logger.info(f"Session reset to new thread: {st.session_state.thread_id}")


def main() -> None:
    """Main application loop for Streamlit UI."""
    # 1. Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    session = get_or_create_session()

    # 2. Render Header and Sidebar Controls
    render_header()
    render_sidebar(on_clear_thread=clear_session)

    tab1, tab2 = st.tabs(["💬 Agent Assistant", "📊 Graph RAG & Benchmark Explorer"])

    with tab1:
        # Display Past Chat History
        render_chat_history(st.session_state.chat_history)

        # Chat Input Handling
        user_prompt = st.chat_input("Ask about a city or enter a follow-up question (e.g. 'Tell me about Tokyo', 'What about next week?')...")

        if user_prompt:
            # Display user query in UI
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Invoke Application Session Layer
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing query with LangGraph workflow..."):
                    try:
                        output_state = session.invoke(user_prompt)
                        
                        # Extract TravelResponse structured object
                        final_response: TravelResponse = output_state.get("final_response")
                        city = output_state.get("city")

                        # Fallback construction if final_response was not populated
                        if not final_response:
                            city_summary = output_state.get("city_summary", "No summary generated.")
                            weather_forecast = output_state.get("weather_forecast") or []
                            image_urls = output_state.get("image_urls") or []
                            
                            final_response = TravelResponse(
                                city_summary=city_summary,
                                weather_forecast=weather_forecast,
                                image_urls=image_urls,
                                weather_error=output_state.get("weather_error"),
                                image_error=output_state.get("image_error"),
                                search_error=output_state.get("search_error"),
                                graph_triples=output_state.get("graph_triples") or [],
                                retrieval_mode=output_state.get("retrieval_mode") or "vector_store",
                            )

                        # Render structured TravelResponse
                        render_travel_response(final_response, city=city)

                        # Store in chat history for persistence across reruns
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "response": final_response,
                            "city": city,
                        })

                    except Exception as e:
                        logger.error(f"Error during graph execution: {e}", exc_info=True)
                        error_msg = f"An unexpected error occurred during execution: {str(e)}"
                        render_llm_error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "is_error": True,
                        })

    with tab2:
        st.header("🕸️ Graph RAG & Benchmark Explorer")
        st.caption("Perform Knowledge Graph traversal and run RAG evaluation benchmarks across Vector RAG, Graph RAG, and Hybrid Multi-Modal RAG.")

        city_choice = st.selectbox("Select Target Destination City", ["Tokyo", "Paris", "New York", "Kyoto", "Snohomish"])
        
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            if st.button("🕸️ Explore Knowledge Graph Subgraph", use_container_width=True):
                graph_svc = GraphRAGService()
                subgraph = graph_svc.get_subgraph(city_choice)
                st.write(f"### Subgraph for `{city_choice}`")
                st.write(f"**Entities ({len(subgraph.entities)}):**")
                for e in subgraph.entities:
                    st.markdown(f"- **{e.name}** (`{e.entity_type}`) - {e.attributes}")
                triples_str = [t.to_fact_string() for t in subgraph.triples]
                render_graph_triples(triples_str)

        with col_g2:
            if st.button("⚡ Run Comparative RAG Benchmark", use_container_width=True):
                benchmarker = RAGBenchmarker()
                report = benchmarker.run_benchmark(city=city_choice)
                st.session_state["last_benchmark"] = report.model_dump()

        if "last_benchmark" in st.session_state:
            render_benchmark_dashboard(st.session_state["last_benchmark"])


if __name__ == "__main__":
    main()

