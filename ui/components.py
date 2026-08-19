from typing import Callable, Optional
import pandas as pd
import streamlit as st
from config.settings import settings
from models.travel import TravelResponse
from models.weather import WeatherDataPoint


def render_header() -> None:
    """Render main title header and welcome tagline."""
    st.title("🌍 Multi-Modal Agentic Travel Assistant")
    st.caption(
        "Powered by LangGraph, Streamlit, and Multi-Modal AI. "
        "Explore cities with intelligent vector knowledge, real-time web fallback, weather forecasts, and visual galleries."
    )
    st.divider()


def render_sidebar(on_clear_thread: Optional[Callable[[], None]] = None) -> None:
    """Render sidebar control panel containing status and thread management.

    Args:
        on_clear_thread: Optional callback function triggered when starting a new thread.
    """
    with st.sidebar:
        st.header("⚙️ System Status")
        
        mode_color = "green" if settings.is_mock() else "orange"
        st.markdown(
            f"**Provider Mode:** `{settings.provider_mode.value}`"
        )
        
        st.markdown(f"**LLM Model:** `{settings.llm_model}` ({settings.llm_provider})")
        st.markdown(f"**Vector Store:** `{settings.vector_store_path}`")
        
        st.divider()
        st.header("💬 Conversation Session")
        
        thread_id = st.session_state.get("thread_id", "Not Initialized")
        st.code(thread_id, language="text")
        
        if st.button("➕ New Conversation Thread", use_container_width=True):
            if on_clear_thread:
                on_clear_thread()
            st.rerun()

        st.divider()
        st.markdown("### 💡 Quick Prompts")
        st.markdown("- *Tell me about Tokyo.*")
        st.markdown("- *What is the weather in Paris for next week?*")
        st.markdown("- *Show me sights in Snohomish.*")


def render_error_states(response: TravelResponse) -> None:
    """Render warning alerts if any partial errors occurred during service retrieval.

    Args:
        response: TravelResponse structured output containing data and potential errors.
    """
    if response.search_error:
        st.warning(f"⚠️ **Knowledge Search Notice:** {response.search_error}")
    if response.weather_error:
        st.warning(f"⚠️ **Weather Service Notice:** {response.weather_error}")
    if response.image_error:
        st.warning(f"⚠️ **Image Service Notice:** {response.image_error}")


def render_city_summary(
    summary: str,
    city: Optional[str] = None,
    search_error: Optional[str] = None,
) -> None:
    """Render formatted city knowledge summary or search error state.

    Args:
        summary: Text summary of the city.
        city: Optional city name header.
        search_error: Optional error message from search/vector service.
    """
    st.subheader(f"📖 Overview{f' - {city}' if city else ''}")
    
    if search_error:
        st.error(f"⚠️ **Knowledge Search Unavailable:** {search_error}")
    
    if summary:
        st.markdown(summary)
    elif not search_error:
        st.info("No detailed city summary available.")


def render_weather_chart(
    forecast: list[WeatherDataPoint],
    weather_error: Optional[str] = None,
) -> None:
    """Render interactive weather forecast visualization, metrics, or error state.

    Args:
        forecast: List of WeatherDataPoint objects.
        weather_error: Optional weather service error message.
    """
    st.subheader("🌤️ 5-7 Day Weather Forecast")

    if weather_error:
        st.error(f"⚠️ **Weather Service Error:** {weather_error}")
        st.info("Unable to retrieve weather forecast for this destination.")
        return
    
    if not forecast:
        st.info("Weather forecast data is currently unavailable.")
        return

    # Convert forecast list to Pandas DataFrame for visualization
    df_data = [
        {
            "Date": dp.date,
            "Temperature (°C)": float(dp.temperature),
            "Condition": dp.condition,
        }
        for dp in forecast
    ]
    df = pd.DataFrame(df_data)

    # Key Weather Metric Summary Cards
    col1, col2, col3 = st.columns(3)
    avg_temp = df["Temperature (°C)"].mean()
    min_temp = df["Temperature (°C)"].min()
    max_temp = df["Temperature (°C)"].max()
    
    with col1:
        st.metric("Avg Temp", f"{avg_temp:.1f} °C")
    with col2:
        st.metric("Min Temp", f"{min_temp:.1f} °C")
    with col3:
        st.metric("Max Temp", f"{max_temp:.1f} °C")

    # Line Chart of Temperatures across Forecast Period
    st.markdown("#### Temperature Trend")
    st.line_chart(df.set_index("Date")["Temperature (°C)"])

    # Detailed Daily Breakdown
    st.markdown("#### Forecast Details")
    cols = st.columns(min(len(forecast), 7))
    for idx, dp in enumerate(forecast[:7]):
        with cols[idx]:
            st.markdown(f"**{dp.date}**")
            st.markdown(f"🌡️ {dp.temperature:.1f} °C")
            st.caption(f"{dp.condition}")


def render_image_gallery(
    urls: list[str],
    image_error: Optional[str] = None,
) -> None:
    """Render responsive image gallery grid or image service error state.

    Args:
        urls: List of image URL strings.
        image_error: Optional image service error message.
    """
    st.subheader("📸 Destination Gallery")

    if image_error:
        st.error(f"⚠️ **Image Service Error:** {image_error}")
        st.info("Unable to load destination gallery images.")
        return
    
    if not urls:
        st.info("No destination images available.")
        return

    # Render gallery in a 3-column grid layout
    cols_per_row = 3
    for i in range(0, len(urls), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            img_idx = i + j
            if img_idx < len(urls):
                url = urls[img_idx]
                with col:
                    st.image(
                        url,
                        caption=f"Image {img_idx + 1}",
                        use_container_width=True,
                    )


def render_graph_triples(triples: list[str]) -> None:
    """Render Knowledge Graph fact triples visualization.

    Args:
        triples: List of string triples (e.g. 'Subject -> Predicate -> Object').
    """
    st.subheader("🕸️ Knowledge Graph Triples")
    if not triples:
        st.info("No Knowledge Graph triples available for this query.")
        return

    st.markdown("Extracted entity relationships and multi-hop facts:")
    cols = st.columns(min(len(triples), 3))
    for idx, triple in enumerate(triples):
        with cols[idx % 3]:
            st.info(f"📌 `{triple}`")


def render_graph_rag_visualization(graph_nodes: list[dict] = None, graph_entities: list[str] = None) -> None:
    """Render interactive Knowledge Graph entity & relationship visualization.

    Args:
        graph_nodes: List of graph node objects with type, category, and metadata.
        graph_entities: List of entity names extracted from graph.
    """
    st.subheader("🕸️ Knowledge Graph Entity Network & RAG Traversal")

    if not graph_nodes and not graph_entities:
        st.info("No Knowledge Graph entity data available for this query.")
        return

    if graph_entities:
        st.markdown("**Discovered Graph Entities & POIs:**")
        entity_badges = " ".join([f"`📍 {e}`" for e in graph_entities])
        st.markdown(entity_badges)

    if graph_nodes:
        st.markdown("#### Entity Subgraph Topology")
        df_nodes = pd.DataFrame(graph_nodes)
        st.dataframe(df_nodes, use_container_width=True)



def render_benchmark_dashboard(report_dict: dict) -> None:
    """Render interactive RAG benchmark comparative report.

    Args:
        report_dict: Dictionary representation of BenchmarkReport.
    """
    st.subheader("📊 RAG Benchmarking Framework Evaluation")
    if not report_dict or "metrics" not in report_dict:
        st.info("Run a benchmark evaluation to compare RAG paradigms.")
        return

    st.markdown(f"**Target Query:** `{report_dict.get('query')}`")
    st.success(f"🏆 **Winner:** `{report_dict.get('winner')}`")
    st.caption(report_dict.get("summary", ""))

    metrics_list = report_dict.get("metrics", [])
    df = pd.DataFrame(metrics_list)
    if not df.empty:
        st.table(df[[
            "mode", "latency_ms", "entity_count", "fact_count",
            "precision_score", "recall_score", "structured_valid"
        ]])


def render_llm_error(error_message: str) -> None:
    """Render structured error alert when LLM or graph execution fails completely.

    Args:
        error_message: Error description or exception string.
    """
    st.error(f"🤖 **LLM / Workflow Failure:** {error_message}")
    st.caption("Please check system configuration, API keys, or network connectivity.")


def render_invalid_response_error(details: Optional[str] = None) -> None:
    """Render error alert when output is malformed or invalid TravelResponse object.

    Args:
        details: Optional detailed message regarding the invalid structure.
    """
    st.error("⚠️ **Invalid Structured Response:** Application output did not match expected TravelResponse schema.")
    if details:
        st.caption(f"Details: {details}")


