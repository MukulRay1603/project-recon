import logging
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.state import ResearchState, Verdict
from src.memory import init_db, load_session
from src.agents.planner import planner_node
from src.agents.retriever import retriever_node
from src.agents.critic import critic_node
from src.agents.synthesizer import synthesizer_node

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing function — decides what happens after the critic
# ---------------------------------------------------------------------------

def route_after_critic(state: ResearchState) -> str:
    """
    Returns the name of the next node based on critic verdict.
    PASS / FORCED_PASS → synthesizer
    STALE / CONTRADICTED / INSUFFICIENT → retriever (retry loop)
    """
    verdict = state.get("critic_verdict", "")

    if verdict in (Verdict.PASS, Verdict.FORCED_PASS):
        logger.info(f"Routing: {verdict} → synthesizer")
        return "synthesizer"

    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        logger.info("Routing: max retries → synthesizer (forced)")
        return "synthesizer"

    logger.info(f"Routing: {verdict} → retriever (retry {retry_count})")
    return "retriever"


def route_after_retriever_retry(state: ResearchState) -> str:
    """After a retry retrieval, always go to critic."""
    return "critic"


# ---------------------------------------------------------------------------
# Session loader — prepended to graph as first node
# ---------------------------------------------------------------------------

def session_loader_node(state: ResearchState) -> ResearchState:
    """
    Loads session context from SQLite before the planner runs.
    This gives the planner access to prior queries and positions.
    """
    session_id = state.get("session_id", "")
    if session_id:
        try:
            ctx = load_session(session_id)
            logger.info(
                f"Session loaded: {len(ctx.prior_positions)} prior positions"
            )
            return {**state, "session_context": ctx}
        except Exception as e:
            logger.warning(f"Session load failed: {e}")

    return state


# ---------------------------------------------------------------------------
# Retry retriever — uses rewritten questions from critic
# ---------------------------------------------------------------------------

def retry_retriever_node(state: ResearchState) -> ResearchState:
    """
    Like the retriever but uses rewritten_questions from the critic
    instead of sub_questions from the planner.
    Merges new results with existing ones.
    """
    rewritten = state.get("rewritten_questions") or []
    if not rewritten:
        logger.warning("Retry retriever: no rewritten questions, skipping")
        return state

    # Swap sub_questions for rewritten ones, run retriever
    retry_state = {**state, "sub_questions": rewritten}
    result = retriever_node(retry_state)

    # Merge new papers with existing (deduplicate by paper_id)
    existing_papers = state.get("retrieved_papers") or []
    new_papers = result.get("retrieved_papers") or []

    seen_ids = {p.paper_id for p in existing_papers if p.paper_id}
    merged = list(existing_papers)
    for p in new_papers:
        if p.paper_id not in seen_ids:
            merged.append(p)
            if p.paper_id:
                seen_ids.add(p.paper_id)

    # Sort merged by hybrid score
    merged.sort(key=lambda p: p.hybrid_score, reverse=True)

    # Merge web results too
    existing_web = state.get("web_results") or []
    new_web = result.get("web_results") or []
    seen_urls = {r.url for r in existing_web}
    merged_web = list(existing_web)
    for r in new_web:
        if r.url not in seen_urls:
            merged_web.append(r)
            seen_urls.add(r.url)

    logger.info(
        f"Retry retriever: merged to {len(merged)} papers, "
        f"{len(merged_web)} web results"
    )

    return {
        **result,
        "retrieved_papers": merged,
        "web_results": merged_web,
        "sub_questions": state.get("sub_questions") or [],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and compile the RECON LangGraph state machine.

    Flow:
    session_loader → planner → retriever → critic
                                               ↓ PASS/FORCED_PASS
                                           synthesizer → END
                                               ↓ STALE/CONTRADICTED/INSUFFICIENT
                                       retry_retriever → critic (loop, max 2x)
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("session_loader", session_loader_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("critic", critic_node)
    graph.add_node("retry_retriever", retry_retriever_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Linear flow: session_loader → planner → retriever → critic
    graph.set_entry_point("session_loader")
    graph.add_edge("session_loader", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")

    # Conditional routing after critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesizer": "synthesizer",
            "retriever": "retry_retriever",
        }
    )

    # Retry loop: retry_retriever → critic
    graph.add_edge("retry_retriever", "critic")

    # Synthesizer is terminal
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run function
# ---------------------------------------------------------------------------

def run_recon(
    query: str,
    session_id: str,
    decay_config: str = "linear",
) -> ResearchState:
    """
    Run the full RECON pipeline for a query.
    Returns the final state.
    """
    init_db()

    graph = build_graph()

    initial_state: ResearchState = {
        "original_query": query,
        "session_id": session_id,
        "session_context": None,
        "sub_questions": [],
        "retrieved_papers": [],
        "citation_graph": {},
        "web_results": [],
        "critic_verdict": "",
        "critic_notes": "",
        "rewritten_questions": [],
        "retry_count": 0,
        "synthesized_position": "",
        "claim_confidences": [],
        "session_update": None,
        "export_md": "",
        "decay_config": decay_config,
        "calibration_bin": "",
        "latency_ms": 0.0,
    }

    start = time.time()
    logger.info(f"RECON run started: '{query[:60]}'")

    final_state = graph.invoke(initial_state)

    elapsed_ms = (time.time() - start) * 1000
    final_state["latency_ms"] = elapsed_ms
    logger.info(f"RECON run complete in {elapsed_ms:.0f}ms")

    return final_state