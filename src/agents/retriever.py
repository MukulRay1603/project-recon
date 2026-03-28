import logging
import time
from dotenv import load_dotenv

from src.state import ResearchState, Paper, WebResult
from src.retriever_utils import (
    search_semantic_scholar,
    search_web,
    build_citation_graph,
)

load_dotenv()
logger = logging.getLogger(__name__)

# How many results to fetch per sub-question per source
S2_LIMIT_PER_QUESTION = 4
WEB_LIMIT_PER_QUESTION = 3

def _to_search_query(question: str) -> str:
    """
    Convert a natural language question to a short keyword query for S2.
    Strips question words and keeps the core noun phrases.
    """
    import re
    # Remove question words and common filler
    stopwords = [
        "what are", "what is", "how does", "how do", "why is", "why are",
        "when did", "where is", "which are", "tell me about",
        "foundational papers on", "recent advances in", "open challenges in",
        "the current state of", "published in", "for llms", "in llms",
        "papers on", "research on", "advances in", "challenges in",
        "were", "was", "the", "a ", "an ", "in ", "of ", "for ", "on ",
    ]
    q = question.lower().strip().rstrip("?")
    for sw in stopwords:
        q = q.replace(sw, " ")
    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    # Cap at 6 words
    words = q.split()[:6]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Retriever node — called by LangGraph
# ---------------------------------------------------------------------------
def retriever_node(state: ResearchState) -> ResearchState:
    """
    Reads:  sub_questions, decay_config
    Writes: retrieved_papers, web_results, citation_graph
    """
    sub_questions = state.get("sub_questions") or []
    decay_config = state.get("decay_config", "linear")

    if not sub_questions:
        logger.warning("Retriever received no sub-questions — returning empty")
        return {
            **state,
            "retrieved_papers": [],
            "web_results": [],
            "citation_graph": {},
        }

    all_papers: list[Paper] = []
    all_web: list[WebResult] = []
    seen_paper_ids: set[str] = set()
    seen_urls: set[str] = set()

    for i, question in enumerate(sub_questions):
        logger.info(f"Retriever: searching for sub-question {i+1}: {question[:60]}")

        # --- Semantic Scholar ---
        s2_query = _to_search_query(question)
        logger.info(f"  S2 keyword query: '{s2_query}'")
        papers = search_semantic_scholar(
            s2_query,
            limit=S2_LIMIT_PER_QUESTION,
            decay_config=decay_config,
        )
        for p in papers:
            if p.paper_id and p.paper_id not in seen_paper_ids:
                seen_paper_ids.add(p.paper_id)
                all_papers.append(p)
            elif not p.paper_id:
                all_papers.append(p)

        # --- Web search ---
        web_results = search_web(question, limit=WEB_LIMIT_PER_QUESTION)
        for r in web_results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_web.append(r)

        # Small pause between sub-questions to be gentle on APIs
        if i < len(sub_questions) - 1:
            time.sleep(1)

    # Sort papers by hybrid score descending
    all_papers.sort(key=lambda p: p.hybrid_score, reverse=True)

    # Build citation graph from retrieved papers
    citation_graph = build_citation_graph(all_papers)

    logger.info(
        f"Retriever complete: {len(all_papers)} papers, "
        f"{len(all_web)} web results, "
        f"{sum(len(v) for v in citation_graph.values())} citation edges"
    )

    return {
        **state,
        "retrieved_papers": all_papers,
        "web_results": all_web,
        "citation_graph": citation_graph,
    }