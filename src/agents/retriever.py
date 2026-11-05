import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from src.state import ResearchState, Paper, WebResult
from src.retriever_utils import (
    search_semantic_scholar,
    search_web,
    build_citation_graph,
    hybrid_score,
)

load_dotenv()
logger = logging.getLogger(__name__)

# How many results to fetch per sub-question per source
S2_LIMIT_PER_QUESTION = 4
WEB_LIMIT_PER_QUESTION = 3
# Sub-questions are retrieved concurrently. S2 calls still respect the
# process-wide rate limiter in retriever_utils; set to 1 for the old
# sequential behaviour.
RETRIEVAL_WORKERS = int(os.environ.get("RECON_RETRIEVAL_WORKERS", "3"))

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

    def _retrieve_one(indexed_question):
        i, question = indexed_question
        logger.info(f"Retriever: searching for sub-question {i+1}: {question[:60]}")
        s2_query = _to_search_query(question)
        logger.info(f"  S2 keyword query: '{s2_query}'")
        papers = search_semantic_scholar(
            s2_query,
            limit=S2_LIMIT_PER_QUESTION,
            decay_config=decay_config,
        )
        web_results = search_web(question, limit=WEB_LIMIT_PER_QUESTION)
        return papers, web_results

    # Fetch in parallel; pool.map returns results in sub-question order, so
    # de-duplication below is deterministic and identical to the old loop.
    workers = max(1, min(RETRIEVAL_WORKERS, len(sub_questions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_question = list(pool.map(_retrieve_one, enumerate(sub_questions)))

    for papers, web_results in per_question:
        for p in papers:
            if p.paper_id and p.paper_id not in seen_paper_ids:
                seen_paper_ids.add(p.paper_id)
                all_papers.append(p)
            elif not p.paper_id:
                all_papers.append(p)
        for r in web_results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_web.append(r)

    # --- Phase 2.2: OpenAlex augmentation ---
    from src.openalex_utils import search_openalex
    existing_dois = {p.doi.lower() for p in all_papers if p.doi}
    existing_ids = {p.paper_id for p in all_papers}

    if len(all_papers) < 12:
        def _oa(question):
            try:
                return search_openalex(question, max_results=3)
            except Exception as e:
                logger.warning(f"OpenAlex augmentation failed for '{question[:40]}': {e}")
                return []

        oa_questions = sub_questions[:2]  # only first 2 sub-questions
        with ThreadPoolExecutor(max_workers=max(1, len(oa_questions))) as pool:
            oa_batches = list(pool.map(_oa, oa_questions))

        for question, oa_results in zip(oa_questions, oa_batches):
            try:
                for r in oa_results:
                    doi_lower = (r.get("doi") or "").lower()
                    pid = r.get("paper_id") or ""
                    # Skip if we already have this paper by DOI or paper_id
                    if doi_lower and doi_lower in existing_dois:
                        continue
                    if pid in existing_ids:
                        continue
                    # Build Paper object from OpenAlex result
                    p = Paper(
                        title=r.get("title") or "",
                        abstract=r.get("abstract") or "",
                        year=r.get("year") or 0,
                        citation_count=r.get("citation_count") or 0,
                        paper_id=pid,
                        authors=[a.strip() for a in (r.get("authors") or "").split(",") if a.strip()],
                        references=[],
                        doi=r.get("doi") or "",
                        source="openalex",
                    )
                    if not p.title or not p.year:
                        continue
                    p.hybrid_score = hybrid_score(
                        semantic_sim=0.3,  # conservative default — no query embedding for OA papers
                        year=p.year,
                        citation_count=p.citation_count,
                        decay_config=decay_config,
                    )
                    all_papers.append(p)
                    existing_dois.add(doi_lower)
                    existing_ids.add(pid)
            except Exception as e:
                logger.warning(f"OpenAlex augmentation failed for '{question[:40]}': {e}")
                continue

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