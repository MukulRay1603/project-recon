import os
import time
import math
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.state import Paper, WebResult

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model — loaded once at module level (CPU, fast)
# ---------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Disk cache — prevents re-fetching on eval loop crashes
# ---------------------------------------------------------------------------
_CACHE_DIR = os.environ.get(
    "RECON_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
)
os.makedirs(_CACHE_DIR, exist_ok=True)

def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def _cache_get(key: str) -> Optional[list]:
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def _cache_set(key: str, data: list) -> None:
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Recency scoring — three formulas for ablation study
# ---------------------------------------------------------------------------
CURRENT_YEAR = datetime.now().year

def recency_score(year: int, decay_config: str = "linear") -> float:
    """
    Returns a 0–1 recency score for a paper given its publication year.
    decay_config: "none" | "linear" | "log"
    """
    if year is None or year == 0:
        return 0.0
    age = max(0, CURRENT_YEAR - year)

    if decay_config == "none":
        return 1.0
    elif decay_config == "linear":
        return max(0.0, 1.0 - (age / 20.0))
    elif decay_config == "log":
        return max(0.0, 1.0 - math.log1p(age) / math.log1p(20))
    else:
        return max(0.0, 1.0 - (age / 20.0))  # default to linear


def authority_score(citation_count: int) -> float:
    """Normalize citation count to 0–1 using log scale."""
    if citation_count <= 0:
        return 0.0
    return min(1.0, math.log1p(citation_count) / math.log1p(10000))


def hybrid_score(
    semantic_sim: float,
    year: int,
    citation_count: int,
    decay_config: str = "linear",
) -> float:
    """
    final_score = semantic_sim × 0.5 + recency × 0.3 + authority × 0.2
    Weights chosen by ablation study (see eval/).
    """
    r = recency_score(year, decay_config)
    a = authority_score(citation_count)
    return round(semantic_sim * 0.5 + r * 0.3 + a * 0.2, 4)


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------
def search_semantic_scholar(
    query: str,
    limit: int = 5,
    decay_config: str = "linear",
    use_cache: bool = True,
) -> list[Paper]:
    """
    Search Semantic Scholar via direct HTTP request (avoids pagination bug).
    Returns a list of Paper objects sorted by hybrid_score descending.
    """
    cache_key = _cache_key(f"s2v2_{query}_{limit}")
    if use_cache:
        cached = _cache_get(cache_key)
        if cached:
            logger.info(f"S2 cache hit: {query[:50]}")
            return [Paper(**p) for p in cached]

    import requests

    s2_key = os.getenv("S2_API_KEY")
    headers = {"x-api-key": s2_key} if s2_key else {}

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,authors,references,paperId,externalIds",
    }

    time.sleep(3)  # rate limit guard

    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"S2 search failed for '{query}': {e}")
        return []

    raw_papers = data.get("data", [])
    if not raw_papers:
        return []

    embedder = get_embedder()
    query_vec = embedder.encode([query])

    papers = []
    for r in raw_papers:
        abstract = r.get("abstract") or ""
        if not abstract:
            abstract = r.get("title") or "No abstract available"
        abstract_vec = embedder.encode([abstract])
        sim = float(cosine_similarity(query_vec, abstract_vec)[0][0])

        year = r.get("year") or 0
        citations = r.get("citationCount") or 0
        authors = [a["name"] for a in r.get("authors") or []]
        references = [
            ref["paperId"] for ref in (r.get("references") or [])
            if ref.get("paperId")
        ]

        doi = (r.get("externalIds") or {}).get("DOI", "") or ""
        paper = Paper(
            title=r.get("title") or "Untitled",
            abstract=abstract,
            year=year,
            citation_count=citations,
            paper_id=r.get("paperId") or "",
            authors=authors,
            references=references,
            doi=doi,
            hybrid_score=hybrid_score(sim, year, citations, decay_config),
            source="semantic_scholar",
        )
        papers.append(paper)

    papers.sort(key=lambda p: p.hybrid_score, reverse=True)

    if use_cache:
        _cache_set(cache_key, [p.__dict__ for p in papers])

    return papers


# ---------------------------------------------------------------------------
# DuckDuckGo web search (with Tavily fallback)
# ---------------------------------------------------------------------------
def search_web(
    query: str,
    limit: int = 5,
    use_cache: bool = True,
) -> list[WebResult]:
    """
    Search the web via DuckDuckGo. Falls back to Tavily if DDG fails.
    Returns a list of WebResult objects.
    """
    cache_key = _cache_key(f"web_{query}_{limit}")
    if use_cache:
        cached = _cache_get(cache_key)
        if cached:
            logger.info(f"Web cache hit: {query[:50]}")
            return [WebResult(**r) for r in cached]

    results = _ddg_search(query, limit)

    if not results:
        logger.warning(f"DDG failed for '{query}', trying Tavily fallback")
        results = _tavily_search(query, limit)

    if use_cache and results:
        _cache_set(cache_key, [r.__dict__ for r in results])

    return results


def _ddg_search(query: str, limit: int) -> list[WebResult]:
    try:
        from ddgs import DDGS
        time.sleep(1)
        # Force English results, safesearch off, recent results
        search_query = f"{query} research paper arxiv"
        with DDGS() as ddgs:
            raw = list(ddgs.text(
                search_query,
                max_results=limit,
                region="wt-wt",      # worldwide — avoids regional override
                safesearch="off",
            ))
        results = []
        for r in raw:
            year = _infer_year(r.get("body", ""))
            results.append(WebResult(
                url=r.get("href", ""),
                snippet=r.get("body", "")[:500],
                title=r.get("title", ""),
                inferred_year=year,
                source="duckduckgo",
            ))
        return results
    except Exception as e:
        logger.warning(f"DDG error: {e}")
        return []


def _tavily_search(query: str, limit: int) -> list[WebResult]:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        response = client.search(query, max_results=limit)
        results = []
        for r in response.get("results", []):
            year = _infer_year(r.get("content", ""))
            results.append(WebResult(
                url=r.get("url", ""),
                snippet=r.get("content", "")[:500],
                title=r.get("title", ""),
                inferred_year=year,
                source="tavily",
            ))
        return results
    except Exception as e:
        logger.warning(f"Tavily error: {e}")
        return []


def _infer_year(text: str) -> Optional[int]:
    """Try to extract a 4-digit year (2000–2026) from a text snippet."""
    import re
    matches = re.findall(r"\b(20[0-2][0-9])\b", text)
    if matches:
        years = [int(y) for y in matches]
        return max(years)
    return None


# ---------------------------------------------------------------------------
# Citation graph builder
# ---------------------------------------------------------------------------
def build_citation_graph(papers: list[Paper]) -> dict:
    """
    Build a citation graph from retrieved papers.
    Returns {paper_id: [list of referenced paper_ids that are also in our set]}
    Only includes edges where both source and target are in our retrieved set.
    """
    paper_ids = {p.paper_id for p in papers}
    graph = {}
    for p in papers:
        graph[p.paper_id] = [
            ref for ref in p.references
            if ref in paper_ids
        ]
    return graph