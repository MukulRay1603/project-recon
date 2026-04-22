"""
OpenAlex API integration for RECON v2.
Provides paper search and DOI-based lookup with citation centrality.

API key required (free at openalex.org/settings/api).
Store as OPENALEX_API_KEY in .env

Rate limits (free tier, April 2026):
- Singleton DOI lookups: unlimited
- List/filter searches: 10,000/day
- Full-text search: 1,000/day
"""

import os
import time
import hashlib
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
TIMEOUT = 8  # seconds — fail fast, OpenAlex is usually quick
MAX_RESULTS_PER_QUERY = 5


def _get_api_key() -> Optional[str]:
    return os.environ.get("OPENALEX_API_KEY")


def _headers() -> dict:
    key = _get_api_key()
    if key:
        return {"Authorization": f"Bearer {key}"}
    logger.debug("OPENALEX_API_KEY not set — requests may be rate limited")
    return {}


def _safe_get(url: str, params: dict) -> Optional[dict]:
    """Single GET with timeout and error handling. Returns parsed JSON or None."""
    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            logger.warning("OpenAlex rate limit hit — skipping")
            return None
        else:
            logger.warning(f"OpenAlex {resp.status_code} for {url}")
            return None
    except requests.exceptions.Timeout:
        logger.warning(f"OpenAlex timeout for {url}")
        return None
    except Exception as e:
        logger.warning(f"OpenAlex error: {e}")
        return None


def _parse_work(work: dict) -> Optional[dict]:
    """
    Parse a single OpenAlex Work object into a flat dict compatible with
    RECON's Paper dataclass fields.

    Returns dict with keys: title, year, doi, abstract, citation_count,
    authors, paper_id, url, source
    Returns None if work is missing essential fields.
    """
    title = work.get("title") or ""
    if not title:
        return None

    year = work.get("publication_year")
    doi = work.get("doi") or ""
    # OpenAlex DOIs come as full URLs — strip to bare DOI
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    citation_count = work.get("cited_by_count") or 0

    # Abstract: OpenAlex stores as inverted index — reconstruct
    abstract = ""
    inv_abstract = work.get("abstract_inverted_index") or {}
    if inv_abstract:
        word_positions = []
        for word, positions in inv_abstract.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        abstract = " ".join(w for _, w in word_positions)

    # Authors
    authorships = work.get("authorships") or []
    authors = []
    for a in authorships[:5]:  # cap at 5
        display = (a.get("author") or {}).get("display_name") or ""
        if display:
            authors.append(display)
    authors_str = ", ".join(authors)

    # Stable ID: prefer DOI, fall back to OpenAlex ID
    openalex_id = work.get("id") or ""
    paper_id = f"openalex:{doi}" if doi else f"openalex:{openalex_id}"

    # URL: prefer DOI link, then OpenAlex page
    url = f"https://doi.org/{doi}" if doi else (work.get("primary_location") or {}).get("landing_page_url") or ""

    return {
        "title": title,
        "year": year,
        "doi": doi,
        "abstract": abstract,
        "citation_count": citation_count,
        "authors": authors_str,
        "paper_id": paper_id,
        "url": url,
        "source": "openalex",
    }


def search_openalex(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """
    Search OpenAlex for papers matching query string.
    Returns list of parsed paper dicts (compatible with Paper dataclass fields).
    Returns empty list on any failure — never raises.
    """
    params = {
        "search": query,
        "filter": "type:article",
        "sort": "cited_by_count:desc",
        "per-page": max_results,
        "select": "id,title,publication_year,doi,cited_by_count,abstract_inverted_index,authorships,primary_location",
    }
    data = _safe_get(f"{OPENALEX_BASE}/works", params)
    if not data:
        return []

    results = []
    for work in (data.get("results") or []):
        parsed = _parse_work(work)
        if parsed:
            results.append(parsed)

    logger.info(f"OpenAlex search '{query[:40]}': {len(results)} results")
    return results


def get_openalex_by_doi(doi: str) -> Optional[dict]:
    """
    Fetch a single paper by DOI. Used for enrichment — getting cited_by_count
    for papers already retrieved from Semantic Scholar.
    Returns parsed paper dict or None.
    """
    if not doi:
        return None
    clean_doi = doi.strip()
    if clean_doi.startswith("https://doi.org/"):
        clean_doi = clean_doi[len("https://doi.org/"):]

    params = {
        "filter": f"doi:{clean_doi}",
        "select": "id,title,publication_year,doi,cited_by_count,abstract_inverted_index,authorships",
    }
    data = _safe_get(f"{OPENALEX_BASE}/works", params)
    if not data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return _parse_work(results[0])


def get_citation_centrality(doi: str, citation_count: Optional[int] = None) -> float:
    """
    Compute normalized citation centrality for a paper.

    If doi is provided, fetches cited_by_count from OpenAlex for accuracy.
    If doi is missing or fetch fails, uses provided citation_count as fallback.

    Formula: min(1.0, log1p(cited_by_count) / log1p(10000))
    This matches the existing authority_score formula in retriever_utils.py
    so the scales are comparable.

    Returns float in [0, 1]. Returns 0.0 on complete failure.
    """
    import math
    count = None

    if doi:
        paper = get_openalex_by_doi(doi)
        if paper:
            count = paper.get("citation_count")

    if count is None:
        count = citation_count or 0

    return min(1.0, math.log1p(count) / math.log1p(10000))
