"""
Edge Reliability Scoring for RECON v2.

Computes a three-signal reliability score for each retrieved paper:

    edge_reliability = (citation_centrality × 0.4)
                     + (recency_signal       × 0.3)
                     + (content_coherence    × 0.3)

Signals:
- citation_centrality: normalized cited_by_count from OpenAlex (or S2 fallback)
  High centrality = foundational paper = high reliability regardless of age
- recency_signal: linear decay max(0, 1 - age/20) — same as RECON v1
  Now one of three inputs, not the whole score
- content_coherence: LLM check — does this paper's abstract still represent
  current scientific understanding? Batched for all papers in one LLM call.

Dominant signal labels (for explainability in synthesizer output):
  FOUNDATIONAL: reliability >= 0.70 AND centrality >= 0.6
  CURRENT:      reliability >= 0.60 AND recency >= 0.7
  DECLINING:    reliability 0.35–0.60
  SUPERSEDED:   reliability < 0.35
"""

import math
import logging
import os
from dataclasses import dataclass
from typing import Optional
import json
import re

logger = logging.getLogger(__name__)

CURRENT_YEAR = 2026

# Reliability thresholds
THRESHOLD_FOUNDATIONAL_RELIABILITY = 0.70
THRESHOLD_FOUNDATIONAL_CENTRALITY = 0.60
THRESHOLD_CURRENT_RELIABILITY = 0.60
THRESHOLD_CURRENT_RECENCY = 0.70
THRESHOLD_DECLINING_LOW = 0.35

# Signal weights
W_CENTRALITY = 0.4
W_RECENCY = 0.3
W_COHERENCE = 0.3


@dataclass
class ReliabilityScore:
    score: float                  # [0, 1] composite reliability
    centrality: float             # normalized citation centrality
    recency: float                # linear decay recency signal
    coherence: float              # LLM content coherence [0, 1]
    dominant_signal: str          # FOUNDATIONAL / CURRENT / DECLINING / SUPERSEDED
    reason: str                   # one-line human-readable explanation


def _compute_centrality(citation_count: int, doi: str = "") -> float:
    """
    Normalized citation centrality.
    Uses OpenAlex cited_by_count if DOI available, else falls back to S2 count.
    Formula: min(1.0, log1p(count) / log1p(10000))
    """
    from src.openalex_utils import get_citation_centrality
    return get_citation_centrality(doi=doi, citation_count=citation_count)


def _compute_recency(year: Optional[int]) -> float:
    """Linear decay: max(0, 1 - age/20). Age 0 = 1.0, age 20+ = 0.0."""
    if not year or year <= 0:
        return 0.0
    age = CURRENT_YEAR - year
    return max(0.0, 1.0 - age / 20.0)


def _compute_coherence_batch(papers: list, query: str) -> list[float]:
    """
    LLM batch coherence check for all papers at once.

    For each paper, asks: does this paper's abstract still represent
    current scientific understanding on this topic?

    Returns a list of float scores [0, 1] in the same order as input papers.
    Falls back to recency-based heuristic if LLM call fails.

    Batched: one LLM call for all papers, not one per paper.
    """
    if not papers:
        return []

    # Build batch prompt
    paper_summaries = []
    for i, p in enumerate(papers):
        abstract_snippet = (p.abstract or "")[:300]
        paper_summaries.append(
            f"Paper {i+1}: [{p.year}] {p.title}\n"
            f"Abstract: {abstract_snippet}"
        )

    papers_text = "\n\n".join(paper_summaries)

    system_prompt = """You are a scientific literature analyst assessing whether papers represent current scientific understanding.

For each paper provided, assign a content_coherence score from 0.0 to 1.0:
- 1.0: Paper's central claims are still the consensus view, no major challenges
- 0.7: Paper is foundational and still cited, but some aspects have been refined
- 0.5: Paper's claims are actively debated; newer work challenges some findings
- 0.3: Paper's central claims have been substantially superseded by newer work
- 0.1: Paper is clearly outdated; its claims contradict current consensus

Respond ONLY with a JSON array of objects, one per paper, in the same order:
[{"paper_index": 1, "coherence": 0.8, "reason": "one sentence"}, ...]

Be concise. No other text."""

    user_prompt = f"""Research query context: {query[:200]}

Papers to assess:
{papers_text}

Return ONLY the JSON array."""

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()

        # Extract JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            scores = [0.5] * len(papers)  # default
            for item in data:
                idx = int(item.get("paper_index", 0)) - 1  # 1-indexed in prompt
                if 0 <= idx < len(papers):
                    scores[idx] = float(item.get("coherence", 0.5))
            return scores

    except Exception as e:
        logger.warning(f"Coherence batch LLM call failed: {e}")

    # Fallback: use recency as coherence proxy
    return [_compute_recency(p.year) for p in papers]


def _dominant_signal(score: float, centrality: float, recency: float, coherence: float) -> str:
    """
    Classify dominant signal for explainability.

    FOUNDATIONAL: high centrality + high coherence — trusted regardless of age
    CURRENT: recent + reliable — recently published and well-supported
    DECLINING: mixed signals — some reliability but losing relevance
    SUPERSEDED: low reliability overall — likely outdated
    """
    # Foundational: highly cited AND content is still coherent with consensus
    # Age is irrelevant for foundational papers — that's the point.
    # When coherence=0.0 (LLM off, recency proxy for old paper), centrality alone qualifies.
    if centrality >= THRESHOLD_FOUNDATIONAL_CENTRALITY and (coherence >= 0.65 or coherence == 0.0):
        return "FOUNDATIONAL"
    # Current: recent paper with good reliability
    elif recency >= THRESHOLD_CURRENT_RECENCY and score >= THRESHOLD_CURRENT_RELIABILITY:
        return "CURRENT"
    elif score >= THRESHOLD_DECLINING_LOW:
        return "DECLINING"
    else:
        return "SUPERSEDED"


def _build_reason(dominant: str, centrality: float, recency: float,
                  coherence: float, year: Optional[int]) -> str:
    """One-line reason string for the trust summary."""
    age = (CURRENT_YEAR - year) if year else None
    age_str = f"{age}yr old" if age is not None else "unknown age"

    if dominant == "FOUNDATIONAL":
        return f"High citation centrality ({centrality:.2f}), {age_str} - foundational work still current"
    elif dominant == "CURRENT":
        return f"Recent ({age_str}), coherence={coherence:.2f} - aligns with current consensus"
    elif dominant == "DECLINING":
        return f"Mixed signals: centrality={centrality:.2f}, recency={recency:.2f}, coherence={coherence:.2f}"
    else:
        return f"Low reliability: {age_str}, centrality={centrality:.2f}, coherence={coherence:.2f} - likely superseded"


def score_papers(papers: list, query: str, use_llm: bool = True) -> dict[str, ReliabilityScore]:
    """
    Main entry point. Scores all papers and returns a dict of paper_id -> ReliabilityScore.

    Args:
        papers: list of Paper objects
        query: the original research query (for coherence context)
        use_llm: if False, skips coherence LLM call (uses recency as fallback)
                 Set False during eval to save Groq API calls.

    Returns:
        dict mapping paper_id -> ReliabilityScore
    """
    if not papers:
        return {}

    # Step 1: Centrality (OpenAlex DOI lookup if available, else S2 count)
    centralities = []
    for p in papers:
        c = _compute_centrality(
            citation_count=getattr(p, "citation_count", 0) or 0,
            doi=getattr(p, "doi", "") or "",
        )
        centralities.append(c)

    # Step 2: Recency
    recencies = [_compute_recency(getattr(p, "year", None)) for p in papers]

    # Step 3: Coherence (batched LLM call)
    if use_llm:
        coherences = _compute_coherence_batch(papers, query)
    else:
        coherences = [_compute_recency(getattr(p, "year", None)) for p in papers]

    # Step 4: Composite score and labeling
    results = {}
    for i, p in enumerate(papers):
        c = centralities[i]
        r = recencies[i]
        co = coherences[i] if i < len(coherences) else r

        score = W_CENTRALITY * c + W_RECENCY * r + W_COHERENCE * co
        dominant = _dominant_signal(score, c, r, co)
        reason = _build_reason(dominant, c, r, co, getattr(p, "year", None))

        results[p.paper_id] = ReliabilityScore(
            score=round(score, 4),
            centrality=round(c, 4),
            recency=round(r, 4),
            coherence=round(co, 4),
            dominant_signal=dominant,
            reason=reason,
        )

    return results
