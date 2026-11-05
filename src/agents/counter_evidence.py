"""
Counter-evidence retrieval (RECON v2.1).

Why this exists
---------------
The README's known limitation: contradiction catch rate is 0% because the
retriever returns papers that are *related* to the query, not papers that
*disagree* with it. Similarity search is biased toward agreement: a paper
that says "X works" is closest in embedding space to other papers that say
"X works".

This node runs one extra, targeted search per sub-question that is built to
surface the opposing camp: critiques, failed replications, "revisiting" /
"rethinking" papers and negative results. Those papers are stored in a
separate state key, ``opposing_papers``, and the critic pairs them against
the top supporting papers when it checks for contradictions.

What it deliberately does NOT do
--------------------------------
* It does not add opposing papers to ``retrieved_papers``. The staleness
  verdict, the INSUFFICIENT checks and the reliability scores are computed
  from exactly the same papers as before, so the 44% staleness number is not
  affected by this change.
* It does not change any verdict threshold or scoring weight. It only gives
  the existing contradiction check better evidence to look at.
* It makes no LLM calls. Queries are built deterministically, so the node is
  cheap, testable and works without an API key.

Toggle with RECON_COUNTER_EVIDENCE=0 to run the old pipeline (for ablations).
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from src.state import ResearchState, Paper

logger = logging.getLogger(__name__)

# Terms that tend to appear in titles/abstracts of papers that push back on
# an earlier result. Kept short: S2 keyword search degrades with long queries.
CRITIQUE_TERMS = [
    "limitations",
    "fails replicate",
    "revisiting",
    "critique",
]

COUNTER_LIMIT_PER_QUESTION = 3
MAX_QUERIES = 4


def is_enabled() -> bool:
    return os.environ.get("RECON_COUNTER_EVIDENCE", "1").strip() not in ("0", "false", "no")


def _keywords(question: str, max_words: int = 4) -> str:
    """Reuse the retriever's keyword extraction, then trim further so the
    critique term still carries weight in the query."""
    from src.agents.retriever import _to_search_query

    return " ".join(_to_search_query(question).split()[:max_words])


_POSITIVE_TO_NEGATIVE = [
    (r"\bimproves?\b", "does not improve"),
    (r"\boutperforms?\b", "does not outperform"),
    (r"\bworks?\b", "does not work"),
    (r"\benables?\b", "fails to enable"),
    (r"\bis effective\b", "is ineffective"),
    (r"\bscales?\b", "does not scale"),
]


def negate_claim(claim: str) -> str | None:
    """Very small rule-based negation of a claim, e.g.
    "dropout improves generalization" -> "dropout does not improve generalization".
    Returns None if no rule applies (we don't guess)."""
    text = claim.strip().rstrip("?.").lower()
    # "Does X improve Y?" -> "x improve y" so the rule below reads naturally
    text = re.sub(r"^(does|do|can|is|are)\s+", "", text)
    for pattern, repl in _POSITIVE_TO_NEGATIVE:
        if re.search(pattern, text):
            return re.sub(pattern, repl, text, count=1)
    return None


def build_counter_queries(original_query: str, sub_questions: list[str]) -> list[str]:
    """Build up to MAX_QUERIES keyword queries aimed at opposing evidence.

    1. If the original query states a claim we can negate, search the negation.
    2. For each sub-question, pair its core keywords with one critique term
       (rotating through CRITIQUE_TERMS so we don't send four near-identical
       queries for the same topic).
    """
    queries: list[str] = []

    negated = negate_claim(original_query or "")
    if negated:
        queries.append(" ".join(negated.split()[:7]))

    topics = [_keywords(q) for q in (sub_questions or [original_query]) if q]
    for i, topic in enumerate(topics):
        if not topic:
            continue
        queries.append(f"{topic} {CRITIQUE_TERMS[i % len(CRITIQUE_TERMS)]}")

    # De-duplicate, keep order
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:MAX_QUERIES]


def counter_evidence_node(state: ResearchState) -> ResearchState:
    """
    Reads:  original_query, sub_questions, retrieved_papers, decay_config
    Writes: opposing_papers
    """
    if not is_enabled():
        return {**state, "opposing_papers": []}

    from src.retriever_utils import search_semantic_scholar

    queries = build_counter_queries(
        state.get("original_query", ""), state.get("sub_questions") or []
    )
    decay_config = state.get("decay_config", "linear")
    supporting_ids = {p.paper_id for p in (state.get("retrieved_papers") or []) if p.paper_id}

    def _fetch(q: str) -> list[Paper]:
        try:
            return search_semantic_scholar(q, limit=COUNTER_LIMIT_PER_QUESTION, decay_config=decay_config)
        except Exception as e:  # never let counter-evidence break the main pipeline
            logger.warning(f"Counter-evidence search failed for '{q}': {e}")
            return []

    # S2 calls go through the shared rate limiter in retriever_utils, so
    # running them in threads is safe.
    with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
        results = list(pool.map(_fetch, queries))

    opposing: list[Paper] = []
    seen: set[str] = set(supporting_ids)
    for papers in results:
        for p in papers:
            if p.paper_id and p.paper_id in seen:
                continue  # already a supporting paper, or a duplicate
            if p.paper_id:
                seen.add(p.paper_id)
            p.source = f"{p.source}+counter"
            opposing.append(p)

    opposing.sort(key=lambda p: p.hybrid_score, reverse=True)
    logger.info(f"Counter-evidence: {len(queries)} queries -> {len(opposing)} opposing candidates")
    return {**state, "opposing_papers": opposing}
