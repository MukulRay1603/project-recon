"""
eval/curate_surveys.py
----------------------
Phase 10 — Dataset curation step 1.

Fetches real survey and benchmark papers from Semantic Scholar across
3 subfields × 3 category hints. Saves to eval/survey_sources.json
and eval/survey_sources_summary.txt.

Run from repo root:
    python eval/curate_surveys.py

Output:
    eval/survey_sources.json        ← machine-readable, used by next step
    eval/survey_sources_summary.txt ← human-readable, for spot-checking

Runtime: ~90 seconds (S2 rate limit sleep built in).
"""

import sys
import os
import json
import time
import logging

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

# ── imports from your existing code ──────────────────────────────────────────
# We call search_semantic_scholar directly — same function your pipeline uses.
# use_cache=False so we always get fresh results (not stale cache from dev runs).
import requests

# ---------------------------------------------------------------------------
# We bypass the wrapper slightly here — we want raw S2 data (no hybrid score
# needed, just title + abstract + year + citation_count + paper_id + authors)
# This avoids loading sentence-transformers just for curation.
# ---------------------------------------------------------------------------

S2_KEY = os.getenv("S2_API_KEY")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}
S2_URL  = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS  = "title,abstract,year,citationCount,authors,paperId,references"

# Minimum abstract length to consider a paper usable as ground truth source
MIN_ABSTRACT_LEN = 150
# Minimum year — older papers won't reflect current state of field
MIN_YEAR = 2019
# Papers per query
LIMIT_PER_QUERY = 8

# ---------------------------------------------------------------------------
# Query plan — 3 subfields × 6 queries each = 18 total
# category_hint:
#   A = consensus questions (survey/overview papers)
#   B = superseded literature (comparison/benchmark papers that show one method
#       beating another, or explicit "we show X is suboptimal" language)
#   C = contested questions (debate/controversy/tradeoff papers)
# ---------------------------------------------------------------------------

QUERY_PLAN = [

    # ── Subfield 1: LLM Efficiency (KV cache, speculative decoding, quant) ──
    {
        "subfield": "llm_efficiency",
        "category_hint": "A",
        "query": "KV cache compression large language models survey",
    },
    {
        "subfield": "llm_efficiency",
        "category_hint": "A",
        "query": "speculative decoding inference acceleration survey",
    },
    {
        "subfield": "llm_efficiency",
        "category_hint": "B",
        "query": "KV cache eviction comparison H2O StreamingLLM benchmark",
    },
    {
        "subfield": "llm_efficiency",
        "category_hint": "B",
        "query": "speculative decoding draft model supersedes baseline",
    },
    {
        "subfield": "llm_efficiency",
        "category_hint": "C",
        "query": "LLM quantization tradeoffs debate accuracy versus compression",
    },
    {
        "subfield": "llm_efficiency",
        "category_hint": "C",
        "query": "KV cache compression controversy limitations open problems",
    },

    # ── Subfield 2: Training Methods (RLHF, LoRA, fine-tuning) ──────────────
    {
        "subfield": "training_methods",
        "category_hint": "A",
        "query": "RLHF reinforcement learning human feedback alignment survey",
    },
    {
        "subfield": "training_methods",
        "category_hint": "A",
        "query": "LoRA low rank adaptation fine-tuning survey overview",
    },
    {
        "subfield": "training_methods",
        "category_hint": "B",
        "query": "DPO direct preference optimization versus PPO comparison",
    },
    {
        "subfield": "training_methods",
        "category_hint": "B",
        "query": "QLoRA quantized LoRA outperforms full fine-tuning benchmark",
    },
    {
        "subfield": "training_methods",
        "category_hint": "C",
        "query": "RLHF alignment debate reward hacking limitations controversy",
    },
    {
        "subfield": "training_methods",
        "category_hint": "C",
        "query": "LoRA versus full fine-tuning tradeoff disagreement",
    },

    # ── Subfield 3: Retrieval & RAG ──────────────────────────────────────────
    {
        "subfield": "rag",
        "category_hint": "A",
        "query": "retrieval augmented generation survey overview 2024",
    },
    {
        "subfield": "rag",
        "category_hint": "A",
        "query": "dense retrieval versus sparse retrieval benchmark survey",
    },
    {
        "subfield": "rag",
        "category_hint": "B",
        "query": "RAG chunking strategy comparison supersedes fixed size",
    },
    {
        "subfield": "rag",
        "category_hint": "B",
        "query": "hybrid retrieval dense sparse outperforms single method",
    },
    {
        "subfield": "rag",
        "category_hint": "C",
        "query": "RAG versus long context LLM debate tradeoff controversy",
    },
    {
        "subfield": "rag",
        "category_hint": "C",
        "query": "retrieval augmented generation limitations open problems debate",
    },
]


# ---------------------------------------------------------------------------
# Fetch function — raw S2 call, no hybrid scoring needed here
# ---------------------------------------------------------------------------

def fetch_s2_papers(query: str, limit: int = LIMIT_PER_QUERY) -> list[dict]:
    """
    Direct S2 REST call. Returns list of raw paper dicts.
    Filters out papers with missing/short abstracts and pre-2019 papers.
    """
    params = {
        "query": query,
        "limit": limit,
        "fields": FIELDS,
    }

    time.sleep(3)  # S2 rate limit guard — same as your pipeline

    try:
        r = requests.get(S2_URL, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ S2 error for '{query[:50]}': {e}")
        return []

    raw = data.get("data", [])
    papers = []
    for p in raw:
        abstract = p.get("abstract") or ""
        year     = p.get("year") or 0

        # Filter: need a real abstract and recent enough year
        if len(abstract) < MIN_ABSTRACT_LEN:
            continue
        if year < MIN_YEAR:
            continue

        papers.append({
            "title":         p.get("title") or "Untitled",
            "abstract":      abstract,
            "year":          year,
            "citation_count": p.get("citationCount") or 0,
            "paper_id":      p.get("paperId") or "",
            "authors":       [a["name"] for a in (p.get("authors") or [])],
            "references":    [
                ref["paperId"] for ref in (p.get("references") or [])
                if ref.get("paperId")
            ],
        })

    return papers


# ---------------------------------------------------------------------------
# Main curation run
# ---------------------------------------------------------------------------

def run_curation() -> None:
    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir   = script_dir  # script lives in eval/, output goes there too
    os.makedirs(eval_dir, exist_ok=True)

    output_json = os.path.join(eval_dir, "survey_sources.json")
    output_txt  = os.path.join(eval_dir, "survey_sources_summary.txt")

    print("=" * 60)
    print("RECON — Survey Source Curation")
    print(f"Queries to run: {len(QUERY_PLAN)}")
    print(f"S2 key present: {bool(S2_KEY)}")
    print(f"Estimated runtime: ~{len(QUERY_PLAN) * 4}s")
    print("=" * 60)

    all_results   = []
    summary_lines = []
    total_papers  = 0
    seen_ids      = set()  # deduplicate across queries

    for i, entry in enumerate(QUERY_PLAN, 1):
        subfield  = entry["subfield"]
        cat_hint  = entry["category_hint"]
        query     = entry["query"]

        print(f"\n[{i:02d}/{len(QUERY_PLAN)}] [{subfield}] [Cat {cat_hint}] {query}")

        papers = fetch_s2_papers(query)

        # Deduplicate by paper_id across all queries
        unique_papers = []
        for p in papers:
            pid = p["paper_id"]
            if pid and pid in seen_ids:
                print(f"  ↳ duplicate skipped: {p['title'][:50]}")
                continue
            if pid:
                seen_ids.add(pid)
            unique_papers.append(p)

        print(f"  ✓ {len(unique_papers)} papers fetched (after dedup)")
        for p in unique_papers:
            print(f"    [{p['year']}] [{p['citation_count']:,} cit] {p['title'][:65]}")

        total_papers += len(unique_papers)

        # Store result
        all_results.append({
            "subfield":     subfield,
            "category_hint": cat_hint,
            "query_used":   query,
            "papers":       unique_papers,
        })

        # Build summary text
        summary_lines.append(
            f"\n{'─'*60}\n"
            f"[{subfield.upper()}] Category hint: {cat_hint}\n"
            f"Query: {query}\n"
            f"Papers ({len(unique_papers)}):\n"
        )
        for p in unique_papers:
            authors_str = ", ".join(p["authors"][:2])
            if len(p["authors"]) > 2:
                authors_str += " et al."
            summary_lines.append(
                f"  • {p['title']}\n"
                f"    {authors_str} | {p['year']} | {p['citation_count']:,} citations\n"
                f"    ID: {p['paper_id']}\n"
                f"    Abstract preview: {p['abstract'][:200]}...\n"
            )

    # ── Save outputs ──────────────────────────────────────────────────────────

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("RECON — Survey Source Curation Summary\n")
        f.write(f"Total papers fetched: {total_papers}\n")
        f.write(f"Unique paper IDs: {len(seen_ids)}\n")
        f.write("".join(summary_lines))

    print("\n" + "=" * 60)
    print(f"✅ Curation complete")
    print(f"   Total papers:     {total_papers}")
    print(f"   Unique paper IDs: {len(seen_ids)}")
    print(f"   JSON output:      eval/survey_sources.json")
    print(f"   Summary output:   eval/survey_sources_summary.txt")
    print("=" * 60)
    print("\nNext step:")
    print("  Paste the contents of eval/survey_sources_summary.txt")
    print("  back into the chat — ground truth extraction follows.")


if __name__ == "__main__":
    run_curation()