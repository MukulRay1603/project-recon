"""
eval/generate_leaderboard.py
-----------------------------
Generates docs/superseded_leaderboard.md from REAL usage data.

Reads the verdict_log SQLite table — populated automatically every time
a researcher runs a query through RECON. Aggregates which papers appear
most frequently in STALE and CONTRADICTED verdict queries.

Every entry in the output leaderboard is backed by actual queries
that real people asked. No pre-written content. No synthetic pairs.

Run from repo root (anytime — safe to run repeatedly):
    python eval/generate_leaderboard.py

The leaderboard updates as usage grows. Run it weekly or after a
significant number of new queries to refresh docs/superseded_leaderboard.md.

Minimum viable: works with as few as 1 verdict_log entry.
Best results:   50+ queries across diverse ML topics.
"""

import sys
import os
import json
import re
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.memory import query_verdict_log, init_db

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(os.path.dirname(EVAL_DIR), "docs")
OUTPUT_MD = os.path.join(DOCS_DIR, "superseded_leaderboard.md")

os.makedirs(DOCS_DIR, exist_ok=True)


# ── Paper aggregation ─────────────────────────────────────────────────────────

def aggregate_stale_papers(log_rows: list[dict]) -> list[dict]:
    """
    From verdict_log rows with STALE or CONTRADICTED verdicts,
    aggregate which papers appear most frequently.

    For each paper, track:
      - How many distinct queries retrieved it in a stale context
      - The queries themselves (for the leaderboard "example queries" column)
      - The mean age when flagged
      - Citation count (as a proxy for how widely it's still being cited)

    Returns list of paper dicts sorted by stale_appearance_count descending.
    """
    paper_stats = defaultdict(lambda: {
        "title": "",
        "year": 0,
        "citation_count": 0,
        "paper_id": "",
        "authors": [],
        "stale_count": 0,
        "contradicted_count": 0,
        "example_queries": [],
        "mean_ages_seen": [],
    })

    for row in log_rows:
        verdict = row.get("verdict", "")
        query   = row.get("query", "")
        mean_age = row.get("mean_age_months", 0) or 0

        try:
            papers = json.loads(row.get("papers_json", "[]"))
        except Exception:
            continue

        for p in papers:
            pid = p.get("paper_id", "") or p.get("title", "")
            if not pid:
                continue

            stats = paper_stats[pid]
            stats["title"]          = p.get("title", "") or stats["title"]
            stats["year"]           = p.get("year", 0) or stats["year"]
            stats["citation_count"] = max(
                stats["citation_count"],
                p.get("citation_count", 0) or 0,
            )
            stats["paper_id"]  = p.get("paper_id", "") or stats["paper_id"]
            stats["authors"]   = p.get("authors", []) or stats["authors"]
            stats["mean_ages_seen"].append(mean_age)

            if verdict == "STALE":
                stats["stale_count"] += 1
                if query and query not in stats["example_queries"]:
                    stats["example_queries"].append(query)
            elif verdict == "CONTRADICTED":
                stats["contradicted_count"] += 1
                if query and query not in stats["example_queries"]:
                    stats["example_queries"].append(query)

    # Filter: only papers that appeared in at least one stale/contradicted query
    filtered = {
        pid: stats for pid, stats in paper_stats.items()
        if stats["stale_count"] + stats["contradicted_count"] > 0
        and stats["title"]
        and stats["year"] > 0
    }

    # Sort by total flagged appearances, then by citation count
    sorted_papers = sorted(
        filtered.values(),
        key=lambda s: (
            s["stale_count"] + s["contradicted_count"],
            s["citation_count"],
        ),
        reverse=True,
    )

    return sorted_papers


def format_authors(authors: list) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al."


# ── Markdown generation ───────────────────────────────────────────────────────

def generate_markdown(
    papers: list[dict],
    total_queries: int,
    stale_queries: int,
    contradicted_queries: int,
    generated_at: str,
) -> str:
    """
    Generate the leaderboard markdown from aggregated paper stats.
    """
    lines = []

    lines += [
        "# RECON — Most Superseded ML Papers Leaderboard",
        "",
        "> **What this is:** Papers that RECON's staleness critic flagged most frequently",
        "> as potentially outdated evidence across real researcher queries.",
        ">",
        "> **How it's generated:** Every query run through RECON is logged. When the critic",
        "> fires a STALE or CONTRADICTED verdict, the retrieved papers are recorded.",
        "> This leaderboard aggregates those logs — every entry is backed by real queries",
        "> from real researchers, not pre-written content.",
        ">",
        "> **What it means for a paper to appear here:** RECON retrieved it in response to",
        "> a researcher's question, but the critic determined the evidence was outdated —",
        "> meaning newer, higher-scoring papers were available on the same topic.",
        "> This does not mean the paper is wrong. It means it may no longer represent",
        "> the current state of the field for the query it was retrieved for.",
        "",
        "---",
        "",
        "## Usage Statistics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total queries logged | {total_queries} |",
        f"| Queries with STALE verdict | {stale_queries} |",
        f"| Queries with CONTRADICTED verdict | {contradicted_queries} |",
        f"| Unique papers flagged | {len(papers)} |",
        f"| Leaderboard generated | {generated_at} |",
        "",
        "---",
        "",
    ]

    if not papers:
        lines += [
            "## No data yet",
            "",
            "RECON hasn't accumulated enough query data to generate a meaningful leaderboard.",
            "Run more queries through the system — every query contributes.",
            "Check back after 20+ queries have been run.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## Papers Most Frequently Flagged as Stale",
        "",
        "Ranked by number of distinct researcher queries in which this paper was retrieved",
        "and the critic flagged STALE or CONTRADICTED evidence.",
        "",
        "| Rank | Paper | Year | Authors | Times Flagged | Stale | Contradicted | Example Query |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for i, p in enumerate(papers[:50], 1):  # cap at 50 rows
        title         = p["title"][:70] + "..." if len(p["title"]) > 70 else p["title"]
        authors       = format_authors(p["authors"])
        year          = p["year"]
        stale         = p["stale_count"]
        contradicted  = p["contradicted_count"]
        total_flagged = stale + contradicted
        example_query = p["example_queries"][0][:60] + "..." if p["example_queries"] else "—"
        # Clean pipe characters from text to avoid breaking markdown tables
        title         = title.replace("|", "—")
        example_query = example_query.replace("|", "—")

        lines.append(
            f"| {i} | **{title}** | {year} | {authors} "
            f"| {total_flagged} | {stale} | {contradicted} | *{example_query}* |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "**Staleness does not mean wrong.** A paper flagged here was retrieved in response",
        "to a researcher's question and was found to be older than the evidence threshold",
        "(mean paper age > 24 months). Newer work on the same topic existed in the",
        "Semantic Scholar index at query time.",
        "",
        "**The leaderboard grows with usage.** Every query through RECON contributes.",
        "After 100+ queries across diverse ML topics, the leaderboard becomes a",
        "meaningful signal about which papers researchers are still citing that",
        "may have been superseded.",
        "",
        "**Coverage is query-driven.** Papers appear here because researchers asked",
        "questions that retrieved them. Subfields that aren't queried won't appear.",
        "This is a feature, not a bug — it reflects actual research patterns.",
        "",
        f"*Generated by [RECON](https://github.com/MukulRay1603/recon) · {generated_at}*",
    ]

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("RECON — Leaderboard Generator")
    print("=" * 55)

    init_db()

    # Load all STALE and CONTRADICTED rows from verdict_log
    stale_rows       = query_verdict_log(verdict_filter=["STALE"], limit=1000)
    contradicted_rows = query_verdict_log(verdict_filter=["CONTRADICTED"], limit=1000)
    all_flagged_rows = stale_rows + contradicted_rows

    # Total query count (all verdicts, for statistics)
    all_rows = query_verdict_log(verdict_filter=None, limit=5000)
    total_queries = len(all_rows)

    print(f"Total queries in verdict_log: {total_queries}")
    print(f"STALE verdicts:               {len(stale_rows)}")
    print(f"CONTRADICTED verdicts:        {len(contradicted_rows)}")

    if not all_flagged_rows:
        print("\n⚠ No STALE or CONTRADICTED verdicts in verdict_log yet.")
        print("  Run some queries through RECON first, then re-run this script.")
        print("  Even a small number of queries will produce a starter leaderboard.")

    # Aggregate paper appearances
    papers = aggregate_stale_papers(all_flagged_rows)
    print(f"Unique papers flagged:        {len(papers)}")

    if papers:
        print("\nTop 10 most frequently flagged papers:")
        for i, p in enumerate(papers[:10], 1):
            total = p["stale_count"] + p["contradicted_count"]
            print(f"  {i:2d}. [{total}x] {p['title'][:60]} ({p['year']})")

    # Generate markdown
    generated_at = datetime.utcnow().strftime("%B %Y")
    md = generate_markdown(
        papers=papers,
        total_queries=total_queries,
        stale_queries=len(stale_rows),
        contradicted_queries=len(contradicted_rows),
        generated_at=generated_at,
    )

    # Write output
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n✅ Leaderboard written → {OUTPUT_MD}")
    print(f"   {len(papers)} papers · {len(md)} chars")

    if total_queries < 20:
        print(
            f"\n💡 Only {total_queries} queries logged so far. "
            "Run more queries through RECON to get a richer leaderboard.\n"
            "   Every query through the Gradio UI or API contributes automatically."
        )
    else:
        print("\n   Embed in README:")
        print("   [Superseded Papers Leaderboard](docs/superseded_leaderboard.md)")


if __name__ == "__main__":
    main()