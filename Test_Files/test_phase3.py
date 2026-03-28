import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from src.retriever_utils import (
    search_semantic_scholar,
    search_web,
    build_citation_graph,
    hybrid_score,
    recency_score,
)

print("=== Phase 3: Retriever Utils ===\n")

# Test 1: Scoring functions
print("--- Scoring functions ---")
print(f"✓ recency linear year=2024: {recency_score(2024, 'linear'):.3f}")
print(f"✓ recency linear year=2019: {recency_score(2019, 'linear'):.3f}")
print(f"✓ recency log    year=2019: {recency_score(2019, 'log'):.3f}")
print(f"✓ recency none   year=2019: {recency_score(2019, 'none'):.3f}")
print(f"✓ hybrid score (sim=0.8, year=2023, citations=500): {hybrid_score(0.8, 2023, 500):.4f}")

# Test 2: Semantic Scholar
print("\n--- Semantic Scholar ---")
papers = search_semantic_scholar("KV cache compression LLM", limit=3)
if papers:
    for p in papers:
        print(f"  ✓ [{p.hybrid_score:.3f}] {p.title[:60]} ({p.year})")
else:
    print("  ⚠ No results (S2 key may not be active yet — expected)")

# Test 3: Web search
print("\n--- DuckDuckGo ---")
results = search_web("KV cache compression large language models 2024", limit=3)
if results:
    for r in results:
        print(f"  ✓ [{r.source}] {r.title[:60]}")
else:
    print("  ⚠ No results from DDG or Tavily")

# Test 4: Citation graph
print("\n--- Citation graph ---")
if papers:
    graph = build_citation_graph(papers)
    print(f"  ✓ Graph nodes: {len(graph)}")
    edges = sum(len(v) for v in graph.values())
    print(f"  ✓ Internal edges: {edges}")
else:
    print("  ⚠ Skipped (no papers retrieved)")

print("\n✅ Phase 3 complete")