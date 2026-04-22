# CHANGELOG — RECON v2 (Edge Reliability)

## [Unreleased]

### Phase 2 — Edge Reliability Scoring + OpenAlex Augmentation
- 2.1: Created src/openalex_utils.py
  - `search_openalex(query, max_results=5)` — keyword search, returns Paper-compatible dicts
  - `get_openalex_by_doi(doi)` — singleton DOI lookup for citation enrichment
  - `get_citation_centrality(doi, citation_count)` — `min(1.0, log1p(count) / log1p(10000))`
  - Abstract reconstruction from OpenAlex inverted index format
  - DOI URL stripping (`https://doi.org/` prefix removed)
  - API key auth via `OPENALEX_API_KEY` env var; graceful degradation without key
- 2.2: Added DOI extraction to retriever_utils.py
  - Added `"externalIds"` to Semantic Scholar API field list
  - Extracts `doi = (r.get("externalIds") or {}).get("DOI", "") or ""`
  - Passes `doi=doi` to Paper constructor
  - Added `doi: str = ""` field to Paper dataclass in state.py (after references, before hybrid_score)
- 2.3: OpenAlex augmentation in retriever.py
  - Triggered when S2 retrieval returns < 12 papers
  - Searches top 2 sub-questions, max 3 results each
  - Deduplicates by DOI and paper_id before merging
  - OpenAlex papers get hybrid_score from semantic sim (0.3) + authority + recency
- 2.4: Created src/reliability.py — three-signal paper reliability scorer
  - Signals: centrality (40%), recency (30%), coherence (30%)
  - Centrality: `min(1.0, log1p(citation_count) / log1p(10000))`
  - Recency: `max(0, 1 - (age_years / 10))`
  - Coherence: batched LLM call (one call for all papers) — scores 0–1 relevance to query
  - Dominant signal labels: FOUNDATIONAL / CURRENT / DECLINING / SUPERSEDED
  - FOUNDATIONAL: `centrality >= 0.60 AND (coherence >= 0.65 OR coherence == 0.0)` — coherence passthrough for LLM-off case
  - Returns `{paper_id: ReliabilityScore}` dict
- 2.5: Wired reliability scorer into critic.py
  - `score_papers()` called after high-score paper check
  - STALE threshold changed from `mean_age > 24` to `mean_reliability < 0.40`
  - Falls back to age-based threshold if scorer returns empty
  - `paper_reliability_scores` added to ResearchState TypedDict
  - Scores serialised to `{pid: rs.__dict__}` in all non-INSUFFICIENT critic return paths
- 2.6: Added per-paper trust summary to synthesizer output
  - Reads `paper_reliability_scores` from state after synthesis
  - Appends `## Evidence Trust Summary` block to `synthesized_position`
  - Color-coded signal labels: HIGH (FOUNDATIONAL/CURRENT), MEDIUM (DECLINING), LOW (SUPERSEDED)
  - Trust summary included in `export_md` via `_build_export_md`
- Warning fix: `_headers()` in openalex_utils.py downgraded from WARNING to DEBUG when API key absent (was spamming 14+ warnings per query during centrality scoring)
- Eval results (recon_linear_v2_full, 130Q):
  - Staleness (STALE verdict): 32.3% (42/130) — down from 52% v1; reliability-based threshold is more precise, fewer false-positive STALE calls
  - Contradiction catch rate: 0% (retriever still returns adjacent-topic papers, not opposing-camp pairs — known gap, deferred)
  - Position accuracy (MATCH): 44.6% (58/130) — unchanged vs v1 43.9%

### Phase 1 — Integrity + Critic Fix
- 1.1: Archived patch_contradiction.py to eval/archived/ with README
- 1.2: Fixed critic_node — STALE and CONTRADICTED checks now run in parallel
  - Removed short-circuit: STALE no longer blocks CONTRADICTED
  - When both fire, CONTRADICTED wins (richer signal)
  - Removed **state spread from all return paths for consistency
  - All 5 return paths now have identical key shape
  - Added missing retry_count to FORCED_PASS path
- 1.3: Ran full 130Q eval on v2 recon_linear
  - Staleness catch rate: 48% (vs 52% v1 — within run-to-run variance, confirmed same methodology)
  - Contradiction catch rate: 0% (root cause: retriever finds adjacent-topic papers, not opposing-camp pairs; gap filter also too coarse for ML field — 2yr filter blocks most recent paper pairs)
  - Position accuracy (MATCH): 44.6% vs v1 43.9%
  - Cat C has no ground truth in ground_truth.json — contradiction_caught is critic-behavior only
  - Gap filter lowered to 1yr and tested on all 30 Cat C questions — reverted, no improvement (CONTRADICTED still 0/30); problem is in the retriever, not the filter

### Phase 0 — Branch Setup
- Created v2-edge-reliability branch from main
- Added CHANGELOG.md
- v1 remains on main branch, deployed to HF Spaces
- All v2 development happens on this branch
