# RECON v2 — Master Context Document
**Last updated:** April 22, 2026  
**Version:** Post-feasibility audit, pre-implementation  
**Purpose:** Single source of truth for all RECON v2 development. Paste this into any new Claude session (Claude.ai or Claude Code). Supersedes the original RECON_MASTER_CONTEXT.md for all v2 work. The v1 doc remains the reference for the existing codebase.

---

## Part 1 — Project Identity

RECON is a multi-agent RAG system built with LangGraph that treats **temporal supersession** as a first-class retrieval failure mode in scientific literature. No existing RAG system or evaluation framework (RAGAS, ARES, TREC RAG) measures or targets this.

**The philosophical claim:** Every RAG system asks "what is relevant?" RECON asks "what should I trust, right now, and why?" That is a different kind of question. RECON is a prototype of a **temporal epistemology layer** for AI retrieval.

**Key links:**
- GitHub: https://github.com/MukulRay1603/project-recon
- HF Space (live, Gradio): https://huggingface.co/spaces/MukulRay/recon
- NASA KG Dataset: https://huggingface.co/datasets/nasa-gesdisc/nasa-eo-knowledge-graph
- NASA GNN Model: https://huggingface.co/nasa-gesdisc/edgraph-gnn-graphsage
- NASA Publications Dataset: https://huggingface.co/datasets/nasa-gesdisc/es-publications-researchareas

**Owner:** Mukul Ray (MukulRay1603 on GitHub, MukulRay on HF)  
**Collaborator:** Prof. Armin Mehrabian, NLP researcher, NASA GES-DISC

---

## Part 2 — Where v1 Stands (Audit-Verified)

### What is built and working
- Full 6-node LangGraph state machine: session_loader → planner → retriever → critic → synthesizer (with retry_retriever loop)
- 4 core agents: planner, retriever, critic, synthesizer
- Hybrid scoring: semantic_sim × 0.5 + recency × 0.3 + authority × 0.2
- Three decay configs: none, linear, log
- Eval harness: 5 architectures × 130 questions, LLM-as-judge, crash-resume
- Deployed on HF Spaces (Gradio UI)
- SQLite session memory with verdict logging

### Eval results (honest, source-verified)
| Architecture | Staleness Catch Rate | Position Accuracy |
|---|---|---|
| Single-pass RAG (baseline) | 0% | 32.3% |
| RECON no decay | 42% | 38.1% |
| RECON log decay | 38% | 36.7% |
| **RECON linear decay** | **52%** | **43.9%** |

- Contradiction catch rate: **0%** — honest result, caused by live bug (see below)
- Benchmark: 130 questions from real ML survey paper supersession chains

### Known bugs and integrity issues

**BUG 1 — STALE fires before CONTRADICTED (LIVE, PRODUCTION)**  
In `critic_node`, the verdict checks are sequential and first-match-wins. STALE (check 4: mean age > 24 months) fires before CONTRADICTED (check 5: LLM pairwise comparison). Any question with papers older than 2 years exits at STALE and never reaches the contradiction check. This is why contradiction catch rate is 0%. This is not an eval artifact — it happens to real users on the HF Space.

**INTEGRITY — patch_contradiction.py still exists**  
`eval/patch_contradiction.py` was documented as "dropped" in the original blueprint. It still exists and is fully functional. It includes a keyword-heuristic boost (`position_acknowledges_debate`) that can override an LLM verdict. If any reported numbers were generated with this file active, that is a research integrity risk. Must be archived before any paper submission. The 0% contradiction catch rate is the honest number.

**BUG 2 — Claim confidence is LLM-generated, not computed**  
The `_extract_claims()` function sends text to a second LLM call. The LLM assigns confidence based on rules ("high = multiple recent 2022+ papers agree") but actual `hybrid_score` values are never passed in. Confidence is an LLM judgment, not a numeric derivation. Not a bug per se, but must be described accurately in any paper.

**Inaccuracies corrected in v1 audit:**
- `retry_count` is incremented in critic, not retry_retriever
- Retriever is sequential, not parallel
- `verdict_log` schema had 4 wrong fields in original doc (corrected in v1 master context)

### What is described but NOT built
- OpenAlex integration (no `search_openalex()` function exists)
- Edge reliability scoring (formula designed, no code)
- `network_reliability` field in synthesizer output
- CrossRef API for DOI fill-in
- 130Q benchmark uploaded as public HF dataset

---

## Part 3 — The Publishability Problem

### Why v1 cannot survive peer review

The 52% staleness catch rate is real but the mechanism is vulnerable to a single devastating critique:

> "Your STALE verdict fires when mean paper age exceeds 24 months. Any script that checks `if avg_age > 2: return 'stale'` achieves the same result. Where is the intelligence in your system?"

This is correct. The STALE check is a metadata threshold. It does not distinguish between a foundational 2003 paper with 10,000 citations and a forgotten 2020 preprint with 5 citations. Both get the same treatment based on age alone.

CONTRADICTED is the verdict that adds real intelligence (LLM content comparison), but it never fires due to Bug 1.

### What makes it publishable

The system must demonstrate that it can:
1. Flag a 2019 paper with low centrality as stale (correct detection)
2. Keep a 2003 foundational paper with high centrality as reliable (correct non-detection)
3. Explain WHY each verdict was assigned (explainability)
4. Use content analysis, not just age (via the coherence signal)

This requires: Bug 1 fix + edge reliability formula + trust summary in output.

---

## Part 4 — Prof. Armin Mehrabian — Full Context

### Who he is
NLP researcher at NASA GES-DISC. Main contributor on the `nasa-eo-knowledge-graph` dataset (150K+ nodes, 7 entity types, trained GraphSAGE GNN). Currently Mukul's professor. Shared the dataset link in class.

### The relationship timeline
1. **In class:** Prof. shared the HF dataset link
2. **First email from Mukul:** Explained RECON, proposed connecting staleness signal to his graph at the edge level, mentioned OpenAlex, promised slides
3. **Prof's response (email, April 2026):** Tried the HF Space, gave five specific points of feedback (see below), closed with "looking forward to seeing the slides"
4. **Status now:** Active, willing collaborator. Expects slides as next artifact.

### His five feedback points — decoded

1. **"I couldn't assess the validity of the results"**  
   → Explainability problem. Output shows verdict but not reasoning. Needs per-paper trust summary showing which signal drove the verdict.

2. **"Are you analyzing content for contradiction, or mostly metadata?"**  
   → Honest question. STALE is metadata-driven. CONTRADICTED uses LLM on abstracts but barely fires (Bug 1). Answer must be honest.

3. **"Recency is not validity. Older papers can be foundational."**  
   → The sharpest critique. Pure age decay is too blunt. A 2003 paper with 10K citations is not stale. This is the core methodological gap.

4. **"Think in terms of edge reliability rather than staleness"**  
   → He renamed the concept. Better framing, stronger paper title, survives the foundational paper objection. This is the pivot.

5. **"Have you considered PageRank or citation centrality?"**  
   → He is pointing at `pagerank_global`, which already exists on every node in his graph. He is telling Mukul the answer to his own question #3.

### What he expects next
- Slides showing RECON's architecture, the staleness metric, and how it maps to his graph
- Visible progress in the HF Space (trust summary, better output)
- Concrete technical direction, not just plans

### Authorship and collaboration structure
- **Paper 1 (solo arXiv cs.IR):** Mukul only. RECON standalone with edge reliability using open citation data (OpenAlex cited_by_count). Acknowledge Armin for suggesting the reframe. Edge reliability on his graph = Future Work.
- **Paper 2 (joint):** Armin leads, Mukul co-authors. Edge reliability applied to NASA KG using `pagerank_global`. GNN edge weighting. Earth science benchmark. Target venue: Armin's preference (ECIR, SIGIR workshop, AGU, ESIP).
- **The boundary between papers:** Paper 1 uses publicly available citation centrality from OpenAlex. Paper 2 uses graph-native PageRank from his proprietary graph. This boundary is clean, honest, and strategically sound.

### Communication principles
- Never ask him for something before showing him something
- Every ask (endorsement, KG access) comes after demonstrated progress
- Conversational, question-driven approach — not formal pitches
- Endorsement ask happens when the draft is ready, attached to the slides

---

## Part 5 — The v2 Architecture

### From v1 to v2: what changes

```
v1 (current):
  session_loader → planner → retriever → critic → synthesizer → END
                                            ↕
                                     retry_retriever

v2 (target):
  session_loader → planner → retriever → critic → synthesizer → explorer → END
                                            ↕
                                     retry_retriever
```

**New agent: Explorer (Agent 5)**  
A Socratic exploration agent that reads the completed synthesis, identifies blind spots the user's query didn't cover, runs targeted searches for those angles, and returns findings the user didn't know to ask about. No published agentic RAG system does this — Speculative RAG pre-fetches anticipated follow-ups but doesn't generate novel questions.

**Upgraded agent: Critic (Agent 3)**  
Moves from single-threshold metadata checks to three-signal edge reliability scoring. No longer short-circuits; runs STALE and CONTRADICTED checks in parallel and combines signals.

**Upgraded agent: Synthesizer (Agent 4)**  
Adds per-paper trust summary showing reliability score, dominant signal, and one-line reasoning.

**Upgraded: Retriever (Agent 2)**  
Adds OpenAlex as a second source alongside Semantic Scholar. Merged by DOI deduplication.

**Upgraded: Follow-up System**  
Context inheritance for follow-up queries. Paper accumulation with bounds. Verdict evolution tracking.

### New state fields (additions to ResearchState)

```python
# Edge reliability
paper_reliability_scores: dict       # {paper_id: ReliabilityScore}
reliability_dominant_signals: dict   # {paper_id: "FOUNDATIONAL" | "CURRENT" | "DECLINING" | "SUPERSEDED"}

# Explorer
explore_mode: bool                   # User toggle, default False
explorer_findings: list[ExplorerFinding]

# Follow-up
is_followup: bool                    # Detected by planner
previous_papers: list[Paper]         # Carried from prior turn
previous_verdict: str                # For verdict evolution
previous_claims: list[Claim]         # For context
turn_number: int                     # Session turn counter
```

### New dataclasses

```python
@dataclass
class ReliabilityScore:
    score: float                     # [0, 1]
    centrality: float                # Normalized cited_by_count
    recency: float                   # Linear decay
    coherence: float                 # LLM content check
    dominant_signal: str             # FOUNDATIONAL / CURRENT / DECLINING / SUPERSEDED
    reason: str                      # One-line explanation

@dataclass
class ExplorerFinding:
    question: str                    # The blind spot question
    answer: str                      # One-paragraph answer
    sources: list[Paper]             # Supporting papers
    relevance_score: float           # Cosine sim to original query
```

### New files (minimizes merge conflicts with v1)

```
src/openalex_utils.py      — OpenAlex API integration
src/reliability.py         — Edge reliability scorer
src/agents/explorer.py     — Explorer agent
eval/archived/             — Directory for archived eval files
eval/archived/patch_contradiction.py  — Moved here with README
```

### Files edited (minimal, surgical changes)

```
src/agents/critic.py       — Reorder verdicts, use reliability scores
src/agents/synthesizer.py  — Add trust summary block
src/agents/planner.py      — Add follow-up detection
src/agents/retriever.py    — Add OpenAlex calls alongside S2
src/retriever_utils.py     — Add OpenAlex merge/dedup logic
src/state.py               — Add new fields
graph.py                   — Add explorer node + edge
app.py                     — Add explorer toggle + trust summary display
```

---

## Part 6 — Edge Reliability Scoring (The Core Upgrade)

### The formula

```
edge_reliability(paper, query) =
    (citation_centrality × 0.4)
  + (recency_signal     × 0.3)
  + (content_coherence  × 0.3)
```

### Signal definitions

**citation_centrality** (w1 = 0.4)  
Paper 1: `min(1.0, log1p(cited_by_count) / log1p(10000))` — from OpenAlex  
Paper 2: normalized `pagerank_global` from NASA KG — graph-native centrality  
High centrality = foundational = high reliability regardless of age.

**recency_signal** (w2 = 0.3)  
RECON's existing linear decay: `max(0, 1 - age/20)`  
Now one of three inputs, not the entire signal.

**content_coherence** (w3 = 0.3)  
LLM check: given 3 recent papers on the same topic (from last 3 years via OpenAlex search), ask "Does [older paper]'s central claim still represent current scientific understanding, or has it been superseded?"  
Returns: float [0, 1] where 1 = fully current, 0 = fully superseded.  
Batched: one LLM call for all papers being scored, not one per paper.

### Dominant signal labels

| Score Range | Label | Meaning |
|---|---|---|
| reliability ≥ 0.70 AND centrality ≥ 0.6 | FOUNDATIONAL | Old but highly cited, still current |
| reliability ≥ 0.60 AND recency ≥ 0.7 | CURRENT | Recent and well-supported |
| reliability 0.35–0.60 | DECLINING | Losing relevance, use with caution |
| reliability < 0.35 | SUPERSEDED | Outdated, newer work has replaced this |

### The foundational paper test (MUST PASS)

**Test case 1:** A 2003 paper with 10,000 citations on the same topic.  
Expected: centrality = ~0.92, recency = 0.0 (age 23), coherence = 0.8 (still cited approvingly)  
Reliability = 0.92×0.4 + 0.0×0.3 + 0.8×0.3 = 0.368 + 0 + 0.24 = **0.608 → CURRENT**  
Correctly NOT flagged as stale despite extreme age.

**Test case 2:** A 2020 paper with 5 citations, contradicted by a 2023 paper.  
Expected: centrality = ~0.12, recency = 0.7 (age 6), coherence = 0.1 (superseded)  
Reliability = 0.12×0.4 + 0.7×0.3 + 0.1×0.3 = 0.048 + 0.21 + 0.03 = **0.288 → SUPERSEDED**  
Correctly flagged despite being only 6 years old.

This test case pair is what makes the paper defensible. If the formula passes both, the metadata critique dies.

### Revised critic logic

```
OLD (v1):
  1. FORCED_PASS if retry_count >= 2
  2. INSUFFICIENT if len(papers) < 3
  3. INSUFFICIENT if < 3 papers with hybrid_score >= 0.40
  4. STALE if mean_age > 24 months          ← fires first, blocks #5
  5. CONTRADICTED if LLM detects conflict   ← never reached
  6. PASS

NEW (v2):
  1. FORCED_PASS if retry_count >= 2
  2. INSUFFICIENT if len(papers) < 3
  3. INSUFFICIENT if < 3 papers with hybrid_score >= 0.40
  4. Compute reliability scores for all papers
  5. Run contradiction check (regardless of age)
  6. Run reliability check (regardless of contradictions)
  7. COMBINE:
     - Both fire → CONTRADICTED (stronger signal)
     - Only contradiction → CONTRADICTED
     - Only low reliability → STALE
     - Neither → PASS
```

No more short-circuiting. Both checks always run. The richer signal wins.

---

## Part 7 — Explorer Agent (The New Agent)

### What it does

After the synthesizer completes, the explorer reads the synthesis and asks: "What did this answer miss that a thorough researcher would want to know?"

### Behavior

1. **Input:** synthesized_position + retrieved_papers + original_query
2. **LLM call 1:** Identify 2-3 blind spots — angles, contradictions, or related phenomena the query didn't cover but the evidence touches on
3. **Relevance gate:** Score each blind spot question against the original query using the embedding model (all-MiniLM-L6-v2). Drop any with cosine similarity < 0.3.
4. **For each blind spot:** Run targeted search (S2 + OpenAlex, 3 papers each)
5. **LLM call 2-3:** One-paragraph answer per blind spot with citations
6. **Output:** 2-3 ExplorerFinding objects appended to output

### Example

User query: "What are the latest approaches to KV Cache compression in large language models?"

Explorer finds:
- "How does KV cache compression interact with speculative decoding?"
- "Are there hardware-specific optimization approaches (GPU memory hierarchy) not covered in algorithmic papers?"
- "The 2024 paper by Zhang et al. claims 4× compression with no quality loss — has this been independently replicated?"

### Design constraints

- **Optional:** Toggle in Gradio UI, default OFF
- **Append-only:** Explorer CANNOT modify the synthesis. If it finds contradicting evidence, it flags it but does not alter the main output.
- **Bounded:** Max 2-3 blind spots. Max 3-4 LLM calls.
- **Progressive output:** Main synthesis shows immediately. Explorer findings stream in below as they complete.
- **Light model option:** Use Llama 3.1 8B for question generation (cheaper, 14,400 RPD on Groq). Use 70B only for answer synthesis.

### Why this is novel

Searched the agentic RAG landscape thoroughly. Speculative RAG pre-fetches anticipated follow-ups. Self-RAG reflects on retrieval quality. CRAG corrects bad retrievals. None generate novel questions that the user didn't ask, search for answers, and present unexpected but relevant findings. This is a genuinely new agent pattern in the RAG taxonomy.

---

## Part 8 — Follow-Up System Upgrade

### Current state (v1)
- session_turns stores query + position + claims
- Planner injects last 3 prior queries to avoid repetition
- Follow-up queries start essentially fresh — papers from prior turns are gone

### Target state (v2)

**Step 1 — Follow-up detection**  
Planner classifies incoming query as "new topic" or "follow-up" via LLM classification.  
Signals: pronouns ("what about..."), topic overlap, explicit references ("you mentioned...").  
Output: `is_followup` boolean in state.

**Step 2 — Context inheritance**  
On follow-up: carry forward top 8 papers, previous verdict, previous claims as planner context.  
Planner generates sub-questions targeting the GAP the follow-up implies, not rediscovering the whole topic.  
On new topic: clean slate (current behavior).

**Step 3 — Paper accumulation with bounds**  
Follow-up merges new papers with carried-forward papers, deduped by paper_id.  
Cap total at 15 papers (top by hybrid_score). Prevents context overflow.

**Step 4 — Verdict evolution**  
If previous verdict exists and new evidence changes the picture, critic explains what changed:  
"Previous: PASS. Updated: STALE — new retrieval found primary source has been superseded."

**Step 5 — Sliding window**  
Last 3 turns: full context (papers, claims, verdict).  
Older turns: compressed to query + verdict + top 3 claims only.  
Prevents token limit issues in sessions with 4+ turns.

---

## Part 9 — Feasibility Audit (Verified April 22, 2026)

### OpenAlex API — FEASIBILITY: GREEN (with caveat)

**Caveat: API key is now REQUIRED.** As of Feb 13, 2026, the polite pool / mailto parameter is deprecated. You must create a free account at openalex.org and use an API key.

**Free tier ($1/day):**
| Action | Daily Limit | Notes |
|---|---|---|
| Singleton lookup (by DOI) | Unlimited | This is what RECON needs most |
| List + filter | 10,000 calls | Searching by topic, year, etc. |
| Search (full-text) | 1,000 calls | Keyword/semantic search |
| Content download | 100 PDFs | Not needed for RECON |

**For RECON's use case:** Each query triggers ~3-6 OpenAlex searches (one per sub-question) + DOI lookups for enrichment. At 1,000 searches/day, RECON can handle ~166-333 queries/day. More than enough for both real-time use and full 130Q eval in a single day.

**Academic researchers:** Can request higher limits for free by emailing support@openalex.org. Worth doing once RECON is active.

**Implementation note:** Add `api_key` parameter to all requests. Store key in `.env` as `OPENALEX_API_KEY`. On HF Spaces, add as a secret.

### Semantic Scholar API — FEASIBILITY: GREEN

**Current RECON approach:** Uses `requests.get()` directly, likely unauthenticated.  
**Recommendation:** Get a free API key for dedicated 1 RPS rate (vs shared pool).  
**Rate limits:** Unauthenticated = shared pool ~100 req/5min. Authenticated = 1 RPS dedicated.  
**RECON's current `sleep(1)` between calls:** Already respects the 1 RPS limit. No changes needed.

### Groq LLM API — FEASIBILITY: YELLOW (binding constraint)

**Free tier (April 2026):**
| Constraint | Llama 3.3 70B | Llama 3.1 8B |
|---|---|---|
| Requests/minute | 30 | 30 |
| Tokens/minute | 6,000 | 6,000 |
| Requests/day | 1,000 | 14,400 |

**LLM calls per query in v2:**
| Agent | Calls | Model |
|---|---|---|
| Planner | 1 | 70B |
| Critic | 1 | 70B |
| Synthesizer | 2 (synthesis + claims) | 70B |
| Reliability coherence (batched) | 1 | 70B or 8B |
| Explorer (if enabled) | 3-4 | 8B for questions, 70B for answers |
| **Total without explorer** | **5** | |
| **Total with explorer** | **8-9** | |

**Impact on daily limits:**
- Without explorer: 1,000 RPD ÷ 5 calls = **200 queries/day** on 70B
- With explorer (using 8B for light tasks): 1,000 RPD on 70B handles ~200 queries. 8B handles the explorer question generation separately (14,400 RPD).
- **For eval runs:** 130 questions × 5-9 calls = 650-1,170 calls. Fits within 1,000 RPD for 70B if explorer is disabled during eval. If not, split eval across 2 days.

**TPM constraint:** 6,000 TPM is tight. Each call uses ~800-2,000 tokens. 5 calls back-to-back could require 4,000-10,000 tokens, exceeding the per-minute budget. Mitigation: the existing `sleep(1)` between retriever calls creates natural spacing. Add similar spacing between LLM calls if 429s appear.

**Mitigation strategies:**
1. Use 8B for lighter tasks (explorer question generation, coherence checks)
2. Batch content coherence into one call (all papers at once, not per-paper)
3. Cache reliability scores — same paper in same session gets cached score
4. Existing backoff logic in eval handles 429s — extend to all LLM calls
5. Developer tier (free with credit card): up to 10x higher limits

### HF Spaces — FEASIBILITY: GREEN

Free tier: 2 vCPU, 16GB RAM. All LLM inference is on Groq, not local. Local compute: only embedding model (all-MiniLM-L6-v2) + retriever logic. Well within limits.

HF Space deploys from `main` branch. The `v2-edge-reliability` branch does not affect the live deployment until explicitly merged.

### Embedding Model — FEASIBILITY: GREEN

`all-MiniLM-L6-v2` is loaded once globally. Used for: semantic similarity in hybrid scoring + relevance gate in explorer. No additional model needed. Inference is local, fast, and free.

---

## Part 10 — Risk Matrix and Fallback Plan

### Risk 1: Explorer generates irrelevant questions
- **Probability:** Medium-High
- **Impact:** Low (explorer is append-only, doesn't touch synthesis)
- **Fallback:** Relevance gate using cosine similarity to original query (drop < 0.3). Cap at 2-3 findings. Toggle is the ultimate fallback — if consistently bad, default to OFF.
- **Design principle:** Explorer CANNOT modify the synthesis. If it fails, core output is intact.

### Risk 2: OpenAlex API unreliable
- **Probability:** Low (well-funded, 1.5B calls/month infrastructure)
- **Impact:** Medium (no cited_by_count for centrality signal)
- **Fallback:** OpenAlex is additive, not replacing S2. If down, fall back to S2's `citationCount` field. Wrap every call in try/except with 5s timeout. Cache results alongside S2 cache.

### Risk 3: Edge reliability changes hurt eval numbers
- **Probability:** Medium
- **Impact:** High (paper's headline number could drop)
- **Fallback:** Run both v1 and v2 scoring in parallel. Record both metrics. If v2 catch rate drops because foundational papers correctly stopped being flagged, that is a PRECISION improvement — the paper argues this. Worst case: report both and frame as "v2 trades recall for precision in temporal assessment."

### Risk 4: Branch diverges too far from main
- **Probability:** Low (if new-files strategy is followed)
- **Fallback:** New functionality in new files. Minimal edits to existing files. Merge Phase 1 to main early to reduce divergence. Do NOT refactor existing code while adding features.

### Risk 5: Explorer adds too much latency
- **Probability:** High (3-4 extra LLM calls + API calls)
- **Impact:** Medium (users on HF Space may not wait)
- **Fallback:** Explorer is opt-in (default OFF). Show synthesis immediately, explorer streams in below. Cap at 2 blind spots. Use 8B for question generation (faster, higher RPD).

### Risk 6: Groq rate limits hit more often
- **Probability:** Medium-High
- **Impact:** Medium (429 errors, degraded experience)
- **Fallback:** Extend existing eval backoff to all LLM calls. Batch coherence checks. Cache reliability scores. Split eval across 2 days. Consider Developer tier (free with credit card, 10x limits).

### Risk 7: Follow-up context exceeds token limits
- **Probability:** Medium (after 3-4 turns)
- **Impact:** Medium (degraded synthesis quality)
- **Fallback:** Sliding window: full context for last 3 turns, compressed for older. Paper cap at 15. Explorer findings not carried forward. Hard limit: if estimated context > 80% of model window, force summary and reset.

### Risk 8: Critic rewrite breaks passing tests
- **Probability:** Medium
- **Impact:** High (regression in existing results)
- **Fallback:** Run 130Q benchmark before AND after every critic change. Diff verdicts question by question. Keep v1 logic as a function for A/B testing. Log all disagreements between v1 and v2 critic.

### Risk 9: Explorer contradicts synthesizer
- **Probability:** Low-Medium
- **Impact:** This is a FEATURE, not a bug
- **Handling:** Explorer findings are labeled supplementary. If contradicting, flag it: "Note: this finding presents evidence that differs from the main synthesis." Do NOT auto-update synthesis.

---

## Part 11 — Publication Strategy (Path C)

### Paper 1 — RECON Standalone (solo arXiv cs.IR)

**Title direction:** "RECON: Edge Reliability Scoring for Temporally-Aware Scientific Retrieval"

**Core argument:** Temporal supersession is a first-class retrieval failure mode. RECON detects it using a three-signal reliability formula (citation centrality + recency + content coherence) that distinguishes foundational from stale papers. The explorer agent proactively discovers blind spots in retrieved evidence.

**Centrality signal:** OpenAlex `cited_by_count` (publicly available, anyone can reproduce)  
**NOT:** `pagerank_global` from the NASA KG (reserved for Paper 2)

**Sections:**
1. Introduction — the temporal flatness problem in RAG
2. Related Work — CRAG, TG-RAG, T-GRAG, DEAN (all confirmed non-overlapping)
3. System Architecture — 7-node LangGraph, 5 agents
4. Edge Reliability Scoring — three-signal formula, foundational paper test
5. Explorer Agent — Socratic blind spot discovery
6. Evaluation — staleness catch rate, contradiction catch rate, position accuracy, calibration
7. Limitations — honest: single-domain benchmark, LLM-dependent coherence
8. Future Work — graph-native extension with heterogeneous KGs using pagerank_global

**Acknowledgment:** "We thank Prof. Armin Mehrabian (NASA GES-DISC) for suggesting the edge reliability reframe and for discussions on citation centrality that shaped the three-signal formula."

**Ask from Armin:** arXiv endorsement (first-time submitter)

### Paper 2 — Joint with Prof. Armin

**Title direction:** "Edge Reliability Scoring for Heterogeneous Scientific Knowledge Graphs"

**Centrality signal upgrade:** Replace `cited_by_count` with `pagerank_global` from NASA KG. Graph-native centrality is structurally richer than flat citation counts.

**Contributions:**
1. Edge reliability scorer applied to all Dataset → Publication edges in NASA KG
2. Staleness-weighted GraphSAGE training (edge weights in message passing, PyG SAGEConv supports natively)
3. Earth science supersession benchmark from dataset version chains (MERRA-2 v5 → v6, etc.)
4. Evaluation: weighted vs unweighted GNN on Dataset → ScienceKeyword link prediction

**Authorship:** Armin first, Mukul second. Clean split: he brings dataset, GNN, domain expertise, venue. Mukul brings the entire mechanism, code, eval harness, benchmark.

**Target venue:** Armin's preference — ECIR, SIGIR workshop, AGU, ESIP, or AI for Earth Science workshop.

### Papers 3+ (branching, future)
- Multi-domain benchmark suite (medical guidelines, legal precedent, climate policy)
- Predictive supersession (citation trajectory analysis → detect papers becoming stale before recognized)
- Temporal embedding fine-tuning (train `recon-temporal-mpnet` where year is a first-class embedding signal)

---

## Part 12 — Novelty Assessment (Verified via Literature Search)

### Genuinely novel in RECON v2
- **Edge reliability scoring** as a three-signal composite for RAG temporal assessment — no published system combines citation centrality + recency + LLM content coherence for trust scoring
- **Explorer agent** — proactive blind spot discovery in agentic RAG. No published system generates novel questions the user didn't ask and searches for answers
- **Staleness catch rate** as a formal RAG evaluation metric — no framework measures temporal supersession
- **Four-verdict taxonomy with failure-mode-specific query rewriting** — CRAG has relevance-based verdicts, not temporal ones
- **130-question superseded-claims benchmark** — no public benchmark targets this

### Not novel (don't claim)
- Multi-agent LangGraph pipelines
- Hybrid retrieval scoring (semantic + recency + authority)
- LLM-as-judge for RAG evaluation
- Using OpenAlex or S2 for paper retrieval

### Closest related work (all confirmed non-overlapping)

| Paper | What it does | Why RECON v2 is different |
|---|---|---|
| CRAG (Jan 2024) | Corrective RAG, relevance evaluator | No temporal supersession, no reliability scoring |
| TG-RAG (Oct 2025) | Temporal GraphRAG | Requires pre-existing edge timestamps — RECON infers them |
| T-GRAG (Aug 2025) | Dynamic GraphRAG temporal conflicts | Same timestamp requirement |
| DEAN (Feb 2024) | Outdated fact detection in KGs | Structural only, no LLM critic, no RAG, no citation centrality |
| FOS Benchmark (Nov 2025) | Temporal scientific graph benchmark | Predicts field pairings, not paper supersession |
| Speculative RAG | Pre-fetches anticipated follow-ups | Doesn't generate novel questions or discover blind spots |

---

## Part 13 — Implementation Checklist (Full, Ordered by Dependency)

### Phase 0 — Branch Setup (30 min)
```
□  Create branch v2-edge-reliability from main
□  Verify HF Space deploys from main only
□  Add CHANGELOG.md to branch
□  Copy this document to the branch as reference
```

### Phase 1 — Integrity + Critic Fix (3 days)
```
□  1.1  Archive patch_contradiction.py → eval/archived/ with README
□  1.2  Fix critic: run STALE + CONTRADICTED in parallel, combine signals
         (see Part 6, "Revised critic logic")
□  1.3  Re-run 130Q eval with fixed critic, record both catch rates
□  1.4  Merge Phase 1 to main (safe — improves live system)

GATE: patch_contradiction archived, contradiction catch rate nonzero
```

### Phase 2 — Edge Reliability (2 weeks)
```
□  2.1  Create src/openalex_utils.py
         - search_openalex(query) → list of papers
         - get_openalex_by_doi(doi) → enriched paper
         - API key from env var OPENALEX_API_KEY
         - try/except with 5s timeout, return empty on failure
□  2.2  Integrate OpenAlex into retriever.py (merge with S2, dedup by DOI)
□  2.3  Create src/reliability.py (three-signal scorer)
         - Must pass foundational paper test (Part 6)
□  2.4  Rewire critic to use reliability scores (Part 6, revised logic)
□  2.5  Add trust summary to synthesizer output (per-paper block)
□  2.6  Update Gradio UI for trust summary display
□  2.7  Full 130Q eval with edge reliability system
□  2.8  Test on 20 Earth science queries (Armin's domain)

GATE: foundational paper test passes, trust summary visible in UI
```

### Phase 3 — Explorer Agent (2 weeks)
```
□  3.1  Create src/agents/explorer.py (Part 7)
         - LLM call for blind spots
         - Relevance gate (cosine sim > 0.3)
         - Targeted search per blind spot
         - LLM answer synthesis
□  3.2  Add ExplorerFinding to state.py
□  3.3  Wire explorer into graph.py (conditional on explore_mode)
□  3.4  Add toggle to Gradio UI ("Deep exploration mode")
□  3.5  Update export_md for explorer findings

GATE: explorer generates relevant findings, toggle works, non-redundant with synthesis
```

### Phase 4 — Follow-Up System (1 week)
```
□  4.1  Follow-up detection in planner (LLM classification)
□  4.2  Context inheritance (papers, verdict, claims from prior turn)
□  4.3  Paper accumulation with cap (max 15)
□  4.4  Verdict evolution (explain what changed)
□  4.5  Sliding window (full context for 3 turns, compressed for older)

GATE: follow-ups build on prior context, no token overflow after 4+ turns
```

### Phase 5 — Paper 1 + Benchmark (3 weeks)
```
□  5.1  Upload 130Q benchmark to HF as public dataset with card
□  5.2  Test with second LLM backbone (Mixtral or similar)
□  5.3  Write Paper 1 (6-8 pages, cs.IR)
□  5.4  Email Armin with draft + updated HF Space link
         (see Part 14 for email template)
□  5.5  Request arXiv endorsement
□  5.6  Submit to arXiv cs.IR

GATE: paper submitted, benchmark public, Armin endorses
```

### Phase 6 — Paper 2 with Armin (6-8 weeks)
```
□  6.1  Get KG access from Armin
□  6.2  Replace cited_by_count with pagerank_global
□  6.3  Score all Dataset → Publication edges
□  6.4  Pipe edge weights into GraphSAGE training
□  6.5  Build Earth science supersession benchmark
□  6.6  Write Paper 2 jointly
□  6.7  Submit to venue Armin selects

GATE: paper submitted to agreed venue
```

### Phase 7 — Branching Out (ongoing)
```
□  7.1  Multi-domain benchmark suite
□  7.2  Predictive supersession (citation trajectory)
□  7.3  Temporal embedding fine-tune
□  7.4  RECON as drop-in critic layer (packaged module)
```

---

## Part 14 — Email to Armin (Send After Phase 2 Complete)

**Subject:** Re: RECON — progress update + slides incoming

```
Hi Prof. Armin,

Quick update on where things stand. I took your feedback
seriously and made real changes to the system, not just plans.

What's different since you last tried it:

• The staleness signal now uses three inputs instead of
  just age: citation centrality (from OpenAlex), recency
  decay, and an LLM content check. A highly cited older
  paper keeps its reliability score. The "foundational
  paper problem" you flagged is fixed.

• Every paper in the output now shows a trust summary —
  reliability score, dominant signal, and a one-line
  reason. You said you couldn't assess the validity of
  the results last time. Try it again and tell me if
  this helps.

• OpenAlex is in the retriever now alongside Semantic
  Scholar, merged by DOI. Better Earth science coverage.

• I also added an exploration mode — the system
  identifies angles the query didn't cover and searches
  for those too. Optional toggle.

The HF Space is updated:
huggingface.co/spaces/MukulRay/recon

I'm writing this up as a paper for arXiv (cs.IR). The
edge reliability reframe you suggested is central to
the contribution, acknowledged in the paper. As a
first-time submitter I'll need an endorsement — would
you be open to that once you've seen the draft?

Also putting together the slides you asked for. Will
send those over separately so you can look through
them before we next talk.

Best,
Mukul
```

---

## Part 15 — Tech Stack Reference

| Component | Technology | Version/Notes |
|---|---|---|
| Orchestration | LangGraph | State machine, 7 nodes in v2 |
| LLM (primary) | Llama 3.3 70B via Groq | Planner (t=0.2), Critic (t=0.1), Synthesizer (t=0.3) |
| LLM (light tasks) | Llama 3.1 8B via Groq | Explorer question gen, coherence checks |
| Embeddings | all-MiniLM-L6-v2 | sentence-transformers, loaded once globally |
| Paper API 1 | Semantic Scholar | ~220M papers, API key recommended |
| Paper API 2 | OpenAlex | ~271M papers, API key REQUIRED, free account |
| Web search | DuckDuckGo (primary) | ddgs library, Tavily fallback |
| Database | SQLite | Sessions, turns, verdict log |
| Deployment | HF Spaces | Gradio UI, 2 vCPU, 16GB RAM |
| Version control | GitHub | main = live, v2-edge-reliability = dev branch |
| Eval | Custom harness | 5 architectures, 130Q, LLM-as-judge, crash-resume |

### Environment variables needed for v2
```
GROQ_API_KEY=...              # Existing
OPENALEX_API_KEY=...          # NEW — get from openalex.org/settings/api
S2_API_KEY=...                # RECOMMENDED — get from semanticscholar.org/product/api
RECON_CACHE_DIR=/tmp/recon_cache  # Existing (HF Spaces)
```

---

## Part 16 — The NASA EO Knowledge Graph (Prof's Dataset)

### What it is
150,351-node heterogeneous knowledge graph of NASA's Earth Observation ecosystem.

### Seven node types
| Node | Count (approx) | Key Fields |
|---|---|---|
| Dataset | Varies | globalId, doi, pagerank_global |
| Publication | ~12K | title, authors, year, doi, abstract, url, pagerank_global |
| ScienceKeyword | Varies | GCMD-controlled vocabulary |
| Instrument | Varies | AIRS, MODIS, etc. |
| Platform | Varies | Aqua, Terra, etc. |
| Project | Varies | MERRA-2, etc. |
| DataCenter | Varies | NASA DAACs |

### Critical schema facts
- Every node has: `globalId`, `doi`, `pagerank_global` plus type-specific fields
- **Relationship properties are null across all edge types** — no timestamps, no weights
- This is the core gap: all edges are treated as equally current
- The GNN trains on all edges equally — a 2008 paper and a 2024 paper contribute the same

### The GNN model
- Architecture: Heterogeneous GraphSAGE (PyTorch Geometric)
- Base embeddings: `nasa-impact/nasa-smd-ibm-st-v2` (fine-tuned)
- Task: Link prediction for missing Dataset → ScienceKeyword edges
- `pagerank_global` already on every node — key for Paper 2's edge reliability work

### Access
- HF dataset is public for download
- Formats: JSON (JSONL), GraphML, Cypher
- KG credentials for direct access need to come from Armin

---

## Part 17 — Coordination Model with Armin

| Phase | Interaction | What Mukul Shows | What Mukul Asks |
|---|---|---|---|
| Phase 1-2 | Solo work, no coordination | Slides + updated HF Space after Phase 2 | Nothing yet |
| Phase 5 | One touchpoint | Paper 1 draft | arXiv endorsement |
| Phase 6 | Real collaboration begins | Full mechanism, code, eval harness | KG access, domain guidance, venue selection |
| Phase 7+ | Independent again | Papers 3+ | Co-authorship if interested |

**Core principle:** Never ask before showing. Every ask comes after demonstrated progress.

---

## Part 18 — How To Use This Document

### In Claude.ai (web chat)
Paste this at the start of a new conversation:
> "Read RECON_V2_MASTER_CONTEXT.md. This is the complete context for the RECON v2 project. It supersedes the v1 master context for all upgrade work. Use this as ground truth for what's built, what's planned, what the risks are, and what order to build things. When you reference anything, cite the Part number."

### In Claude Code
Place this file in the repo root. Reference it at session start:
> "Read RECON_V2_MASTER_CONTEXT.md before doing anything."

### What this document does NOT cover
- Line-by-line code walkthrough (see the v1 RECON_MASTER_CONTEXT.md for that)
- The full conversation history with Armin (summarized in Part 4)
- Alternative approaches that were considered and rejected (documented in prior Claude conversations)

---

## Part 19 — Version History

| Date | Change |
|---|---|
| April 22, 2026 | Initial v2 master context created. Incorporates: v1 code audit findings, Armin's 5-point feedback, edge reliability formula, explorer agent design, follow-up system plan, full feasibility audit (OpenAlex API key requirement, Groq rate limits, S2 limits), risk matrix with 9 identified risks, publication strategy (Path C: two papers with clean boundary), and complete implementation checklist across 7 phases. |
