# RECON — Full Project Context & Research Direction
**Last updated:** April 2026  
**Purpose:** Complete handoff document for Claude Code sessions. Contains everything discussed across the research planning conversation — architecture, eval results, collaboration context, professor feedback, and the revised technical direction.

---

## 1. What RECON Is

RECON is a four-agent LangGraph state machine for temporally-aware scientific literature retrieval. Its defining contribution is treating **temporal supersession** as a first-class retrieval failure mode — something no existing RAG evaluation framework (RAGAS, ARES, TREC RAG) currently measures.

**The core argument:** A 2019 paper with 800 citations scores high on cosine similarity and high on authority. If a 2023 paper explicitly refutes its claims, retrieving the 2019 paper as evidence produces a confident but stale answer. Standard RAG has no mechanism to detect this. RECON's critic does.

**Repos:**
- GitHub: https://github.com/MukulRay1603/project-recon
- HF Space (live, Gradio): https://huggingface.co/spaces/MukulRay/recon

---

## 2. Architecture (Source-Verified)

```
session_loader → planner → retriever → critic
                                           │
                        ┌──────────────────┴──────────────────┐
                        │ PASS / FORCED_PASS                   │ STALE / CONTRADICTED / INSUFFICIENT
                        ▼                                      ▼
                   synthesizer → END              retry_retriever → critic (max 2 retries)
```

### Agent Responsibilities

**session_loader** (`graph.py:51`)
- Loads prior session context from SQLite before planner runs
- Fails silently — pipeline continues if session load fails

**planner** (`planner.py`)
- LLM: Groq LLaMA 3.3-70B, temperature 0.2
- Decomposes query into 2–3 temporally-typed sub-questions: foundational / recent / contested
- Session-aware: injects last 3 prior queries to avoid repetition
- Fallback: uses raw query if LLM output unparseable

**retriever** (`retriever.py` + `retriever_utils.py`)
- Semantic Scholar REST API via direct `requests.get()` to `graph/v1/paper/search`
- `sleep(3)` rate limit guard per S2 call
- Each paper scored: `hybrid_score = semantic_sim × 0.5 + recency × 0.3 + authority × 0.2`
- Three recency decay configs: `none` / `linear` / `log` (parameterized via `decay_config` state field)
- Linear decay: `max(0, 1 − age/20)` where age = current_year − paper_year
- Log decay: `max(0, 1 − log(1+age)/log(21))`
- Authority: `min(1.0, log(1+citations)/log(10001))` (log-normalized)
- DuckDuckGo web search in parallel (`ddgs`, `region="wt-wt"`)
- Tavily fallback if DDG fails
- Results cached to `data/cache/{md5_hash}.json`
- On HF Spaces: cache dir is `/tmp/recon_cache` via `RECON_CACHE_DIR` env var

**critic** (`critic.py`)
- LLM: Groq LLaMA 3.3-70B, temperature 0.1
- Verdict order (sequential, first match wins):
  1. `FORCED_PASS` — retry_count ≥ 2 (hard ceiling)
  2. `INSUFFICIENT` — fewer than 3 papers retrieved
  3. `INSUFFICIENT` — fewer than 3 papers with hybrid_score ≥ 0.40
  4. `STALE` — mean paper age > 24 months
  5. `CONTRADICTED` — LLM pairwise check on top-4 papers, only pairs with ≥ 2yr gap
  6. `PASS` — all checks clear
- On non-PASS: LLM rewrites sub-questions with strategy: `broaden` / `recent` / `probe_contradiction`
- Rewritten questions stored in state as `rewritten_questions`
- `calibration_bin` field set to verdict for eval aggregation

**retry_retriever** (`graph.py:74`)
- Uses `rewritten_questions` from critic instead of `sub_questions` from planner
- Merges new papers with existing set — deduplication by `paper_id`
- Re-sorts merged set by `hybrid_score` descending
- Merges web results by URL deduplication
- Increments `retry_count`

**synthesizer** (`synthesizer.py`)
- LLM: Groq LLaMA 3.3-70B, temperature 0.3
- Produces four-section brief: Overview / Key Findings / Active Debates / Outlook
- Per-claim confidence scoring: HIGH / MEDIUM / LOW based on hybrid_score + year
- Inline citations formatted as `[Author et al., Year]`
- Calls `log_verdict()` on every completed run — populates `verdict_log` table
- Calls `save_turn()` — persists query + position + claims to `session_turns` table
- Generates `export_md` field — full session as downloadable markdown

### State Schema (`state.py`)
```python
class ResearchState(TypedDict):
    original_query: str
    session_id: str
    session_context: Optional[SessionContext]
    sub_questions: list[str]
    retrieved_papers: list[Paper]
    citation_graph: dict                    # {paper_id: [cited_ids]}
    web_results: list[WebResult]
    critic_verdict: str                     # PASS/STALE/CONTRADICTED/INSUFFICIENT/FORCED_PASS
    critic_notes: str
    rewritten_questions: list[str]
    retry_count: int
    synthesized_position: str
    claim_confidences: list[Claim]
    session_update: Optional[SessionUpdate]
    export_md: str
    decay_config: str                       # "none" | "linear" | "log"
    calibration_bin: str
    latency_ms: float
```

### Verdict Constants
```
PASS | STALE | CONTRADICTED | INSUFFICIENT | FORCED_PASS
```

---

## 3. Database Schema

SQLite at `data/sessions.db` locally, `/tmp/recon_sessions.db` on HF Spaces.

**sessions:** `session_id TEXT PK, created_at TEXT, updated_at TEXT`  
**session_turns:** `id INTEGER PK, session_id FK, query TEXT, position TEXT, claim_json TEXT`  
**verdict_log:** `id INTEGER PK, session_id FK, query TEXT, verdict TEXT, retry_count INTEGER, decay_config TEXT, latency_ms REAL, timestamp TEXT`

---

## 4. Evaluation Results (Real, Source-Verified)

| Architecture | Staleness Catch Rate | Position Accuracy |
|---|---|---|
| Single-pass RAG (baseline) | 0% | 32.3% |
| RECON no decay | 42% | 38.1% |
| RECON log decay | 38% | 36.7% |
| **RECON linear decay** | **52%** | **43.9%** |

- Benchmark: 130 questions sourced from real ML survey paper supersession chains
- Evaluation method: LLM-as-judge via Groq on 5 architecture variants with backoff
- Contradiction catch rate: 0% — known limitation, acknowledged honestly in results
- Contradiction detection inflated by post-hoc heuristics was discovered and dropped (E8 in blueprint)

---

## 5. Tech Stack Rules (Never Break These)

| Rule | Detail |
|---|---|
| S2 API | Direct `requests.get()` to `graph/v1/paper/search` ONLY. Never `semanticscholar` library |
| DDG package | `ddgs` (not `duckduckgo-search`). Always `region="wt-wt"` |
| Gradio | NOT in `requirements.txt`. Only in `sdk_version: 6.10.0` in README YAML |
| SQLite path | `data/sessions.db` locally, `/tmp/recon_sessions.db` on HF via `SESSION_DB_PATH` env var |
| Cache dir | `data/cache/` locally, `/tmp/recon_cache` on HF via `RECON_CACHE_DIR` env var |
| Python | 3.12 locally (3.13 on HF) |
| Verdict strings | `PASS / STALE / CONTRADICTED / INSUFFICIENT / FORCED_PASS` — exact case everywhere |
| Eval numbers | Real only. 52% staleness catch rate, 43.9% position accuracy |

---

## 6. Novelty Assessment

### What is genuinely novel
- **Staleness catch rate** as a formal RAG evaluation metric. No existing framework (RAGAS, ARES, TREC RAG) measures temporal supersession as a failure mode.
- **Four-verdict critic with failure-mode-specific query rewriting.** CRAG (2024) has three verdicts based on relevance (Correct/Incorrect/Ambiguous) — it does not detect temporal supersession. RECON's STALE verdict is a distinct concept from relevance failure.
- **130-question superseded-claims benchmark.** No public benchmark targets ML literature staleness detection.

### What is not novel
- Multi-agent LangGraph pipelines — common pattern
- Hybrid retrieval scoring (semantic + recency + authority) — exists in literature
- LLM-as-judge for RAG evaluation — standard practice

### Closest related work
- **CRAG (Jan 2024)** — corrective RAG with relevance evaluator. No temporal supersession concept.
- **TG-RAG (Oct 2025)** — temporal GraphRAG. Requires pre-existing timestamps on graph edges. NASA KG has none.
- **T-GRAG (Aug 2025)** — dynamic GraphRAG for temporal conflicts. Same limitation as TG-RAG.
- **DEAN (Feb 2024)** — outdated fact detection in KGs using structural contrastive learning. No LLM critic, no RAG connection, no scientific domain application.
- **FOS Benchmark (Nov 2025)** — temporal scientific graph benchmark for interdisciplinary link prediction. Different task.

---

## 7. The NASA EO Knowledge Graph — Professor's Dataset

**HF Dataset:** https://huggingface.co/datasets/nasa-gesdisc/nasa-eo-knowledge-graph  
**HF Model (GNN):** https://huggingface.co/nasa-gesdisc/edgraph-gnn-graphsage  
**HF Publications Dataset:** https://huggingface.co/datasets/nasa-gesdisc/es-publications-researchareas  
**DOI:** 10.57967/hf/3463  
**Version:** v1.2.0, October 2025  
**Total nodes:** 150,351  

### The Seven Node Types
| Node Type | What It Represents |
|---|---|
| Dataset | Satellite/EO datasets from NASA DAACs + 184 providers |
| Publication | Scientific papers citing those datasets |
| ScienceKeyword | GCMD-controlled vocabulary tags (Ozone, Precipitation, etc.) |
| Instrument | Sensors used to collect data (AIRS, MODIS, etc.) |
| Platform | Satellites carrying instruments (Aqua, Terra, etc.) |
| Project | Scientific missions (MERRA-2, etc.) |
| DataCenter | NASA DAACs and affiliated institutions |

### Known Schema Facts (from dataset card)
- Every node has properties: `globalId`, `doi`, `pagerank_global`, and node-type-specific fields
- Publication nodes have: `title`, `authors`, `year`, `doi`, `abstract`, `url`
- **Relationship properties are null across all types** — edges carry no weight, no timestamp, no metadata
- Available formats: JSON (JSONL), GraphML, Cypher (Neo4j)

### The GNN Model
- Architecture: Heterogeneous GraphSAGE (PyTorch Geometric)
- Base embeddings: `nasa-impact/nasa-smd-ibm-st-v2` (fine-tuned)
- Task: Link prediction for missing Dataset → ScienceKeyword edges
- Purpose: Find datasets in the archive that are missing keyword tags they should have

### The Companion Publications Dataset
- ~12K GES-DISC citing publications, each classified into 20 applied research areas
- Built by fine-tuning NASA's own LLM on labeled abstracts
- 87% classification accuracy into research areas
- This is the supervision signal for the GNN

### The Core Gap (Our Identified Problem)
The graph knows **what** is connected. It does not know **whether those connections are still trustworthy today.**

- All Dataset → Publication edges are treated identically regardless of publication year
- A 2008 paper and a 2024 paper carry the same weight during GNN message passing
- The GNN's keyword predictions are being shaped by stale signal
- The graph has no mechanism to distinguish a foundational citation from an outdated one

The professor's own published work (NTRS 2024) explicitly identifies dataset version supersession as an open problem: *"Datasets undergo a life cycle where older versions are replaced by newer versions... It is challenging when publications citing a dataset need to be traced over the entire lifecycle."* — This is the gap we are filling.

---

## 8. The Collaboration — Prof Armin Mehrabian

**Who:** NLP expert, NASA GES-DISC researcher. Main contributor on `nasa-eo-knowledge-graph` (commit author: `arminmehrabian`). The professor who shared the dataset link with Mukul in class.

**Status:** Active, interested. Said "let us work on it and see if we can improve and publish it" after being shown RECON.

**His feedback on RECON (received April 2026):**
After trying the HF Space, he sent five specific points:

1. *"I had no idea how to assess the validity of the results"* — output lacks explainability about WHY something was flagged. The verdict alone isn't enough for a domain expert to trust it.
2. *"Are you actually analyzing content for contradiction, or mostly relying on metadata?"* — Honest methodological question. Currently STALE is mostly metadata-driven (age). CONTRADICTED uses LLM pairwise check on abstracts.
3. *"I would be careful not to treat recency as validity. Older papers can be foundational."* — **The sharpest critique.** Pure age-based decay is too blunt. A 2003 paper with 10,000 citations is not stale.
4. *"Think in terms of edge reliability rather than just staleness."* — He reframed the concept. Not "is this paper old?" but "how much should we trust this edge right now?" This is a richer and more defensible framing.
5. *"Have you considered combining your signal with something like PageRank or citation centrality to preserve important older work?"* — He is pointing at `pagerank_global`, which already exists on every node in his graph. This is him telling us the answer.

---

## 9. The Revised Technical Direction — Edge Reliability Scoring

### The Pivot
**Old framing:** Staleness detection — flag papers that are old  
**New framing:** Edge Reliability Scoring — score how much each Dataset→Publication edge should be trusted right now

This survives the foundational paper objection. A 1998 paper with high PageRank and thousands of citations is reliable. A 2022 paper with 3 citations that contradicts it may be less reliable (or an emerging challenger — needs content signal to distinguish).

### The New Formula

```
edge_reliability(dataset_id, publication_id) =
    (citation_centrality × w1)
  + (recency_signal     × w2)
  + (content_coherence  × w3)
```

**citation_centrality** — `pagerank_global` from the professor's graph nodes, normalized to [0,1]. High PageRank = foundational = high reliability contribution regardless of age.

**recency_signal** — RECON's existing linear decay formula: `max(0, 1 − age/20)`. Now just one component, not the whole score. Tunable weight `w2`.

**content_coherence** — Does this paper's abstract still align with current scientific consensus on this topic? This is the LLM component. Query Semantic Scholar + OpenAlex for papers on the same topic published in the last 3 years, run a lightweight LLM check: "Does [older paper] make claims that are contradicted or superseded by [newer paper]?" Binary or scored output. This is the content analysis Armin asked about.

**Suggested starting weights:** w1=0.4, w2=0.3, w3=0.3 — to be ablated.

### What Changes in RECON

| Current RECON | Revised RECON (v2) |
|---|---|
| STALE = mean paper age > 24 months | STALE = low edge_reliability across all three signals |
| Recency is the primary signal | Recency is one of three weighted inputs |
| No use of citation network position | `pagerank_global` from graph feeds directly in |
| Contradiction is binary LLM check | Contradiction weighted by both papers' centrality |
| Output shows verdict only | Output shows which signal drove the verdict + network trust score |
| Metric: staleness catch rate | Metric: edge reliability precision + staleness catch rate (both) |

### The Explainability Fix
The prof said he couldn't assess the output. The fix: synthesizer adds a trust summary per paper:

```
[Smith et al., 2019]  
Network reliability: LOW  
Reason: low PageRank (0.12), newer work by Chen et al. 2023 addresses same claim with higher centrality
```

This gives a domain expert exactly what they need to sanity-check the verdict.

---

## 10. OpenAlex — The New Data Source

The professor mentioned OpenAlex during the in-person conversation as a better source than Semantic Scholar alone for Earth science papers.

**What it is:** Fully open scholarly database, 271M+ works, free REST API, CC0 license. Replacement for the discontinued Microsoft Academic Graph. Better coverage of Earth science journals and institutional repositories than Semantic Scholar.

**API:** `api.openalex.org/works?filter=doi:YOUR_DOI`  
Returns: title, year, cited_by_count, abstract, open_access status, referenced_works, citing_works

**Plan:** Run Semantic Scholar + OpenAlex in parallel, deduplicate by DOI, merge into unified paper pool before the hybrid scorer runs. This directly addresses the paywall concern raised in conversation.

### Retriever Stack (v2)
```
Query
  ↓
Semantic Scholar API  (~220M papers, strong for ML/CS)
  +
OpenAlex API          (~271M papers, strong for Earth science, fully open)
  +
CrossRef API          (DOI resolution, metadata fill-in for gaps)
  ↓
Merged, deduplicated by DOI
  ↓
RECON hybrid scorer   (semantic × 0.5 + recency × 0.3 + authority × 0.2)
  ↓
RECON critic v2       (PASS / STALE / CONTRADICTED / INSUFFICIENT)
  ↓
Synthesizer with trust summary per claim
```

---

## 11. Paper Strategy

### Paper 1 — RECON Standalone (arXiv, cs.IR)
**Owner:** Mukul (solo)  
**Status:** Ready to write. System is live. Eval is done.  
**Target:** arXiv cs.IR preprint — timestamps the contribution  
**Key ask from prof:** arXiv endorsement (first-time submitter needs it — he can click once)  
**Sections:**
1. Introduction — temporal supersession as an unaddressed RAG failure mode
2. Related Work — CRAG, Self-RAG, TG-RAG, DEAN — positioned clearly against each
3. System Design — four-agent architecture, four-verdict critic, decay ablation
4. Evaluation — 52% vs 0% result, 130Q benchmark, 5-architecture comparison
5. Limitations — 0% contradiction catch rate (honest), single domain, single LLM
6. Future Work — edge reliability extension, NASA KG application (seeds Paper 2)

**Important:** The revised "edge reliability" concept introduced from prof's feedback DOES NOT dilute Paper 1. Paper 1 is about the staleness catch rate metric and benchmark. Paper 2 is the graph extension.

### Paper 2 — Joint with Prof Armin
**Owner:** Prof Armin leads, Mukul co-authors  
**Target:** ECIR, SIGIR, or AI for Earth Science workshop  
**Contribution:** Edge reliability scoring for heterogeneous scientific knowledge graphs — no timestamps needed, scores derived from PageRank + recency + content coherence  
**Mukul brings:** The edge reliability mechanism, staleness scorer code, evaluation harness, OpenAlex integration  
**Prof brings:** Dataset access + credentials, GNN training infrastructure, domain expertise, institutional affiliation, venue selection  
**Authorship:** Prof first, Mukul second — this is his dataset, his domain. Second author on a NASA GES-DISC paper is a strong outcome at this stage.

---

## 12. Immediate Next Steps (Ordered)

### This week
- [ ] Integrate OpenAlex API as second retriever source in `retriever.py`
  - Endpoint: `api.openalex.org/works?filter=doi:X`
  - Merge with S2 results, deduplicate by DOI
  - Test on 20 questions from the existing benchmark

- [ ] Add `network_reliability` field to the synthesizer output
  - Show which signal (age / centrality / content) drove each verdict
  - This directly addresses prof's "I couldn't assess the output" feedback

- [ ] Upload 130Q benchmark as a public HF dataset
  - Dataset card: what it contains, how it was built, what it measures
  - Required before arXiv paper — paper cites it

### Next 2 weeks
- [ ] Prototype edge_reliability scorer
  - Input: `(publication_id, pagerank_score, year, topic_query)`
  - Output: reliability score [0,1] + which signal dominated
  - Does NOT require KG credentials yet — can test on publication nodes from the public JSON

- [ ] Ask prof for arXiv endorsement
  - Natural timing: after showing him the OpenAlex integration + trust summary in the UI
  - One sentence ask: "I'm planning to write this up for arXiv cs.IR — would you be okay endorsing my submission as a first-time submitter?"

- [ ] Write the RECON arXiv paper
  - 6–8 pages
  - Do NOT include Paper 2 material — keep it clean and focused on the standalone contribution
  - Mention edge reliability as future work

### Before Paper 2
- [ ] Get KG credentials from prof (needed to run GNN training with weighted edges)
- [ ] Ask technical questions about publication nodes:
  - Is the `year` field populated for all publication nodes?
  - Are DOIs resolvable to Semantic Scholar/OpenAlex for all publications?
  - What does `pagerank_global` represent exactly — is it global graph PageRank or something else?
- [ ] Build Earth science staleness benchmark from `es-publications-researchareas`
  - Find papers citing old dataset versions vs papers citing newer versions with improvement language
  - Target: ~50–100 verified supersession chains

---

## 13. Open Questions (Unresolved, Need Discussion)

**Technical:**
- What weight split between w1 (centrality), w2 (recency), w3 (content) performs best? Needs ablation.
- How to handle papers with no `pagerank_global` (new nodes, disconnected nodes)?
- Should content_coherence use the paper's abstract only, or full text where available (OpenAlex has some open access full text)?
- For a paper that is old BUT has high and *growing* citation trajectory — should that override age? Dynamic signal vs static snapshot?

**Research design:**
- Is the right evaluation metric "edge reliability precision" — i.e., do the edges we score LOW actually correspond to superseded science? Need ground truth for that.
- Can the 130Q ML benchmark be partially adapted to test the revised formula, or does it need a new benchmark?
- The contradiction catch rate is 0%. Armin's edge reliability framing might actually unlock this — a contradiction is just two edges pointing in opposite directions with conflicting content. Worth exploring.

**Collaboration:**
- What level of involvement does prof want in the RECON paper (Paper 1)? Acknowledgement only, or something more? Don't assume — ask when the time is right.
- What venues does he prefer for Paper 2? His prior work appeared at NTRS, AGU, ESIP — might prefer an Earth science venue over a pure IR venue.

---

## 14. Key People & Links Reference

| Item | Detail |
|---|---|
| Mukul's GitHub | https://github.com/MukulRay1603/project-recon |
| RECON HF Space | https://huggingface.co/spaces/MukulRay/recon |
| NASA KG Dataset | https://huggingface.co/datasets/nasa-gesdisc/nasa-eo-knowledge-graph |
| NASA GNN Model | https://huggingface.co/nasa-gesdisc/edgraph-gnn-graphsage |
| NASA Publications Dataset | https://huggingface.co/datasets/nasa-gesdisc/es-publications-researchareas |
| OpenAlex API docs | https://docs.openalex.org |
| Semantic Scholar API | https://api.semanticscholar.org/graph/v1/paper/search |
| CrossRef API | https://api.crossref.org/works/{doi} |
| CRAG paper (related work) | https://arxiv.org/abs/2401.15884 |
| TG-RAG paper (related work) | https://arxiv.org/abs/2510.13590 |
| DEAN paper (related work) | https://arxiv.org/abs/2402.03732 |
| Prof's NTRS paper on dataset lifecycle | https://ntrs.nasa.gov/citations/20240010838 |

---

## 15. How To Use This Document in Claude Code

When starting a new session in the RECON repo with Claude Code, paste this at the start:

> "Read RECON_CONTEXT.md in the project root for full context on what RECON is, where we are in the research, and what needs to be built next. Then do a code audit — read the key source files (graph.py, critic.py, retriever.py, retriever_utils.py, synthesizer.py, state.py) and tell me: (1) the current state of the codebase vs what's described in the context doc, (2) what's already implemented vs what's planned, (3) any discrepancies or things that look wrong."

That gives Claude Code everything it needs to do a proper audit and continue from exactly here.
