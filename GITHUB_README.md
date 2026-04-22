# RECON -- Temporally-Aware Scientific Literature Retrieval

> A multi-agent RAG system that asks not just "what is relevant?" but "what should I trust, right now?"

**Live demo:** https://huggingface.co/spaces/MukulRay/recon  
**Status:** Active development -- v2 (edge reliability) deployed

---

## The Problem

Standard RAG retrieves the most semantically similar papers. It has no mechanism to detect when those papers have been superseded by newer work.

A 2019 paper with 800 citations scores high on cosine similarity and high on authority. If a 2023 paper explicitly refutes its central claims, retrieving the 2019 paper produces a confident but stale answer. RECON detects this -- and explains why.

---

## What's New in v2

v2 replaces the age-based staleness threshold with a three-signal **edge reliability formula**:

```
edge_reliability = (citation_centrality x 0.4)
                 + (recency_signal       x 0.3)
                 + (content_coherence    x 0.3)
```

A 2003 paper with 10,000 citations scores **FOUNDATIONAL** -- high centrality overrides age.  
A 2020 paper with 5 citations, superseded by newer work, scores **SUPERSEDED**.  
Pure age-based detection cannot make this distinction.

---

## Architecture

```
session_loader -> planner -> retriever -> critic -> synthesizer -> END
                                             |
                                      retry_retriever (max 2)
```

| Agent | Role |
|---|---|
| **Planner** | Decomposes query into 2-3 temporally-typed sub-questions (foundational / recent / open) |
| **Retriever** | Fetches papers from Semantic Scholar + OpenAlex, deduplicated by DOI. Hybrid scoring: semantic x 0.5 + recency x 0.3 + authority x 0.2 |
| **Critic** | Computes edge reliability scores, then issues verdict: PASS / STALE / CONTRADICTED / INSUFFICIENT / FORCED_PASS. On non-PASS: rewrites sub-questions with failure-specific strategy |
| **Synthesizer** | Four-section brief (Overview / Key Findings / Active Debates / Outlook) with per-claim citations and per-paper trust summary |

---

## Edge Reliability Scoring (`src/reliability.py`)

Each retrieved paper receives a `ReliabilityScore` with:

- `score` -- composite [0, 1]
- `centrality` -- `min(1.0, log1p(cited_by_count) / log1p(10000))` from OpenAlex
- `recency` -- `max(0, 1 - age/20)` linear decay
- `coherence` -- LLM batch check: does this paper's abstract still represent current consensus?
- `dominant_signal` -- `FOUNDATIONAL` / `CURRENT` / `DECLINING` / `SUPERSEDED`
- `reason` -- one-line explanation

The synthesizer appends a trust summary to every response so domain experts can verify verdict reasoning.

---

## Evaluation

130-question benchmark across three categories: consensus claims (Cat A), superseded claims (Cat B), contested claims (Cat C). Ground truth sourced from real ML survey paper supersession chains.

| Architecture | Staleness Catch Rate | Position Accuracy | False Positives |
|---|---|---|---|
| Single-pass RAG (baseline) | 0% | 32.3% | -- |
| Naive multi-agent | 0% | 44.6% | -- |
| RECON v1 (age-based STALE) | 52% | 43.9% | 8% |
| **RECON v2 (edge reliability)** | **44%** | **44.6%** | **2%** |

v2 trades some staleness recall for substantially lower false-positive rate. The reliability formula correctly preserves foundational papers that v1 would incorrectly flag as stale.

**Known limitation:** Contradiction catch rate is 0% -- the retriever returns topically adjacent papers rather than opposing-camp papers. This is a retrieval problem, not a critic problem. Addressed in future work.

---

## Repository Structure

```
src/
  agents/
    planner.py          -- query decomposition, temporally-typed sub-questions
    retriever.py        -- S2 + OpenAlex fetch, hybrid scoring, DOI dedup
    critic.py           -- edge reliability scoring, verdict logic, retry
    synthesizer.py      -- synthesis, trust summary, claim extraction
  openalex_utils.py     -- OpenAlex API (search, DOI lookup, citation centrality)
  reliability.py        -- three-signal edge reliability scorer
  retriever_utils.py    -- hybrid_score, recency_score, authority_score, S2 API
  state.py              -- ResearchState TypedDict, Paper/Claim dataclasses
  memory.py             -- SQLite session persistence
  graph.py              -- LangGraph state machine, node wiring
app.py                  -- Gradio UI
eval/
  run_eval.py           -- 5-architecture evaluation harness, LLM-as-judge
  questions.json        -- 130-question benchmark
  ground_truth.json     -- ground truth for Cat A/B
  results/              -- eval CSVs
  archived/             -- patch_contradiction.py (archived, not used in reported metrics)
```

---

## Setup

```bash
git clone https://github.com/MukulRay1603/project-recon
cd project-recon
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=...
OPENALEX_API_KEY=...   # free at openalex.org/settings/api
S2_API_KEY=...         # optional but recommended
TAVILY_API_KEY=...     # optional fallback web search
```

```bash
python app.py
```

---

## Tech Stack

| Component | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM | Llama 3.3 70B via Groq |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Paper APIs | Semantic Scholar + OpenAlex |
| Web search | DuckDuckGo (Tavily fallback) |
| Session memory | SQLite |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

---

## Author

Mukul Ray -- MS Applied ML, University of Maryland College Park  
GitHub: [@MukulRay1603](https://github.com/MukulRay1603)
