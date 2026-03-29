---
title: RECON
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: true
license: mit
short_description: Multi-agent ML literature research with staleness detection
---

# RECON — Multi-Agent Research Navigator

**Temporally-aware ML literature research. Live Semantic Scholar. Staleness detection.**

RECON is a four-agent LangGraph system that retrieves live ML papers, evaluates evidence quality using a four-verdict critic, and synthesizes research positions with per-claim confidence scoring.

## What makes it different from standard RAG

Standard RAG retrieves the most semantically similar chunk with no mechanism to detect whether that chunk has been superseded. A 2019 paper cited 600 times and never contradicted is strong evidence. A 2019 paper that a 2023 paper explicitly refutes is weak evidence — regardless of its cosine similarity score. RECON's critic reasons about this distinction.

## Architecture

```
session_loader → planner → retriever → critic → synthesizer
                                           ↓ STALE/CONTRADICTED/INSUFFICIENT
                                     retry_retriever → critic (max 2x)
```

**Four agents:**
- **Planner** — decomposes query into temporally-typed sub-questions (foundational / recent / contested)
- **Retriever** — searches Semantic Scholar (200M+ papers) + DuckDuckGo with hybrid scoring
- **Critic** — four-verdict taxonomy: PASS / STALE / CONTRADICTED / INSUFFICIENT
- **Synthesizer** — structured position with inline citations and per-claim confidence

## Eval results (130-question ground truth dataset)

| Architecture | Position Acc | Staleness Catch | Latency |
|---|---|---|---|
| Single-agent RAG | 32.3% | 0% | 4.8s |
| Naive multi-agent | 44.6% | 0% | 23.9s |
| **RECON (linear decay)** | **43.9%** | **52%** | **17.1s** |

RECON catches 52% of superseded claims vs 0% for single-pass RAG on Category B questions sourced from real survey paper supersession chains.

## Tech stack

- **Orchestration:** LangGraph
- **LLM:** Groq / LLaMA 3.3-70B
- **Retrieval:** Semantic Scholar REST API + DuckDuckGo
- **Embeddings:** all-MiniLM-L6-v2
- **Session memory:** SQLite
- **Eval:** Ragas + LLM-as-judge

## GitHub

[github.com/MukulRay1603/project-recon](https://github.com/MukulRay1603/project-recon)