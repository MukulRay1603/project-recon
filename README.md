---
title: RECON
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: true
license: mit
short_description: Multi-agent ML literature research with staleness detection
---

# RECON — Temporally-Aware Scientific Retrieval

A multi-agent RAG system that detects when retrieved scientific evidence has been superseded by newer work.

**Try it:** Enter any research question. RECON retrieves papers from Semantic Scholar and OpenAlex, scores their reliability using a three-signal formula (citation centrality + recency + content coherence), and flags stale or contradicted evidence before synthesizing an answer.

Each paper in the results shows a reliability label: FOUNDATIONAL, CURRENT, DECLINING, or SUPERSEDED.

**Evaluation:** 44% staleness catch rate on a 130-question benchmark of real scientific supersession chains. Single-pass RAG baseline: 0%.

---
Built by Mukul Ray | [GitHub](https://github.com/MukulRay1603/project-recon)
