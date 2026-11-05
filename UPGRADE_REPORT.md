# RECON upgrade report — branch `upgrade/recon-2026-11`

Date: 2026-11-05. Not merged. The Hugging Face Space was not touched. The README results table is unchanged.

**Summary:** three small changes. (1) A counter-evidence retrieval step aimed at the 0% contradiction catch rate. (2) Parallel sub-question retrieval behind a shared Semantic Scholar rate limiter, with retries. (3) LLM timeouts, unit tests and CI. **None of the 130-question benchmark numbers were re-measured.** This sandbox had no Groq key, and its network policy blocked `api.semanticscholar.org`. Every claim below is labelled as tested or not tested.

---

## 1. What changed and why

### 1.1 Counter-evidence retrieval (main limitation)
**Problem:** the README says the contradiction catch rate is 0% because retrieval returns *related* papers, not *opposing* ones. Similarity search favours agreement: a paper saying "X works" sits closest in embedding space to other papers saying "X works". The critic's contradiction check (`_detect_contradictions`) only looks at the top 4 papers by hybrid score, so it rarely sees a real critique.

**Change:** a new graph node, `counter_evidence`, sits between `retriever` and `critic` (`src/agents/counter_evidence.py`):
- It builds up to 2 keyword queries (`RECON_COUNTER_QUERIES`) aimed at the opposing camp:
  - **Negation of the claim**, when a small rule applies: "dropout improves generalization" becomes "dropout does not improve generalization". There is no LLM, and no guessing when no rule matches.
  - **Topic keywords plus a critique term**: `limitations`, `fails replicate`, `revisiting` or `critique`.
- It runs those Semantic Scholar searches. Papers already in the supporting set are dropped, and the rest go into a **separate state key, `opposing_papers`**.
- The critic gets a new function, `_detect_counter_contradictions`. It pairs the top 2 supporting papers with the top 2 opposing papers and asks the **same** LLM contradiction question as before. That is at most 4 extra LLM calls per critic pass. There is no 2-year-gap rule, because a critique can come out the same year as the paper it critiques. An opposing paper may not be older than the paper it supposedly refutes.

**What deliberately did not change:**
- Opposing papers are **not** added to `retrieved_papers`. Reliability scores, the STALE threshold, the INSUFFICIENT checks and the synthesizer all see the same papers as before. A unit test checks this.
- No thresholds, weights or verdict rules changed. `patch_contradiction.py` stays archived.
- `RECON_COUNTER_EVIDENCE=0` turns the step off for a clean ablation.

**A bug found on the way (not fixed, on purpose):** `_to_search_query` in `src/agents/retriever.py` removes stopwords by substring replacement. "Is there consensus on whether..." comes out as `"is re consensus whe"`. I wrote whole-word keyword extraction for the counter-evidence queries only, and left the main retriever alone. Fixing it there would change main retrieval, and with it the staleness number, which would mix two effects in one comparison. It is worth fixing as a separate, separately measured change.

### 1.2 Parallel retrieval and rate limiting (speed)
**Problem:** `search_semantic_scholar` slept a fixed 3 s before *every* call. On top of that, `retriever_node` handled sub-questions one at a time, with a 1 s pause between them, plus a 0.5 s pause after each OpenAlex call.

**Change** (`src/retriever_utils.py`, `src/agents/retriever.py`):
- **`RateLimiter`:** a thread-safe, process-wide limit on the minimum gap between S2 request *starts*. The gap is 1.05 s with `S2_API_KEY` (S2's limit for keyed requests is 1 per second) and 3.0 s without a key (same as before), overridable with `S2_MIN_INTERVAL`. The first request doesn't wait, and time already spent on other work counts toward the gap.
- **Thread pool for sub-questions:** `RECON_RETRIEVAL_WORKERS`, default 3; set it to 1 for the old sequential order. `pool.map` keeps sub-question order, so de-duplication gives exactly the same paper list as the old loop (tested). The two OpenAlex calls also run concurrently.
- **Thread safety:** the embedding model now loads under a lock, and cache files are written to a temp file then renamed, so a parallel reader never sees a half-written file.

### 1.3 Reliability
- **`_get_with_retry`** for S2:
  - 15 s timeout (`S2_TIMEOUT`) and up to 3 retries with exponential backoff (1, 2, 4 s, capped at 30 s).
  - Retries on 429, 5xx, timeouts and connection errors; uses `Retry-After` when it's longer. Other 4xx errors aren't retried.
  - Each attempt goes through the rate limiter. Previously a single 429 meant zero papers for that sub-question.
- **LLM clients:** every Groq client (planner, critic, synthesizer, reliability scorer) now has an explicit 30 s timeout and 2 retries (`src/llm_config.py`). Before, a hung request could stall a whole run.
- **Lazy imports:** `retriever_utils` imports `sentence_transformers`/`sklearn` only when needed, so the unit tests and CI don't install torch.
- **Tests and CI:**
  - 18 unit tests in `tests/` need no network or keys, and run in about 1.5 s.
  - `.github/workflows/tests.yml` runs them on every push and PR, using `requirements-dev.txt`.
  - The existing `Test_Files/` scripts are manual live-API checks and are not part of CI.
- **Eval script:** `eval/run_v2_eval.py` now takes `RECON_EVAL_OUTPUT`, so ablation runs don't overwrite each other. It also prints a breakdown: Cat B caught through STALE vs through CONTRADICTED, the Cat C contradiction rate, and Cat A flags. It only reports; scoring is unchanged.

---

## 2. What was tested, with real results

| Check | Result |
|---|---|
| `python -m pytest -q tests` | **18 passed** (about 1.5 s). Covers claim negation, query building, keyword extraction, the node's de-duplication, error handling and kill switch, critic pairing rules, staleness scoring ignoring opposing papers, rate-limiter spacing (single thread and 5 threads), retry/backoff/Retry-After/give-up/no-retry-on-400, parallel vs sequential retrieval giving identical output, and LLM client timeouts. |
| Graph compiles | Nodes: `session_loader, planner, retriever, counter_evidence, critic, retry_retriever, synthesizer` |
| Existing scripts `Test_Files/test_phase2.py`, `test_db.py`, `test_phase3.py` | Ran without errors. `test_phase3` found no S2 or web results because the sandbox has no network access to those APIs, so its retrieval path was not really exercised. |
| Simulated latency, `python eval/bench_retrieval_latency.py --simulate --repeats 2` (fixed fake latencies: S2 0.8 s, web 1.5 s; **real** limiter and **real** `retriever_node`; 3 sub-questions) | See below |

Simulated retrieval-node time for 3 sub-questions:

| Configuration | Old code (emulated) | New, 1 worker | New, 3 workers |
|---|---|---|---|
| No S2 key (3.0 s spacing) | 17.9 s | 8.3 s | 8.3 s |
| With S2 key (1.05 s spacing) | 17.9 s | 6.9 s | 4.4 s |

How to read this table:
- These are **synthetic**. They show how the code behaves, not production latency.
- **Without a key, most of the gain comes from the rate limiter, not the threads.** The old code slept 3 s even when no request had been made recently. With 3 s spacing, the S2 limit becomes the bottleneck and threads add nothing.
- Threads help once the spacing is short (with a key): web search and embedding overlap with the S2 wait.
- The ~17.9 s figure is the emulated *retriever node alone* under these fake latencies. It is **not** the same thing as the ~17 s end-to-end figure in your notes.

## 3. What was NOT tested

- **The 130-question evaluation.** No Groq key was available, so there are no new staleness, contradiction, position-accuracy or false-positive numbers. The README results stay as they were.
- **Real S2/OpenAlex/DDG calls and real latency.** The network policy blocked `api.semanticscholar.org`.
- **Whether counter-evidence queries actually return opposing papers**, and whether the LLM then labels them as contradictions. This is the main open question.
- **End-to-end latency.** Piece 1 adds up to 2 S2 requests and 4 LLM calls per critic pass, which can cancel out part of piece 2's saving. Measure both.
- **Groq rate limits under the extra calls**, and DuckDuckGo behaviour under 3 concurrent searches.
- **CI on GitHub:** the workflow file is pushed, but I haven't seen it run.

## 4. Commands to measure it later

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

```bash
python -m pytest -q tests
```

Contradiction ablation, with the step off and then on, writing to separate files:

```bash
RECON_COUNTER_EVIDENCE=0 RECON_EVAL_OUTPUT=eval/results/v21_counter_off.csv python eval/run_v2_eval.py
```

```bash
RECON_COUNTER_EVIDENCE=1 RECON_EVAL_OUTPUT=eval/results/v21_counter_on.csv python eval/run_v2_eval.py
```

Compare the printed "Contradiction catch rate (Cat C)" and "Cat A flagged" lines between the two runs. **Watch this:** `staleness_caught()` in `eval/run_eval.py` counts STALE **or** CONTRADICTED. More CONTRADICTED verdicts on Cat B would therefore raise the "staleness" number without the staleness detector getting any better. Report "Cat B caught via STALE" separately, and don't claim a staleness gain from this change.

Retrieval latency with real APIs, sequential vs parallel (no LLM key needed):

```bash
python eval/bench_retrieval_latency.py --live --repeats 3
```

End-to-end latency for one query, sequential vs parallel:

```bash
RECON_RETRIEVAL_WORKERS=1 python Test_Files/test_check.py
```

```bash
RECON_RETRIEVAL_WORKERS=3 python Test_Files/test_check.py
```

The counter-evidence results are cached on disk (`data/cache`), so you can re-run cheaply.

## 5. Likely interview questions

**Counter-evidence retrieval**
- *Why was the contradiction rate 0%?* Retrieval ranks by similarity, which favours papers that agree. The contradiction check only looked at the top 4 of those, so it never saw an opposing paper. The problem was in retrieval, not in the critic.
- *Why not just lower the contradiction threshold or change the prompt?* That trades precision for recall on the same evidence. I archived a patch like that earlier because it inflated the score. This change changes the evidence, not the judge.
- *Why rule-based negation instead of asking the LLM for a negated query?* It's free, deterministic and unit-testable, and it doesn't add to Groq rate-limit pressure. The downside is coverage: only 6 of the 30 Cat C questions match a negation rule. The critique-term query covers the rest. An LLM-written query is the obvious next experiment.
- *Why a separate `opposing_papers` key?* So the staleness signal stays the same, and any change in the metric can be traced to the contradiction path. It also allows a clean on/off ablation.
- *Risks?* More false CONTRADICTED verdicts on consensus questions (Cat A); I would watch the Cat A flag rate. Also more retry loops (CONTRADICTED triggers a retry), and extra latency and LLM calls.
- *Why no year-gap rule for opposing pairs?* Critiques and failed replications often come out in the same year as the work they critique. I only require the opposing paper not to be older.

**Parallel retrieval**
- *How do you stay within S2's rate limit when running in parallel?* One process-wide limiter, guarded by a lock, reserves the next start time for each request. Threads overlap the waiting and the other I/O, but the request rate stays capped.
- *Why threads and not asyncio?* The work is I/O-bound, `requests` and the DDG client are synchronous, and LangGraph nodes here are synchronous. A thread pool was the smallest change. The GIL doesn't matter for network waits.
- *Is the output identical?* Yes. `pool.map` keeps order and de-duplication runs after all results are in; a test checks parallel and sequential give the same output.
- *Where did the speedup come from?* In the simulation without a key, almost all of it comes from removing the unconditional 3 s sleep. Threads only help once the S2 spacing is short (with a key). I measured that instead of assuming it.

**Reliability**
- *Which errors do you retry?* 429, 5xx, timeouts and connection errors, with exponential backoff and Retry-After honoured. Other 4xx errors mean a bad request, and retrying won't fix them.
- *Why bounded retries?* Every extra attempt also adds latency, and the callers already fall back cleanly when there are no results.
- *How does CI avoid needing API keys and torch?* Heavy imports are lazy and the network calls are monkeypatched. The live-API scripts stay as manual checks.

## 6. Resume wording the evidence supports

Supported now:
- "Added a counter-evidence retrieval stage (claim negation + critique/replication queries) to a multi-agent LangGraph pipeline to address a 0% contradiction-detection baseline; isolated from staleness scoring for clean ablation."
- "Parallelized sub-question retrieval behind a thread-safe, process-wide API rate limiter with exponential-backoff retries; added a no-network unit test suite and GitHub Actions CI."

**Not supported yet:**
- Any new contradiction catch rate.
- Any latency reduction in production.
- Any change to the 44% staleness figure.

Run the commands in section 4 first. Then quote only the measured numbers, including the Cat A false-positive rate next to any contradiction gain.
