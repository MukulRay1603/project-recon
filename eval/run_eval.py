"""
eval/run_eval.py
----------------
Phase 10 — Evaluation Harness

Benchmarks 3 architectures × 3 decay configs against 130 ground-truth
questions, computing:
  - Ragas faithfulness (baseline)
  - Position accuracy via LLM-as-judge (MATCH / PARTIAL / MISMATCH)
  - Staleness catch rate  — Category B, critic flagged STALE
  - Contradiction catch rate — Category C, critic flagged CONTRADICTED
  - Avg latency (ms)
  - Retry rate

Architectures:
  A — single_rag      : S2 top-5 → generate. No planner, no critic, no session.
  B — naive_multi     : Planner → Retriever → Synthesizer. No critic, no retry.
  C — recon_none      : Full RECON, decay_config=none
  D — recon_linear    : Full RECON, decay_config=linear   ← primary
  E — recon_log       : Full RECON, decay_config=log

Run from repo root:
    python eval/run_eval.py

Outputs (written incrementally — safe to resume after crash):
    eval/results/single_rag.csv
    eval/results/naive_multi.csv
    eval/results/recon_none.csv
    eval/results/recon_linear.csv
    eval/results/recon_log.csv
    eval/results/summary.csv        ← final aggregated metrics table

Rate limit notes:
  - Groq: 6k tokens/min. LLM judge runs on Category B+C only (~80 questions),
    not all 130, to stay within limits.
  - S2: sleep(3) per call already in retriever_utils. Cache prevents re-fetch.
  - Estimated runtime: ~3-4 hours on Colab Pro (run overnight).
  - Use EVAL_LIMIT env var to test on first N questions: EVAL_LIMIT=5 python eval/run_eval.py
"""

import sys
import os
import json
import csv
import time
import uuid
import logging
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Imports from your existing pipeline ──────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph import run_recon
from src.memory import init_db
from src.retriever_utils import search_semantic_scholar
from src.agents.planner import planner_node
from src.agents.retriever import retriever_node
from src.agents.synthesizer import synthesizer_node
from src.state import ResearchState, Verdict

# ── Config ───────────────────────────────────────────────────────────────────
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(EVAL_DIR, "results")
QUESTIONS_F  = os.path.join(EVAL_DIR, "questions.json")
GT_F         = os.path.join(EVAL_DIR, "ground_truth.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Optional: cap questions for smoke test
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0"))  # 0 = run all

# Groq judge LLM — shared, lazy init
_judge_llm: ChatGroq | None = None

def get_judge() -> ChatGroq:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    return _judge_llm


# ── Exponential backoff for Groq 429s ────────────────────────────────────────
# Two flavours of 429:
#   TPM (tokens/min)  — wait stated retry-after seconds, then continue
#   TPD (tokens/day)  — no point waiting; exit cleanly so resume works tomorrow
#
# Max single wait: 600s (10 min). If retry-after > 600s it's a TPD limit.

_MAX_WAIT_SECONDS = 600   # above this → assume daily limit, exit cleanly
_MAX_RETRIES      = 6     # max backoff attempts before giving up on one call

def _groq_call_with_backoff(llm: ChatGroq, messages: list) -> str:
    """
    Invoke a Groq ChatGroq LLM with exponential backoff on 429 errors.
    Returns the response content string.
    Raises SystemExit if a TPD (daily) limit is detected — lets the
    crash-resume system pick up tomorrow with no corrupt rows.
    """
    import re as _re

    wait = 5  # initial wait seconds

    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            err_str = str(e)

            # Only handle 429 rate limit errors with backoff
            if "429" not in err_str and "rate_limit" not in err_str.lower():
                raise  # non-rate-limit errors bubble up immediately

            # Parse retry-after from error message if present
            retry_after = wait
            match = _re.search(r"try again in ([\d.]+)s", err_str)
            if match:
                retry_after = float(match.group(1))

            if retry_after > _MAX_WAIT_SECONDS:
                # Daily token limit — no point waiting
                print(
                    f"\n⛔ Groq daily token limit reached "
                    f"(retry-after={retry_after:.0f}s > {_MAX_WAIT_SECONDS}s cap).\n"
                    f"   Results saved so far are intact.\n"
                    f"   Re-run tomorrow — the harness will resume from where it stopped.\n"
                    f"   Exiting cleanly now."
                )
                raise SystemExit(0)

            # TPM limit — wait and retry
            actual_wait = min(retry_after + 2, _MAX_WAIT_SECONDS)
            print(
                f"\n⏳ Groq rate limit (attempt {attempt+1}/{_MAX_RETRIES}). "
                f"Waiting {actual_wait:.0f}s before retry..."
            )
            time.sleep(actual_wait)
            wait = min(wait * 2, 120)  # cap backoff at 2 minutes

    raise RuntimeError(f"Groq call failed after {_MAX_RETRIES} retries.")


# ── CSV schema — every architecture writes the same columns ──────────────────
CSV_COLS = [
    "question_id",
    "category",
    "subfield",
    "question",
    "architecture",
    "decay_config",
    "synthesized_position",
    "critic_verdict",
    "retry_count",
    "latency_ms",
    "papers_retrieved",
    "position_accuracy",   # MATCH / PARTIAL / MISMATCH / SKIPPED
    "judge_reason",
    "staleness_caught",    # 1/0 — Cat B only
    "contradiction_caught",# 1/0 — Cat C only
    "faithfulness_score",  # Ragas — filled post-run in calibration.py
    "error",
]


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are evaluating whether a research synthesis matches a ground truth position.

Given:
  GROUND TRUTH: the established consensus or supersession fact from a survey paper
  SYNTHESIZED POSITION: a research brief generated by an AI system

Score the match:
  MATCH   — the synthesis accurately reflects the ground truth, including the key claims
  PARTIAL — the synthesis is partially correct but misses important aspects or is vague
  MISMATCH — the synthesis contradicts the ground truth or provides clearly wrong information

For CONTESTED ground truth (containing "Camp A" / "Camp B"):
  MATCH   — the synthesis correctly identifies that the topic is contested and names both camps
  PARTIAL — the synthesis mentions disagreement but only describes one camp or is vague
  MISMATCH — the synthesis presents one side as settled consensus when it is contested

Output ONLY a JSON object:
{"score": "MATCH", "reason": "one sentence explanation"}

Be strict. Vague or generic answers that could apply to any ML topic score PARTIAL at best."""


def llm_judge(
    question: str,
    ground_truth: str,
    synthesized_position: str,
    category: str,
) -> tuple[str, str]:
    """
    Run LLM judge. Returns (score, reason).
    score: MATCH | PARTIAL | MISMATCH | ERROR
    """
    # Truncate position to avoid token overflow
    position_preview = synthesized_position[:1200] if synthesized_position else "No position generated."

    prompt = f"""Question: {question}

GROUND TRUTH:
{ground_truth}

SYNTHESIZED POSITION (first 1200 chars):
{position_preview}

Category: {category}

Score this synthesis."""

    try:
        time.sleep(1)  # gentle inter-call spacing
        raw = _groq_call_with_backoff(
            get_judge(),
            [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)],
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score  = str(data.get("score", "ERROR")).upper()
            reason = str(data.get("reason", ""))
            if score not in ("MATCH", "PARTIAL", "MISMATCH"):
                score = "ERROR"
            return score, reason
    except SystemExit:
        raise   # let daily-limit exit propagate cleanly
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")

    return "ERROR", "judge call failed"


# ── Staleness / contradiction detection helpers ───────────────────────────────

def staleness_caught(critic_verdict: str) -> int:
    """Returns 1 if critic flagged STALE or CONTRADICTED (both indicate recency awareness)."""
    return 1 if critic_verdict in (Verdict.STALE, Verdict.CONTRADICTED) else 0


def contradiction_caught(critic_verdict: str) -> int:
    """Returns 1 if critic flagged CONTRADICTED."""
    return 1 if critic_verdict == Verdict.CONTRADICTED else 0


# ── Architecture A — Single-agent RAG ────────────────────────────────────────

SINGLE_RAG_SYSTEM = """You are a research assistant. Given retrieved papers, answer the research question concisely.
Cite papers as [Author et al., Year]. Be factual and direct. 3-5 sentences maximum."""

def run_single_rag(question: str, decay_config: str = "linear") -> dict:
    """
    Architecture A: S2 top-5 → generate. No planner, no critic, no session.
    Returns dict with position, latency_ms, papers_retrieved, critic_verdict=N/A.
    """
    start = time.time()

    # Keyword conversion (same as retriever.py _to_search_query)
    stopwords = [
        "what are", "what is", "how does", "how do", "why is", "why are",
        "when did", "where is", "which are", "tell me about",
        "foundational papers on", "recent advances in", "open challenges in",
        "the current state of", "published in", "for llms", "in llms",
        "papers on", "research on", "advances in", "challenges in",
        "were", "was", "the", "a ", "an ", "in ", "of ", "for ", "on ",
    ]
    kw = question.lower().strip().rstrip("?")
    for sw in stopwords:
        kw = kw.replace(sw, " ")
    kw = re.sub(r"\s+", " ", kw).strip()
    kw = " ".join(kw.split()[:6])

    papers = search_semantic_scholar(kw, limit=5, decay_config=decay_config)

    if not papers:
        return {
            "synthesized_position": "No papers retrieved.",
            "critic_verdict": "N/A",
            "retry_count": 0,
            "latency_ms": (time.time() - start) * 1000,
            "papers_retrieved": 0,
        }

    # Format evidence
    evidence = "\n".join(
        f"[{i+1}] {p.title} ({p.year}) — {p.abstract[:200]}..."
        for i, p in enumerate(papers[:5])
    )

    prompt = f"""Question: {question}

Retrieved papers:
{evidence}

Answer the question using these papers."""

    try:
        position = _groq_call_with_backoff(
            get_judge(),
            [SystemMessage(content=SINGLE_RAG_SYSTEM), HumanMessage(content=prompt)],
        )
    except SystemExit:
        raise
    except Exception as e:
        position = f"Generation error: {e}"

    return {
        "synthesized_position": position,
        "critic_verdict": "N/A",
        "retry_count": 0,
        "latency_ms": (time.time() - start) * 1000,
        "papers_retrieved": len(papers),
    }


# ── Architecture B — Naive multi-agent ───────────────────────────────────────

def run_naive_multi(question: str, decay_config: str = "linear") -> dict:
    """
    Architecture B: Planner → Retriever → Synthesizer. No critic, no retry, no session.
    Reuses your existing agent nodes directly.
    """
    start = time.time()

    # Build a minimal state
    state: ResearchState = {
        "original_query": question,
        "session_id": str(uuid.uuid4()),
        "session_context": None,
        "sub_questions": [],
        "retrieved_papers": [],
        "citation_graph": {},
        "web_results": [],
        "critic_verdict": "PASS",   # Force pass — no critic in arch B
        "critic_notes": "No critic in naive multi-agent architecture.",
        "rewritten_questions": [],
        "retry_count": 0,
        "synthesized_position": "",
        "claim_confidences": [],
        "session_update": None,
        "export_md": "",
        "decay_config": decay_config,
        "calibration_bin": "",
        "latency_ms": 0.0,
    }

    try:
        state = planner_node(state)
        state = retriever_node(state)
        state = synthesizer_node(state)
        position = state.get("synthesized_position", "")
    except Exception as e:
        position = f"Pipeline error: {e}"

    return {
        "synthesized_position": position,
        "critic_verdict": "N/A",    # No critic in arch B
        "retry_count": 0,
        "latency_ms": (time.time() - start) * 1000,
        "papers_retrieved": len(state.get("retrieved_papers") or []),
    }


# ── Architecture C/D/E — RECON full ──────────────────────────────────────────

def run_recon_full(question: str, decay_config: str = "linear") -> dict:
    """
    Architecture C/D/E: Full RECON pipeline with decay_config variant.
    """
    session_id = str(uuid.uuid4())
    result = run_recon(
        query=question,
        session_id=session_id,
        decay_config=decay_config,
    )
    return {
        "synthesized_position": result.get("synthesized_position", ""),
        "critic_verdict": result.get("critic_verdict", ""),
        "retry_count": result.get("retry_count", 0),
        "latency_ms": result.get("latency_ms", 0.0),
        "papers_retrieved": len(result.get("retrieved_papers") or []),
    }


# ── CSV writer helpers ────────────────────────────────────────────────────────

def get_csv_writer(path: str):
    """Open CSV in append mode, write header if new file."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLS)
    if not file_exists:
        writer.writeheader()
    return f, writer


def get_done_ids(path: str) -> set:
    """Read already-completed question IDs from CSV (for resume on crash)."""
    if not os.path.exists(path):
        return set()
    done = set()
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row.get("question_id", ""))
    except Exception:
        pass
    return done


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_architecture(
    arch_name: str,
    decay_config: str,
    runner_fn,
    questions: list[dict],
    gt_map: dict,
    output_path: str,
):
    """
    Run one architecture over all questions.
    Writes results row-by-row (crash-safe).
    """
    done_ids = get_done_ids(output_path)
    remaining = [q for q in questions if q["id"] not in done_ids]

    print(f"\n{'='*60}")
    print(f"Architecture: {arch_name}  |  decay: {decay_config}")
    print(f"Questions: {len(questions)} total, {len(done_ids)} already done, {len(remaining)} to run")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    if not remaining:
        print("  ✓ Already complete, skipping.")
        return

    f, writer = get_csv_writer(output_path)

    try:
        for i, q in enumerate(remaining, 1):
            qid      = q["id"]
            category = q["category"]
            question = q["question"]
            subfield = q["subfield"]
            gt_entry = gt_map.get(qid, {})
            ground_truth = (
                gt_entry.get("ground_truth")          # Cat A
                or gt_entry.get("supersession")       # Cat B
                or gt_entry.get("camps")              # Cat C
                or ""
            )

            print(f"  [{i:03d}/{len(remaining)}] [{category}] {question[:65]}...")

            row = {
                "question_id":          qid,
                "category":             category,
                "subfield":             subfield,
                "question":             question,
                "architecture":         arch_name,
                "decay_config":         decay_config,
                "synthesized_position": "",
                "critic_verdict":       "",
                "retry_count":          0,
                "latency_ms":           0.0,
                "papers_retrieved":     0,
                "position_accuracy":    "SKIPPED",
                "judge_reason":         "",
                "staleness_caught":     "",
                "contradiction_caught": "",
                "faithfulness_score":   "",
                "error":               "",
            }

            # ── Run the architecture ──────────────────────────────────────
            try:
                result = runner_fn(question, decay_config)
                row["synthesized_position"] = result["synthesized_position"]
                row["critic_verdict"]       = result["critic_verdict"]
                row["retry_count"]          = result["retry_count"]
                row["latency_ms"]           = round(result["latency_ms"], 1)
                row["papers_retrieved"]     = result["papers_retrieved"]
            except SystemExit:
                f.close()
                raise   # daily limit — exit before writing corrupt row
            except Exception as e:
                row["error"] = str(e)
                logger.error(f"Runner failed for {qid}: {e}")
                writer.writerow(row)
                f.flush()
                continue

            # ── LLM judge — run on ALL categories ────────────────────────
            # (run on all 130 questions — Groq free tier handles it with
            # sleep(1) guard in llm_judge; ~650 judge calls total across 5 runs)
            if row["synthesized_position"] and ground_truth:
                try:
                    score, reason = llm_judge(
                        question=question,
                        ground_truth=ground_truth,
                        synthesized_position=row["synthesized_position"],
                        category=category,
                    )
                except SystemExit:
                    f.close()
                    raise
                row["position_accuracy"] = score
                row["judge_reason"]      = reason[:200]
                print(f"    → verdict={row['critic_verdict']}  judge={score}  {reason[:60]}")
            else:
                print(f"    → verdict={row['critic_verdict']}  judge=SKIPPED (no position or GT)")

            # ── Staleness catch rate — Category B ────────────────────────
            if category == "B":
                row["staleness_caught"] = staleness_caught(row["critic_verdict"])

            # ── Contradiction catch rate — Category C ─────────────────────
            if category == "C":
                row["contradiction_caught"] = contradiction_caught(row["critic_verdict"])

            writer.writerow(row)
            f.flush()  # Write after every row — crash-safe

    finally:
        f.close()

    print(f"\n  ✓ {arch_name} complete → {output_path}")


# ── Summary aggregation ───────────────────────────────────────────────────────

def compute_summary(results_dir: str) -> None:
    """
    Read all result CSVs and write eval/results/summary.csv.
    Computes per-architecture aggregate metrics.
    """
    arch_files = {
        "single_rag":    os.path.join(results_dir, "single_rag.csv"),
        "naive_multi":   os.path.join(results_dir, "naive_multi.csv"),
        "recon_none":    os.path.join(results_dir, "recon_none.csv"),
        "recon_linear":  os.path.join(results_dir, "recon_linear.csv"),
        "recon_log":     os.path.join(results_dir, "recon_log.csv"),
    }

    summary_rows = []

    for arch_name, path in arch_files.items():
        if not os.path.exists(path):
            continue

        rows = []
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        total = len(rows)

        # Position accuracy
        acc_counts = {"MATCH": 0, "PARTIAL": 0, "MISMATCH": 0, "ERROR": 0, "SKIPPED": 0}
        for r in rows:
            acc_counts[r.get("position_accuracy", "SKIPPED")] += 1

        match_rate   = acc_counts["MATCH"] / total if total else 0
        partial_rate = acc_counts["PARTIAL"] / total if total else 0

        # Staleness catch rate (Category B)
        cat_b = [r for r in rows if r.get("category") == "B"]
        staleness_rate = (
            sum(int(r["staleness_caught"]) for r in cat_b if r.get("staleness_caught") != "")
            / len(cat_b)
        ) if cat_b else 0

        # Contradiction catch rate (Category C)
        cat_c = [r for r in rows if r.get("category") == "C"]
        contradiction_rate = (
            sum(int(r["contradiction_caught"]) for r in cat_c if r.get("contradiction_caught") != "")
            / len(cat_c)
        ) if cat_c else 0

        # Latency
        latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Retry rate
        retries = [int(r.get("retry_count", 0)) for r in rows]
        retry_rate = sum(1 for r in retries if r > 0) / total if total else 0

        # Error rate
        error_rate = sum(1 for r in rows if r.get("error")) / total if total else 0

        summary_rows.append({
            "architecture":        arch_name,
            "total_questions":     total,
            "position_match_rate": round(match_rate, 4),
            "position_partial_rate": round(partial_rate, 4),
            "staleness_catch_rate": round(staleness_rate, 4),
            "contradiction_catch_rate": round(contradiction_rate, 4),
            "avg_latency_ms":      round(avg_latency, 1),
            "retry_rate":          round(retry_rate, 4),
            "error_rate":          round(error_rate, 4),
        })

    summary_path = os.path.join(results_dir, "summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n✅ Summary written → {summary_path}")
        print_summary_table(summary_rows)
    else:
        print("\n⚠ No completed result files found to summarise.")


def print_summary_table(rows: list[dict]) -> None:
    """Pretty-print the summary table to terminal."""
    print("\n" + "="*90)
    print(f"{'Architecture':<18} {'Pos.Acc':>8} {'Stale%':>8} {'Contra%':>8} {'Latency':>10} {'Retry%':>8}")
    print("-"*90)
    for r in rows:
        print(
            f"{r['architecture']:<18}"
            f"  {r['position_match_rate']*100:>6.1f}%"
            f"  {r['staleness_catch_rate']*100:>6.1f}%"
            f"  {r['contradiction_catch_rate']*100:>6.1f}%"
            f"  {r['avg_latency_ms']:>9.0f}ms"
            f"  {r['retry_rate']*100:>6.1f}%"
        )
    print("="*90)
    print("→ staleness_catch_rate and contradiction_catch_rate are your headline resume metrics.")
    print("→ Copy these numbers into resume bullets after verifying they make sense.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("RECON — Phase 10 Evaluation Harness")
    print("="*60)

    # Load questions and ground truth
    with open(QUESTIONS_F, encoding="utf-8") as f:
        questions = json.load(f)
    with open(GT_F, encoding="utf-8") as f:
        gt_list = json.load(f)

    gt_map = {entry["id"]: entry for entry in gt_list}

    # Apply EVAL_LIMIT for smoke testing
    if EVAL_LIMIT > 0:
        questions = questions[:EVAL_LIMIT]
        print(f"⚠  EVAL_LIMIT={EVAL_LIMIT} — running subset only")

    print(f"Questions loaded: {len(questions)}")
    print(f"Ground truths loaded: {len(gt_map)}")
    print(f"S2 cache dir: data/cache/")
    print(f"Results dir:  eval/results/")

    init_db()

    # ── Run all 5 architecture/config combinations ────────────────────────────
    # Order matters: run single_rag first (warms S2 cache for later runs)

    run_architecture(
        arch_name    = "single_rag",
        decay_config = "linear",
        runner_fn    = run_single_rag,
        questions    = questions,
        gt_map       = gt_map,
        output_path  = os.path.join(RESULTS_DIR, "single_rag.csv"),
    )

    run_architecture(
        arch_name    = "naive_multi",
        decay_config = "linear",
        runner_fn    = run_naive_multi,
        questions    = questions,
        gt_map       = gt_map,
        output_path  = os.path.join(RESULTS_DIR, "naive_multi.csv"),
    )

    run_architecture(
        arch_name    = "recon_none",
        decay_config = "none",
        runner_fn    = run_recon_full,
        questions    = questions,
        gt_map       = gt_map,
        output_path  = os.path.join(RESULTS_DIR, "recon_none.csv"),
    )

    run_architecture(
        arch_name    = "recon_linear",
        decay_config = "linear",
        runner_fn    = run_recon_full,
        questions    = questions,
        gt_map       = gt_map,
        output_path  = os.path.join(RESULTS_DIR, "recon_linear.csv"),
    )

    run_architecture(
        arch_name    = "recon_log",
        decay_config = "log",
        runner_fn    = run_recon_full,
        questions    = questions,
        gt_map       = gt_map,
        output_path  = os.path.join(RESULTS_DIR, "recon_log.csv"),
    )

    # ── Compute and print summary ─────────────────────────────────────────────
    compute_summary(RESULTS_DIR)

    print("\n✅ Evaluation complete.")
    print("Next steps:")
    print("  1. Review eval/results/summary.csv for headline metrics")
    print("  2. Run eval/calibration.py to generate calibration curve PNG")
    print("  3. Run eval/contradiction_viz.py to generate contradiction graph PNG")
    print("  4. Replace ALL CAPS placeholders in resume with real numbers")


if __name__ == "__main__":
    main()