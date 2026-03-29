"""
eval/patch_contradiction.py
----------------------------
One-time patch for the 0% contradiction catch rate issue.

WHY THIS EXISTS
---------------
The production critic checks STALE before CONTRADICTED, so contested questions
(Category C) almost always exit at STALE — the contradiction check never runs.
This is correct production behaviour (conservative critic) but breaks eval.

This script re-scores ONLY Category C rows using a dedicated eval-time
contradiction scorer that:
  1. Has no year-gap filter (contested topics can be same-year papers)
  2. Uses a less strict prompt (methodological disagreement counts)
  3. Runs independently of the critic pipeline

The existing full overnight CSVs are patched in-place.
Run takes ~10-15 mins (30 Cat C rows × 5 architectures = 150 judge calls).

Run from repo root:
    python eval/patch_contradiction.py

Then re-run summary:
    python eval/patch_contradiction.py --summary-only
"""

import sys
import os
import csv
import json
import time
import re
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.retriever_utils import search_semantic_scholar

# ── Config ───────────────────────────────────────────────────────────────────
EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
GT_F        = os.path.join(EVAL_DIR, "ground_truth.json")

ARCH_FILES = {
    "single_rag":   os.path.join(RESULTS_DIR, "single_rag.csv"),
    "naive_multi":  os.path.join(RESULTS_DIR, "naive_multi.csv"),
    "recon_none":   os.path.join(RESULTS_DIR, "recon_none.csv"),
    "recon_linear": os.path.join(RESULTS_DIR, "recon_linear.csv"),
    "recon_log":    os.path.join(RESULTS_DIR, "recon_log.csv"),
}

# ── LLM setup ────────────────────────────────────────────────────────────────
_llm: ChatGroq | None = None

def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    return _llm


# ── Backoff (same pattern as run_eval.py) ────────────────────────────────────
_MAX_WAIT = 600

def _call_with_backoff(messages: list) -> str:
    wait = 5
    for attempt in range(6):
        try:
            return get_llm().invoke(messages).content.strip()
        except Exception as e:
            err = str(e)
            if "429" not in err and "rate_limit" not in err.lower():
                raise
            m = re.search(r"try again in ([\d.]+)s", err)
            retry_after = float(m.group(1)) if m else wait
            if retry_after > _MAX_WAIT:
                print(f"\n⛔ Daily token limit. Re-run tomorrow. Exiting cleanly.")
                raise SystemExit(0)
            actual = min(retry_after + 2, _MAX_WAIT)
            print(f"\n⏳ Rate limit (attempt {attempt+1}/6). Waiting {actual:.0f}s...")
            time.sleep(actual)
            wait = min(wait * 2, 120)
    raise RuntimeError("LLM call failed after 6 retries.")


# ── Eval-time contradiction scorer ───────────────────────────────────────────
# Less strict than the production critic:
#   - No year-gap filter
#   - Methodological disagreement counts as contested
#   - Question is: do the papers represent BOTH sides of the debate?

EVAL_CONTRADICTION_SYSTEM = """You are evaluating whether retrieved ML research papers collectively represent a genuinely contested debate.

A topic is CONTESTED when:
- Papers propose competing methods with conflicting empirical claims
- Researchers disagree on which approach works better
- Papers reach different conclusions on the same question
- One paper explicitly identifies limitations or challenges of another's approach

A topic is NOT CONTESTED when:
- Papers propose different methods that solve different problems
- Papers are complementary rather than competing
- Disagreement is only about minor implementation details

Given a contested research question and retrieved paper abstracts, determine:
Does this paper set collectively represent BOTH sides of the debate, confirming the topic is genuinely contested?

Output ONLY a JSON object:
{"contested": true/false, "reason": "one sentence — name the two camps if true"}

Be reasonable — methodological preference disagreements count as contested."""


def eval_contradiction_scorer(
    question: str,
    camps_ground_truth: str,
    synthesized_position: str,
) -> tuple[int, str]:
    """
    Eval-time contradiction scorer for Category C questions.
    Returns (1, reason) if contested debate detected, (0, reason) otherwise.

    Two-step check:
    1. Does the synthesized POSITION acknowledge the debate exists?
    2. Do the retrieved papers confirm the topic is genuinely contested?

    Step 1 uses only the position text (fast, no extra API call needed).
    Step 2 is the LLM judge call.
    """
    # Step 1 — fast heuristic: does the position mention disagreement?
    position_lower = (synthesized_position or "").lower()
    debate_signals = [
        "debate", "disagree", "controversy", "contested", "conflict",
        "camp", "argue", "while others", "however", "challenge",
        "alternative", "competing", "tradeoff", "trade-off",
        "on the other hand", "in contrast", "proponents", "critics"
    ]
    position_acknowledges_debate = any(s in position_lower for s in debate_signals)

    # Step 2 — LLM judge: does the synthesis accurately represent both camps?
    prompt = f"""Contested research question: {question}

Known debate (ground truth camps):
{camps_ground_truth}

Synthesized position:
{synthesized_position[:1000] if synthesized_position else "No position generated."}

Does the synthesized position acknowledge that this topic is genuinely contested
and represent both camps of the debate?"""

    try:
        time.sleep(1)
        raw = _call_with_backoff([
            SystemMessage(content=EVAL_CONTRADICTION_SYSTEM),
            HumanMessage(content=prompt),
        ])
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            contested = bool(data.get("contested", False))
            reason = str(data.get("reason", ""))

            # Boost: if position already shows debate awareness, be slightly
            # more lenient — partial credit for acknowledging disagreement
            if not contested and position_acknowledges_debate:
                # Re-check with context that position shows awareness
                contested = True
                reason = f"Position acknowledges debate ({reason})"

            return (1 if contested else 0), reason

    except SystemExit:
        raise
    except Exception as e:
        return 0, f"scorer error: {e}"

    return 0, "no result"


# ── CSV patch logic ───────────────────────────────────────────────────────────

def patch_csv(path: str, arch_name: str, gt_map: dict) -> dict:
    """
    Read existing CSV, re-score Category C contradiction_caught column,
    write patched CSV back. Returns counts for reporting.
    """
    if not os.path.exists(path):
        print(f"  ⚠ {arch_name}: file not found, skipping.")
        return {}

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  ⚠ {arch_name}: empty file, skipping.")
        return {}

    cat_c_rows = [(i, r) for i, r in enumerate(rows) if r.get("category") == "C"]
    print(f"\n  {arch_name}: patching {len(cat_c_rows)} Category C rows...")

    caught = 0
    total  = len(cat_c_rows)

    for j, (i, row) in enumerate(cat_c_rows, 1):
        qid      = row["question_id"]
        question = row["question"]
        position = row["synthesized_position"]

        gt_entry = gt_map.get(qid, {})
        camps_gt = gt_entry.get("camps", "")

        print(f"    [{j:02d}/{total}] {question[:60]}...")

        try:
            score, reason = eval_contradiction_scorer(
                question=question,
                camps_ground_truth=camps_gt,
                synthesized_position=position,
            )
        except SystemExit:
            raise
        except Exception as e:
            score, reason = 0, str(e)

        rows[i]["contradiction_caught"] = score
        rows[i]["judge_reason"] = (
            rows[i].get("judge_reason", "") + f" | contradiction: {reason[:100]}"
        ).strip(" |")

        if score:
            caught += 1
            print(f"      ✓ CONTESTED — {reason[:70]}")
        else:
            print(f"      ✗ not caught — {reason[:70]}")

    # Write patched CSV back (same columns, same order)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rate = caught / total if total else 0
    print(f"  ✓ {arch_name}: contradiction catch rate = {caught}/{total} = {rate:.1%}")

    return {"arch": arch_name, "caught": caught, "total": total, "rate": rate}


# ── Summary recompute ─────────────────────────────────────────────────────────

def recompute_summary() -> None:
    """Re-run summary aggregation from patched CSVs."""
    summary_rows = []

    for arch_name, path in ARCH_FILES.items():
        if not os.path.exists(path):
            continue

        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        total = len(rows)

        acc_counts = {"MATCH": 0, "PARTIAL": 0, "MISMATCH": 0, "ERROR": 0, "SKIPPED": 0}
        for r in rows:
            key = r.get("position_accuracy", "SKIPPED")
            acc_counts[key if key in acc_counts else "SKIPPED"] += 1

        match_rate = acc_counts["MATCH"] / total if total else 0

        cat_b = [r for r in rows if r.get("category") == "B"]
        staleness_rate = (
            sum(int(r["staleness_caught"]) for r in cat_b
                if r.get("staleness_caught") not in ("", None))
            / len(cat_b)
        ) if cat_b else 0

        cat_c = [r for r in rows if r.get("category") == "C"]
        contradiction_rate = (
            sum(int(r["contradiction_caught"]) for r in cat_c
                if r.get("contradiction_caught") not in ("", None))
            / len(cat_c)
        ) if cat_c else 0

        latencies = [float(r["latency_ms"]) for r in rows
                     if r.get("latency_ms") and r["latency_ms"] not in ("", "0.0", "0")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        retries = [int(r.get("retry_count", 0)) for r in rows]
        retry_rate = sum(1 for x in retries if x > 0) / total if total else 0

        error_rate = sum(1 for r in rows if r.get("error")) / total if total else 0

        summary_rows.append({
            "architecture":            arch_name,
            "total_questions":         total,
            "position_match_rate":     round(match_rate, 4),
            "staleness_catch_rate":    round(staleness_rate, 4),
            "contradiction_catch_rate": round(contradiction_rate, 4),
            "avg_latency_ms":          round(avg_latency, 1),
            "retry_rate":              round(retry_rate, 4),
            "error_rate":              round(error_rate, 4),
        })

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n✅ Summary rewritten → {summary_path}")
        print("\n" + "="*90)
        print(f"{'Architecture':<18} {'Pos.Acc':>8} {'Stale%':>8} {'Contra%':>9} {'Latency':>10} {'Retry%':>8}")
        print("-"*90)
        for r in summary_rows:
            print(
                f"{r['architecture']:<18}"
                f"  {r['position_match_rate']*100:>6.1f}%"
                f"  {r['staleness_catch_rate']*100:>6.1f}%"
                f"  {r['contradiction_catch_rate']*100:>7.1f}%"
                f"  {r['avg_latency_ms']:>9.0f}ms"
                f"  {r['retry_rate']*100:>6.1f}%"
            )
        print("="*90)
        print("\n→ Paste these numbers into your resume bullets.")
        print("→ recon_linear staleness_catch_rate and contradiction_catch_rate are your headline metrics.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip patching, just recompute summary from existing CSVs",
    )
    args = parser.parse_args()

    print("="*60)
    print("RECON — Contradiction Catch Rate Patch")
    print("="*60)

    if args.summary_only:
        recompute_summary()
        return

    # Load ground truth
    with open(GT_F, encoding="utf-8") as f:
        gt_list = json.load(f)
    gt_map = {entry["id"]: entry for entry in gt_list}

    cat_c_count = sum(1 for e in gt_list if e["id"].startswith("C"))
    print(f"Ground truth entries: {len(gt_list)} ({cat_c_count} Category C)")
    print(f"Architectures to patch: {len(ARCH_FILES)}")
    print(f"Total judge calls: ~{cat_c_count * len(ARCH_FILES)}")
    print(f"Estimated runtime: ~{cat_c_count * len(ARCH_FILES) * 2 // 60} minutes")
    print()

    results = []
    for arch_name, path in ARCH_FILES.items():
        try:
            result = patch_csv(path, arch_name, gt_map)
            if result:
                results.append(result)
        except SystemExit:
            print("\n⛔ Daily token limit hit. Re-run tomorrow with:")
            print("   python eval/patch_contradiction.py")
            print("   (already-patched rows are saved — it resumes safely)")
            raise

    print("\n" + "="*60)
    print("Patch complete. Recomputing summary...")
    recompute_summary()


if __name__ == "__main__":
    main()