import logging
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ResearchState, Paper, Verdict

load_dotenv()
logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now().year
_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    return _llm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CONTRADICTION_SYSTEM = """You are evaluating whether two ML research papers contradict each other.

Given two paper abstracts, determine if paper B explicitly refutes, supersedes, or contradicts the main claims of paper A.

Output ONLY a JSON object:
{"contradicts": true/false, "reason": "one sentence explanation"}

Be strict — only mark contradicts=true if there is a clear, explicit disagreement on a specific technical claim.
Do not mark contradicts=true just because papers propose different methods."""

REWRITE_SYSTEM = """You are rewriting research sub-questions to improve search results.

Given the original sub-questions and a rewrite strategy, output ONLY a JSON array of 2 new questions.
Make questions shorter and more keyword-focused for academic search.
Format: ["question 1", "question 2"]
No preamble."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_age_months(papers: list[Paper]) -> float:
    """Returns mean age of papers in months."""
    ages = [(CURRENT_YEAR - p.year) * 12 for p in papers if p.year and p.year > 0]
    return sum(ages) / len(ages) if ages else 9999.0


def _check_contradiction(paper_a: Paper, paper_b: Paper) -> tuple[bool, str]:
    """Ask the LLM whether paper_b contradicts paper_a."""
    if not paper_a.abstract or not paper_b.abstract:
        return False, "Missing abstract"

    prompt = f"""Paper A ({paper_a.year}): {paper_a.title}
Abstract: {paper_a.abstract[:400]}

Paper B ({paper_b.year}): {paper_b.title}
Abstract: {paper_b.abstract[:400]}

Does paper B explicitly contradict or supersede paper A?"""

    try:
        response = get_llm().invoke([
            SystemMessage(content=CONTRADICTION_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return bool(data.get("contradicts", False)), data.get("reason", "")
    except Exception as e:
        logger.warning(f"Contradiction check failed: {e}")

    return False, ""


def _detect_contradictions(papers: list[Paper]) -> list[tuple[str, str, str]]:
    """Check top 4 papers for contradictions. Only checks pairs with 2+ year gap."""
    contradictions = []
    top_papers = papers[:4]

    for i, pa in enumerate(top_papers):
        for pb in top_papers[i+1:]:
            if abs((pa.year or 0) - (pb.year or 0)) < 2:
                continue
            older = pa if (pa.year or 0) < (pb.year or 0) else pb
            newer = pa if (pa.year or 0) > (pb.year or 0) else pb

            contradicts, reason = _check_contradiction(older, newer)
            if contradicts:
                contradictions.append((older.title, newer.title, reason))
                logger.info(f"Contradiction: '{older.title[:40]}' vs '{newer.title[:40]}'")

    return contradictions


def _rewrite_questions(sub_questions: list[str], strategy: str) -> list[str]:
    """Rewrite sub-questions using LLM based on strategy."""
    if not sub_questions:
        return []

    strategy_instructions = {
        "broaden": "Broaden the questions to retrieve more papers. Use more general terms.",
        "recent": "Rewrite to focus on very recent work (2023-2025). Add 'recent', 'latest', '2024' keywords.",
        "probe_contradiction": "Rewrite to explore the specific disagreement. Focus on the contested claim.",
    }

    instruction = strategy_instructions.get(strategy, strategy_instructions["broaden"])
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions[:2]))

    prompt = f"""Strategy: {instruction}

Original questions:
{questions_text}

Output 2 improved search queries as a JSON array."""

    try:
        response = get_llm().invoke([
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            return [q for q in questions if isinstance(q, str)][:2]
    except Exception as e:
        logger.warning(f"Question rewrite failed: {e}")

    return sub_questions[:2]


# ---------------------------------------------------------------------------
# Critic node
# ---------------------------------------------------------------------------

def critic_node(state: ResearchState) -> ResearchState:
    """
    Reads:  retrieved_papers, retry_count, sub_questions
    Writes: critic_verdict, critic_notes, rewritten_questions,
            retry_count, calibration_bin
    """
    papers = state.get("retrieved_papers") or []
    retry_count = state.get("retry_count", 0)

    # FORCED PASS — max retries reached
    if retry_count >= 2:
        logger.info("Critic: max retries reached, forcing PASS")
        return {
            "critic_verdict": Verdict.FORCED_PASS,
            "critic_notes": "Max retries reached. Passing with available evidence.",
            "rewritten_questions": [],
            "retry_count": retry_count,
            "calibration_bin": Verdict.FORCED_PASS,
        }

    # INSUFFICIENT — not enough papers
    if len(papers) < 3:
        logger.info(f"Critic: insufficient papers ({len(papers)})")
        rewritten = _rewrite_questions(state.get("sub_questions") or [], "broaden")
        return {
            "critic_verdict": Verdict.INSUFFICIENT,
            "critic_notes": f"Only {len(papers)} papers retrieved. Need at least 3.",
            "rewritten_questions": rewritten,
            "retry_count": retry_count + 1,
            "calibration_bin": Verdict.INSUFFICIENT,
        }

    # INSUFFICIENT — scores too low
    high_score_papers = [p for p in papers if p.hybrid_score >= 0.40]
    if len(high_score_papers) < 3:
        logger.info("Critic: insufficient high-score papers")
        rewritten = _rewrite_questions(state.get("sub_questions") or [], "broaden")
        return {
            "critic_verdict": Verdict.INSUFFICIENT,
            "critic_notes": "Fewer than 3 papers with hybrid_score >= 0.40.",
            "rewritten_questions": rewritten,
            "retry_count": retry_count + 1,
            "calibration_bin": Verdict.INSUFFICIENT,
        }

    # --- Run STALE and CONTRADICTED checks in parallel (both always run) ---
    mean_age = _mean_age_months(papers)
    is_stale = mean_age > 24

    contradictions = _detect_contradictions(papers)
    is_contradicted = len(contradictions) > 0

    # --- Combine signals: CONTRADICTED wins when both fire ---
    if is_contradicted and is_stale:
        verdict = Verdict.CONTRADICTED
        contradiction_details = "; ".join(f"'{c[0]}' vs '{c[1]}': {c[2]}" for c in contradictions)
        notes = f"CONTRADICTED (also stale, mean age {mean_age:.0f} months). Contradictions found: {contradiction_details}"
        strategy = "probe_contradiction"
    elif is_contradicted:
        verdict = Verdict.CONTRADICTED
        contradiction_details = "; ".join(f"'{c[0]}' vs '{c[1]}': {c[2]}" for c in contradictions)
        notes = f"Contradictions found: {contradiction_details}"
        strategy = "probe_contradiction"
    elif is_stale:
        verdict = Verdict.STALE
        notes = f"Evidence is stale (mean age {mean_age:.0f} months > 24 month threshold)"
        strategy = "recent"
    else:
        # PASS — all checks clear
        return {
            "critic_verdict": Verdict.PASS,
            "critic_notes": f"Evidence passes all checks (mean age {mean_age:.0f} months, {len(papers)} papers, no contradictions detected)",
            "retry_count": retry_count,
            "rewritten_questions": [],
            "calibration_bin": Verdict.PASS,
        }

    # --- Non-PASS path: rewrite questions and return ---
    sub_questions = state.get("sub_questions") or []
    rewritten = _rewrite_questions(sub_questions, strategy)
    return {
        "critic_verdict": verdict,
        "critic_notes": notes,
        "rewritten_questions": rewritten,
        "retry_count": retry_count + 1,
        "calibration_bin": verdict,
    }