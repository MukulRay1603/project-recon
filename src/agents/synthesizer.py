import logging
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ResearchState, Paper, Claim, SessionUpdate, WebResult
from src.memory import save_turn, log_verdict

load_dotenv()
logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now().year

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    return _llm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM = """You are the Synthesizer agent in RECON, a multi-agent ML research navigator.

Your job: synthesize retrieved evidence into a structured research brief.

OUTPUT FORMAT — follow this exactly:

## Overview
One paragraph (3-4 sentences) summarizing the current state of the field. Lead with the most important finding.

## Key Findings
3-5 bullet points. Each bullet = one concrete finding with a citation. Bold the key technical term.
Format: • **Term**: finding [Citation]

## Active Debates
1-2 paragraphs on where the field actively disagrees. Name both camps explicitly with citations.
If no contradiction was flagged, write one paragraph on open questions instead.

## Outlook
2-3 sentences on where the field is heading and what problems remain unsolved.

---

Rules:
- Every factual claim MUST have an inline citation: [Author et al., Year]
- Use ONLY the provided papers as sources. Do not invent citations.
- Be precise and technical. This is for ML researchers.
- Bold key technical terms using **term** markdown syntax.
- Keep each section tight — no filler sentences."""


CLAIM_EXTRACTOR_SYSTEM = """You are extracting individual claims from a research synthesis.

Given a research position, extract the 4-6 most important factual claims.
For each claim, assess confidence based on how many sources support it and how recent they are.

Output ONLY a JSON array:
[
  {
    "text": "the specific claim in one sentence",
    "source_title": "paper title that supports this claim",
    "source_year": 2024,
    "confidence": "high",
    "flagged": false
  }
]

Confidence levels:
- high: multiple recent papers (2022+) agree
- medium: older consensus or single recent paper
- low: single source, contested, or pre-2020

flagged: true if the claim is contested or came from a CONTRADICTED verdict."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_evidence(papers: list[Paper], web_results: list[WebResult]) -> str:
    """Format retrieved evidence for the LLM prompt."""
    lines = ["=== Retrieved Papers ==="]
    for i, p in enumerate(papers[:8], 1):
        authors = p.authors[:2] if p.authors else ["Unknown"]
        author_str = ", ".join(authors)
        if len(p.authors) > 2:
            author_str += " et al."
        lines.append(
            f"\n[{i}] {p.title}\n"
            f"    Authors: {author_str} | Year: {p.year} | Citations: {p.citation_count}\n"
            f"    Abstract: {p.abstract[:300]}..."
        )

    if web_results:
        lines.append("\n=== Web Sources ===")
        for i, r in enumerate(web_results[:3], 1):
            lines.append(f"\n[W{i}] {r.title}\n    {r.snippet[:200]}...")

    return "\n".join(lines)


def _parse_position(raw: str) -> tuple[str, str]:
    """Extract position and summary from LLM output."""
    position = raw.strip()
    summary = ""

    outlook_match = re.search(r"## Outlook\s*(.*?)$", raw, re.DOTALL)
    if outlook_match:
        summary = outlook_match.group(1).strip()[:300]

    return position, summary


def _extract_claims(
    position: str,
    papers: list[Paper],
    is_contradicted: bool,
) -> list[Claim]:
    """Use LLM to extract and score individual claims from the position."""
    paper_titles = "\n".join(
        f"- {p.title} ({p.year})" for p in papers[:8]
    )

    prompt = f"""Research position:
{position[:1500]}

Available source papers:
{paper_titles}

Contradicted verdict: {is_contradicted}

Extract 4-6 key claims as a JSON array."""

    try:
        response = get_llm().invoke([
            SystemMessage(content=CLAIM_EXTRACTOR_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            claims_data = json.loads(match.group())
            claims = []
            for c in claims_data:
                if not isinstance(c, dict):
                    continue
                claims.append(Claim(
                    text=c.get("text", ""),
                    source_title=c.get("source_title", ""),
                    source_year=int(c.get("source_year", 0)),
                    confidence=c.get("confidence", "medium"),
                    flagged=bool(c.get("flagged", False)),
                ))
            return claims
    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")

    return []


def _mean_age_months(papers: list[Paper]) -> float:
    """Compute mean age of retrieved papers in months."""
    ages = [
        (CURRENT_YEAR - p.year) * 12
        for p in papers
        if p.year and p.year > 0
    ]
    return round(sum(ages) / len(ages), 1) if ages else 0.0


# ---------------------------------------------------------------------------
# Synthesizer node
# ---------------------------------------------------------------------------

def synthesizer_node(state: ResearchState) -> ResearchState:
    """
    Reads:  retrieved_papers, web_results, original_query,
            critic_verdict, session_context, session_id
    Writes: synthesized_position, claim_confidences,
            session_update, export_md
    """
    papers        = state.get("retrieved_papers") or []
    web_results   = state.get("web_results") or []
    query         = state.get("original_query", "")
    critic_verdict = state.get("critic_verdict", "")
    critic_notes  = state.get("critic_notes", "")
    session_id    = state.get("session_id", "")
    session_ctx   = state.get("session_context")
    retry_count   = state.get("retry_count", 0)

    is_contradicted = critic_verdict == "CONTRADICTED"

    # Build prior context block
    prior_context = ""
    if session_ctx and session_ctx.prior_positions:
        recent = session_ctx.prior_positions[-2:]
        prior_context = "\n\nPrior session context:\n" + "\n".join(
            f"- {p[:200]}" for p in recent
        )

    # Format evidence
    evidence = _format_evidence(papers, web_results)

    # Build the synthesis prompt
    verdict_note = ""
    if is_contradicted:
        verdict_note = "\nNOTE: The critic detected a contradiction in the evidence. Explicitly name both camps.\n"
    elif critic_verdict == "FORCED_PASS":
        verdict_note = "\nNOTE: Evidence is limited. Acknowledge uncertainty where appropriate.\n"

    user_prompt = f"""Research query: {query}
{verdict_note}
{evidence}
{prior_context}

Synthesize a research position on this query using the evidence above."""

    logger.info(f"Synthesizer: generating position for '{query[:50]}'")

    try:
        response = get_llm().invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()
    except Exception as e:
        logger.error(f"Synthesizer LLM call failed: {e}")
        raw = f"POSITION:\nUnable to synthesize position due to error: {e}\n\nSUMMARY:\nError occurred during synthesis."

    position, summary = _parse_position(raw)

    if not position:
        position = raw

    # Extract claims with confidence scoring
    claims = _extract_claims(position, papers, is_contradicted)
    logger.info(f"Synthesizer: extracted {len(claims)} claims")

    # Build session update
    contradictions_found = []
    if is_contradicted and critic_notes:
        contradictions_found = [critic_notes]

    session_update = SessionUpdate(
        query=query,
        position=position,
        claim_confidences=claims,
        contradictions_found=contradictions_found,
    )

    # Persist turn to SQLite
    if session_id:
        try:
            save_turn(session_id, session_update)
            logger.info(f"Synthesizer: session turn saved for {session_id}")
        except Exception as e:
            logger.warning(f"Session save failed: {e}")

    # ── Log verdict for leaderboard generation ────────────────────────────────
    # Every completed pipeline run is logged — verdict, papers, query, age.
    # This table accumulates real usage data that generates the superseded
    # paper leaderboard from actual queries, not pre-written content.
    try:
        log_verdict(
            query=query,
            verdict=critic_verdict or "PASS",
            papers=papers,
            critic_notes=critic_notes,
            mean_age_months=_mean_age_months(papers),
            retry_count=retry_count,
            session_id=session_id,
        )
        logger.info(f"Synthesizer: verdict logged ({critic_verdict})")
    except Exception as e:
        logger.warning(f"Verdict log failed (non-fatal): {e}")
    # ── End verdict logging ───────────────────────────────────────────────────

    # --- Phase 2.6: Trust summary block ---
    reliability_scores = state.get("paper_reliability_scores", {})
    if reliability_scores and papers:
        trust_lines = ["\n\n---\n## Evidence Trust Summary\n"]
        for p in papers[:8]:  # same top-8 window synthesizer already uses
            rs = reliability_scores.get(p.paper_id)
            if not rs:
                continue
            # Format author citation
            if p.authors and len(p.authors) > 0:
                first_author = p.authors[0] if isinstance(p.authors, list) else str(p.authors).split(",")[0]
                last_name = first_author.strip().split()[-1] if first_author.strip() else "Unknown"
                citation_label = f"{last_name} et al., {p.year}"
            else:
                citation_label = f"Paper {p.year}"

            # Signal label to human-readable
            signal_map = {
                "FOUNDATIONAL": "HIGH — foundational work",
                "CURRENT": "HIGH — current",
                "DECLINING": "MEDIUM — declining relevance",
                "SUPERSEDED": "LOW — likely superseded",
            }
            dominant = rs.get("dominant_signal", "DECLINING") if isinstance(rs, dict) else rs.dominant_signal
            score_val = rs.get("score", 0.0) if isinstance(rs, dict) else rs.score
            reason_val = rs.get("reason", "") if isinstance(rs, dict) else rs.reason
            reliability_label = signal_map.get(dominant, f"score={score_val:.2f}")

            trust_lines.append(
                f"**[{citation_label}]**  \n"
                f"Reliability: {reliability_label}  \n"
                f"Reason: {reason_val}\n"
            )
        position = position + "".join(trust_lines)

    # Build markdown export
    export_md = _build_export_md(query, position, summary, claims, state)

    return {
        **state,
        "synthesized_position": position,
        "claim_confidences": claims,
        "session_update": session_update,
        "export_md": export_md,
    }


def _build_export_md(
    query: str,
    position: str,
    summary: str,
    claims: list[Claim],
    state: ResearchState,
) -> str:
    """Build the full session markdown export."""
    lines = [
        f"# RECON Research Note",
        f"",
        f"**Query:** {query}",
        f"**Verdict:** {state.get('critic_verdict', 'N/A')}",
        f"**Papers used:** {len(state.get('retrieved_papers') or [])}",
        f"",
        f"---",
        f"",
        f"## Position",
        f"",
        position,
        f"",
    ]

    if summary:
        lines += [f"**Summary:** {summary}", ""]

    if claims:
        lines += ["## Claims", ""]
        for c in claims:
            flag = " ⚠️" if c.flagged else ""
            lines.append(
                f"- **[{c.confidence.upper()}]** {c.text} "
                f"— *{c.source_title} ({c.source_year})*{flag}"
            )
        lines.append("")

    critic_notes = state.get("critic_notes", "")
    if critic_notes:
        lines += ["## Critic notes", "", critic_notes, ""]

    return "\n".join(lines)