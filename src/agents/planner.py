import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ResearchState, SessionContext

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — shared across agents, lazy init
# ---------------------------------------------------------------------------
_llm: ChatGroq | None = None

def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return _llm


# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You are the Planner agent in RECON, a multi-agent ML research navigator.

Your job: decompose the user's research query into exactly 2-3 sub-questions.
Each sub-question must target a DIFFERENT temporal range:
  - Foundational: seminal/classic work that established the field
  - Recent advances: work from the last 2-3 years
  - Open/contested: where the field actively disagrees or has open problems

Rules:
- Output ONLY a numbered list. No preamble, no explanation.
- Each sub-question must be self-contained and searchable on its own.
- If session context is provided, do NOT re-ask questions already answered.
- Keep each sub-question under 20 words.

Example output:
1. What are the foundational methods for KV cache compression in transformers?
2. What are the most effective KV cache compression techniques published in 2023-2024?
3. What are the open challenges and contested approaches in KV cache compression?"""


# ---------------------------------------------------------------------------
# Planner node — called by LangGraph
# ---------------------------------------------------------------------------
def planner_node(state: ResearchState) -> ResearchState:
    """
    Reads: original_query, session_context
    Writes: sub_questions
    """
    query = state["original_query"]
    session_ctx: SessionContext = state.get("session_context") or SessionContext()

    # Build context block for the LLM
    context_block = ""
    if session_ctx.prior_queries:
        prior = "\n".join(f"- {q}" for q in session_ctx.prior_queries[-3:])
        context_block = f"\n\nAlready answered in this session:\n{prior}\nDo not repeat these."

    user_prompt = f"Research query: {query}{context_block}"

    logger.info(f"Planner decomposing: {query[:60]}")

    try:
        response = get_llm().invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()
    except Exception as e:
        logger.error(f"Planner LLM call failed: {e}")
        # Fallback: use the original query as a single sub-question
        return {**state, "sub_questions": [query]}

    # Parse numbered list
    sub_questions = _parse_numbered_list(raw)

    if not sub_questions:
        logger.warning("Planner returned unparseable output, using raw query")
        sub_questions = [query]

    logger.info(f"Planner produced {len(sub_questions)} sub-questions")
    for i, q in enumerate(sub_questions, 1):
        logger.info(f"  {i}. {q}")

    return {**state, "sub_questions": sub_questions}


def _parse_numbered_list(text: str) -> list[str]:
    """Parse '1. question\n2. question' into a list of strings."""
    import re
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match lines starting with a number and period/dot
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:  # skip very short lines
                questions.append(q)
    return questions[:3]  # max 3 sub-questions