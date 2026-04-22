from typing import TypedDict, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses — typed objects passed through the graph
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    title: str
    abstract: str
    year: int
    citation_count: int
    paper_id: str
    authors: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # list of paper_ids
    doi: str = ""
    hybrid_score: float = 0.0
    source: str = "semantic_scholar"  # or "web"


@dataclass
class WebResult:
    url: str
    snippet: str
    title: str
    inferred_year: Optional[int] = None
    hybrid_score: float = 0.0
    source: str = "web"


@dataclass
class Claim:
    text: str
    source_title: str
    source_year: int
    confidence: str          # "high" | "medium" | "low"
    flagged: bool = False    # True if contested or contradicted


@dataclass
class SessionContext:
    prior_positions: list[str] = field(default_factory=list)
    flagged_contradictions: list[str] = field(default_factory=list)
    prior_queries: list[str] = field(default_factory=list)


@dataclass
class SessionUpdate:
    position: str
    query: str
    claim_confidences: list[Claim] = field(default_factory=list)
    contradictions_found: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verdict constants — used by Critic agent
# ---------------------------------------------------------------------------

class Verdict:
    PASS = "PASS"
    STALE = "STALE"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"
    FORCED_PASS = "FORCED_PASS"


# ---------------------------------------------------------------------------
# LangGraph state — the single TypedDict shared across all agents
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    # --- Input ---
    original_query: str
    session_id: str

    # --- Planner output ---
    session_context: Optional[SessionContext]
    sub_questions: list[str]

    # --- Retriever output ---
    retrieved_papers: list[Paper]
    citation_graph: dict                  # {paper_id: [cited_paper_ids]}
    web_results: list[WebResult]

    # --- Critic output ---
    critic_verdict: str                   # one of Verdict constants
    critic_notes: str
    rewritten_questions: list[str]
    retry_count: int

    # --- Synthesizer output ---
    synthesized_position: str
    claim_confidences: list[Claim]
    session_update: Optional[SessionUpdate]
    export_md: str                        # NEW v2 — full session as markdown

    # --- Eval / config ---
    decay_config: str                     # "none" | "linear" | "log"
    calibration_bin: str                  # filled by critic for eval aggregation
    latency_ms: float