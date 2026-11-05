"""RECON Gradio app.

Run locally:  python app.py
The UI streams progress from the LangGraph pipeline (src.graph.stream_recon)
so each agent's step, the critic's verdict and any retries are visible.
"""
import html
import logging
import os
import re
import tempfile
import uuid

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

try:  # warms the embedding model on Spaces; optional for local UI work
    import preload  # noqa: F401
except Exception as e:  # pragma: no cover
    logger.warning(f"Embedding model preload skipped: {e}")


# ---------------------------------------------------------------------------
# Environment checks (shown to the user, never the key itself)
# ---------------------------------------------------------------------------

def env_status() -> dict:
    return {
        "llm": bool(os.getenv("GROQ_API_KEY")),
        "s2_key": bool(os.getenv("S2_API_KEY")),
    }


LLM_MISSING_MSG = (
    "The language model is not configured, so RECON cannot plan, judge or write an answer. "
    "Set GROQ_API_KEY in a local .env file (or in the Space secrets) and restart the app."
)


def friendly_error(e: Exception) -> str:
    text = str(e)
    low = text.lower()
    if "api_key" in low or "401" in low or "invalid api key" in low or "authentication" in low:
        return "The language model rejected the API key. Check GROQ_API_KEY and restart."
    if "429" in low or "rate limit" in low:
        return "A service is rate limiting requests right now. Wait a minute and try again."
    if "timeout" in low or "timed out" in low:
        return "A service took too long to respond. Try again, or try a shorter question."
    if "connection" in low or "network" in low:
        return "A network request failed. Check your connection and try again."
    return "Something went wrong while running the pipeline."


# ---------------------------------------------------------------------------
# Rendering helpers (pure functions, unit tested)
# ---------------------------------------------------------------------------

def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


STEPS = [
    ("planner", "Plan", "Split the question into sub-questions"),
    ("retriever", "Retrieve", "Search Semantic Scholar, OpenAlex and the web"),
    ("counter_evidence", "Counter-evidence", "Search for critiques and failed replications"),
    ("critic", "Critic", "Score reliability, check staleness and contradictions"),
    ("synthesizer", "Synthesize", "Write the answer with citations"),
]

VERDICT_INFO = {
    "PASS": ("Pass", "ok", "The evidence passed the staleness and contradiction checks."),
    "FORCED_PASS": ("Forced pass", "warn", "The retry limit was reached, so the answer uses the evidence available."),
    "STALE": ("Stale", "warn", "Average reliability of the retrieved papers was low, so the critic asked for more recent work."),
    "CONTRADICTED": ("Contradicted", "bad", "The critic's LLM check judged that one paper refutes or supersedes another."),
    "INSUFFICIENT": ("Insufficient", "bad", "Too few relevant papers were found, so the critic broadened the search."),
}

SIGNAL_TONE = {"FOUNDATIONAL": "ok", "CURRENT": "info", "DECLINING": "warn", "SUPERSEDED": "bad"}


def empty_html(text: str) -> str:
    return f"<div class='rc-empty'>{esc(text)}</div>"


def progress_html(events: list, running: str | None = None, failed: bool = False) -> str:
    """events: list of (node, summary). running: node currently expected."""
    done = {n for n, _ in events}
    summaries = {}
    for n, s in events:
        summaries[n] = s  # last summary wins (critic may run more than once)
    retries = sum(1 for n, _ in events if n == "retry_retriever")
    items = []
    for node, label, desc in STEPS:
        if node in done:
            state, icon = "done", "&#10003;"
        elif node == running:
            state, icon = ("failed", "!") if failed else ("running", "")
        else:
            state, icon = "pending", ""
        detail = summaries.get(node) or desc
        items.append(
            f"<li class='rc-step rc-{state}'><span class='rc-dot'>{icon}</span>"
            f"<div><div class='rc-step-label'>{esc(label)}</div>"
            f"<div class='rc-step-detail'>{esc(detail)}</div></div></li>"
        )
    retry_note = ""
    if retries:
        retry_note = (f"<div class='rc-note'>The critic sent the search back {retries} "
                      f"time{'s' if retries != 1 else ''} with rewritten queries.</div>")
    return f"<ol class='rc-steps'>{''.join(items)}</ol>{retry_note}"


def summarize_event(node: str, state: dict) -> str:
    if node == "planner":
        n = len(state.get("sub_questions") or [])
        return f"{n} sub-question{'s' if n != 1 else ''}"
    if node in ("retriever", "retry_retriever"):
        p = len(state.get("retrieved_papers") or [])
        w = len(state.get("web_results") or [])
        return f"{p} papers, {w} web results"
    if node == "counter_evidence":
        n = len(state.get("opposing_papers") or [])
        return f"{n} candidate opposing paper{'s' if n != 1 else ''}" if n else "No opposing papers found"
    if node == "critic":
        v = state.get("critic_verdict", "")
        return f"Verdict: {VERDICT_INFO.get(v, (v,))[0]}"
    if node == "synthesizer":
        return "Answer written"
    return ""


def verdict_html(state: dict) -> str:
    verdict = state.get("critic_verdict") or ""
    if not verdict:
        return empty_html("The critic's verdict and its reasons will appear here.")
    label, tone, meaning = VERDICT_INFO.get(verdict, (verdict, "info", ""))
    retries = state.get("retry_count", 0) or 0
    notes = state.get("critic_notes") or ""
    limit = ""
    if retries >= 2 and verdict not in ("PASS", "FORCED_PASS"):
        limit = "<p class='rc-note'>The retry limit was reached, so the answer was written from the evidence available.</p>"
    rewritten = state.get("rewritten_questions") or []
    rw = ""
    if rewritten:
        rw = ("<div class='rc-sub'>Rewritten queries for the next search</div><ul class='rc-list'>"
              + "".join(f"<li>{esc(q)}</li>" for q in rewritten) + "</ul>")
    latency = state.get("latency_ms") or 0
    meta = [f"{len(state.get('retrieved_papers') or [])} papers",
            f"{retries} retr{'ies' if retries != 1 else 'y'}"]
    if latency:
        meta.append(f"{latency / 1000:.1f} s")
    return (
        f"<div class='rc-card rc-tone-{tone}'>"
        f"<div class='rc-verdict'><span class='rc-pill rc-tone-{tone}'>{esc(label)}</span>"
        f"<span class='rc-meta'>{esc(' · '.join(meta))}</span></div>"
        f"<p>{esc(meaning)}</p>"
        + (f"<div class='rc-sub'>Critic notes</div><p class='rc-mono'>{esc(notes)}</p>" if notes else "")
        + limit + rw + "</div>"
    )


def _signal(rs) -> str | None:
    if not rs:
        return None
    return rs.get("dominant_signal") if isinstance(rs, dict) else getattr(rs, "dominant_signal", None)


def paper_cards_html(papers, reliability_scores=None, notes: str = "", limit: int = 10,
                     empty: str = "No papers yet.") -> str:
    reliability_scores = reliability_scores or {}
    if not papers:
        return empty_html(empty)
    cards = []
    for p in papers[:limit]:
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "") if p.authors else "Unknown authors"
        abstract = p.abstract or ""
        if len(abstract) > 220:
            abstract = abstract[:220].rsplit(" ", 1)[0] + "..."
        badges = []
        sig = _signal(reliability_scores.get(p.paper_id))
        if sig:
            badges.append(f"<span class='rc-pill rc-tone-{SIGNAL_TONE.get(sig, 'info')}'>{esc(sig.title())}</span>")
        if notes and p.title and p.title in notes:
            badges.append("<span class='rc-pill rc-tone-bad'>Named by critic</span>")
        year = p.year or "n.d."
        cards.append(
            f"<div class='rc-paper'><div class='rc-paper-head'><div class='rc-paper-title'>{esc(p.title)}</div>"
            f"<div class='rc-badges'>{''.join(badges)}</div></div>"
            f"<div class='rc-meta'>{esc(authors)} · {esc(year)} · {p.citation_count:,} citations · "
            f"score {p.hybrid_score:.2f}</div>"
            f"<div class='rc-abstract'>{esc(abstract)}</div></div>"
        )
    more = f"<div class='rc-meta'>Showing {limit} of {len(papers)}.</div>" if len(papers) > limit else ""
    return "".join(cards) + more


def claims_html(claims) -> str:
    if not claims:
        return empty_html("Claims and their confidence will appear here after the answer is written.")
    tone = {"high": "ok", "medium": "warn", "low": "bad"}
    rows = "".join(
        f"<tr><td><span class='rc-pill rc-tone-{tone.get(c.confidence, 'info')}'>{esc(c.confidence.title())}</span></td>"
        f"<td>{esc(c.text)}{' <span class=rc-pill>Contested</span>' if c.flagged else ''}</td>"
        f"<td>{esc(c.source_title)}</td><td>{esc(c.source_year)}</td></tr>"
        for c in claims
    )
    return ("<div class='rc-table-wrap'><table class='rc-table'><thead><tr><th>Confidence</th><th>Claim</th>"
            f"<th>Source</th><th>Year</th></tr></thead><tbody>{rows}</tbody></table></div>")


def session_html(ctx, session_id: str) -> str:
    queries = list(getattr(ctx, "prior_queries", []) or [])
    if not queries:
        return empty_html("No questions in this session yet. Follow-up questions reuse this session's context.")
    items = "".join(f"<li>{esc(q)}</li>" for q in queries)
    return (f"<div class='rc-meta'>Session {esc(session_id[:8])} · {len(queries)} question"
            f"{'s' if len(queries) != 1 else ''}</div><ul class='rc-list'>{items}</ul>")


def highlight_citations(text: str) -> str:
    return re.sub(
        r"(\[[A-Za-z][^,\[\]]{1,40},?\s*(?:et al\.?)?,?\s*\d{4}[a-z]?\])",
        r"**\1**",
        text or "",
    )


# ---------------------------------------------------------------------------
# Pipeline runner (generator for streaming UI updates)
# ---------------------------------------------------------------------------

NODE_ORDER = ["planner", "retriever", "counter_evidence", "critic", "synthesizer"]


def _next_node(node: str, state: dict) -> str | None:
    if node == "critic":
        v = state.get("critic_verdict")
        if v in ("PASS", "FORCED_PASS") or (state.get("retry_count") or 0) >= 2:
            return "synthesizer"
        return "critic"  # retry retrieval, then the critic again
    if node == "retry_retriever":
        return "critic"
    if node in NODE_ORDER and node != "synthesizer":
        return NODE_ORDER[NODE_ORDER.index(node) + 1]
    if node == "session_loader":
        return "planner"
    return None


def run_query(query, session_id, decay_config, history, stream_fn=None, status=None):
    """Yields tuples: (history, session_id, progress, verdict, papers, opposing, claims, session, export_md)."""
    status = status or env_status()
    history = list(history or [])
    query = (query or "").strip()
    session_id = session_id or str(uuid.uuid4())
    blank = (empty_html("No papers yet."), empty_html("No counter-evidence yet."))

    if not query:
        yield (history, session_id, progress_html([]), verdict_html({}), *blank,
               claims_html([]), gr.update(), gr.update())
        return

    history.append({"role": "user", "content": query})

    if not status["llm"]:
        history.append({"role": "assistant", "content": LLM_MISSING_MSG})
        err = f"<div class='rc-card rc-tone-bad'><p>{esc(LLM_MISSING_MSG)}</p></div>"
        yield (history, session_id, progress_html([], running="planner", failed=True), err, *blank,
               claims_html([]), gr.update(), gr.update())
        return

    if stream_fn is None:
        from src.graph import stream_recon as stream_fn

    events: list = []
    state: dict = {}
    running = "planner"
    yield (history, session_id, progress_html(events, running), empty_html("Waiting for the critic..."),
           *blank, claims_html([]), gr.update(), gr.update())
    try:
        for node, state in stream_fn(query=query, session_id=session_id, decay_config=decay_config):
            if node == "__done__":
                break
            if node != "session_loader":
                events.append((node, summarize_event(node, state)))
            running = _next_node(node, state)
            yield (history, session_id, progress_html(events, running), verdict_html(state),
                   paper_cards_html(state.get("retrieved_papers"), state.get("paper_reliability_scores"),
                                    state.get("critic_notes", "")),
                   blank[1], claims_html([]), gr.update(), gr.update())
    except Exception as e:
        logger.exception("Pipeline error")
        msg = friendly_error(e)
        history.append({"role": "assistant", "content": msg})
        err = (f"<div class='rc-card rc-tone-bad'><p>{esc(msg)}</p>"
               f"<p class='rc-meta'>Details: {esc(type(e).__name__)}: {esc(str(e)[:200])}</p></div>")
        yield (history, session_id, progress_html(events, running, failed=True), err,
               paper_cards_html(state.get("retrieved_papers")), blank[1], claims_html([]),
               gr.update(), gr.update())
        return

    answer = state.get("synthesized_position") or "No answer was generated."
    history.append({"role": "assistant", "content": highlight_citations(answer)})
    try:
        from src.memory import load_session
        sess = session_html(load_session(session_id), session_id)
    except Exception:
        sess = empty_html("Session memory is unavailable.")
    opposing_note = ("<p class='rc-note'>Found by a search for critiques and replications. The critic checks these "
                     "against the top papers. Being listed here does not mean a contradiction was found. "
                     "They are not used in the staleness score.</p>")
    opp = state.get("opposing_papers") or []
    yield (history, session_id, progress_html(events, None), verdict_html(state),
           paper_cards_html(state.get("retrieved_papers"), state.get("paper_reliability_scores"),
                            state.get("critic_notes", "")),
           opposing_note + paper_cards_html(opp, notes=state.get("critic_notes", ""),
                                            empty="No counter-evidence candidates were found for this question.")
           if opp else empty_html("No counter-evidence candidates were found for this question."),
           claims_html(state.get("claim_confidences")), sess, state.get("export_md", ""))


def export_md_file(export_md_content, session_id):
    if not (export_md_content or "").strip():
        return gr.update(value=None, visible=False)
    path = os.path.join(tempfile.gettempdir(), f"recon_{(session_id or 'session')[:8]}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_md_content)
    return gr.update(value=path, visible=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """

:root { --rc-muted: #475569; } .dark { --rc-muted: #a3b1c6; }
footer { display: none !important; }
.rc-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: var(--body-text-color); }
.rc-header p { margin: 4px 0 0; color: var(--rc-muted); font-size: 0.95rem; max-width: 70ch; }
.rc-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.rc-chips .rc-pill { white-space: normal; }
.rc-banner { border: 1px dashed #b45309; background: rgba(245,158,11,0.12); color: var(--body-text-color);
  padding: 8px 12px; border-radius: 8px; margin-top: 10px; font-weight: 600; }
.rc-section { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--rc-muted); margin: 4px 0 6px; }
.rc-empty { color: var(--rc-muted); font-size: 0.9rem; padding: 10px 2px; }
.rc-pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  border: 1px solid var(--block-border-color); color: var(--body-text-color); white-space: nowrap; }
.rc-tone-ok { --tone: #15803d; } .rc-tone-info { --tone: #1d4ed8; }
.rc-tone-warn { --tone: #b45309; } .rc-tone-bad { --tone: #b91c1c; }
.dark .rc-tone-ok { --tone: #4ade80; } .dark .rc-tone-info { --tone: #93c5fd; }
.dark .rc-tone-warn { --tone: #fbbf24; } .dark .rc-tone-bad { --tone: #f87171; }
.rc-pill[class*="rc-tone-"] { color: var(--tone); border-color: var(--tone);
  background: color-mix(in srgb, var(--tone) 10%, transparent); }
.rc-card { border: 1px solid var(--block-border-color); border-left: 4px solid var(--tone, #64748b);
  border-radius: 8px; padding: 12px 14px; color: var(--body-text-color); }
.rc-card p { margin: 6px 0; line-height: 1.5; font-size: 0.92rem; }
.rc-verdict { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.rc-verdict .rc-pill { font-size: 0.95rem; padding: 3px 12px; }
.rc-meta { color: var(--rc-muted); font-size: 0.82rem; }
.rc-sub { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--rc-muted); margin-top: 10px; }
.rc-mono { font-size: 0.85rem !important; overflow-wrap: anywhere; }
.rc-note { color: var(--rc-muted); font-size: 0.85rem; margin: 8px 0; }
.rc-list { margin: 4px 0 0; padding-left: 18px; font-size: 0.88rem; color: var(--body-text-color); }
.rc-steps { list-style: none; padding: 0; margin: 0; }
.rc-step { display: flex; gap: 10px; align-items: flex-start; padding: 6px 0; }
.rc-dot { flex: 0 0 22px; height: 22px; border-radius: 50%; border: 2px solid var(--block-border-color);
  display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; }
.rc-done .rc-dot { background: #15803d; border-color: #15803d; color: #fff; }
.rc-running .rc-dot { border-color: #1d4ed8; border-top-color: transparent; animation: rc-spin 0.9s linear infinite; }
.rc-failed .rc-dot { background: #b91c1c; border-color: #b91c1c; color: #fff; }
.rc-pending .rc-step-label { color: var(--rc-muted); }
.rc-step-label { font-weight: 600; font-size: 0.92rem; color: var(--body-text-color); }
.rc-step-detail { font-size: 0.82rem; color: var(--rc-muted); }
@keyframes rc-spin { to { transform: rotate(360deg); } }
.rc-paper { border: 1px solid var(--block-border-color); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; }
.rc-paper-head { display: flex; gap: 8px; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; }
.rc-paper-title { font-weight: 600; color: var(--body-text-color); flex: 1 1 260px; }
.rc-badges { display: flex; gap: 6px; flex-wrap: wrap; }
.rc-abstract { font-size: 0.86rem; color: var(--rc-muted); line-height: 1.5; margin-top: 4px; }
.rc-table-wrap { overflow-x: auto; }
.rc-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; color: var(--body-text-color); }
.rc-table th { text-align: left; font-size: 0.78rem; text-transform: uppercase; color: var(--rc-muted);
  border-bottom: 1px solid var(--block-border-color); padding: 6px 8px; }
.rc-table td { border-bottom: 1px solid var(--block-border-color); padding: 8px; vertical-align: top; }
"""

EXAMPLES = [
    "What is the current state of KV cache compression for long-context LLMs?",
    "Is speculative decoding lossless in practice?",
    "What superseded RLHF with PPO for aligning language models?",
    "Does quantization to 4 bits hurt LLM reasoning ability?",
]


def header_html(status: dict, banner: str | None = None) -> str:
    llm = ("<span class='rc-pill rc-tone-ok'>Language model ready</span>" if status["llm"]
           else "<span class='rc-pill rc-tone-bad'>Language model not configured: set GROQ_API_KEY</span>")
    s2 = ("<span class='rc-pill rc-tone-ok'>Semantic Scholar key set</span>" if status["s2_key"]
          else "<span class='rc-pill rc-tone-warn'>No Semantic Scholar key: slower public rate limit</span>")
    ban = f"<div class='rc-banner'>{esc(banner)}</div>" if banner else ""
    return (
        "<div class='rc-header'><h1>RECON</h1>"
        "<p>Research assistant for ML literature. It retrieves papers, checks whether the evidence is stale "
        "or contradicted, and writes a cited answer. Each paper gets a reliability label.</p>"
        f"<div class='rc-chips'>{llm}{s2}</div>{ban}</div>"
    )


def build_demo(stream_fn=None, status=None, banner: str | None = None) -> gr.Blocks:
    status = status or env_status()

    with gr.Blocks(title="RECON") as demo:
        gr.HTML(header_html(status, banner))
        session_id_state = gr.State("")
        export_md_state = gr.State("")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=280):
                chatbot = gr.Chatbot(
                    label="Answer",
                    height=440,
                    render_markdown=True,
                    placeholder="Ask a research question. RECON shows each step as it runs, "
                                "then answers with citations.",
                )
                query_input = gr.Textbox(
                    placeholder="Ask about an ML research topic",
                    label="Research question",
                    lines=2,
                    max_lines=5,
                )
                with gr.Row():
                    submit_btn = gr.Button("Research", variant="primary", scale=3)
                    new_session_btn = gr.Button("New session", variant="secondary", scale=1, min_width=120)
                gr.Examples(EXAMPLES, inputs=query_input, label="Example questions")
                with gr.Accordion("Settings", open=False):
                    decay_dropdown = gr.Dropdown(
                        choices=["linear", "log", "none"], value="linear",
                        label="Recency decay used in paper ranking",
                    )

            with gr.Column(scale=2, min_width=260):
                gr.HTML("<div class='rc-section'>Pipeline</div>")
                progress_output = gr.HTML(progress_html([]))
                gr.HTML("<div class='rc-section'>Critic verdict</div>")
                verdict_output = gr.HTML(verdict_html({}))

        with gr.Tabs():
            with gr.Tab("Papers used"):
                papers_output = gr.HTML(empty_html("Papers will appear here, each with a reliability label."))
            with gr.Tab("Counter-evidence"):
                opposing_output = gr.HTML(empty_html("Candidate opposing papers will appear here."))
            with gr.Tab("Claims"):
                claims_output = gr.HTML(claims_html([]))
            with gr.Tab("Session"):
                session_output = gr.HTML(empty_html("No questions in this session yet."))
                export_btn = gr.Button("Export session as Markdown", variant="secondary")
                export_file = gr.File(label="Download", visible=False)

        outputs = [chatbot, session_id_state, progress_output, verdict_output, papers_output,
                   opposing_output, claims_output, session_output, export_md_state]

        def on_submit(query, session_id, decay_config, history):
            yield from run_query(query, session_id, decay_config, history, stream_fn=stream_fn, status=status)

        for trigger in (submit_btn.click, query_input.submit):
            trigger(on_submit, [query_input, session_id_state, decay_dropdown, chatbot], outputs,
                    concurrency_limit=2).then(lambda: "", None, query_input)

        def new_session():
            return ([], str(uuid.uuid4()), progress_html([]), verdict_html({}),
                    empty_html("Papers will appear here, each with a reliability label."),
                    empty_html("Candidate opposing papers will appear here."), claims_html([]),
                    empty_html("No questions in this session yet."), "", gr.update(value=None, visible=False))

        new_session_btn.click(new_session, None, outputs + [export_file])
        export_btn.click(export_md_file, [export_md_state, session_id_state], export_file)

    return demo


if __name__ == "__main__":
    from src.memory import init_db

    init_db()
    build_demo().launch(css=CSS)
