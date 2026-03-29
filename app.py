import gradio as gr
import uuid
import os
import tempfile
import logging
from dotenv import load_dotenv

from src.graph import run_recon
from src.memory import init_db, load_session

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

init_db()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VERDICT_META = {
    "PASS":        ("✅", "#22c55e", "Pass"),
    "FORCED_PASS": ("⚠️", "#f59e0b", "Forced Pass"),
    "STALE":       ("🕰️", "#f59e0b", "Stale"),
    "CONTRADICTED":("⚡", "#ef4444", "Contradicted"),
    "INSUFFICIENT":("📉", "#ef4444", "Insufficient"),
}

CONF_META = {
    "high":   ("🟢", "#22c55e"),
    "medium": ("🟡", "#f59e0b"),
    "low":    ("🔴", "#ef4444"),
}

def _highlight_citations(text: str) -> str:
    """Wrap [Author et al., Year] citations in styled spans."""
    import re
    return re.sub(
        r"(\[[A-Za-z][^,\[\]]{1,40},?\s*(?:et al\.?)?,?\s*\d{4}[a-z]?\])",
        r'<span style="background:#1e3a5f;color:#93c5fd;padding:1px 5px;'
        r'border-radius:4px;font-size:0.88em;font-weight:500">\1</span>',
        text
    )

def _paper_cards_html(papers) -> str:
    """Render retrieved papers as styled cards."""
    if not papers:
        return "<p style='color:#6b7280;font-style:italic'>No papers retrieved.</p>"

    cards = []
    for p in papers[:8]:
        score_color = "#22c55e" if p.hybrid_score >= 0.6 else "#f59e0b" if p.hybrid_score >= 0.4 else "#ef4444"
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "") if p.authors else "Unknown"
        abstract_preview = (p.abstract[:180] + "...") if p.abstract and len(p.abstract) > 180 else (p.abstract or "")

        cards.append(f"""
<div style="border:1px solid #2d3748;border-radius:8px;padding:12px 14px;
            margin-bottom:8px;background:#1a1f2e;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px">
    <span style="font-weight:600;font-size:0.92em;color:#e2e8f0;flex:1;margin-right:8px">{p.title}</span>
    <span style="background:{score_color}22;color:{score_color};border:1px solid {score_color}44;
                 padding:2px 8px;border-radius:12px;font-size:0.78em;white-space:nowrap;font-weight:600">
      {p.hybrid_score:.3f}
    </span>
  </div>
  <div style="color:#94a3b8;font-size:0.8em;margin-bottom:6px">
    {authors} · {p.year} · {p.citation_count:,} citations · <span style="color:#64748b">{p.source}</span>
  </div>
  <div style="color:#9ca3af;font-size:0.8em;line-height:1.5">{abstract_preview}</div>
</div>""")

    return "".join(cards)


def _verdict_badge_html(verdict: str, notes: str, retry: int,
                        papers: int, latency: float, decay: str,
                        rewritten: list) -> str:
    emoji, color, label = VERDICT_META.get(verdict, ("❓", "#6b7280", verdict))

    rw_html = ""
    if rewritten:
        items = "".join(f"<li style='color:#94a3b8;font-size:0.82em'>{q}</li>" for q in rewritten)
        rw_html = f"<div style='margin-top:10px'><div style='color:#64748b;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px'>Rewritten queries</div><ul style='margin:0;padding-left:16px'>{items}</ul></div>"

    return f"""
<div style="border:1px solid {color}44;border-radius:10px;padding:14px 16px;background:{color}11;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
    <span style="font-size:1.4em">{emoji}</span>
    <span style="font-size:1.1em;font-weight:700;color:{color}">{label}</span>
    <span style="margin-left:auto;color:#64748b;font-size:0.8em">{latency:.0f}ms</span>
  </div>
  <div style="color:#cbd5e1;font-size:0.88em;line-height:1.6;margin-bottom:8px">{notes}</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    <span style="color:#64748b;font-size:0.8em">📄 {papers} papers</span>
    <span style="color:#64748b;font-size:0.8em">🔁 {retry} retries</span>
    <span style="color:#64748b;font-size:0.8em">📐 {decay} decay</span>
  </div>
  {rw_html}
</div>"""


def _claims_html(claims) -> str:
    if not claims:
        return "<p style='color:#6b7280;font-style:italic'>No claims extracted.</p>"

    rows = ""
    for c in claims:
        emoji, color = CONF_META.get(c.confidence, ("⚪", "#6b7280"))
        flag = " <span title='Contested claim' style='color:#f59e0b'>⚠️</span>" if c.flagged else ""
        rows += f"""
<tr style="border-bottom:1px solid #2d3748">
  <td style="padding:8px 10px;white-space:nowrap">
    <span style="color:{color};font-weight:600;font-size:0.82em">{emoji} {c.confidence.upper()}</span>
  </td>
  <td style="padding:8px 10px;color:#e2e8f0;font-size:0.84em;line-height:1.5">{c.text}{flag}</td>
  <td style="padding:8px 10px;color:#94a3b8;font-size:0.8em;white-space:nowrap">{c.source_title[:35]}...</td>
  <td style="padding:8px 10px;color:#64748b;font-size:0.8em">{c.source_year}</td>
</tr>"""

    return f"""
<table style="width:100%;border-collapse:collapse;font-family:inherit">
  <thead>
    <tr style="border-bottom:2px solid #374151">
      <th style="padding:8px 10px;text-align:left;color:#6b7280;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em">Confidence</th>
      <th style="padding:8px 10px;text-align:left;color:#6b7280;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em">Claim</th>
      <th style="padding:8px 10px;text-align:left;color:#6b7280;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em">Source</th>
      <th style="padding:8px 10px;text-align:left;color:#6b7280;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em">Year</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


def _session_html(session_ctx, session_id: str) -> str:
    turns = len(session_ctx.prior_queries)
    if turns == 0:
        return f"<p style='color:#6b7280;font-size:0.85em'>Session <code>{session_id[:8]}...</code> — no turns yet.</p>"

    items = "".join(
        f"<li style='color:#94a3b8;font-size:0.83em;padding:3px 0;border-bottom:1px solid #2d3748'>{q[:70]}</li>"
        for q in session_ctx.prior_queries
    )

    contradictions = ""
    if session_ctx.flagged_contradictions:
        c_items = "".join(
            f"<li style='color:#fca5a5;font-size:0.8em;padding:2px 0'>{c[:80]}</li>"
            for c in session_ctx.flagged_contradictions[:3]
        )
        contradictions = f"""
<div style="margin-top:10px">
  <div style="color:#ef4444;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">⚡ Contradictions flagged</div>
  <ul style="margin:0;padding-left:16px">{c_items}</ul>
</div>"""

    return f"""
<div style="font-family:inherit">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
    <code style="background:#1e293b;color:#7dd3fc;padding:2px 8px;border-radius:4px;font-size:0.82em">{session_id[:8]}...</code>
    <span style="color:#64748b;font-size:0.82em">{turns} turn{"s" if turns != 1 else ""}</span>
  </div>
  <div style="color:#64748b;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">Queries</div>
  <ul style="margin:0;padding-left:16px">{items}</ul>
  {contradictions}
</div>"""


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def run_query(query, session_id, decay_config, history):
    if not query.strip():
        yield history, session_id, "", "", "", "", "", None
        return

    if not session_id.strip():
        session_id = str(uuid.uuid4())

    history = history + [{"role": "user", "content": query}]
    yield history, session_id, \
        _verdict_badge_html("", "🔍 Running pipeline...", 0, 0, 0, decay_config, []), \
        "", "", "", "", None

    try:
        result = run_recon(query=query, session_id=session_id, decay_config=decay_config)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        history = history + [{"role": "assistant", "content": f"❌ Error: {e}"}]
        yield history, session_id, f"<p style='color:#ef4444'>❌ {e}</p>", "", "", "", "", None
        return

    position = result.get("synthesized_position", "No position generated.")
    highlighted = _highlight_citations(position)
    history = history + [{"role": "assistant", "content": highlighted}]

    verdict       = result.get("critic_verdict", "N/A")
    critic_notes  = result.get("critic_notes", "")
    retry_count   = result.get("retry_count", 0)
    latency       = result.get("latency_ms", 0)
    papers_used   = len(result.get("retrieved_papers") or [])
    rewritten     = result.get("rewritten_questions") or []

    verdict_html  = _verdict_badge_html(verdict, critic_notes, retry_count,
                                         papers_used, latency, decay_config, rewritten)
    claims_html   = _claims_html(result.get("claim_confidences") or [])
    papers_html   = _paper_cards_html(result.get("retrieved_papers") or [])
    session_ctx   = load_session(session_id)
    session_html  = _session_html(session_ctx, session_id)
    export_md     = result.get("export_md", "")

    yield history, session_id, verdict_html, claims_html, papers_html, session_html, export_md, None


def export_md_file(export_md_content, session_id):
    if not export_md_content.strip():
        return None
    try:
        path = os.path.join(tempfile.gettempdir(), f"recon_{session_id[:8]}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(export_md_content)
        return path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None


def new_session():
    new_id = str(uuid.uuid4())
    return new_id, [], "", "", "", "", "", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
.chatbot-wrap .message-wrap { font-size: 0.92em; line-height: 1.7; }
footer { display: none !important; }
"""

with gr.Blocks(title="RECON") as demo:

    gr.HTML("""
<div style="padding:20px 0 10px;border-bottom:1px solid #2d3748;margin-bottom:20px">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
    <span style="font-size:1.6em">🔍</span>
    <h1 style="margin:0;font-size:1.5em;font-weight:700;color:#f1f5f9">RECON</h1>
    <span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;border-radius:12px;
                 font-size:0.75em;font-weight:600;letter-spacing:0.04em">MULTI-AGENT</span>
  </div>
  <p style="margin:0;color:#64748b;font-size:0.88em">
    Temporally-aware ML literature research · Live Semantic Scholar · Staleness detection · Contradiction flagging
  </p>
</div>
""")

    session_id_state = gr.State(str(uuid.uuid4()))
    export_md_state  = gr.State("")

    with gr.Row(equal_height=False):

        # ── Left column ──────────────────────────────────────────────────
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="Research Position",
                height=480,
                render_markdown=True,
                elem_classes=["chatbot-wrap"],
            )

            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="e.g. What is the current state of KV cache compression in LLMs?",
                    label="Research Query",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("🔍 Research", variant="primary", scale=1, min_width=120)

            with gr.Row():
                decay_dropdown = gr.Dropdown(
                    choices=["linear", "log", "none"],
                    value="linear",
                    label="Recency decay",
                    scale=1,
                )
                new_session_btn = gr.Button("🔄 New Session", scale=1)
                session_display = gr.Textbox(
                    label="Session ID",
                    interactive=False,
                    scale=2,
                )

            with gr.Accordion("📄 Retrieved Papers", open=False):
                papers_output = gr.HTML(
                    value="<p style='color:#6b7280;font-style:italic;padding:8px'>Run a query to see retrieved papers.</p>"
                )

            with gr.Accordion("📊 Claim Confidence Table", open=True):
                claims_output = gr.HTML(
                    value="<p style='color:#6b7280;font-style:italic;padding:8px'>Run a query to see claim confidence scores.</p>"
                )

        # ── Right column ─────────────────────────────────────────────────
        with gr.Column(scale=2):

            gr.HTML("<div style='color:#94a3b8;font-size:0.78em;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px'>Critic Debug Panel</div>")
            critic_output = gr.HTML(
                value="<p style='color:#6b7280;font-style:italic;font-size:0.88em'>Critic verdict will appear here.</p>"
            )

            gr.HTML("<hr style='border-color:#2d3748;margin:14px 0'>")

            gr.HTML("<div style='color:#94a3b8;font-size:0.78em;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px'>Session Memory</div>")
            session_output = gr.HTML(
                value="<p style='color:#6b7280;font-style:italic;font-size:0.88em'>Session history will appear here.</p>"
            )

            gr.HTML("<hr style='border-color:#2d3748;margin:14px 0'>")

            export_btn  = gr.Button("📥 Export Session (.md)", variant="secondary")
            export_file = gr.File(label="Download")

    # ── Events ───────────────────────────────────────────────────────────

    def on_submit(query, session_id, decay_config, history):
        for r in run_query(query, session_id, decay_config, history):
            chat, sid, critic, claims, papers, session, export_md, _ = r
            yield chat, sid, critic, claims, papers, session, export_md, sid

    submit_btn.click(
        fn=on_submit,
        inputs=[query_input, session_id_state, decay_dropdown, chatbot],
        outputs=[chatbot, session_id_state, critic_output, claims_output,
                 papers_output, session_output, export_md_state, session_display],
    )

    query_input.submit(
        fn=on_submit,
        inputs=[query_input, session_id_state, decay_dropdown, chatbot],
        outputs=[chatbot, session_id_state, critic_output, claims_output,
                 papers_output, session_output, export_md_state, session_display],
    )

    new_session_btn.click(
        fn=new_session,
        outputs=[session_id_state, chatbot, critic_output, claims_output,
                 papers_output, session_output, export_md_state, export_file],
    )

    export_btn.click(
        fn=export_md_file,
        inputs=[export_md_state, session_id_state],
        outputs=[export_file],
    )

demo.launch()