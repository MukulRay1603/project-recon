import gradio as gr
import uuid
import os
import tempfile
import logging
from dotenv import load_dotenv

from src.graph import run_recon
from src.memory import init_db, load_session, export_session_md

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

init_db()

# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_query(query, session_id, decay_config, history):
    if not query.strip():
        yield history, session_id, "*No query entered.*", "*No query entered.*", "*No session.*", "", None
        return

    if not session_id.strip():
        session_id = str(uuid.uuid4())

    # Add user message
    history = history + [{"role": "user", "content": query}]
    yield history, session_id, "🔍 Running pipeline...", "", "*Updating...*", "", None

    try:
        result = run_recon(
            query=query,
            session_id=session_id,
            decay_config=decay_config,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        history = history + [{"role": "assistant", "content": f"❌ Error: {e}"}]
        yield history, session_id, f"❌ Error: {e}", "", "", "", None
        return

    position = result.get("synthesized_position", "No position generated.")
    history = history + [{"role": "assistant", "content": position}]

    # Critic panel
    verdict = result.get("critic_verdict", "N/A")
    critic_notes = result.get("critic_notes", "")
    retry_count = result.get("retry_count", 0)
    latency = result.get("latency_ms", 0)
    papers_used = len(result.get("retrieved_papers") or [])

    verdict_emoji = {
        "PASS": "✅", "FORCED_PASS": "⚠️",
        "STALE": "🕰️", "CONTRADICTED": "⚡", "INSUFFICIENT": "📉",
    }.get(verdict, "❓")

    critic_panel = f"""### Critic Debug Panel

**Verdict:** {verdict_emoji} `{verdict}`
**Notes:** {critic_notes}
**Retries:** {retry_count}
**Papers used:** {papers_used}
**Latency:** {latency:.0f}ms
**Decay config:** {result.get('decay_config', 'linear')}
"""
    if result.get("rewritten_questions"):
        critic_panel += "\n**Rewritten questions:**\n"
        for q in result["rewritten_questions"]:
            critic_panel += f"- {q}\n"

    # Claims table
    claims = result.get("claim_confidences") or []
    if claims:
        claims_md = "### Claim Confidence Table\n\n"
        claims_md += "| Confidence | Claim | Source | Year | Flag |\n"
        claims_md += "|-----------|-------|--------|------|------|\n"
        for c in claims:
            flag = "⚠️" if c.flagged else ""
            claim_text = c.text[:80] + "..." if len(c.text) > 80 else c.text
            source = c.source_title[:35] + "..." if len(c.source_title) > 35 else c.source_title
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c.confidence, "⚪")
            claims_md += f"| {conf_emoji} {c.confidence.upper()} | {claim_text} | {source} | {c.source_year} | {flag} |\n"
    else:
        claims_md = "*No claims extracted.*"

    # Session sidebar
    session_ctx = load_session(session_id)
    session_md = f"### Session `{session_id[:8]}...`\n\n"
    session_md += f"**Turns:** {len(session_ctx.prior_queries)}\n\n"
    if session_ctx.prior_queries:
        session_md += "**Queries:**\n"
        for q in session_ctx.prior_queries:
            session_md += f"- {q[:60]}\n"
    if session_ctx.flagged_contradictions:
        session_md += "\n**Contradictions flagged:**\n"
        for c in session_ctx.flagged_contradictions[:3]:
            session_md += f"- {c[:80]}\n"

    export_md = result.get("export_md", "")

    yield history, session_id, critic_panel, claims_md, session_md, export_md, None


def export_md_file(export_md_content, session_id):
    if not export_md_content.strip():
        return None
    try:
        fname = f"recon_session_{session_id[:8]}.md"
        path = os.path.join(tempfile.gettempdir(), fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(export_md_content)
        return path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None


def new_session():
    new_id = str(uuid.uuid4())
    return new_id, [], "*Critic verdict will appear here.*", "*Run a query to see claims.*", "*Session history will appear here.*", "", None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RECON — Multi-Agent Research Navigator") as demo:

    gr.Markdown("""
# 🔍 RECON — Multi-Agent Research Navigator
**Temporally-aware ML literature research with staleness detection and contradiction flagging.**
Enter a research query about any ML topic. RECON retrieves live papers, evaluates evidence quality, and synthesizes a cited research position.
""")

    session_id_state = gr.State(str(uuid.uuid4()))
    export_md_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="Research Position",
                height=500,
            )

            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="e.g. What is the current state of KV cache compression in LLMs?",
                    label="Research Query",
                    scale=4,
                    lines=2,
                )
                submit_btn = gr.Button("🔍 Research", variant="primary", scale=1)

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

            claims_output = gr.Markdown(
                value="*Run a query to see claim confidence scores.*"
            )

        with gr.Column(scale=2):

            critic_output = gr.Markdown(
                value="*Critic verdict will appear here after each query.*"
            )

            gr.Markdown("---")

            session_output = gr.Markdown(
                value="*Session history will appear here.*"
            )

            gr.Markdown("---")

            export_btn = gr.Button("📥 Export Session (.md)", variant="secondary")
            export_file = gr.File(label="Download")

    # ---------------------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------------------

    def on_submit(query, session_id, decay_config, history):
        for result in run_query(query, session_id, decay_config, history):
            chat, sid, critic, claims, session, export_md, _ = result
            yield chat, sid, critic, claims, session, export_md, sid

    submit_btn.click(
        fn=on_submit,
        inputs=[query_input, session_id_state, decay_dropdown, chatbot],
        outputs=[chatbot, session_id_state, critic_output, claims_output,
                 session_output, export_md_state, session_display],
    )

    query_input.submit(
        fn=on_submit,
        inputs=[query_input, session_id_state, decay_dropdown, chatbot],
        outputs=[chatbot, session_id_state, critic_output, claims_output,
                 session_output, export_md_state, session_display],
    )

    new_session_btn.click(
        fn=new_session,
        outputs=[session_id_state, chatbot, critic_output,
                 claims_output, session_output, export_md_state, export_file],
    )

    export_btn.click(
        fn=export_md_file,
        inputs=[export_md_state, session_id_state],
        outputs=[export_file],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )