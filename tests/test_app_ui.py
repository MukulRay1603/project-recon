"""UI helper and runner tests. No LLM, no network (uses the sample stream)."""
import pytest

gr = pytest.importorskip("gradio")

import app  # noqa: E402
from scripts.ui_sample_app import sample_stream  # noqa: E402
from src.state import Paper  # noqa: E402


def _collect(gen):
    return list(gen)


def test_missing_llm_key_shows_clear_message_and_does_not_run():
    def boom(**k):
        raise AssertionError("pipeline must not run without a key")
    out = _collect(app.run_query("q?", "", "linear", [], stream_fn=boom, status={"llm": False, "s2_key": False}))
    history = out[-1][0]
    assert history[-1]["content"] == app.LLM_MISSING_MSG
    assert "GROQ_API_KEY" in out[-1][3]


def test_empty_query_is_a_no_op():
    out = _collect(app.run_query("   ", "", "linear", [], status={"llm": True, "s2_key": True}))
    assert out[-1][0] == []


def test_streams_progress_and_final_state():
    out = _collect(app.run_query("kv cache?", "sid", "linear", [], stream_fn=sample_stream,
                                 status={"llm": True, "s2_key": True}))
    assert len(out) > 3  # one update per agent step
    final = out[-1]
    assert final[0][-1]["role"] == "assistant"
    assert "Answer written" in final[2] and "sent the search back 1 time" in final[2]
    assert "Pass" in final[3]
    assert "Superseded" in final[4]
    assert "does not mean a contradiction was found" in final[5]


def test_pipeline_error_is_friendly():
    def failing(**k):
        yield "planner", {"sub_questions": ["a"]}
        raise RuntimeError("Error code: 429 rate limit exceeded")
    out = _collect(app.run_query("q?", "s", "linear", [], stream_fn=failing, status={"llm": True, "s2_key": True}))
    assert "rate limiting" in out[-1][0][-1]["content"]
    assert "rc-failed" in out[-1][2]


def test_paper_titles_are_escaped():
    p = Paper(title="<script>x</script>", abstract="a < b", year=2024, citation_count=1, paper_id="p")
    html = app.paper_cards_html([p])
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_ui_copy_has_no_em_dashes():
    src = open(app.__file__, encoding="utf-8").read()
    assert "—" not in src


def test_demo_builds():
    assert app.build_demo(stream_fn=sample_stream, status={"llm": True, "s2_key": False}) is not None
