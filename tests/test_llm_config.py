from src.agents import critic, planner, synthesizer
from src.llm_config import LLM_TIMEOUT_S, LLM_MAX_RETRIES


def test_all_llm_clients_have_timeout_and_bounded_retries(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-not-used")  # client is built, never called
    for mod in (critic, planner, synthesizer):
        monkeypatch.setattr(mod, "_llm", None)
        llm = mod.get_llm()
        assert llm.request_timeout == LLM_TIMEOUT_S
        assert llm.max_retries == LLM_MAX_RETRIES
