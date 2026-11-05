"""Shared LLM client settings.

Before this, ChatGroq clients were created with library defaults, so a hung
Groq request could stall a whole pipeline run. Every client now gets an
explicit per-request timeout and a bounded number of retries (the Groq SDK
retries 429/5xx with backoff). Callers already catch exceptions and fall back,
so a timeout degrades one step instead of hanging the run.
"""
import os

LLM_TIMEOUT_S = float(os.environ.get("RECON_LLM_TIMEOUT_S", "30"))
LLM_MAX_RETRIES = int(os.environ.get("RECON_LLM_MAX_RETRIES", "2"))
