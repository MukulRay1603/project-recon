# RECON UI report: branch `ui/polish-2026-11`

This branch was built on top of `upgrade/recon-2026-11`. Nothing is merged, and the Hugging Face Space was not redeployed. Scoring, thresholds, verdict rules and the main retriever's `_to_search_query` are unchanged.

## Flaws found in the old UI

I ran the old `app.py` at 1280 px and 390 px wide. These are the problems I found:

1. **Invisible title.** The colors were written for a dark theme (for example, title text `#f1f5f9`) and hardcoded, but the app renders in light mode. The "RECON" title was close to invisible, and paper cards would have shown as dark boxes on a white page.
2. **Styles never applied.** The `CSS` string was defined but never passed to Gradio, so none of it took effect.
3. **No progress.** The whole pipeline ran as one blocking call, and the user saw only "Running pipeline..." with a 0 ms timer. Retries were not visible while they happened.
4. **Empty states gave no guidance.** The chat area was a 480 px blank box with no hint and no example questions. A large empty "Download" box was always visible.
5. **Developer-facing labels.** "Critic Debug Panel" was jargon. A read-only "Session ID" box sat empty on load. Raw hybrid scores were the most prominent badge on each paper card.
6. **Bad phone layout.** The critic verdict and session panel ended up below everything else, and the query box was squeezed next to the button.
7. **No help when a key is missing.** Without `GROQ_API_KEY`, the user got a raw exception string.
8. **Bugs:**
   - Paper titles, abstracts and claims were inserted into the page as raw HTML, with no escaping.
   - The function rendering paper cards used a mutable default argument (`reliability_scores: dict = {}`).
   - `app.py` called `demo.launch()` on import, so it couldn't be imported in tests.
   - `import preload` crashed the app when `sentence_transformers` wasn't installed.
   - The session copy contained an em dash.

## What changed

- **`src/graph.py`:** a new `stream_recon()` runs the same compiled graph with `stream_mode="updates"`. It yields the state after each node, plus a final `("__done__", state)` with the latency. `run_recon()` behaves exactly as before.
- **`app.py`, rewritten.** It still uses the same Chatbot ("re use the chat that worked before"). The page now has:
  - **Header:** a plain description and status chips. One says whether the language model is configured. The other says whether a Semantic Scholar key is set; without one, the public rate limit makes runs slower.
  - **Pipeline tracker:** Plan, Retrieve, Counter-evidence, Critic, Synthesize. Each step shows a live state and a short result, such as "3 papers, 3 web results" or "Verdict: Stale". It also notes how many times the critic sent the search back.
  - **Critic verdict card:** the verdict, a plain explanation of what it means, the critic's own notes (the "why"), and the retry count. It lists rewritten queries when there are any, and notes when the retry limit forced an answer.
  - **Tabs:**
    - **Papers used:** each paper carries its reliability label (Foundational, Current, Declining or Superseded), and a "Named by critic" tag when the critic's notes mention it.
    - **Counter-evidence:** a separate tab for opposing candidates, with a caveat. Being listed does not mean a contradiction was found, and these papers are not used in the staleness score.
    - **Claims:** the claims table, which scrolls sideways on narrow screens.
    - **Session:** the session history, plus Markdown export. The download box only appears once there is a file.
  - **Example questions:** four, which can be clicked to fill the question box. A collapsed "Settings" section holds the recency-decay control.
  - **Clear messages:**
    - **Missing key:** the pipeline doesn't start, and the user sees an explicit instruction to set `GROQ_API_KEY` in `.env` or the Space secrets.
    - **Other failures:** the user gets a plain message for authentication problems, rate limits, timeouts and network errors, plus the short technical detail. The tracker marks the step that failed.
  - **Theme-aware colors:** text uses Gradio's theme variables, so it works in light and dark mode. Muted text is `#475569` on light backgrounds, for higher contrast than the default.
  - **Fixes:**
    - All dynamic text is HTML-escaped.
    - The mutable default argument is gone.
    - The app only launches under `if __name__ == "__main__"`, and `css` is passed to `launch()` as Gradio 6 requires.
    - The preload import is optional.
    - The UI copy contains no em dashes, and a test checks this.
  - **Mobile:** removed a `max-width` override on `.gradio-container` that pushed the page to 451 px on a 390 px phone. There is now no horizontal scroll at 390 px. Columns stack, and the status chips wrap.
- **Sample data, for UI checks only:**
  - `scripts/ui_sample_app.py` launches the same UI with invented placeholder papers and verdicts.
  - A banner on that page says "SAMPLE DATA ... not RECON output", and every sample title starts with "Sample paper:".
  - `app.py` never imports this file.
- **Screenshots** in `docs/ui/`, all taken in headless Chromium at 2x:
  - Desktop (1280 px) and phone (390 px): empty state, result and counter-evidence tab. All of these use sample data.
  - `desktop-missing-key.png`: the real `python app.py` with no key.
- **Tests and lint:**
  - New tests: `tests/test_app_ui.py` (7 tests) and `tests/test_stream_recon.py` (1 test).
  - Gradio and ruff are added to `requirements-dev.txt`. CI now runs `ruff check --isolated --select E4,E7,E9,F` on the files touched in v2.1 and in this branch.

## What was tested

- `python -m pytest -q tests`: **26 passed.** The new tests cover:
  - the missing-key path, which never calls the pipeline
  - an empty query
  - streaming progress through to the final state, using the sample stream
  - a friendly message for a 429 error
  - HTML escaping
  - no em dashes in the UI source
  - the demo building
  - `stream_recon` node order and the merged final state, with stub nodes
- Ruff (the rules above): all checks pass.
- **Headless browser:** Chromium at 1280 px and 390 px.
  - At 390 px the page has no horizontal scroll (`scrollWidth` 390); before the fix it was 451.
  - The flow ran end to end against the sample stream: fill a question, click Research, wait for "Answer written", open the Counter-evidence tab.
  - The real `python app.py` with no `GROQ_API_KEY` showed the missing-key message and stopped before planning.

## What was not tested

- **A real query end to end.** There was no Groq key, and this sandbox blocks Semantic Scholar and Hugging Face. Every result-state screenshot uses sample data. Real latency, real verdicts and how real papers render are unverified.
- **Dark mode.** I set the colors and didn't take a screenshot.
- **The Hugging Face Space build.** I didn't redeploy. Gradio 6.10.0 matches the README's `sdk_version`.
- **Two users at once.** Each click is capped at 2 concurrent runs (`concurrency_limit=2`), but I didn't load-test it.
- **Benchmark numbers.** None are shown or changed, and the UI makes no claim that contradiction detection improved.

## Run it locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Real app (needs `GROQ_API_KEY` in `.env`; `S2_API_KEY` is optional):

```bash
python app.py
```

Sample-data UI for design review only (no keys needed):

```bash
python scripts/ui_sample_app.py
```

Regenerate the screenshots while the sample app is running:

```bash
node scripts/ui_screenshots.js http://127.0.0.1:7862 docs/ui
```

Tests and lint:

```bash
python -m pytest -q tests
```

```bash
ruff check --isolated --select E4,E7,E9,F app.py scripts tests src/graph.py src/agents/counter_evidence.py src/llm_config.py
```
