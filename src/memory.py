import sqlite3
import json
import os
from datetime import datetime
from src.state import SessionContext, SessionUpdate, Claim


_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions.db")
DB_PATH = os.environ.get("SESSION_DB_PATH", _DEFAULT_DB)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at app startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_turns (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL,
                query             TEXT NOT NULL,
                position          TEXT NOT NULL,
                claim_json        TEXT NOT NULL,   -- JSON list of Claim dicts
                contradictions    TEXT NOT NULL,   -- JSON list of strings
                created_at        TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verdict_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                query           TEXT NOT NULL,
                verdict         TEXT NOT NULL,  -- PASS/STALE/CONTRADICTED/INSUFFICIENT/FORCED_PASS
                mean_age_months REAL,           -- mean age of retrieved papers in months
                retry_count     INTEGER,
                papers_json     TEXT NOT NULL,  -- JSON list of {title, year, citation_count, paper_id}
                critic_notes    TEXT,
                session_id      TEXT
            )
        """)
        conn.commit()


def log_verdict(
    query: str,
    verdict: str,
    papers: list,
    critic_notes: str = "",
    mean_age_months: float = 0.0,
    retry_count: int = 0,
    session_id: str = "",
) -> None:
    """
    Log every critic verdict to verdict_log for leaderboard generation.

    Called from synthesizer_node after every completed pipeline run.
    Each row represents one query where the critic fired a specific verdict,
    along with the papers that were retrieved and evaluated.

    This is the raw data that generates the real, query-driven superseded
    paper leaderboard — not a pre-written document.

    Args:
        query:           The original research question
        verdict:         PASS / STALE / CONTRADICTED / INSUFFICIENT / FORCED_PASS
        papers:          List of Paper dataclass objects retrieved for this query
        critic_notes:    Human-readable explanation from the critic
        mean_age_months: Mean age of retrieved papers in months
        retry_count:     Number of retries before this verdict
        session_id:      Session UUID (optional, for traceability)
    """
    now = datetime.utcnow().isoformat()

    # Serialise paper metadata — just what's needed for the leaderboard
    papers_data = []
    for p in papers[:8]:  # cap at 8 — same as synthesizer display limit
        try:
            papers_data.append({
                "title":          getattr(p, "title", "") or "",
                "year":           getattr(p, "year", 0) or 0,
                "citation_count": getattr(p, "citation_count", 0) or 0,
                "paper_id":       getattr(p, "paper_id", "") or "",
                "authors":        (getattr(p, "authors", []) or [])[:2],
                "hybrid_score":   round(getattr(p, "hybrid_score", 0.0) or 0.0, 4),
            })
        except Exception:
            continue

    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO verdict_log
                    (created_at, query, verdict, mean_age_months,
                     retry_count, papers_json, critic_notes, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    query,
                    verdict,
                    round(mean_age_months, 1),
                    retry_count,
                    json.dumps(papers_data),
                    critic_notes or "",
                    session_id or "",
                ),
            )
            conn.commit()
    except Exception as e:
        # Non-fatal — never let logging break the pipeline
        import logging
        logging.getLogger(__name__).warning(f"verdict_log insert failed: {e}")


def query_verdict_log(
    verdict_filter: list[str] | None = None,
    min_count: int = 1,
    limit: int = 500,
) -> list[dict]:
    """
    Query the verdict log for leaderboard generation.

    Args:
        verdict_filter: List of verdicts to include, e.g. ['STALE', 'CONTRADICTED'].
                        None = all verdicts.
        min_count:      Minimum number of times a paper must appear to be included.
        limit:          Max rows to return from the log.

    Returns:
        List of raw verdict_log rows as dicts.
    """
    with _get_conn() as conn:
        if verdict_filter:
            placeholders = ",".join("?" * len(verdict_filter))
            rows = conn.execute(
                f"""
                SELECT * FROM verdict_log
                WHERE verdict IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (*verdict_filter, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM verdict_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    return [dict(r) for r in rows]


def load_session(session_id: str) -> SessionContext:
    """
    Load prior positions and contradictions for this session.
    Returns an empty SessionContext if the session doesn't exist yet.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT query, position, contradictions
            FROM session_turns
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,)
        ).fetchall()

    if not rows:
        return SessionContext()

    prior_positions = [r["position"] for r in rows]
    prior_queries = [r["query"] for r in rows]
    flagged_contradictions = []
    for r in rows:
        flagged_contradictions.extend(json.loads(r["contradictions"]))

    return SessionContext(
        prior_positions=prior_positions,
        prior_queries=prior_queries,
        flagged_contradictions=flagged_contradictions,
    )


def save_turn(session_id: str, update: SessionUpdate) -> None:
    """
    Persist one completed turn (query + synthesized position + claims).
    Creates the session row if it doesn't exist.
    """
    now = datetime.utcnow().isoformat()

    claim_json = json.dumps([
        {
            "text": c.text,
            "source_title": c.source_title,
            "source_year": c.source_year,
            "confidence": c.confidence,
            "flagged": c.flagged,
        }
        for c in update.claim_confidences
    ])

    contradictions_json = json.dumps(update.contradictions_found)

    with _get_conn() as conn:
        # Upsert the session header row
        conn.execute(
            """
            INSERT INTO sessions (session_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (session_id, now, now),
        )
        # Insert the turn
        conn.execute(
            """
            INSERT INTO session_turns
                (session_id, query, position, claim_json, contradictions, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, update.query, update.position,
             claim_json, contradictions_json, now),
        )
        conn.commit()


def export_session_md(session_id: str) -> str:
    """
    Export the full session as a markdown research note.
    Returns the markdown string (not saved to disk here — caller decides).
    """
    with _get_conn() as conn:
        session_row = conn.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        turns = conn.execute(
            """
            SELECT query, position, claim_json, contradictions, created_at
            FROM session_turns
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,)
        ).fetchall()

    if not session_row:
        return "# Session not found\n"

    lines = [
        f"# RECON Research Session",
        f"**Session ID:** `{session_id}`  ",
        f"**Started:** {session_row['created_at']}  ",
        f"**Turns:** {len(turns)}",
        "",
        "---",
        "",
    ]

    for i, turn in enumerate(turns, 1):
        claims = json.loads(turn["claim_json"])
        contradictions = json.loads(turn["contradictions"])

        lines += [
            f"## Turn {i}: {turn['query']}",
            "",
            "### Position",
            turn["position"],
            "",
        ]

        if claims:
            lines += ["### Claims", ""]
            for c in claims:
                flag = " ⚠️" if c["flagged"] else ""
                lines.append(
                    f"- **[{c['confidence'].upper()}]** {c['text']} "
                    f"— *{c['source_title']} ({c['source_year']})*{flag}"
                )
            lines.append("")

        if contradictions:
            lines += ["### Contradictions flagged", ""]
            for contradiction in contradictions:
                lines.append(f"- {contradiction}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def delete_session(session_id: str) -> None:
    """Hard delete a session and all its turns."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()