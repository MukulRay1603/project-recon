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
        conn.commit()


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