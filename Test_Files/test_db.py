import sqlite3
conn = sqlite3.connect("data/sessions.db")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("Tables:", tables)

count = conn.execute("SELECT COUNT(*) FROM verdict_log").fetchone()[0]
print(f"verdict_log rows: {count}")

if count > 0:
    row = conn.execute(
        "SELECT query, verdict, mean_age_months, retry_count FROM verdict_log LIMIT 1"
    ).fetchone()
    print(f"Sample: query='{row[0][:50]}' verdict={row[1]} mean_age={row[2]}mo retry={row[3]}")

conn.close()
