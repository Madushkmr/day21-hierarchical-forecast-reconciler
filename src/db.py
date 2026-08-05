"""SQLite persistence: runs, per-node forecast points, and drift alerts."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    season_length INTEGER NOT NULL,
    reconciliation_strategy TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    node_name TEXT NOT NULL,
    node_level TEXT NOT NULL,      -- 'product' | 'category' | 'total'
    step_offset INTEGER NOT NULL,  -- 0-indexed weeks ahead
    point REAL NOT NULL,
    lower REAL NOT NULL,
    upper REAL NOT NULL,
    bottom_up_weight REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    node_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    pct_change REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forecasts_run ON forecasts(run_id);
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id);
"""


@contextmanager
def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path):
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_run(db_path, forecast_horizon, season_length, reconciliation_strategy, node_forecasts, alerts):
    """node_forecasts: {node_name: {"level":..., "point":[...], "lower":[...],
    "upper":[...], "bottom_up_weight": float}}
    alerts: list of {"node_name","severity","message","pct_change"}
    """
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO runs (created_at, forecast_horizon, season_length, reconciliation_strategy) VALUES (?,?,?,?)",
            (now, forecast_horizon, season_length, reconciliation_strategy),
        )
        run_id = cur.lastrowid
        for node_name, data in node_forecasts.items():
            for step, (point, lower, upper) in enumerate(zip(data["point"], data["lower"], data["upper"])):
                conn.execute(
                    "INSERT INTO forecasts (run_id, node_name, node_level, step_offset, point, lower, upper, bottom_up_weight) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, node_name, data["level"], step, point, lower, upper, data["bottom_up_weight"]),
                )
        for a in alerts:
            conn.execute(
                "INSERT INTO alerts (run_id, node_name, severity, message, pct_change, created_at) VALUES (?,?,?,?,?,?)",
                (run_id, a["node_name"], a["severity"], a["message"], a["pct_change"], now),
            )
        return run_id


def list_runs(db_path, limit=20):
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(db_path, run_id):
    with connect(db_path) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            return None
        forecasts = conn.execute(
            "SELECT * FROM forecasts WHERE run_id=? ORDER BY node_name, step_offset", (run_id,)
        ).fetchall()
        alerts = conn.execute(
            "SELECT * FROM alerts WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return {
            "run": dict(run),
            "forecasts": [dict(f) for f in forecasts],
            "alerts": [dict(a) for a in alerts],
        }


def get_latest_run_id(db_path):
    with connect(db_path) as conn:
        row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None


def list_alerts(db_path, limit=50):
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
