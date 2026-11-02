from __future__ import annotations
import sqlite3
from contextlib import closing
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "retention.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        retailer_id TEXT NOT NULL,
        action TEXT NOT NULL,
        owner TEXT,
        planned_date TEXT,
        status TEXT NOT NULL DEFAULT 'Planned',
        actual_cost REAL DEFAULT 0,
        outcome TEXT DEFAULT 'Pending',
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit()
    return con


def add_intervention(retailer_id, action, owner, planned_date, status, actual_cost, outcome, notes):
    with closing(_connect()) as con, con:
        con.execute("INSERT INTO interventions (retailer_id, action, owner, planned_date, status, actual_cost, outcome, notes) VALUES (?,?,?,?,?,?,?,?)",
                    (retailer_id, action, owner, str(planned_date), status, float(actual_cost or 0), outcome, notes))


def get_interventions():
    with closing(_connect()) as con:
        return pd.read_sql_query("SELECT * FROM interventions ORDER BY created_at DESC, id DESC", con)


def update_intervention(record_id, status, actual_cost, outcome, notes):
    with closing(_connect()) as con, con:
        con.execute("UPDATE interventions SET status=?, actual_cost=?, outcome=?, notes=? WHERE id=?",
                    (status, float(actual_cost or 0), outcome, notes, int(record_id)))
