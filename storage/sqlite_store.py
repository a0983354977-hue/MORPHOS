"""SQLite store for metadata, events and state snapshots.

High-frequency trajectories live in the binary WAL; SQLite keeps run
metadata, observer/anomaly events, and periodic serialized snapshots
used for crash recovery.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at   REAL,
    steps      INTEGER DEFAULT 0,
    note       TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER,
    sim_t    REAL,
    wall     REAL NOT NULL,
    level    TEXT NOT NULL,
    message  TEXT NOT NULL,
    data     TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER,
    created_at REAL NOT NULL,
    sim_t      REAL,
    step       INTEGER,
    key        TEXT NOT NULL,
    blob       BLOB NOT NULL
);
"""


class Store:
    def __init__(self, path: str, check_same_thread: bool = True):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=check_same_thread)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.run_id: int | None = None

    def begin_run(self, note: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, note) VALUES (?, ?)",
            (time.time(), note))
        self.conn.commit()
        self.run_id = int(cur.lastrowid)
        return self.run_id

    def end_run(self, steps: int) -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at=?, steps=? WHERE id=?",
            (time.time(), steps, self.run_id))
        self.conn.commit()

    def log_event(self, level: str, message: str, sim_t: float = 0.0,
                  data: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (run_id, sim_t, wall, level, message, data)"
            " VALUES (?,?,?,?,?,?)",
            (self.run_id, float(sim_t), time.time(), level, message,
             json.dumps(data) if data is not None else None))
        self.conn.commit()

    def save_snapshot(self, blob: bytes, sim_t: float, step: int,
                      key: str = "latest") -> int:
        cur = self.conn.execute(
            "INSERT INTO snapshots (run_id, created_at, sim_t, step, key,"
            " blob) VALUES (?,?,?,?,?,?)",
            (self.run_id, time.time(), float(sim_t), int(step), key, blob))
        self.conn.commit()
        return int(cur.lastrowid)

    def load_snapshot(self, key: str = "latest"):
        row = self.conn.execute(
            "SELECT blob, sim_t, step FROM snapshots WHERE key=?"
            " ORDER BY id DESC LIMIT 1", (key,)).fetchone()
        if row is None:
            return None
        return {"blob": row[0], "sim_t": row[1], "step": row[2]}

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?",
                                (key,)).fetchone()
        return row[0] if row else default

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
