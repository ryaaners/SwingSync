import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def save(self, item: dict):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO analyses VALUES (?, ?)", (item["id"], json.dumps(item)))

    def get(self, identifier: str):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM analyses WHERE id = ?", (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM analyses ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, identifier: str):
        with self.connect() as db:
            db.execute("DELETE FROM analyses WHERE id = ?", (identifier,))

    def update_annotations(self, identifier, update):
        """Read/compare/UPDATE atomically; a deleted analysis is never recreated."""
        from .annotations import apply_annotations
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload FROM analyses WHERE id = ?', (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            result = apply_annotations(json.loads(row[0]), update)
            db.execute('UPDATE analyses SET payload = ? WHERE id = ?', (json.dumps(result), identifier))
        return result
