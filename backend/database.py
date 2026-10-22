import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FINDSAFE_DATA_DIR", str(ROOT / "data")))
DB_PATH = DATA_DIR / "missing_persons.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS missing_persons (
                case_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                age INTEGER,
                contact_number TEXT,
                last_seen_location TEXT,
                image_path TEXT NOT NULL,
                vector_id TEXT UNIQUE
            )
        """)


def insert_person(case_id, full_name, age, contact_number, last_seen_location, image_path, vector_id):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO missing_persons
            (case_id, full_name, age, contact_number, last_seen_location, image_path, vector_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (case_id, full_name, age, contact_number, last_seen_location, image_path, vector_id))


def get_all_persons():
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM missing_persons ORDER BY rowid DESC").fetchall()
        return [dict(row) for row in rows]


def get_person_by_vector_id(vector_id):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM missing_persons WHERE vector_id = ?", (vector_id,)
        ).fetchone()
        return dict(row) if row else None


init_db()
print("SQLite database ready")
