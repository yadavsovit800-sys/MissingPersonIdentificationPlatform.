import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "missing_persons.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
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

    conn.commit()
    conn.close()


def insert_person(
    case_id,
    full_name,
    age,
    contact_number,
    last_seen_location,
    image_path,
    vector_id
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO missing_persons
        (case_id, full_name, age, contact_number,
         last_seen_location, image_path, vector_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        full_name,
        age,
        contact_number,
        last_seen_location,
        image_path,
        vector_id
    ))

    conn.commit()
    conn.close()


def get_all_persons():
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM missing_persons
        ORDER BY rowid DESC
    """).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_person_by_vector_id(vector_id):
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    row = conn.execute("""
        SELECT * FROM missing_persons
        WHERE vector_id = ?
    """, (vector_id,)).fetchone()

    conn.close()

    return dict(row) if row else None


init_db()

print("SQLite database ready")