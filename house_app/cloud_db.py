import logging
import threading
import time

from config import DATABASE_CONFIG

from .settings import RATINGS_DB_URL


logger = logging.getLogger(__name__)


def get_ratings_db_connection():
    """Return a connection to the ratings DB in SQLite Cloud."""
    if not RATINGS_DB_URL:
        raise RuntimeError(
            "RATINGS_DB_URL is not set. Add it to your environment variables."
        )
    try:
        import sqlitecloud
    except Exception as import_err:
        raise RuntimeError(
            "sqlitecloud package is required. Run: pip install sqlitecloud"
        ) from import_err

    return sqlitecloud.connect(RATINGS_DB_URL)


def init_ratings_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rating (
            zpid INTEGER PRIMARY KEY,
            rating TEXT CHECK (rating IN ('yes','no','maybe')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def init_notes_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            zpid INTEGER PRIMARY KEY,
            note TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def init_for_review_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS for_review (
            zpid INTEGER PRIMARY KEY,
            zillow_url TEXT NOT NULL,
            added_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def load_ratings_dict(conn):
    try:
        rows = conn.execute("SELECT zpid, rating FROM rating").fetchall()
        return dict(rows) if rows else {}
    except Exception:
        return {}


def save_rating(conn, zpid_value, rating_value: str):
    if rating_value in (None, ""):
        conn.execute("DELETE FROM rating WHERE zpid = ?", (int(zpid_value),))
        conn.commit()
        return

    if rating_value not in ("yes", "no", "maybe"):
        return

    conn.execute(
        """
        INSERT INTO rating (zpid, rating, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(zpid) DO UPDATE SET
            rating = excluded.rating,
            updated_at = datetime('now')
        """,
        (int(zpid_value), rating_value),
    )
    conn.commit()


def load_notes_dict(conn):
    try:
        rows = conn.execute("SELECT zpid, note FROM notes").fetchall()
        return dict(rows) if rows else {}
    except Exception:
        return {}


def save_note(conn, zpid_value, note_value: str):
    if note_value in (None, ""):
        conn.execute("DELETE FROM notes WHERE zpid = ?", (int(zpid_value),))
        conn.commit()
        return

    conn.execute(
        """
        INSERT INTO notes (zpid, note, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(zpid) DO UPDATE SET
            note = excluded.note,
            updated_at = datetime('now')
        """,
        (int(zpid_value), note_value),
    )
    conn.commit()


def load_user_data(password_correct: bool):
    ratings_dict = {}
    notes_dict = {}
    if not password_correct:
        return ratings_dict, notes_dict

    try:
        ratings_conn = get_ratings_db_connection()
        ratings_dict = load_ratings_dict(ratings_conn)
        notes_dict = load_notes_dict(ratings_conn)
        ratings_conn.close()
    except Exception as exc:
        logger.error("Could not load ratings and notes: %s", exc)

    return ratings_dict, notes_dict


def database_keepalive():
    if not RATINGS_DB_URL:
        logger.info("Database keepalive: RATINGS_DB_URL not set, skipping keepalive")
        return

    try:
        logger.info("Database keepalive: Running keepalive query...")
        conn = get_ratings_db_connection()
        result = conn.execute("SELECT datetime('now') as current_time").fetchone()
        current_time = result[0] if result else "unknown"
        count_result = conn.execute("SELECT COUNT(*) FROM rating").fetchone()
        rating_count = count_result[0] if count_result else 0
        conn.close()
        logger.info(
            "Database keepalive successful at %s. Rating count: %s",
            current_time,
            rating_count,
        )
    except Exception as exc:
        logger.error("Database keepalive failed: %s", exc)


def initialize_databases():
    if not RATINGS_DB_URL:
        logger.info("Database initialization: RATINGS_DB_URL not set, skipping cloud database initialization")
        return

    try:
        logger.info("Initializing cloud database tables...")
        ratings_conn = get_ratings_db_connection()
        init_ratings_db(ratings_conn)
        init_notes_db(ratings_conn)
        init_for_review_db(ratings_conn)
        ratings_conn.close()
        logger.info("Cloud database tables initialized successfully")
    except Exception as exc:
        logger.error("Failed to initialize cloud database tables: %s", exc)


def start_keepalive_thread():
    if not RATINGS_DB_URL:
        logger.info("Database keepalive thread: RATINGS_DB_URL not set, not starting keepalive thread")
        return

    def keepalive_worker():
        logger.info("Database keepalive thread started - will ping every 3 hours")
        while True:
            try:
                database_keepalive()
                time.sleep(DATABASE_CONFIG["keepalive_interval_seconds"])
            except Exception as exc:
                logger.error("Keepalive thread error: %s", exc)
                time.sleep(DATABASE_CONFIG["keepalive_retry_seconds"])

    keepalive_thread = threading.Thread(target=keepalive_worker, daemon=True)
    keepalive_thread.start()
    logger.info("Database keepalive background thread started")
