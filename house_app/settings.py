import logging
import os
import re
from pathlib import Path

from config import APP_CONFIG, DATABASE_CONFIG


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / DATABASE_CONFIG["local_db_path"])


def _load_ratings_db_url() -> str:
    ratings_db_url = os.environ.get(APP_CONFIG["ratings_db_url_env"], "")
    if ratings_db_url:
        return ratings_db_url

    try:
        secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            content = secrets_path.read_text(encoding="utf-8")
            match = re.search(r'RATINGS_DB_URL\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    except Exception as exc:
        logger.warning("Could not read secrets.toml: %s", exc)

    return ""


SECRET_KEY = os.environ.get(APP_CONFIG["secret_key_env"])
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required but not set")

RATINGS_DB_URL = _load_ratings_db_url()
logger.info("RATINGS_DB_URL environment variable: %s", "SET" if RATINGS_DB_URL else "NOT SET")
if RATINGS_DB_URL:
    logger.info("RATINGS_DB_URL starts with: %s...", RATINGS_DB_URL[:20])

CORRECT_PASSWORD = os.environ.get(APP_CONFIG["correct_password_env"])
if not CORRECT_PASSWORD:
    raise RuntimeError("APP_PASSWORD environment variable is required but not set")
