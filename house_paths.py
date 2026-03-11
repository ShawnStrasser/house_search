from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DB_PATH = REPO_ROOT / "property_data.db"
SCRAPER_CHECKPOINT_PATH = REPO_ROOT / "scraper_checkpoint.json"
TEMPLATE_DIR = REPO_ROOT / "templates"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"
CAPTCHA_DIR = TOOLS_DIR / "captcha"
CAPTCHA_RECORDINGS_DIR = CAPTCHA_DIR / "recordings"
CAPTCHA_BUTTON_TEMPLATE = CAPTCHA_DIR / "captcha_button.png"
CAPTCHA_FULL_TEMPLATE = CAPTCHA_DIR / "captcha.png"
CAPTCHA_PROFILE_PATH = CAPTCHA_DIR / "mouse_profile.json"
