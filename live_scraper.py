#!/usr/bin/env python3
"""
Live Feature Extraction for Real Estate Properties from Zillow using Gemini
This script is designed to update high-scoring listings with better AI models.
It combines scraping and AI-powered feature extraction with intelligent filtering
to focus on properties that are worth re-analyzing.
The process is as follows:
1.  Navigate through Zillow search pages.
2.  For each property listing, check filtering criteria:
    - If listing exists in properties table, update price only
    - Otherwise, process fully with AI analysis
3.  For qualifying listings:
    - Scrape fresh property data and update the properties table
    - Extract comprehensive text and high-quality images
    - Send data to Google's Gemini API for feature analysis
    - Update the property_features table with new results
4.  Continue until all pages are processed or no more qualifying listings found.
"""

#SEARCH_URL = 'https://www.zillow.com/homes/for_sale/?searchQueryState=%7B%22isMapVisible%22%3Atrue%2C%22mapBounds%22%3A%7B%22west%22%3A-156.63478397587753%2C%22east%22%3A-112.82130741337754%2C%22south%22%3A37.003092903893084%2C%22north%22%3A50.98652477441754%7D%2C%22filterState%22%3A%7B%22sort%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22price%22%3A%7B%22max%22%3A700000%7D%2C%22mp%22%3A%7B%22max%22%3A3406%7D%2C%22beds%22%3A%7B%22min%22%3A3%7D%2C%22baths%22%3A%7B%22min%22%3A2%7D%2C%22tow%22%3A%7B%22value%22%3Afalse%7D%2C%22mf%22%3A%7B%22value%22%3Afalse%7D%2C%22con%22%3A%7B%22value%22%3Afalse%7D%2C%22land%22%3A%7B%22value%22%3Afalse%7D%2C%22apa%22%3A%7B%22value%22%3Afalse%7D%2C%22apco%22%3A%7B%22value%22%3Afalse%7D%2C%22lot%22%3A%7B%22min%22%3A87120%7D%7D%2C%22isListVisible%22%3Atrue%2C%22mapZoom%22%3A6%2C%22customRegionId%22%3A%227ca8b68bbaX1-CRlexe5hovo8mh_116nid%22%7D'
SEARCH_URL = 'https://www.zillow.com/homes/for_sale/?searchQueryState=%7B%22pagination%22%3A%7B%7D%2C%22isMapVisible%22%3Atrue%2C%22mapBounds%22%3A%7B%22west%22%3A-130.02589725712753%2C%22east%22%3A-117.23781131962754%2C%22south%22%3A37.646090441738295%2C%22north%22%3A50.230773715727246%7D%2C%22mapZoom%22%3A6%2C%22customRegionId%22%3A%227ca8b68bbaX1-CRlexe5hovo8mh_116nid%22%2C%22filterState%22%3A%7B%22sort%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22price%22%3A%7B%22min%22%3A350000%2C%22max%22%3A710000%7D%2C%22mp%22%3A%7B%22min%22%3Anull%2C%22max%22%3A3406%7D%2C%22beds%22%3A%7B%22min%22%3A3%2C%22max%22%3Anull%7D%2C%22baths%22%3A%7B%22min%22%3A2%2C%22max%22%3Anull%7D%2C%22tow%22%3A%7B%22value%22%3Afalse%7D%2C%22mf%22%3A%7B%22value%22%3Afalse%7D%2C%22con%22%3A%7B%22value%22%3Afalse%7D%2C%22land%22%3A%7B%22value%22%3Afalse%7D%2C%22apa%22%3A%7B%22value%22%3Afalse%7D%2C%22apco%22%3A%7B%22value%22%3Afalse%7D%2C%22lot%22%3A%7B%22min%22%3A87120%2C%22max%22%3Anull%7D%2C%22sqft%22%3A%7B%22min%22%3A2000%7D%7D%2C%22isListVisible%22%3Atrue%2C%22usersSearchTerm%22%3A%22%22%7D'

import os
import re
import json
import base64
import time
import random
import sys
import subprocess
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
import io

import duckdb
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from PIL import Image

import winsound
from feature_extraction import (
    FEATURE_PROPERTIES, 
    house_features_func, 
    feature_extraction_tool, 
    RUBRIC_INSTRUCTIONS,
    create_property_features_table,
    save_features_to_db
)
# --- CONFIGURATION ---
DB_PATH = "property_data.db"  # Main database with properties, property_scores, and property_features tables
MODEL = "gemini-pro-latest"
MAX_IMAGES_TO_EXTRACT = 80  # Max images to feed to the model per property
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
_CACHED_TELEGRAM_CHAT_ID: Optional[str] = TELEGRAM_CHAT_ID

# Validate API key is present
if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required but not set")

# Global speed multiplier for all wait times (0.5 = 2x faster, 1.0 = normal speed, 2.0 = 2x slower)
SPEED_MULTIPLIER = 1

# How many days must have passed since last_updated before an existing active listing is re-checked
LISTING_RECHECK_DAYS = 3

# Scraper settings
HEADLESS_MODE = False
VERBOSE_LOGGING = True

# Populated at runtime from the real browser session - used for image downloads
BROWSER_UA: Optional[str] = None
# Browser mode options:
# - 'launch_chrome': launch Chrome with remote debugging + dedicated debug profile, then attach
# - 'attach': attach to an already-running Chrome that already exposes the remote debugging port
BROWSER_MODE = 'launch_chrome'
CHROME_REMOTE_DEBUGGING_URL = 'http://127.0.0.1:9222'
CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
CHROME_USER_DATA_DIR = r'C:\Users\shawn\chrome-cdp-profile'

# Initialize Google Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

# Checkpoint file for resume-after-interruption
CHECKPOINT_FILE = "scraper_checkpoint.json"

def save_checkpoint(page_num: int, listing_index: int) -> None:
    """Save scraping progress so a run can resume after interruption."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "current_page_num": page_num,
                "resume_listing_index": listing_index,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"    -> ⚠️ Could not save checkpoint: {e}")

def load_checkpoint() -> Optional[dict]:
    """Load a previous checkpoint if one exists."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def clear_checkpoint() -> None:
    """Remove the checkpoint file on successful completion."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception:
        pass


def get_model_startup_info(model_name: str) -> str:
    """
    Return human-readable model metadata for startup logging.
    """
    try:
        full_name = model_name if model_name.startswith('models/') else f"models/{model_name}"
        model_info = genai.get_model(full_name)
        base_model_id = getattr(model_info, 'base_model_id', '') or 'N/A'
        version = getattr(model_info, 'version', '') or 'N/A'
        display_name = getattr(model_info, 'display_name', '') or 'N/A'
        return (
            f"🤖 Gemini model configured: {full_name} "
            f"(display='{display_name}', version='{version}', base_model_id='{base_model_id}')"
        )
    except Exception as e:
        return f"🤖 Gemini model configured: models/{model_name} (metadata lookup failed: {e})"




# --- DATABASE FUNCTIONS ---

def get_for_review_links() -> list[str]:
    """
    Get all links from the for_review table in the SQLite Cloud database.
    Returns a list of Zillow URLs that should be processed first.
    """
    try:
        # Get RATINGS_DB_URL from environment (same as flask app)
        import os
        RATINGS_DB_URL = os.getenv('RATINGS_DB_URL', '')
        
        if not RATINGS_DB_URL:
            if VERBOSE_LOGGING:
                print("⚠️ RATINGS_DB_URL not set, cannot retrieve for_review links.")
            return []
        
        # Connect to SQLite Cloud database (same pattern as flask app)
        try:
            import sqlitecloud
        except ImportError:
            if VERBOSE_LOGGING:
                print("⚠️ sqlitecloud package not available, cannot retrieve for_review links.")
            return []
        
        conn = sqlitecloud.connect(RATINGS_DB_URL)
        cursor = conn.execute("SELECT zillow_url FROM for_review ORDER BY added_at ASC")
        links = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if VERBOSE_LOGGING and links:
            print(f"🔍 Found {len(links)} links in for_review table to process first.")
        
        return links
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"⚠️ Could not retrieve for_review links: {e}")
        return []

def remove_from_for_review(listing_url: str) -> None:
    """
    Remove a successfully processed listing from the for_review table in SQLite Cloud.
    """
    try:
        import os
        RATINGS_DB_URL = os.getenv('RATINGS_DB_URL', '')
        
        if not RATINGS_DB_URL:
            return
        
        try:
            import sqlitecloud
        except ImportError:
            return
        
        conn = sqlitecloud.connect(RATINGS_DB_URL)
        conn.execute("DELETE FROM for_review WHERE zillow_url = ?", [listing_url])
        conn.close()
        
        if VERBOSE_LOGGING:
            print(f"    -> ✅ Removed from for_review table")
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"    -> ⚠️ Could not remove from for_review table: {e}")

def ensure_properties_status_column(conn: duckdb.DuckDBPyConnection) -> None:
    """Add the properties.status and last_updated columns if this database predates them."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info('properties')").fetchall()}

    if 'status' not in columns:
        conn.execute("ALTER TABLE properties ADD COLUMN status TEXT")
        conn.commit()
        if VERBOSE_LOGGING:
            print(f"✅ Added status column to properties table.")
        columns = columns | {'status'}

    if 'last_updated' not in columns:
        conn.execute("ALTER TABLE properties ADD COLUMN last_updated TIMESTAMP")
        conn.commit()
        if VERBOSE_LOGGING:
            print(f"✅ Added last_updated column to properties table.")
        columns = columns | {'last_updated'}

    if VERBOSE_LOGGING:
        print(f"✅ properties table columns: {sorted(columns)}")

def setup_database(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Connect to DuckDB and create all necessary tables for the new simplified schema.
    """
    conn = duckdb.connect(db_path)
    
    # Create the simplified properties table (only zpid, source_url, price)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            zpid INTEGER PRIMARY KEY,
            source_url TEXT,
            price INTEGER,
            status TEXT,
            last_updated TIMESTAMP
        )
    """)
    ensure_properties_status_column(conn)
    
    # Create the property_features table with the updated schema
    create_property_features_table(conn)

    # Create the manual_inspection table for listings needing human review
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manual_inspection (
            zpid INTEGER,
            listing_url TEXT,
            reason TEXT,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create the property_scores table for filtering (may already exist from previous runs)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS property_scores (
            zpid INTEGER PRIMARY KEY,
            total_score REAL
        )
    """)
    
    conn.commit()
    print(f"✅ Database setup complete at '{db_path}'")
    return conn

def normalize_listing_status(status: Any) -> Optional[str]:
    """Return a clean status string or None if Zillow did not provide one."""
    if status is None:
        return None

    normalized = str(status).strip()
    return normalized or None

def log_status_debug_context(
    context: str,
    zpid: int,
    listing_url: str,
    property_data: Optional[Dict[str, Any]] = None,
    html_content: Optional[str] = None,
) -> None:
    """Print focused debug context for status-related issues."""
    if not VERBOSE_LOGGING:
        return

    property_data = property_data or {}
    title_match = re.search(r"<title>(.*?)</title>", html_content or "", re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "N/A"

    print(f"    -> DEBUG [{context}] ZPID={zpid}")
    print(f"       URL: {listing_url}")
    print(f"       Extracted keys: {sorted(property_data.keys())}")
    print(f"       Status: {property_data.get('status')!r}")
    print(f"       Price: {property_data.get('price')!r}")
    print(f"       Has __NEXT_DATA__: {'__NEXT_DATA__' in (html_content or '')}")
    print(f"       Page title: {title[:160]}")

def sleep_jitter(base_seconds: float, jitter_ratio: float = 0.35) -> None:
    """Sleep with randomized jitter to avoid bot-like timing."""
    delta = base_seconds * jitter_ratio
    duration = max(0.2, random.uniform(base_seconds - delta, base_seconds + delta))
    time.sleep(duration * SPEED_MULTIPLIER)

def mouse_wander(page, steps: int = 3) -> None:
    """
    Drift the mouse naturally using Gaussian offsets from the current position.
    Simulates an idle hand moving while reading — not teleporting to random corners.
    """
    try:
        viewport = page.viewport_size or {"width": 1280, "height": 800}
        x = viewport["width"] / 2 + random.gauss(0, 80)
        y = viewport["height"] / 2 + random.gauss(0, 80)
        for _ in range(steps):
            x = max(50, min(viewport["width"] - 50, x + random.gauss(0, 120)))
            y = max(50, min(viewport["height"] - 50, y + random.gauss(0, 80)))
            page.mouse.move(x, y)
            time.sleep(abs(random.gauss(0.2, 0.1)))
    except Exception:
        pass

def slightly_shuffle(lst: list, chunk_size: int = 3) -> list:
    """
    Shuffle within small chunks rather than completely randomizing.
    Nearby items may swap positions but the overall order stays close to original.
    Mimics how a human eye skips around a list slightly rather than reading top-to-bottom.
    """
    result = list(lst)
    for i in range(0, len(result), chunk_size):
        chunk = result[i:i + chunk_size]
        random.shuffle(chunk)
        result[i:i + chunk_size] = chunk
    return result

def human_dwell_pause(page) -> None:
    """
    Log-normal dwell time mimicking how humans read property pages.
    Median ~5s, with a right tail reaching ~40s for thorough reads.
    Includes occasional scrolling during the pause to simulate reading.
    mu=1.6, sigma=0.9 → median e^1.6 ≈ 5s; 95th percentile ~30s.
    """
    pause = random.lognormvariate(1.6, 0.9) * SPEED_MULTIPLIER
    pause = max(1.0, min(pause, 60.0))
    if VERBOSE_LOGGING and pause > 12:
        print(f"    -> 📖 Reading pause ({pause:.0f}s)...")
    
    # Break the pause into chunks with occasional scrolling
    chunks = max(1, int(pause / 3))
    for i in range(chunks):
        time.sleep(pause / chunks)
        # 40% chance to scroll a bit during reading (quick mouse wheel bursts)
        if random.random() < 0.4 and i < chunks - 1:
            try:
                num_ticks = random.randint(2, 4)
                for _ in range(num_ticks):
                    page.mouse.wheel(0, human_scroll_px())
                    time.sleep(random.uniform(0.01, 0.03))
            except Exception:
                pass

def human_scroll_px() -> int:
    """
    Single mouse wheel tick distance.
    Real mouse wheels typically scroll 120-300px per tick.
    """
    return random.randint(120, 300)

def scroll_like_mouse_wheel(page, target_distance: int = None) -> None:
    """
    Simulate realistic mouse wheel scrolling - multiple rapid small scrolls.
    Real users do 3-8 rapid wheel ticks, then pause briefly to scan content.
    Pass a negative target_distance to scroll upward.
    """
    if target_distance is None:
        target_distance = random.randint(800, 2000)
    
    direction = -1 if target_distance < 0 else 1
    remaining = abs(target_distance)
    ticks = 0
    max_ticks = 15  # Safety limit
    
    while remaining > 0 and ticks < max_ticks:
        tick_distance = human_scroll_px()
        try:
            page.mouse.wheel(0, tick_distance * direction)
        except Exception:
            pass
        remaining -= tick_distance
        ticks += 1
        # Tiny delay between wheel ticks (10-30ms)
        time.sleep(random.uniform(0.01, 0.03))
    
    # Brief pause after the burst to "look at content"
    time.sleep(random.uniform(0.2, 0.5))

def get_telegram_chat_id() -> Optional[str]:
    """
    Resolve Telegram chat id from env or recent bot updates.
    """
    global _CACHED_TELEGRAM_CHAT_ID
    if _CACHED_TELEGRAM_CHAT_ID:
        return str(_CACHED_TELEGRAM_CHAT_ID)

    if not TELEGRAM_TOKEN:
        return None

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        response = requests.get(url, timeout=15).json()
        if response.get("ok") and response.get("result"):
            latest = response["result"][-1]
            message = latest.get("message") or latest.get("edited_message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is not None:
                _CACHED_TELEGRAM_CHAT_ID = str(chat_id)
                return _CACHED_TELEGRAM_CHAT_ID
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"⚠️ Could not resolve Telegram chat id: {e}")
    return None

def send_telegram_message(text: str) -> bool:
    """
    Best-effort Telegram alert sender.
    """
    if not TELEGRAM_TOKEN:
        if VERBOSE_LOGGING:
            print("ℹ️ TELEGRAM_TOKEN not set. Skipping Telegram alert.")
        return False

    chat_id = get_telegram_chat_id()
    if not chat_id:
        if VERBOSE_LOGGING:
            print(
                "⚠️ Telegram chat id not found. Send your bot any message first, "
                "or set TELEGRAM_CHAT_ID env var."
            )
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            return True
        if VERBOSE_LOGGING:
            print(f"⚠️ Telegram send failed: {res.text}")
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"⚠️ Telegram send error: {e}")
    return False

def is_anti_bot_challenge_present(page) -> bool:
    """Detect visible challenge overlays or hard-block pages."""
    try:
        challenge_selectors = [
            '#captcha-container',
            'iframe[title*="recaptcha"]',
            'iframe#px-captcha-modal',
            'iframe[src*="perimeterx"]',
            'iframe[src*="captcha"]',
        ]
        for selector in challenge_selectors:
            try:
                if page.is_visible(selector, timeout=800):
                    return True
            except Exception:
                continue

        try:
            page_title = (page.title() or "").lower()
        except Exception:
            page_title = ""

        try:
            body_text = (page.locator("body").inner_text(timeout=2000) or "").lower()
        except Exception:
            body_text = ""

        challenge_phrases = [
            "access to this page has been denied",
            "verify you're a human",
            "press & hold",
            "unusual traffic",
            "captcha",
            "perimeterx",
        ]
        if any(phrase in page_title for phrase in challenge_phrases):
            return True
        if any(phrase in body_text for phrase in challenge_phrases):
            return True

        return False
    except Exception:
        return False

def pause_for_challenge_resolution(page, context_label: str) -> bool:
    """
    Pause for manual intervention whenever anti-bot challenge is detected.
    Returns True when challenge appears resolved, False if user skips.
    """
    if not is_anti_bot_challenge_present(page):
        return True

    print(f"🤖 Anti-bot challenge detected during {context_label}.")
    print("🧭 Please solve it in the open browser window.")
    send_telegram_message(f"🤖 Captcha/challenge detected: {context_label}")

    solver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solve_captcha.py")
    auto_solver_attempted = False
    try:
        if os.path.exists(solver_path):
            auto_solver_attempted = True
            print("🤖 Attempting automatic captcha solve with solve_captcha.py...")
            run_result = subprocess.run(
                [sys.executable, solver_path],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if VERBOSE_LOGGING:
                print(f"🤖 Auto-solver exit code: {run_result.returncode}")
        else:
            if VERBOSE_LOGGING:
                print(f"⚠️ Auto-solver not found at: {solver_path}")
    except subprocess.TimeoutExpired:
        if VERBOSE_LOGGING:
            print("⚠️ Auto-solver timed out.")
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"⚠️ Auto-solver execution error: {e}")

    if auto_solver_attempted:
        print("⏳ Waiting up to 20s for captcha challenge to clear...")
        auto_solve_deadline = time.time() + 20
        while time.time() < auto_solve_deadline:
            try:
                page.wait_for_load_state('domcontentloaded', timeout=3000)
            except Exception:
                pass
            if not is_anti_bot_challenge_present(page):
                print("✅ Auto-solver resolved captcha challenge.")
                send_telegram_message(f"✅ Captcha solved automatically: {context_label}")
                return True
            sleep_jitter(2.0, 0.4)

        print("⚠️ Auto-solver failed to resolve captcha challenge.")
        send_telegram_message(f"❌ Captcha auto-solve failed: {context_label}")

    while True:
        user_input = input("Press Enter after solving (or type 'skip' to skip this listing): ").strip().lower()
        if user_input == "skip":
            return False

        try:
            page.wait_for_load_state('domcontentloaded', timeout=20000)
        except Exception:
            pass
        sleep_jitter(1.5, 0.4)

        if not is_anti_bot_challenge_present(page):
            print("✅ Challenge appears resolved. Continuing.")
            return True

        print("⚠️ Challenge still detected. Please complete it and press Enter again.")

def is_chrome_debug_endpoint_ready() -> bool:
    """Return True when Chrome's DevTools endpoint is reachable."""
    try:
        response = requests.get(f"{CHROME_REMOTE_DEBUGGING_URL.rstrip('/')}/json/version", timeout=2)
        return bool(response.ok)
    except Exception:
        return False

def launch_debug_chrome() -> None:
    """
    Launch Chrome with remote debugging enabled using a non-default user data dir.
    Newer Chrome builds reject remote debugging on the default profile.
    """
    if is_chrome_debug_endpoint_ready():
        if VERBOSE_LOGGING:
            print("✅ Chrome DevTools endpoint already available. Reusing existing debug Chrome session.")
        return

    os.makedirs(CHROME_USER_DATA_DIR, exist_ok=True)
    chrome_command = [
        CHROME_PATH,
        '--remote-debugging-port=9222',
        '--remote-debugging-address=127.0.0.1',
        f'--user-data-dir={CHROME_USER_DATA_DIR}',
    ]

    print("🚀 Launching Chrome with remote debugging...")
    print(f"   Chrome path: {CHROME_PATH}")
    print(f"   Debug profile: {CHROME_USER_DATA_DIR}")
    subprocess.Popen(chrome_command)

    for _ in range(25):
        if is_chrome_debug_endpoint_ready():
            print("✅ Chrome remote debugging is ready.")
            return
        time.sleep(1)

    raise RuntimeError(
        "Chrome remote debugging endpoint was not reachable at "
        f"{CHROME_REMOTE_DEBUGGING_URL}. Ensure Chrome can launch and the debug port is not blocked."
    )

def create_browser_session(playwright):
    """
    Attach to Chrome exposed via remote debugging.
    Reuses existing Zillow search/listing tabs when available to support checkpoint resume.
    """
    if BROWSER_MODE == 'launch_chrome':
        launch_debug_chrome()
    elif BROWSER_MODE != 'attach':
        raise ValueError("BROWSER_MODE must be either 'launch_chrome' or 'attach'.")

    print(f"🔌 Attaching to existing Chrome at {CHROME_REMOTE_DEBUGGING_URL}...")
    print("   Make sure Chrome is open with remote debugging enabled.")
    browser = playwright.chromium.connect_over_cdp(CHROME_REMOTE_DEBUGGING_URL)
    if not browser.contexts:
        raise RuntimeError(
            "Connected to Chrome, but no browser contexts were found. "
            "Open at least one Chrome window first."
        )
    context = browser.contexts[0]
    owns_browser = False

    # Always open fresh tabs so that pre-existing Zillow pages (with active response
    # headers) don't cause ERR_BLOCKED_BY_RESPONSE on Playwright-driven goto() calls.
    page = context.new_page()
    listing_page = context.new_page()

    _stealth = Stealth()
    _stealth.apply_stealth_sync(page)
    _stealth.apply_stealth_sync(listing_page)
    return browser, context, page, listing_page, owns_browser

def extract_prices_from_search_page(page) -> Dict[int, int]:
    """
    Extract all ZPIDs and their prices from the current search results page.
    Returns a dictionary mapping ZPID -> price.
    """
    try:
        prices_dict = page.evaluate("""
            () => {
                const cards = document.querySelectorAll('article[data-test="property-card"]');
                const results = {};
                
                cards.forEach((card) => {
                    // Extract ZPID from the link href within the card
                    const link = card.querySelector('a[href*="_zpid"]');
                    const href = link ? link.href : null;
                    const zpidMatch = href ? href.match(/\\/(\\d+)_zpid\\//) : null;
                    const zpid = zpidMatch ? parseInt(zpidMatch[1]) : null;  // Keep as integer!
                    
                    // Find price element
                    const priceElement = card.querySelector('[data-test="property-card-price"]');
                    const priceText = priceElement ? priceElement.textContent.trim() : null;
                    
                    if (zpid && priceText) {
                        // Parse price (remove $ and commas, convert to int)
                        const priceMatch = priceText.match(/\\$([\\d,]+)/);
                        if (priceMatch) {
                            const price = parseInt(priceMatch[1].replace(/,/g, ''));
                            results[zpid] = price;
                        }
                    }
                });
                
                return results;
            }
        """)
        
        # Convert string keys to integers (JavaScript returns object keys as strings)
        prices_dict = {int(k): v for k, v in prices_dict.items()}
        
        if VERBOSE_LOGGING:
            print(f"    -> Extracted prices for {len(prices_dict)} properties from search page")
        
        return prices_dict
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"    -> ⚠️ Error extracting prices from search page: {e}")
        return {}

def update_price_for_existing_zpid(conn: duckdb.DuckDBPyConnection, zpid: int, new_price: int) -> bool:
    """
    Update the price for an existing ZPID without doing full processing.
    Returns True if successfully updated, False otherwise.
    
    Args:
        conn: Database connection
        zpid: Property ZPID
        new_price: New price extracted from search results
    """
    try:
        # Check current price in database
        current_price = conn.execute("SELECT price FROM properties WHERE zpid = ?", [zpid]).fetchone()[0]

        # Only update if price has changed
        if current_price != new_price:
            conn.execute("UPDATE properties SET price = ? WHERE zpid = ?", [new_price, zpid])
            conn.commit()
            if VERBOSE_LOGGING:
                print(f"    -> 💰 Price updated for ZPID {zpid}: ${current_price} → ${new_price}")

        return True

    except Exception as e:
        print(f"❌ CRITICAL: Error updating price for ZPID {zpid}: {e}")
        print("🛑 Price update failure - stopping execution!")
        raise e

_INACTIVE_STATUS_KEYWORDS = ('sold', 'off market', 'pending')

def _is_inactive_status(status: Optional[str]) -> bool:
    """Return True if the status string indicates a listing is no longer active."""
    if not status:
        return False
    lower = status.lower()
    return any(kw in lower for kw in _INACTIVE_STATUS_KEYWORDS)

def refresh_existing_listing_status(
    page,
    conn: duckdb.DuckDBPyConnection,
    zpid: int,
    listing_url: str,
    fallback_price: Optional[int] = None
) -> bool:
    """
    Lightly refresh status/price for an existing listing without re-running AI analysis.

    Active listings get a full human-like dwell so the browser session looks natural.
    Inactive listings (sold, off market, pending) are processed quickly — the page is
    loaded just long enough to read the HTML, then we move on immediately.
    """
    try:
        if VERBOSE_LOGGING:
            print(f"    -> Refreshing status for existing ZPID {zpid}")

        # Bring tab to front and navigate
        page.bring_to_front()
        page.goto(listing_url, wait_until='load', timeout=60000)

        # Brief pause + challenge check before reading HTML
        sleep_jitter(1.5, 0.3)
        if not pause_for_challenge_resolution(page, f"status refresh (ZPID {zpid})"):
            if VERBOSE_LOGGING:
                print(f"    -> ⏭️ Skipping status refresh for ZPID {zpid} after unresolved challenge.")
            return False

        listing_html = page.content()
        property_data = extract_property_data_from_html(listing_html)
        latest_status = normalize_listing_status(property_data.get('status'))
        latest_price = property_data.get('price', fallback_price)

        # Only dwell / wander on active listings — inactive ones can be closed right away
        if _is_inactive_status(latest_status):
            if VERBOSE_LOGGING:
                print(f"    -> ⚡ Inactive status ({latest_status!r}) — skipping dwell.")
        else:
            mouse_wander(page)
            sleep_jitter(3, 0.4)

        updates = ["last_updated = CURRENT_TIMESTAMP"]
        values = []

        if latest_price is not None:
            updates.append("price = ?")
            values.append(latest_price)

        if latest_status is not None:
            updates.append("status = ?")
            values.append(latest_status)

        if len(updates) == 1:
            # Only last_updated — no price or status extracted; still stamp the timestamp
            # so we don't re-check this listing immediately on the next run.
            if VERBOSE_LOGGING:
                print(f"    -> ⚠️ No status or price extracted for ZPID {zpid}")
                log_status_debug_context("status_refresh_no_updates", zpid, listing_url, property_data, listing_html)

        values.append(zpid)
        conn.execute(f"UPDATE properties SET {', '.join(updates)} WHERE zpid = ?", values)
        conn.commit()

        if VERBOSE_LOGGING:
            print(
                f"    -> ✅ Refreshed existing ZPID {zpid} "
                f"(price: ${latest_price if latest_price is not None else 'unchanged'}, "
                f"status: {latest_status or 'blank'}, last_updated: now)"
            )

        return True

    except Exception as e:
        print(f"❌ CRITICAL: Error refreshing status for ZPID {zpid}: {e}")
        print(f"🔍 DEBUG: listing_url={listing_url}, fallback_price={fallback_price}")
        print(traceback.format_exc())
        print("🛑 Status refresh failure - stopping execution!")
        raise e

def should_process_zpid(conn: duckdb.DuckDBPyConnection, zpid: int) -> bool:
    """
    Check if a ZPID should be processed based on the following criteria:
    1. If ZPID exists in BOTH properties AND property_features tables, skip full processing (only update price)
    2. Otherwise, process it fully
    """
    try:
        # Check if ZPID exists in properties table
        property_result = conn.execute("SELECT 1 FROM properties WHERE zpid = ?", [zpid]).fetchone()
        
        # Check if ZPID exists in property_features table  
        features_result = conn.execute("SELECT 1 FROM property_features WHERE zpid = ?", [zpid]).fetchone()
        
        # Only skip processing if it exists in BOTH properties AND property_features
        if property_result is not None and features_result is not None:
            return False  # Already fully processed, only needs price update
        
        # If in properties but NOT in features, or not in properties at all, process it
        if property_result is not None and features_result is None:
            if VERBOSE_LOGGING:
                print(f"    -> ZPID {zpid} in properties but missing features. Will process fully.")
        
        # Check property_scores table for additional context
        score_result = conn.execute("SELECT total_score FROM property_scores WHERE zpid = ?", [zpid]).fetchone()
        if score_result is not None:
            total_score = score_result[0]
            if VERBOSE_LOGGING:
                print(f"    -> ZPID {zpid} has score ({total_score:.1f}). Will process fully.")
            return True
        else:
            # Not in property_scores table, so process it fully
            if VERBOSE_LOGGING:
                print(f"    -> ZPID {zpid} not in property_scores table. Will process fully.")
            return True

    except Exception as e:
        print(f"❌ CRITICAL: Error checking ZPID {zpid}: {e}")
        print("🛑 Database check failure - stopping execution!")
        raise e


def add_to_manual_inspection(conn: duckdb.DuckDBPyConnection, zpid: int, listing_url: str, reason: str) -> None:
    """Record a listing that requires manual inspection."""
    try:
        conn.execute("INSERT INTO manual_inspection (zpid, listing_url, reason) VALUES (?, ?, ?)", [zpid, listing_url, reason])
        conn.commit()
        if VERBOSE_LOGGING:
            print(f"    -> 📝 Saved to manual_inspection for ZPID {zpid}: {reason}")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to insert into manual_inspection for ZPID {zpid}: {e}")

def extract_property_data_from_html(html_content: str) -> Dict[str, Any]:
    """
    Extract comprehensive property data from HTML content using meta tags and __NEXT_DATA__.
    This is adapted from scraper.py to update properties table with fresh data.
    """
    property_data = {}
    
    # Extract from meta description which contains accurate data
    desc_match = re.search(r'<meta name="description" content="([^"]+)"', html_content)
    if desc_match:
        description = desc_match.group(1)
        property_data['description'] = description
        
        # Extract photo count
        photo_match = re.search(r'has (\d+) photos', description)
        if photo_match:
            property_data['photo_count'] = int(photo_match.group(1))
        
        # Extract price
        price_match = re.search(r'\$(\d+(?:,\d+)*)', description)
        if price_match:
            property_data['price'] = int(price_match.group(1).replace(',', ''))
        
        # Extract beds and baths
        beds_baths_match = re.search(r'(\d+) beds?, (\d+(?:\.\d+)?) baths?', description)
        if beds_baths_match:
            property_data['beds'] = int(beds_baths_match.group(1))
            property_data['baths'] = float(beds_baths_match.group(2))
        
        # Extract home size
        size_match = re.search(r'(\d+(?:,\d+)*)\s+(?:Square\s+)?Feet', description, re.IGNORECASE)
        if size_match:
            property_data['home_size_sqft'] = int(size_match.group(1).replace(',', ''))
        
        # Extract address
        address_match = re.search(r'located at ([^,]+(?:,[^,]+)*)', description)
        if address_match:
            property_data['full_address'] = address_match.group(1).strip()
        
        # Extract year built
        year_match = re.search(r'built in (\d{4})', description)
        if year_match:
            property_data['year_built'] = int(year_match.group(1))
    
    # Try to extract from __NEXT_DATA__ script
    next_data_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', html_content, re.DOTALL)
    if next_data_match:
        try:
            next_data = json.loads(next_data_match.group(1))
            props = next_data.get('props', {}).get('pageProps', {})
            
            # Extract various property details from the structured data
            if 'gdpClientCache' in props:
                cache = props['gdpClientCache']
                for cache_key, cache_data in cache.items():
                    if isinstance(cache_data, dict) and 'property' in cache_data:
                        prop = cache_data['property']
                        
                        # Extract basic info
                        if 'zpid' in prop:
                            property_data['zpid'] = prop['zpid']
                        if 'homeType' in prop:
                            property_data['home_type'] = prop['homeType']
                        if 'propertyType' in prop:
                            property_data['property_type'] = prop['propertyType']
                        if 'homeStatus' in prop:
                            property_data['status'] = normalize_listing_status(prop['homeStatus'])
                        if 'daysOnZillow' in prop:
                            property_data['days_on_zillow'] = prop['daysOnZillow']
                        
                        # Extract location data
                        if 'latitude' in prop:
                            property_data['latitude'] = prop['latitude']
                        if 'longitude' in prop:
                            property_data['longitude'] = prop['longitude']
                        if 'address' in prop:
                            addr = prop['address']
                            if 'zipcode' in addr:
                                property_data['zip_code'] = addr['zipcode']
                            if 'county' in addr:
                                property_data['county'] = addr['county']
                        
                        # Extract lot size
                        if 'lotSize' in prop:
                            property_data['lot_size_sqft'] = prop['lotSize']
                            property_data['lot_size_acres'] = prop['lotSize'] / 43560.0
                        
                        break
        except json.JSONDecodeError as e:
            if VERBOSE_LOGGING:
                print(f"    -> ⚠️ Could not parse __NEXT_DATA__ JSON: {e}")

    # Fallback: extract status from DOM gallery status pill (visible on live page)
    # Structure: div.StyledGalleryStatusContainer > ... > span with "Off market" / "Active" / etc.
    if property_data.get('status') is None:
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(class_=lambda c: c and 'GalleryStatus' in str(c)):
            text = tag.get_text(strip=True)
            if text and len(text) < 50:  # status is short, not a card description
                property_data['status'] = normalize_listing_status(text)
                break
    return property_data

def update_properties_table(conn: duckdb.DuckDBPyConnection, zpid: int, listing_url: str, html_content: str) -> tuple[bool, int]:
    """
    Update the properties table with minimal data (zpid, source_url, price only).
    Returns (success, photo_count) where photo_count is needed for image extraction.
    """
    try:
        # Extract property data from HTML
        property_data = extract_property_data_from_html(html_content)
        
        # Prepare simplified data for insertion (only zpid, source_url, price)
        insert_data = {
            'zpid': zpid,
            'source_url': listing_url,
            'price': property_data.get('price'),
            'status': property_data.get('status')
        }
        
        # Get photo_count for internal use (but don't save to db)
        photo_count = property_data.get('photo_count', 0)

        if VERBOSE_LOGGING and property_data.get('status') is None:
            print(f"    -> ⚠️ No status extracted for ZPID {zpid} during full processing")
            log_status_debug_context("full_process_missing_status", zpid, listing_url, property_data, html_content)
        
        # Filter out None values and build SQL
        filtered_data = {k: v for k, v in insert_data.items() if v is not None}
        
        # Require at least zpid and source_url
        if 'zpid' not in filtered_data or 'source_url' not in filtered_data:
            print(f"    -> ❌ CRITICAL: Missing required data for ZPID {zpid}")
            print("🛑 Property data extraction failure - stopping execution!")
            raise Exception(f"CRITICAL FAILURE: Missing required zpid or source_url for ZPID {zpid}")
            
        columns = ', '.join(filtered_data.keys())
        placeholders = ', '.join(['?' for _ in filtered_data])
        
        # Delete existing record first, then insert fresh data (stamp last_updated = now)
        conn.execute("DELETE FROM properties WHERE zpid = ?", [zpid])
        conn.execute(
            f"INSERT INTO properties ({columns}, last_updated) VALUES ({placeholders}, CURRENT_TIMESTAMP)",
            list(filtered_data.values())
        )
        conn.commit()
        
        if VERBOSE_LOGGING:
            print(
                f"    -> 💾 Properties table updated for ZPID {zpid} "
                f"(price: ${filtered_data.get('price', 'N/A')}, status: {filtered_data.get('status', 'blank')})"
            )
        return True, photo_count
        
    except Exception as e:
        print(f"❌ CRITICAL DATABASE ERROR updating properties for ZPID {zpid}: {e}")
        print(f"🔍 DEBUG INFO:")
        print(f"    - Listing URL: {listing_url}")
        print(f"    - Filtered data keys: {list(filtered_data.keys())}")
        print(f"    - Values: {list(filtered_data.values())}")
        print(traceback.format_exc())
        # FAIL FAST - raise the error to stop execution
        raise e



# --- IMAGE HANDLING ---

def download_image(url: str) -> Optional[bytes]:
    """Download an image from a URL and return its content as bytes."""
    try:
        ua = BROWSER_UA or 'Mozilla/5.0'
        response = requests.get(url, headers={'User-Agent': ua}, timeout=20)
        if response.status_code == 200:
            return response.content
        if VERBOSE_LOGGING:
            print(f"    -> ⚠️ Failed to download image {url} (status: {response.status_code})")
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"    -> ❌ Error downloading image {url}: {e}")
    return None


# --- SCRAPING & EXTRACTION LOGIC ---

def extract_comprehensive_text_content(html_content: str) -> str:
    """
    Extract and clean all meaningful text content from the listing's HTML.
    (This function is adapted from scraper.py)
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        main_content = soup.find('div', {'data-test': 'hdp-for-sale-page-content'}) or soup

        # Preserve natural line boundaries. Using a space separator can collapse the
        # whole page into one line, and a single blocked token (e.g. "cookie")
        # would then incorrectly remove all extracted text.
        text_content = main_content.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]

        blocked_tokens = ("cookie", "privacy", "terms", "consent", "do not sell")
        meaningful_lines: List[str] = []
        for line in lines:
            if len(line) <= 15 or len(line.split()) <= 2:
                continue
            lower_line = line.lower()
            # Skip legal/navigation boilerplate when it's short, but do not discard
            # long descriptive listing text just because one token appears.
            if any(token in lower_line for token in blocked_tokens) and len(line) < 120:
                continue
            meaningful_lines.append(line)

        # Fallback: if aggressive filtering removed everything, return a relaxed
        # extraction rather than failing the entire listing.
        if not meaningful_lines:
            relaxed_lines = [
                line for line in lines
                if len(line) > 15 and len(line.split()) > 2
            ]
            meaningful_lines = relaxed_lines

        return '\n'.join(meaningful_lines[:200]) # Limit lines to keep it manageable
    except Exception as e:
        print(f"⚠️ Error extracting comprehensive text: {e}")
        return ""

def _preview_text(value: str, max_len: int = 220) -> str:
    """Normalize whitespace and return a short preview for logs."""
    if not value:
        return ""
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len] + "..."

def build_text_extraction_diagnostics(page, zpid: int, listing_url: str, listing_html: str, extracted_text: str) -> str:
    """
    Build actionable diagnostics when listing text extraction returns empty content.
    """
    diagnostics = {
        "zpid": zpid,
        "listing_url": listing_url,
        "page_url": "unknown",
        "page_title": "unknown",
        "ready_state": "unknown",
        "html_length": len(listing_html or ""),
        "extracted_text_length": len(extracted_text or ""),
        "body_text_length": None,
        "main_content_selector_found": None,
        "json_ld_count": None,
        "captcha_signals": [],
        "html_preview": _preview_text(listing_html, 280),
        "body_text_preview": "",
    }

    try:
        diagnostics["page_url"] = page.url or "unknown"
    except Exception:
        pass

    try:
        diagnostics["page_title"] = page.title() or "unknown"
    except Exception:
        pass

    try:
        diagnostics["ready_state"] = page.evaluate("() => document.readyState") or "unknown"
    except Exception:
        pass

    try:
        diagnostics["main_content_selector_found"] = bool(page.query_selector('div[data-test="hdp-for-sale-page-content"]'))
    except Exception:
        pass

    try:
        diagnostics["json_ld_count"] = page.locator('script[type="application/ld+json"]').count()
    except Exception:
        pass

    try:
        body_text = page.locator("body").inner_text(timeout=2000) or ""
        diagnostics["body_text_length"] = len(body_text)
        diagnostics["body_text_preview"] = _preview_text(body_text, 280)
        lower_body = body_text.lower()
        for signal in ["captcha", "verify you are human", "access denied", "robot", "unusual traffic"]:
            if signal in lower_body:
                diagnostics["captcha_signals"].append(signal)
    except Exception as body_error:
        diagnostics["body_text_preview"] = f"<body_text_unavailable: {body_error}>"

    return (
        "    -> ❌ CRITICAL: Could not extract text content!\n"
        "🛑 Text extraction diagnostics:\n"
        f"    - ZPID: {diagnostics['zpid']}\n"
        f"    - Listing URL: {diagnostics['listing_url']}\n"
        f"    - Current page URL: {diagnostics['page_url']}\n"
        f"    - Page title: {diagnostics['page_title']}\n"
        f"    - Document readyState: {diagnostics['ready_state']}\n"
        f"    - HTML length: {diagnostics['html_length']}\n"
        f"    - Extracted text length: {diagnostics['extracted_text_length']}\n"
        f"    - body.innerText length: {diagnostics['body_text_length']}\n"
        f"    - Main content selector present (div[data-test='hdp-for-sale-page-content']): {diagnostics['main_content_selector_found']}\n"
        f"    - JSON-LD script count: {diagnostics['json_ld_count']}\n"
        f"    - Captcha/access signals found: {diagnostics['captcha_signals'] if diagnostics['captcha_signals'] else 'none'}\n"
        f"    - HTML preview: {diagnostics['html_preview'] or '<empty>'}\n"
        f"    - Body text preview: {diagnostics['body_text_preview'] or '<empty>'}"
    )

def process_listing(page, zpid: int, listing_url: str, conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Processes a single listing: scrapes data, updates properties table, calls AI model, saves features, and saves analysis results.
    """
    try:
        # Bring this tab to the front so PX sees activity in the active tab
        page.bring_to_front()
        
        if VERBOSE_LOGGING:
            print(f"    -> Navigating to listing: {listing_url}")
        page.goto(listing_url, wait_until='load', timeout=60000)
        human_dwell_pause(page)
        mouse_wander(page)
        if not pause_for_challenge_resolution(page, f"listing processing (ZPID {zpid})"):
            add_to_manual_inspection(conn, zpid, listing_url, "Blocked by captcha/access denied")
            return False

        # 1. Extract comprehensive text and HTML for the model and properties update
        listing_html = page.content()
        comprehensive_text = extract_comprehensive_text_content(listing_html)
        if not comprehensive_text:
            print(build_text_extraction_diagnostics(page, zpid, listing_url, listing_html, comprehensive_text))
            raise Exception(
                f"CRITICAL FAILURE: No text content extracted for ZPID {zpid} at {listing_url}. "
                "See text extraction diagnostics above."
            )

        # 2. Update the properties table with fresh data and get photo count
        success, photo_count = update_properties_table(conn, zpid, listing_url, listing_html)
        if not success:
            print("    -> ⚠️ Could not update properties table. Continuing with feature extraction...")
            photo_count = 0

        # 3. Image extraction logic (mirrors scraper.py)
        if VERBOSE_LOGGING:
            print(f"    -> Looking for property images...")
        all_image_urls = set()
        gallery_opened = False
        
        if photo_count > 0:
            try:
                gallery_container = page.locator('div[data-testid="hollywood-gallery-images"]')
                if gallery_container.is_visible(timeout=3000):
                    gallery_btn = gallery_container.locator('button[data-testid="gallery-see-all-photos-button"]')
                    if gallery_btn.is_visible():
                        gallery_btn.click(timeout=5000)
                        sleep_jitter(4, 0.3)
                        if not pause_for_challenge_resolution(page, f"gallery open (ZPID {zpid})"):
                            add_to_manual_inspection(conn, zpid, listing_url, "Blocked by captcha while opening gallery")
                            return False
                        gallery_opened = True
                        if VERBOSE_LOGGING:
                            print("    -> Gallery opened successfully!")
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"    -> ⚠️ Could not open photo gallery: {e}")

        image_selectors = [
            'img[src*="photos.zillowstatic.com"]',
            'img[data-src*="photos.zillowstatic.com"]',
            'picture img[src*="photos.zillowstatic.com"]'
        ]

        if not gallery_opened:
            add_to_manual_inspection(conn, zpid, listing_url, "Failed to open photo gallery")
            if VERBOSE_LOGGING:
                print("    -> Recorded for manual inspection and skipping AI analysis for this listing.")
            return False
        
        if VERBOSE_LOGGING:
            print("    -> Scrolling inside the gallery to load more images...")
        
        # Dynamic scrolling that targets the correct scrollable container
        scroll_attempts = 0
        max_scrolls = 120
        last_image_count = 0
        consecutive_no_new_images = 0
        
        # Use the smaller of photo_count or MAX_IMAGES_TO_EXTRACT to prevent extra images
        target_image_count = min(photo_count, MAX_IMAGES_TO_EXTRACT) if photo_count > 0 else MAX_IMAGES_TO_EXTRACT
        
        if VERBOSE_LOGGING and photo_count > 0:
            print(f"    -> Property has {photo_count} photos, limiting extraction to {target_image_count} images")
        
        while len(all_image_urls) < target_image_count and scroll_attempts < max_scrolls:
            # Check if we're at the bottom of the scrollable container FIRST
            try:
                at_bottom = page.evaluate("""
                    () => {
                        const container = document.querySelector('#__c11n_40v5e');
                        if (container) {
                            return container.scrollTop + container.clientHeight >= container.scrollHeight - 50;
                        }
                        return false;
                    }
                """)
                if at_bottom:
                    if VERBOSE_LOGGING:
                        print("    -> Reached bottom of gallery container.")
                    break
            except Exception:
                pass
            
            # Scroll like a real mouse wheel - rapid small ticks
            try:
                # Do a burst of 5-10 rapid wheel ticks
                num_ticks = random.randint(5, 10)
                for _ in range(num_ticks):
                    tick_px = human_scroll_px()
                    page.evaluate(f"document.querySelector('#__c11n_40v5e').scrollBy(0, {tick_px})")
                    time.sleep(random.uniform(0.01, 0.03))  # Tiny delay between ticks
            except Exception:
                # Fallback to page-level mouse wheel
                scroll_like_mouse_wheel(page, random.randint(600, 1200))
            
            scroll_attempts += 1
            # Brief pause to "look at images" after the scroll burst
            time.sleep(random.uniform(0.3, 0.7))
            
            # NOW collect images after scrolling and waiting
            for selector in image_selectors:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    src = elem.get_attribute('src') or elem.get_attribute('data-src')
                    if src and 'photos.zillowstatic.com' in src:
                        if any(size in src for size in ['_1536', '_1024', 'cc_ft_', 'uncropped_scaled_within', 'p_f']):
                            all_image_urls.add(src.split('?')[0])
            
            current_image_count = len(all_image_urls)
            if VERBOSE_LOGGING and scroll_attempts % 3 == 0:  # Log every 3rd attempt
                print(f"    -> Collected {current_image_count} images so far (target: {target_image_count})")
            
            # Check if we found new images in this iteration
            if current_image_count == last_image_count:
                consecutive_no_new_images += 1
                if consecutive_no_new_images >= 5:  # Increased patience
                    if VERBOSE_LOGGING:
                        print(f"    -> No new images for 5 iterations. Stopping scroll.")
                    break
            else:
                consecutive_no_new_images = 0
                last_image_count = current_image_count
        
        # Force-jump to very bottom of the gallery container to catch any remaining images
        # that scroll bursts may have missed (e.g. a few images near the end of a long gallery)
        try:
            page.evaluate("""
                () => {
                    const c = document.querySelector('#__c11n_40v5e');
                    if (c) c.scrollTop = c.scrollHeight;
                }
            """)
            time.sleep(1.0)
            for selector in image_selectors:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    src = elem.get_attribute('src') or elem.get_attribute('data-src')
                    if src and 'photos.zillowstatic.com' in src:
                        if any(size in src for size in ['_1536', '_1024', 'cc_ft_', 'uncropped_scaled_within', 'p_f']):
                            all_image_urls.add(src.split('?')[0])
        except Exception:
            pass

        # Final collection pass after exiting loop
        if VERBOSE_LOGGING:
            print(f"    -> Doing final image collection pass...")
        for selector in image_selectors:
            elements = page.query_selector_all(selector)
            for elem in elements:
                src = elem.get_attribute('src') or elem.get_attribute('data-src')
                if src and 'photos.zillowstatic.com' in src:
                    if any(size in src for size in ['_1536', '_1024', 'cc_ft_', 'uncropped_scaled_within', 'p_f']):
                        all_image_urls.add(src.split('?')[0])
        
        # Fallback: scan page-level images if we still need more
        if len(all_image_urls) < target_image_count:
            for selector in image_selectors:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    src = elem.get_attribute('src') or elem.get_attribute('data-src')
                    if src and 'photos.zillowstatic.com' in src:
                        if any(size in src for size in ['_1536', '_1024', 'cc_ft_', 'uncropped_scaled_within', 'p_f']):
                            all_image_urls.add(src.split('?')[0])
                            if len(all_image_urls) >= target_image_count:
                                break
                if len(all_image_urls) >= target_image_count:
                    break

        if VERBOSE_LOGGING:
            print(f"    -> Found {len(all_image_urls)} unique high-quality images.")
        
        # Check if we found significantly fewer images than expected
        if photo_count > 0 and len(all_image_urls) < (target_image_count - 2):
            shortage = target_image_count - len(all_image_urls)
            print(f"    -> ❌ IMAGE COLLECTION FAILURE:")
            print(f"       Expected: {target_image_count} images (property has {photo_count} photos)")
            print(f"       Found: {len(all_image_urls)} images")
            print(f"       Shortage: {shortage} images")
            print(f"       Scroll attempts: {scroll_attempts}/{max_scrolls}")
            print(f"       Last check - at bottom: {at_bottom if 'at_bottom' in locals() else 'unknown'}")
            
            # Debug: Check what's in the DOM
            try:
                all_zillow_imgs = page.query_selector_all('img[src*="photos.zillowstatic.com"]')
                print(f"       Total Zillow images in DOM: {len(all_zillow_imgs)}")
                
                # Sample first few to see what formats we're getting
                sample_srcs = []
                for elem in all_zillow_imgs[:5]:
                    src = elem.get_attribute('src') or elem.get_attribute('data-src')
                    if src:
                        # Show just the size marker part
                        for marker in ['_1536', '_1024', '_768', '_384', 'cc_ft_', 'uncropped_scaled_within', 'p_f']:
                            if marker in src:
                                sample_srcs.append(marker)
                                break
                        else:
                            sample_srcs.append('(other)')
                if sample_srcs:
                    print(f"       Sample image formats found: {sample_srcs}")
            except Exception as e:
                print(f"       Could not debug DOM: {e}")
            
            print(f"    -> ⚠️ CRITICAL FAILURE: Only found {len(all_image_urls)}/{target_image_count} images for ZPID {zpid} - skipping to next listing")
            add_to_manual_inspection(conn, zpid, listing_url, f"Image collection shortage: found {len(all_image_urls)}/{target_image_count} images")
            return False

        # 4. Download images into memory and prepare API payload for Gemini
        model_content = [RUBRIC_INSTRUCTIONS, f"Listing text:\n\n{comprehensive_text}"]
        
        downloaded_count = 0
        failed_downloads = 0
        for img_url in list(all_image_urls):
            if downloaded_count >= target_image_count:
                break
            image_bytes = download_image(img_url)
            if image_bytes:
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    model_content.append(img)
                    downloaded_count += 1
                except Exception as e:
                    failed_downloads += 1
                    if VERBOSE_LOGGING:
                        print(f"    -> ⚠️ Could not process image {img_url}: {e}")
            else:
                failed_downloads += 1

        if downloaded_count == 0:
            print("    -> ❌ CRITICAL: No images could be downloaded/processed for analysis!")
            print("🛑 This indicates a serious problem with image extraction or network connectivity.")
            print("🔧 Process cannot continue without images for AI analysis.")
            raise Exception("CRITICAL FAILURE: No images available for AI analysis - this must be fixed!")
        
        # Check if we successfully downloaded significantly fewer than we found
        if len(all_image_urls) > 0 and downloaded_count < (len(all_image_urls) - 2):
            print(f"    -> ❌ IMAGE DOWNLOAD FAILURE:")
            print(f"       Found: {len(all_image_urls)} image URLs")
            print(f"       Downloaded: {downloaded_count} images")
            print(f"       Failed: {failed_downloads} downloads")
            print("🛑 STOPPING EXECUTION - Too many download failures!")
            raise Exception(f"CRITICAL FAILURE: Only downloaded {downloaded_count}/{len(all_image_urls)} images for ZPID {zpid}")

        if VERBOSE_LOGGING:
            print(f"    -> Prepared {downloaded_count} images for AI analysis.")

        # 5. Call Gemini API for feature extraction
        response = None
        try:
            model = genai.GenerativeModel(MODEL)
            response = model.generate_content(
                model_content,
                tools=[feature_extraction_tool],
                tool_config={'function_calling_config': 'ANY'},
                # Deterministic settings for consistent feature extraction
                generation_config=genai.GenerationConfig(
                    temperature=0.0,    # No creativity - completely deterministic
                    top_k=1             # Always pick the most likely token
                ),
                # It's good practice to set safety settings to avoid blocking responses
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Extract features from the function call response
            function_call = response.candidates[0].content.parts[0].function_call
            if not function_call or not function_call.name == "extract_house_features":
                raise Exception("Model did not return the expected function call.")

            features = {key: getattr(value, 'string_value', value) for key, value in function_call.args.items()}
            features['zpid'] = zpid
            
            if VERBOSE_LOGGING:
                print(f"    -> ✅ AI analysis complete.")

        except Exception as e:
            error_message = str(e)
            print(f"    -> ❌ AI analysis failed: {e}")
            
            # ALL AI failures should stop execution immediately - no graceful failures!
            print("🛑 AI ANALYSIS FAILED - STOPPING EXECUTION IMMEDIATELY!")
            print(f"🚨 Error details: {error_message}")
            if response is not None and hasattr(response, 'prompt_feedback'):
                print(f"    -> Prompt Feedback: {response.prompt_feedback}")
            print("🔧 This requires immediate attention before continuing.")
            raise Exception(f"AI ANALYSIS FAILURE: {error_message}")

        # 6. Save the extracted features to the database
        if not save_features_to_db(conn, features, verbose=VERBOSE_LOGGING):
            print(f"    -> ❌ CRITICAL: Failed to save features for ZPID {zpid}")
            print("🛑 Database save failure - this must be fixed!")
            raise Exception(f"CRITICAL FAILURE: Could not save features for ZPID {zpid}")

        # 7. Processing complete
        
        return True

    except Exception as e:
        print(f"    -> ❌ A critical error occurred while processing ZPID {zpid}: {e}")
        print("🛑 ALL ERRORS ARE CRITICAL - STOPPING EXECUTION!")
        raise e


def update_existing_listings(page, conn: duckdb.DuckDBPyConnection) -> None:
    """
    Pre-scrape step: iterate all active listings in the DB and refresh their price/status.

    A listing is considered "active" when its status is NULL (unknown) or does NOT contain
    known inactive keywords (sold, off market, pending).  Listings whose last_updated is
    within the past LISTING_RECHECK_DAYS days are skipped — they were checked recently.

    Uses `refresh_existing_listing_status` (which also sets last_updated = now) so every
    visited listing gets a fresh timestamp regardless of whether data changed.
    """
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=LISTING_RECHECK_DAYS)

    rows = conn.execute("""
        SELECT zpid, source_url, price
        FROM properties
        WHERE
            -- Only active / unknown-status listings
            (
                status IS NULL
                OR (
                    UPPER(status) NOT LIKE '%SOLD%'
                    AND UPPER(status) NOT LIKE '%OFF%MARKET%'
                    AND UPPER(status) NOT LIKE '%PENDING%'
                )
            )
            -- Not checked recently
            AND (last_updated IS NULL OR last_updated < ?)
        ORDER BY last_updated ASC NULLS FIRST
    """, [cutoff]).fetchall()

    if not rows:
        print("✅ Pre-update: no active listings need refreshing (all checked within the past "
              f"{LISTING_RECHECK_DAYS} days).")
        return

    print(f"\n🔄 Pre-update: refreshing {len(rows)} active listing(s) not checked in the past "
          f"{LISTING_RECHECK_DAYS} day(s)...")

    updated = 0
    skipped = 0
    for i, (zpid, source_url, fallback_price) in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] ZPID {zpid} — {source_url[:70]}")
        try:
            ok = refresh_existing_listing_status(page, conn, zpid, source_url, fallback_price)
            if ok:
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"    -> ❌ Error refreshing ZPID {zpid}: {e}")
            print("🛑 Pre-update failure — stopping execution!")
            raise e

        sleep_jitter(2, 0.4)

    print(f"✅ Pre-update complete: {updated} updated, {skipped} skipped.\n")


def run_live_extraction():
    """
    Main function to orchestrate the Zillow scraping and feature extraction process.
    """
    with sync_playwright() as p:
        print("--- 🏠 Starting Zillow Live Feature Extractor (Gemini Edition) ---")
        print(get_model_startup_info(MODEL))
        winsound.Beep(1000, 500)
        send_telegram_message("🚀 live_scraper.py started (beep played).")
        
        db_conn = setup_database()
        browser = None
        page = None
        listing_page = None
        owns_browser = False
        
        browser, context, page, listing_page, owns_browser = create_browser_session(p)

        global BROWSER_UA
        BROWSER_UA = page.evaluate("navigator.userAgent")
        if VERBOSE_LOGGING:
            print(f"🌐 Browser UA: {BROWSER_UA}")

        # Detect whether we can resume from a saved checkpoint.
        # Only offer resume if the browser is already sitting on a Zillow page
        # (meaning it was left open from the previous interrupted run).
        checkpoint = load_checkpoint()
        is_resuming = False
        resume_page_num = 1
        resume_listing_index = 0
        completed_successfully = False
        if checkpoint:
            try:
                current_tab_url = page.url or ''
            except Exception:
                current_tab_url = ''
            if 'zillow.com' in current_tab_url:
                print(f"\n📂 Checkpoint found: page {checkpoint['current_page_num']}, "
                      f"listing index {checkpoint['resume_listing_index']} "
                      f"(saved {checkpoint['saved_at']})")
                print(f"   Browser tab is already on Zillow: {current_tab_url[:70]}")
                answer = input("Resume from checkpoint? (y/n): ").strip().lower()
                if answer == 'y':
                    is_resuming = True
                    resume_page_num = checkpoint['current_page_num']
                    resume_listing_index = checkpoint['resume_listing_index']
                    print(f"▶️  Resuming at page {resume_page_num}, listing index {resume_listing_index}.")

        try:
            if BROWSER_MODE == 'launch_chrome':
                print("🧑‍💻 Chrome launch+attach mode enabled.")
                print("   A dedicated debug Chrome window was opened for this scraper.")
                print("   💡 TIP: To preserve session state and avoid fresh captchas,")
                print("      keep Chrome open between runs and switch to BROWSER_MODE='attach'")
            else:
                print("🧑‍💻 Chrome attach mode enabled.")
                print("   Using your already-open Chrome window.")
            print("   Sign in / solve any challenge in that window before continuing.")
            input("Press Enter when Chrome is ready for the scraper to take over...")

            if is_resuming:
                print("▶️  Resuming — reloading search tab to ensure fresh DOM...")
                page.bring_to_front()
                page.reload(wait_until='load', timeout=60000)
                sleep_jitter(3, 0.35)
                mouse_wander(page)
                if not pause_for_challenge_resolution(page, "resume startup"):
                    raise Exception("Challenge detected on resume startup")
            else:
                # Step 1: Start from Zillow homepage to avoid captcha
                print("🌐 Navigating to Zillow homepage first...")
                page.bring_to_front()
                page.goto('https://www.zillow.com/', wait_until='domcontentloaded')
                sleep_jitter(3, 0.35)
                mouse_wander(page)
                if not pause_for_challenge_resolution(page, "homepage load"):
                    raise Exception("Challenge unresolved on homepage load")

                # Step 2: Go to the search page
                print("🔍 Loading search results page...")
                page.goto(SEARCH_URL, wait_until='load', timeout=60000)
                sleep_jitter(3, 0.35)
                mouse_wander(page)

                if not pause_for_challenge_resolution(page, "search results page load"):
                    raise Exception("Challenge unresolved on search results page")
            
            # ── Pre-scrape: refresh price/status for all stale active listings ──────
            update_existing_listings(listing_page, db_conn)
            # Return focus to the search tab before the main scraping loop begins
            page.bring_to_front()
            sleep_jitter(2, 0.35)

            current_page_num = resume_page_num
            last_break_time = time.time()
            next_break_interval = random.uniform(10 * 60, 15 * 60)  # first break in 10-15 min
            while True:
                print(f"\n--- Scraping Page {current_page_num} ---")
                if not pause_for_challenge_resolution(page, f"search page {current_page_num}"):
                    raise Exception("Challenge unresolved while on search results page")

                # Robust scroll to load all listings (mirrors scraper.py)
                print("🔄 Scrolling down to load all listings on the page...")
                search_results_selector = "#grid-search-results"
                search_results_container = page.locator(search_results_selector)

                if search_results_container.is_visible(timeout=5000):
                    if VERBOSE_LOGGING:
                        print("    -> Found search results container. Scrolling within it...")
                    last_listing_count = 0
                    scroll_attempts = 0
                    while scroll_attempts < 20: # Increased safety limit
                        current_listing_count = len(page.query_selector_all('article[data-test="property-card"]'))

                        if current_listing_count > 0 and current_listing_count == last_listing_count:
                            if VERBOSE_LOGGING:
                                print("    -> No new listings loaded on scroll. Assuming end of page.")
                            break

                        if VERBOSE_LOGGING:
                            print(f"    -> Found {current_listing_count} listings. Scrolling for more...")
                        last_listing_count = current_listing_count

                        search_results_container.hover()
                        # Mouse wheel burst - 3-6 rapid ticks
                        num_ticks = random.randint(3, 6)
                        for _ in range(num_ticks):
                            page.mouse.wheel(0, human_scroll_px())
                            time.sleep(random.uniform(0.01, 0.03))
                        
                        time.sleep(random.uniform(0.4, 0.8))  # Brief pause after burst
                        scroll_attempts += 1
                else:
                    if VERBOSE_LOGGING:
                        print("    -> ⚠️ Could not find specific search results container, falling back to body scroll.")
                    last_height = page.evaluate('document.body.scrollHeight')
                    scroll_attempts = 0
                    while scroll_attempts < 15: # Safety break
                        # Mouse wheel burst scrolling
                        scroll_like_mouse_wheel(page, random.randint(800, 1500))
                        
                        new_height = page.evaluate('document.body.scrollHeight')
                        if new_height == last_height:
                            break
                        last_height = new_height
                        scroll_attempts += 1

                print("✅ Reached the bottom of the page.")

                # Extract prices from search results page ONCE (no need to open individual listings)
                zpid_to_price = extract_prices_from_search_page(page)

                # Collect all listing URLs and the next page link BEFORE navigating away
                listing_links_on_page = []
                selectors = ['a[href*="/homedetails/"]', '.property-card a', 'a[href*="/zpid/"]']
                for selector in selectors:
                    links = page.query_selector_all(selector)
                    for link in links:
                        href = link.get_attribute('href')
                        if href and ('/homedetails/' in href or '/zpid/' in href):
                            full_url = href if href.startswith('http') else 'https://www.zillow.com' + href
                            listing_links_on_page.append(full_url)

                # Remove duplicates
                listing_links_on_page = sorted(list(set(listing_links_on_page)))

                # Slightly shuffle search results — swaps within small groups of 3 so the
                # order is recognizably similar but not robotically top-to-bottom.
                listing_links_on_page = slightly_shuffle(listing_links_on_page)

                # Add for_review links to the TOP of the list, but only on the FIRST PAGE
                # (prepended after shuffling so they stay at the front)
                if current_page_num == 1:
                    for_review_links = get_for_review_links()
                    if for_review_links:
                        listing_links_on_page = for_review_links + listing_links_on_page
                        print(f"📋 Added {len(for_review_links)} links from for_review table to the top of the list.")

                print(f"🔍 Found {len(listing_links_on_page)} unique listings on this page.")

                # Process each listing
                for i, link in enumerate(listing_links_on_page):
                    # On a resumed run, skip listings we already processed on this page
                    if is_resuming and current_page_num == resume_page_num and i < resume_listing_index:
                        print(f"    ⏩ Skipping listing {i+1} (already processed before interruption).")
                        continue

                    # Strategic break: pause for 2-5 minutes every 10-15 minutes to mimic
                    # a human stepping away from the screen between browsing sessions.
                    elapsed = time.time() - last_break_time
                    if elapsed >= next_break_interval:
                        break_secs = random.uniform(2 * 60, 5 * 60)
                        print(f"\n☕ Strategic break — pausing for {break_secs / 60:.1f} min to look human...")
                        time.sleep(break_secs)
                        last_break_time = time.time()
                        next_break_interval = random.uniform(10 * 60, 15 * 60)
                        print("▶️  Resuming scraping.")

                    zpid_match = re.search(r'/(\d+)_zpid/', link)
                    if not zpid_match:
                        print(f"    ⚠️ Could not extract ZPID from {link}. Skipping.")
                        continue

                    zpid = int(zpid_match.group(1))

                    print(f"Processing listing {i+1}/{len(listing_links_on_page)} on page {current_page_num} (ZPID: {zpid})...")

                    # Check if this ZPID should be processed
                    should_process = should_process_zpid(db_conn, zpid)

                    if not should_process:
                        # Already fully processed — price/status refreshed during the
                        # pre-scrape update_existing_listings() step above.
                        if VERBOSE_LOGGING:
                            print(f"    -> ⏭️  ZPID {zpid} already in DB — skipping.")
                        continue

                    try:
                        processed_ok = process_listing(listing_page, zpid, link, db_conn)
                        
                        # Bring search page back to front after processing listing
                        page.bring_to_front()
                        
                        # Scroll back up and dwell — the page is at the bottom from initial load.
                        # Use JS scrollBy instead of mouse wheel to avoid zooming Zillow's map panel.
                        try:
                            scroll_px = random.randint(600, 1200)
                            page.evaluate(f"window.scrollBy({{top: -{scroll_px}, left: 0, behavior: 'instant'}})")
                            mouse_wander(page)
                            human_dwell_pause(page)
                        except Exception:
                            pass
                        
                        # Only remove from for_review on successful processing
                        if processed_ok:
                            remove_from_for_review(link)
                        else:
                            if VERBOSE_LOGGING:
                                print("    -> ⚠️ Skipping for_review removal due incomplete/blocked processing.")
                        
                    except Exception as e:
                        print(f"    -> ❌ An error occurred while scraping ZPID {zpid}: {e}")
                        print("🛑 ALL ERRORS ARE CRITICAL - STOPPING ENTIRE PROCESS!")
                        raise e

                    sleep_jitter(2, 0.4) # Be respectful

                    # Checkpoint: record that we finished this listing index
                    save_checkpoint(current_page_num, i + 1)

                # Navigate to the next page — re-query fresh to avoid stale ElementHandle
                next_page_button = page.query_selector('a[title="Next page"]')
                if next_page_button:
                    print("\n➡️ Clicking 'Next Page'...")
                    next_page_button.click()
                    page.wait_for_load_state('load', timeout=60000)
                    sleep_jitter(5, 0.35)
                    current_page_num += 1
                    resume_listing_index = 0  # full page, no skip needed
                    save_checkpoint(current_page_num, 0)
                else:
                    if len(listing_links_on_page) == 0:
                        # Zero listings is almost certainly a page-load failure, not a real
                        # end-of-results. Preserve the checkpoint so the next run can retry.
                        print("⚠️  0 listings found on this page — likely a load failure, NOT end of results.")
                        print("   Checkpoint preserved. Re-run the script to retry this page.")
                        break
                    print("✅ No more pages found. All done!")
                    clear_checkpoint()
                    completed_successfully = True
                    break

        except Exception as e:
            print(f"❌ An error occurred in the main process: {e}")
            print("🛑 ALL ERRORS ARE CRITICAL - CANNOT CONTINUE!")
            print("💡 Browser left open for debugging. Close manually when done.")
            send_telegram_message("FAILURE")
            raise e
        
        finally:
            print("🛑 Closing database connection.")
            # Play an alert noise before exiting
            winsound.Beep(1000, 500)
            if completed_successfully:
                send_telegram_message("🛑 live_scraper.py finished.")
            db_conn.close()
            # Don't close browser or pages - leave them open for debugging

if __name__ == "__main__":
    run_live_extraction()
