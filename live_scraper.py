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

SEARCH_URL = 'https://www.zillow.com/homes/for_sale/?searchQueryState=%7B%22isMapVisible%22%3Atrue%2C%22mapBounds%22%3A%7B%22west%22%3A-156.63478397587753%2C%22east%22%3A-112.82130741337754%2C%22south%22%3A37.003092903893084%2C%22north%22%3A50.98652477441754%7D%2C%22filterState%22%3A%7B%22sort%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22price%22%3A%7B%22max%22%3A700000%7D%2C%22mp%22%3A%7B%22max%22%3A3406%7D%2C%22beds%22%3A%7B%22min%22%3A3%7D%2C%22baths%22%3A%7B%22min%22%3A2%7D%2C%22tow%22%3A%7B%22value%22%3Afalse%7D%2C%22mf%22%3A%7B%22value%22%3Afalse%7D%2C%22con%22%3A%7B%22value%22%3Afalse%7D%2C%22land%22%3A%7B%22value%22%3Afalse%7D%2C%22apa%22%3A%7B%22value%22%3Afalse%7D%2C%22apco%22%3A%7B%22value%22%3Afalse%7D%2C%22lot%22%3A%7B%22min%22%3A87120%7D%7D%2C%22isListVisible%22%3Atrue%2C%22mapZoom%22%3A6%2C%22customRegionId%22%3A%227ca8b68bbaX1-CRlexe5hovo8mh_116nid%22%7D'


import os
import re
import json
import base64
import time
from typing import Optional, Dict, Any, List
import io

import duckdb
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from playwright.sync_api import sync_playwright
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
winsound.Beep(1000, 500)
# --- CONFIGURATION ---
DB_PATH = "property_data.db"  # Main database with properties, property_scores, and property_features tables
MODEL = "gemini-2.5-pro" # Using a Gemini model for high quality analysis
MAX_IMAGES_TO_EXTRACT = 80  # Max images to feed to the model per property
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate API key is present
if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required but not set")

# Global speed multiplier for all wait times (0.5 = 2x faster, 1.0 = normal speed, 2.0 = 2x slower)
SPEED_MULTIPLIER = 0.3

# Scraper settings
HEADLESS_MODE = False
SLOW_MOTION_SPEED = 100
VERBOSE_LOGGING = True

# Initialize Google Gemini client
genai.configure(api_key=GOOGLE_API_KEY)




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
            price INTEGER
        )
    """)
    
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

def update_price_for_existing_zpid(conn: duckdb.DuckDBPyConnection, zpid: int, listing_url: str, html_content: str) -> bool:
    """
    Update the price for an existing ZPID without doing full processing.
    Returns True if successfully updated, False otherwise.
    """
    try:
        # Extract fresh property data from HTML
        property_data = extract_property_data_from_html(html_content)

        # Check if we have a new price
        new_price = property_data.get('price')

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

def should_process_zpid(conn: duckdb.DuckDBPyConnection, zpid: int) -> bool:
    """
    Check if a ZPID should be processed based on the following criteria:
    1. If ZPID exists in properties table, update price and skip full processing
    2. Otherwise, process it fully
    """
    try:
        # Check if ZPID exists in properties table
        property_result = conn.execute("SELECT 1 FROM properties WHERE zpid = ?", [zpid]).fetchone()
        if property_result is not None:
            return False  # This will be handled by price update logic in main loop

        # Check property_scores table
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
                            property_data['status'] = prop['homeStatus']
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
        except json.JSONDecodeError:
            pass
    
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
            'price': property_data.get('price')
        }
        
        # Get photo_count for internal use (but don't save to db)
        photo_count = property_data.get('photo_count', 0)
        
        # Filter out None values and build SQL
        filtered_data = {k: v for k, v in insert_data.items() if v is not None}
        
        # Require at least zpid and source_url
        if 'zpid' not in filtered_data or 'source_url' not in filtered_data:
            print(f"    -> ❌ CRITICAL: Missing required data for ZPID {zpid}")
            print("🛑 Property data extraction failure - stopping execution!")
            raise Exception(f"CRITICAL FAILURE: Missing required zpid or source_url for ZPID {zpid}")
            
        columns = ', '.join(filtered_data.keys())
        placeholders = ', '.join(['?' for _ in filtered_data])
        
        # Delete existing record first, then insert fresh data
        conn.execute("DELETE FROM properties WHERE zpid = ?", [zpid])
        conn.execute(f"INSERT INTO properties ({columns}) VALUES ({placeholders})", 
                    list(filtered_data.values()))
        conn.commit()
        
        if VERBOSE_LOGGING:
            print(f"    -> 💾 Properties table updated for ZPID {zpid} (price: ${filtered_data.get('price', 'N/A')})")
        return True, photo_count
        
    except Exception as e:
        print(f"❌ CRITICAL DATABASE ERROR updating properties for ZPID {zpid}: {e}")
        print(f"🔍 DEBUG INFO:")
        print(f"    - Filtered data keys: {list(filtered_data.keys())}")
        print(f"    - Values: {list(filtered_data.values())}")
        # FAIL FAST - raise the error to stop execution
        raise e



# --- IMAGE HANDLING ---

def download_image(url: str) -> Optional[bytes]:
    """Download an image from a URL and return its content as bytes."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
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
        text_content = main_content.get_text(separator=' ', strip=True)
        
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        meaningful_lines = [
            line for line in lines
            if len(line) > 15 and len(line.split()) > 2 and 'cookie' not in line.lower()
        ]
        
        return '\n'.join(meaningful_lines[:200]) # Limit lines to keep it manageable
    except Exception as e:
        print(f"⚠️ Error extracting comprehensive text: {e}")
        return ""

def process_listing(page, zpid: int, listing_url: str, conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Processes a single listing: scrapes data, updates properties table, calls AI model, saves features, and saves analysis results.
    """
    try:
        if VERBOSE_LOGGING:
            print(f"    -> Navigating to listing: {listing_url}")
        page.goto(listing_url, wait_until='load', timeout=60000)
        time.sleep(3 * SPEED_MULTIPLIER)

        # 1. Extract comprehensive text and HTML for the model and properties update
        listing_html = page.content()
        comprehensive_text = extract_comprehensive_text_content(listing_html)
        if not comprehensive_text:
            print("    -> ❌ CRITICAL: Could not extract text content!")
            print("🛑 This indicates a serious problem with HTML parsing or page structure.")
            raise Exception("CRITICAL FAILURE: No text content extracted - this must be fixed!")

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
                        time.sleep(4 * SPEED_MULTIPLIER)
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
        max_scrolls = 15
        last_image_count = 0
        consecutive_no_new_images = 0
        
        # Use the smaller of photo_count or MAX_IMAGES_TO_EXTRACT to prevent extra images
        target_image_count = min(photo_count, MAX_IMAGES_TO_EXTRACT) if photo_count > 0 else MAX_IMAGES_TO_EXTRACT
        
        if VERBOSE_LOGGING and photo_count > 0:
            print(f"    -> Property has {photo_count} photos, limiting extraction to {target_image_count} images")
        
        while len(all_image_urls) < target_image_count and scroll_attempts < max_scrolls:
            # Collect images before scrolling
            for selector in image_selectors:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    src = elem.get_attribute('src') or elem.get_attribute('data-src')
                    if src and 'photos.zillowstatic.com' in src:
                        if any(size in src for size in ['_1536', '_1024', 'cc_ft_', 'uncropped_scaled_within', 'p_f']):
                            all_image_urls.add(src.split('?')[0])
            
            current_image_count = len(all_image_urls)
            
            if current_image_count >= target_image_count:
                break
            
            # Check if we found new images in this iteration
            if current_image_count == last_image_count:
                consecutive_no_new_images += 1
                if consecutive_no_new_images >= 3:
                    break
            else:
                consecutive_no_new_images = 0
                last_image_count = current_image_count
            
            # Check if we're at the bottom of the scrollable container
            try:
                at_bottom = page.evaluate("""
                    () => {
                        const container = document.querySelector('#__c11n_40v5e');
                        if (container) {
                            return container.scrollTop + container.clientHeight >= container.scrollHeight - 10;
                        }
                        return false;
                    }
                """)
                if at_bottom:
                    break
            except Exception:
                pass
            
            # Scroll the correct container
            try:
                page.evaluate("document.querySelector('#__c11n_40v5e').scrollBy(0, 1500)")
            except Exception:
                page.mouse.wheel(0, 1500)
            
            scroll_attempts += 1
            time.sleep(2 * SPEED_MULTIPLIER)
        
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

        # 4. Download images into memory and prepare API payload for Gemini
        model_content = [RUBRIC_INSTRUCTIONS, f"Listing text:\n\n{comprehensive_text}"]
        
        downloaded_count = 0
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
                    if VERBOSE_LOGGING:
                        print(f"    -> ⚠️ Could not process image {img_url}: {e}")

        if downloaded_count == 0:
            print("    -> ❌ CRITICAL: No images could be downloaded/processed for analysis!")
            print("🛑 This indicates a serious problem with image extraction or network connectivity.")
            print("🔧 Process cannot continue without images for AI analysis.")
            raise Exception("CRITICAL FAILURE: No images available for AI analysis - this must be fixed!")

        if VERBOSE_LOGGING:
            print(f"    -> Prepared {downloaded_count} images for AI analysis.")

        # 5. Call Gemini API for feature extraction
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
            if hasattr(response, 'prompt_feedback'):
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


def run_live_extraction():
    """
    Main function to orchestrate the Zillow scraping and feature extraction process.
    """
    with sync_playwright() as p:
        print("--- 🏠 Starting Zillow Live Feature Extractor (Gemini Edition) ---")
        
        db_conn = setup_database()
        
        browser = p.chromium.launch(
            headless=HEADLESS_MODE, 
            slow_mo=SLOW_MOTION_SPEED,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        try:
            # Step 1: Start from Zillow homepage to avoid captcha
            print("🌐 Navigating to Zillow homepage first...")
            page.goto('https://www.zillow.com/', wait_until='domcontentloaded')
            time.sleep(3 * SPEED_MULTIPLIER)

            # Step 2: Go to the search page
            print("🔍 Loading search results page...")
            page.goto(SEARCH_URL, wait_until='load', timeout=60000)
            time.sleep(5 * SPEED_MULTIPLIER)

            # CAPTCHA check
            if page.is_visible('#captcha-container') or page.is_visible('iframe[title*="recaptcha"]'):
                print("🤖 CAPTCHA detected! Please solve it manually.")
                input("Press Enter after solving captcha...")
            
            current_page_num = 1
            while True:
                print(f"\n--- Scraping Page {current_page_num} ---")

                # Robust scroll to load all listings (mirrors scraper.py)
                print("🔄 Scrolling down to load all listings on the page...")
                search_results_selector = "#grid-search-results"
                search_results_container = page.locator(search_results_selector)

                if search_results_container.is_visible(timeout=5000):
                    if VERBOSE_LOGGING:
                        print("    -> Found search results container. Scrolling within it...")
                    last_listing_count = 0
                    scroll_attempts = 0
                    while scroll_attempts < 15: # safety break to prevent infinite loops
                        current_listing_count = len(page.query_selector_all('article[data-test="property-card"]'))

                        if current_listing_count > 0 and current_listing_count == last_listing_count:
                            if VERBOSE_LOGGING:
                                print("    -> No new listings loaded on scroll. Assuming end of page.")
                            break

                        if VERBOSE_LOGGING:
                            print(f"    -> Found {current_listing_count} listings. Scrolling for more...")
                        last_listing_count = current_listing_count

                        search_results_container.hover()
                        page.mouse.wheel(0, 1500) # Scroll down
                        time.sleep(3 * SPEED_MULTIPLIER)
                        scroll_attempts += 1
                else:
                    if VERBOSE_LOGGING:
                        print("    -> ⚠️ Could not find specific search results container, falling back to body scroll.")
                    last_height = page.evaluate('document.body.scrollHeight')
                    scroll_attempts = 0
                    while scroll_attempts < 10: # Safety break
                        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                        time.sleep(3 * SPEED_MULTIPLIER)
                        new_height = page.evaluate('document.body.scrollHeight')
                        if new_height == last_height:
                            break
                        last_height = new_height
                        scroll_attempts += 1

                print("✅ Reached the bottom of the page.")

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
                next_page_button = page.query_selector('a[title="Next page"]')

                # Add for_review links to the TOP of the list, but only on the FIRST PAGE
                if current_page_num == 1:
                    for_review_links = get_for_review_links()
                    if for_review_links:
                        # Add for_review links to the beginning of the list
                        listing_links_on_page = for_review_links + listing_links_on_page
                        print(f"📋 Added {len(for_review_links)} links from for_review table to the top of the list.")

                print(f"🔍 Found {len(listing_links_on_page)} unique listings on this page.")

                # Process each listing
                for i, link in enumerate(listing_links_on_page):
                    zpid_match = re.search(r'/(\d+)_zpid/', link)
                    if not zpid_match:
                        print(f"    ⚠️ Could not extract ZPID from {link}. Skipping.")
                        continue

                    zpid = int(zpid_match.group(1))

                    print(f"Processing listing {i+1}/{len(listing_links_on_page)} on page {current_page_num} (ZPID: {zpid})...")

                    # Check if this ZPID should be processed
                    should_process = should_process_zpid(db_conn, zpid)

                    if not should_process:
                        # Check if this is an existing ZPID that needs price update
                        existing_property = db_conn.execute(
                            "SELECT 1 FROM properties WHERE zpid = ?", [zpid]
                        ).fetchone()

                        if existing_property is not None:

                            # Use a new page to get fresh HTML for price update
                            listing_page = context.new_page()
                            try:
                                listing_page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                                listing_page.goto(link, wait_until='load', timeout=60000)
                                time.sleep(3 * SPEED_MULTIPLIER)
                                listing_html = listing_page.content()

                                update_price_for_existing_zpid(db_conn, zpid, link, listing_html)

                            except Exception as e:
                                print(f"    -> ❌ Error updating price for ZPID {zpid}: {e}")
                                print("🛑 Price update failure - continuing with next listing...")
                            finally:
                                listing_page.close()

                            time.sleep(1 * SPEED_MULTIPLIER) # Be respectful
                            continue
                        else:
                            # Truly skip this ZPID
                            continue

                    # Use a new page for each listing to avoid state conflicts
                    listing_page = context.new_page()
                    try:
                        listing_page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                        process_listing(listing_page, zpid, link, db_conn)
                    except Exception as e:
                        print(f"    -> ❌ An error occurred while scraping ZPID {zpid}: {e}")
                        print("🛑 ALL ERRORS ARE CRITICAL - STOPPING ENTIRE PROCESS!")
                        raise e
                    finally:
                        listing_page.close()

                    time.sleep(2 * SPEED_MULTIPLIER) # Be respectful

                # Navigate to the next page
                if next_page_button:
                    print("\n➡️ Clicking 'Next Page'...")
                    next_page_button.click()
                    page.wait_for_load_state('load', timeout=60000)
                    time.sleep(5 * SPEED_MULTIPLIER)
                    current_page_num += 1
                else:
                    print("✅ No more pages found. All done!")
                    break

        except Exception as e:
            print(f"❌ An error occurred in the main process: {e}")
            print("🛑 ALL ERRORS ARE CRITICAL - CANNOT CONTINUE!")
            raise e
        
        finally:
            print("🛑 Closing browser and database connection.")
            # Play an alert noise before exiting
            winsound.Beep(1000, 500)
            db_conn.close()
            browser.close()

if __name__ == "__main__":
    run_live_extraction()
