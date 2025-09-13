import os
import sqlite3
import duckdb
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from html import escape
import logging
import threading
import time
from datetime import datetime
from config import (
    DEFAULT_SCORING_PARAMETERS, DEFAULT_FEATURE_WEIGHTS, 
    DRIVE_TIME_EMOJIS, RISK_EMOJIS, CRIME_EMOJIS,
    APP_CONFIG, DATABASE_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get(APP_CONFIG["secret_key_env"])
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is required but not set")

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), DATABASE_CONFIG["local_db_path"])

# Cloud-only ratings database URL. Try multiple sources for flexibility
RATINGS_DB_URL = os.environ.get(APP_CONFIG["ratings_db_url_env"], "")

# For local development, also try reading from .streamlit/secrets.toml
if not RATINGS_DB_URL:
    try:
        secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                import re
                content = f.read()
                # Simple regex to extract the URL from TOML
                match = re.search(r'RATINGS_DB_URL\s*=\s*"([^"]+)"', content)
                if match:
                    RATINGS_DB_URL = match.group(1)
    except Exception as e:
        logger.warning(f"Could not read secrets.toml: {e}")

# Debug: Log the environment variable status
logger.info(f"RATINGS_DB_URL environment variable: {'SET' if RATINGS_DB_URL else 'NOT SET'}")
if RATINGS_DB_URL:
    # Only log first 20 chars for security
    logger.info(f"RATINGS_DB_URL starts with: {RATINGS_DB_URL[:20]}...")
CORRECT_PASSWORD = os.environ.get(APP_CONFIG["correct_password_env"])
if not CORRECT_PASSWORD:
    raise RuntimeError("APP_PASSWORD environment variable is required but not set")

def get_drive_time_color(drive_time_minutes):
    """Get color for drive time based on categories"""
    if pd.isna(drive_time_minutes) or drive_time_minutes is None:
        return "#999999"  # Gray for no data
    
    drive_time = float(drive_time_minutes)
    
    if drive_time < 5:
        return "#006400"    # Dark green - excellent
    elif drive_time < 10:
        return "#228B22"    # Forest green - very good
    elif drive_time < 20:
        return "#000000"    # Black - acceptable
    elif drive_time < 40:
        return "#FF8C00"    # Dark orange - concerning
    else:
        return "#DC143C"    # Crimson red - poor

def get_school_rating_color(school_rating):
    """Get color for school rating based on 1-10 scale (higher is better)"""
    if pd.isna(school_rating) or school_rating is None:
        return "#999999"  # Gray for no data
    
    rating = float(school_rating)
    
    if rating >= 8.5:
        return "#006400"    # Dark green - excellent (8.5-10)
    elif rating >= 7.0:
        return "#228B22"    # Forest green - very good (7.0-8.4)
    elif rating >= 5.5:
        return "#000000"    # Black - acceptable (5.5-6.9)
    elif rating >= 4.0:
        return "#FF8C00"    # Dark orange - concerning (4.0-5.4)
    else:
        return "#DC143C"    # Crimson red - poor (1.0-3.9)

def get_risk_emoji(risk_level):
    """Get emoji for risk level (0-4 scale)"""
    if pd.isna(risk_level) or risk_level is None:
        return "❓"  # Question mark for unknown
    
    risk_int = min(4, max(0, int(round(float(risk_level)))))
    return RISK_EMOJIS.get(risk_int, "❓")

def get_crime_emoji(crime_icon_level):
    """Get emoji for crime level (0-4 scale where 0 is worst)"""
    if pd.isna(crime_icon_level) or crime_icon_level is None:
        return "❓"  # Question mark for unknown
        
    crime_level = min(4, max(0, int(round(float(crime_icon_level)))))
    return CRIME_EMOJIS.get(crime_level, "❓")

def extract_city_from_address(address):
    """Extract City, State from address"""
    try:
        if address is None:
            return ""
        text = str(address).strip()
        if not text or text.lower() == "nan":
            return ""
        parts = [p.strip() for p in text.split(',') if p.strip()]
        if len(parts) >= 3:
            city = parts[1]
            state_token = parts[2].split()[0] if parts[2] else ""
            letters_only = ''.join(ch for ch in state_token if ch.isalpha())
            state_abbrev = (letters_only[:2] if letters_only else state_token[:2]).upper()
            return f"{city}, {state_abbrev}".strip(', ')
        if len(parts) >= 2:
            return parts[1]
        return ""
    except Exception:
        return ""

def normalize_weights(weights: dict) -> dict:
    """Normalize weights to sum to 1"""
    total_positive_weight = sum(w for w in weights.values() if w > 0)
    if total_positive_weight == 0:
        return weights
    
    normalized = {}
    for key, weight in weights.items():
        if weight > 0:
            normalized[key] = weight / total_positive_weight
        else:
            normalized[key] = weight / total_positive_weight
            
    return normalized

def generate_scoring_sql(weights: dict, params: dict, financing_filter: list = None) -> str:
    """Generate SQL query for property scoring"""
    sql = """
    WITH base_features AS (
        SELECT
            p.zpid,
            p.source_url,
            p.price,
            pf.full_address,
            pf.city,
            pf.county,
            pf.beds,
            pf.baths,
            pf.home_size_sqft,
            pf.lot_size_acres,
            pf.financing_eligibility,
            pf.financing_eligibility_explanation,
            pf.elementary_school_distance_miles,
            pf.elementary_school_rating,
            pf.middle_school_distance_miles,
            pf.middle_school_rating,
            pf.high_school_distance_miles,
            pf.high_school_rating,
            pf.privacy_level,
            pf.view_rating,
            pf.view_type,
            pf.waterfront_quality,
            pf.land_usability,
            pf.vegetation_density,
            pf.road_exposure,
            pf.negative_features_severity,
            pf.negative_features_list,
            pf.positive_features_score,
            pf.positive_features_list,
            pf.flood_risk_severity,
            pf.fire_risk_severity,
            pf.wind_risk_severity,
            pf.heat_risk_severity,
            pf.air_risk_severity,
            pf.dedicated_office,
            pf.kitchen_quality,
            pf.bathroom_quality,
            pf.general_interior_quality,
            pf.general_exterior_quality,
            pf.house_style,
            pf.general_assessment,
            c.violent_100k,
            c.property_100k,
            g.drive_time,
            g.name as grocery_name,
            g.rating as grocery_rating,
            g.user_ratings_total as grocery_ratings_count
        FROM properties p
        LEFT JOIN property_features pf ON p.zpid = pf.zpid
        LEFT JOIN crime c ON p.zpid = c.zpid
        LEFT JOIN grocery g ON p.zpid = g.zpid
    ),
    
    normalized_scores AS (
        SELECT
            zpid,
            
            -- Price: Lower is better
            (1 - ((price - MIN(price) OVER()) / NULLIF(MAX(price) OVER() - MIN(price) OVER(), 0))) * 100 AS price_score,

            -- Beds & Baths: More is better, capped
            LEAST(beds, 5) / 5.0 * 100 AS beds_score,
            LEAST(baths, 4) / 4.0 * 100 AS baths_score,

            -- Home Size: Diminishing returns
            CASE
                WHEN home_size_sqft >= {home_size_tier2_sqft} THEN 100.0
                WHEN home_size_sqft > {home_size_tier1_sqft} THEN 80.0 + ((home_size_sqft - {home_size_tier1_sqft}) / ({home_size_tier2_sqft} - {home_size_tier1_sqft})) * 20.0
                ELSE (COALESCE(home_size_sqft, 0) / {home_size_tier1_sqft}) * 80.0
            END AS home_size_sqft_score,

            -- Lot Size: More is better up to cap
            LEAST(COALESCE(lot_size_acres, 0) / {lot_size_cap_acres}, 1.0) * 100 AS lot_size_acres_score,

            -- AI Quality Ratings (1-5 scale to 0-100)
            (COALESCE(kitchen_quality, 3) - 1) * 25.0 AS kitchen_quality_score,
            (COALESCE(bathroom_quality, 3) - 1) * 25.0 AS bathroom_quality_score,
            (COALESCE(general_interior_quality, 3) - 1) * 25.0 AS general_interior_quality_score,
            (COALESCE(general_exterior_quality, 3) - 1) * 25.0 AS general_exterior_quality_score,
            (COALESCE(house_style, 3) - 1) * 25.0 AS house_style_score,
            (COALESCE(privacy_level, 3) - 1) * 25.0 AS privacy_level_score,
            (COALESCE(view_rating, 3) - 1) * 25.0 AS view_rating_score,
            (COALESCE(land_usability, 3) - 1) * 25.0 AS land_usability_score,
            (COALESCE(waterfront_quality, 1) - 1) * 25.0 AS waterfront_quality_score,
            (COALESCE(road_exposure, 3) - 1) * 25.0 AS road_exposure_score,
            (COALESCE(vegetation_density, 3) - 1) * 25.0 AS vegetation_density_score,
            (COALESCE(positive_features_score, 3) - 1) * 25.0 AS positive_features_score_score,

            -- Boolean Features
            CAST(dedicated_office AS INTEGER) * 100 AS dedicated_office_score,
            
            -- Drive Time Score (negative feature - higher drive time = worse score)
            (1 - (COALESCE(drive_time, {drive_time_max_minutes}) / {drive_time_max_minutes})) * 100 AS drive_time_score,

            -- Risk Score (lower risk is better, assuming 1-10 scale)
            (1 - ( (
                COALESCE(flood_risk_severity, 5) + 
                COALESCE(fire_risk_severity, 5) + 
                COALESCE(wind_risk_severity, 5) +
                COALESCE(heat_risk_severity, 5) +
                COALESCE(air_risk_severity, 5)
            ) / 5.0 - 1) / 9.0) * 100 AS avg_risk_severity_score,

            -- School Score
            ( (
                COALESCE(elementary_school_rating, 5) +
                COALESCE(middle_school_rating, 5) +
                COALESCE(high_school_rating, 5)
            ) / 3.0) * 10 AS avg_school_rating_score,
            
            -- School Distance Score
            CASE
                WHEN (
                    COALESCE(elementary_school_distance_miles, {school_dist_zero_points_miles}) +
                    COALESCE(middle_school_distance_miles, {school_dist_zero_points_miles}) +
                    COALESCE(high_school_distance_miles, {school_dist_zero_points_miles})
                ) / 3.0 <= {school_dist_full_points_miles} THEN 100.0
                WHEN (
                    COALESCE(elementary_school_distance_miles, {school_dist_zero_points_miles}) +
                    COALESCE(middle_school_distance_miles, {school_dist_zero_points_miles}) +
                    COALESCE(high_school_distance_miles, {school_dist_zero_points_miles})
                ) / 3.0 >= {school_dist_zero_points_miles} THEN 0.0
                ELSE (1.0 - (
                    ((
                        COALESCE(elementary_school_distance_miles, {school_dist_zero_points_miles}) +
                        COALESCE(middle_school_distance_miles, {school_dist_zero_points_miles}) +
                        COALESCE(high_school_distance_miles, {school_dist_zero_points_miles})
                    ) / 3.0) - {school_dist_full_points_miles}
                ) / ({school_dist_zero_points_miles} - {school_dist_full_points_miles})) * 100.0
            END AS avg_school_distance_score,

            -- Crime Scores (lower is better)
            (1 - (COALESCE(violent_100k, {max_violent_crime_100k} / 2.0) / {max_violent_crime_100k})) * 100 AS violent_crime_score,
            (1 - (COALESCE(property_100k, {max_property_crime_100k} / 2.0) / {max_property_crime_100k})) * 100 AS property_crime_score,
            
            -- Average Crime Score
            ((1 - (COALESCE(violent_100k, {max_violent_crime_100k} / 2.0) / {max_violent_crime_100k})) * 100 + 
             (1 - (COALESCE(property_100k, {max_property_crime_100k} / 2.0) / {max_property_crime_100k})) * 100) / 2.0 AS avg_crime_severity_score

        FROM base_features
    )
    """.format(**params)

    # Build weighted sum
    weighted_sum_parts = []
    valid_features = [f for f in weights.keys()]

    for feature in valid_features:
        weight = weights[feature]
        if weight < 0:
            weighted_sum_parts.append(f"((100 - {feature}_score) / 100.0 * {abs(weight)})")
        else:
            weighted_sum_parts.append(f"({feature}_score * {weight})")

    weighted_sum_sql = " + ".join([p for f, p in zip(valid_features, weighted_sum_parts) if weights[f] > 0])
    penalty_sum_sql = " + ".join([p for f, p in zip(valid_features, weighted_sum_parts) if weights[f] < 0])

    school_dist_zero = params['school_dist_zero_points_miles']
    # Build financing filter WHERE clause
    financing_conditions = []
    if financing_filter:
        if 'eligible' in financing_filter:
            financing_conditions.append("bf.financing_eligibility = TRUE")
        if 'not_eligible' in financing_filter:
            financing_conditions.append("bf.financing_eligibility = FALSE")
        if 'unknown' in financing_filter:
            financing_conditions.append("bf.financing_eligibility IS NULL")
    
    financing_where = ""
    if financing_conditions:
        financing_where = "WHERE " + " OR ".join(financing_conditions)
    
    final_select = f"""
    SELECT
        ns.zpid,
        ({weighted_sum_sql}) - ({penalty_sum_sql if penalty_sum_sql else '0'}) AS total_score,
        bf.source_url,
        bf.county,
        bf.full_address,
        bf.price,
        bf.beds,
        bf.baths,
        bf.home_size_sqft,
        bf.lot_size_acres,
        bf.kitchen_quality,
        bf.bathroom_quality,
        bf.general_interior_quality,
        bf.general_exterior_quality,
        bf.house_style,
        bf.privacy_level,
        bf.view_rating,
        bf.view_type,
        bf.land_usability,
        bf.waterfront_quality,
        bf.road_exposure,
        bf.vegetation_density,
        bf.positive_features_score,
        bf.positive_features_list,
        bf.negative_features_severity,
        bf.negative_features_list,
        bf.dedicated_office,
        bf.drive_time,
        bf.grocery_name,
        bf.grocery_rating,
        bf.grocery_ratings_count,
        bf.general_assessment,
        ( (COALESCE(bf.flood_risk_severity, 5) + COALESCE(bf.fire_risk_severity, 5) + COALESCE(bf.wind_risk_severity, 5) + COALESCE(bf.heat_risk_severity, 5) + COALESCE(bf.air_risk_severity, 5)) / 5.0) as avg_risk_severity,
        ( (COALESCE(bf.elementary_school_rating, 5) + COALESCE(bf.middle_school_rating, 5) + COALESCE(bf.high_school_rating, 5)) / 3.0) as avg_school_rating,
        ( (COALESCE(bf.elementary_school_distance_miles, {school_dist_zero}) + COALESCE(bf.middle_school_distance_miles, {school_dist_zero}) + COALESCE(bf.high_school_distance_miles, {school_dist_zero})) / 3.0) as avg_school_distance,
        bf.violent_100k as violent_100k_raw,
        bf.property_100k as property_100k_raw,
        COALESCE(bf.violent_100k, 360.0) as violent_100k,
        COALESCE(bf.property_100k, 2763.0) as property_100k,
        CASE 
            WHEN bf.violent_100k IS NULL OR bf.property_100k IS NULL THEN NULL
            ELSE (bf.violent_100k * 2.0 + bf.property_100k) / 3.0
        END as avg_crime_severity_raw,
        ((COALESCE(bf.violent_100k, 360.0) * 2.0 + COALESCE(bf.property_100k, 2763.0)) / 3.0) as avg_crime_severity,
        bf.financing_eligibility,
        bf.financing_eligibility_explanation
        
    FROM normalized_scores ns
    JOIN base_features bf ON ns.zpid = bf.zpid
    {financing_where}
    ORDER BY total_score DESC
    """
    
    return sql + final_select

def get_scored_properties(sql_query):
    """Execute scoring query and return DataFrame"""
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        return conn.execute(sql_query).df()
    finally:
        conn.close()

def get_ratings_db_connection():
    """Returns a connection to the ratings DB in SQLite Cloud (cloud-only)."""
    if not RATINGS_DB_URL:
        raise RuntimeError(
            "RATINGS_DB_URL is not set. Add it to your environment variables."
        )
    try:
        import sqlitecloud  # Lazy import
    except Exception as import_err:
        raise RuntimeError(
            "sqlitecloud package is required. Run: pip install sqlitecloud"
        ) from import_err
    # Example URL format:
    # sqlitecloud://<host>:<port>/<database>?apikey=<API_KEY>
    conn = sqlitecloud.connect(RATINGS_DB_URL)
    return conn

def init_ratings_db(_conn):
    """Ensures the ratings table exists."""
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rating (
            zpid INTEGER PRIMARY KEY,
            rating TEXT CHECK (rating IN ('yes','no','maybe')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    _conn.commit()

def init_notes_db(_conn):
    """Ensures the notes table exists."""
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            zpid INTEGER PRIMARY KEY,
            note TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    _conn.commit()

def load_ratings_dict(_conn):
    """Loads current ratings into a dictionary with zpid as key and rating as value.
    Avoids pandas.read_sql to support both sqlite3 and sqlitecloud connections.
    """
    try:
        cursor = _conn.execute("SELECT zpid, rating FROM rating")
        rows = cursor.fetchall()
        if not rows:
            return {}  # empty
        # Rows are tuples like (zpid, rating)
        return dict(rows)
    except Exception:
        return {}  # empty if table missing

def save_rating(_conn, zpid_value, rating_value: str):
    """Upserts or deletes a rating for a given zpid based on rating_value ('' deletes)."""
    if rating_value in (None, ""):
        _conn.execute("DELETE FROM rating WHERE zpid = ?", (int(zpid_value),))
        _conn.commit()
        return
    if rating_value not in ("yes", "no", "maybe"):
        return
    _conn.execute(
        """
        INSERT INTO rating (zpid, rating, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(zpid) DO UPDATE SET
            rating = excluded.rating,
            updated_at = datetime('now')
        """,
        (int(zpid_value), rating_value),
    )
    _conn.commit()

def load_notes_dict(_conn):
    """Loads current notes into a dictionary with zpid as key and note as value."""
    try:
        cursor = _conn.execute("SELECT zpid, note FROM notes")
        rows = cursor.fetchall()
        if not rows:
            return {}  # empty
        # Rows are tuples like (zpid, note)
        return dict(rows)
    except Exception:
        return {}  # empty if table missing

def save_note(_conn, zpid_value, note_value: str):
    """Upserts or deletes a note for a given zpid based on note_value ('' or None deletes)."""
    if note_value in (None, ""):
        _conn.execute("DELETE FROM notes WHERE zpid = ?", (int(zpid_value),))
        _conn.commit()
        return
    _conn.execute(
        """
        INSERT INTO notes (zpid, note, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(zpid) DO UPDATE SET
            note = excluded.note,
            updated_at = datetime('now')
        """,
        (int(zpid_value), note_value),
    )
    _conn.commit()

def database_keepalive():
    """Runs a simple query on the SQLite Cloud database to keep it active."""
    if not RATINGS_DB_URL:
        logger.info("Database keepalive: RATINGS_DB_URL not set, skipping keepalive")
        return
    
    try:
        logger.info("Database keepalive: Running keepalive query...")
        conn = get_ratings_db_connection()
        
        # Run a simple query to keep the connection active
        result = conn.execute("SELECT datetime('now') as current_time").fetchone()
        current_time = result[0] if result else "unknown"
        
        # Also ensure our tables exist (lightweight operation)
        init_ratings_db(conn)
        init_notes_db(conn)
        
        # Get a count of ratings as a meaningful keepalive query
        count_result = conn.execute("SELECT COUNT(*) FROM rating").fetchone()
        rating_count = count_result[0] if count_result else 0
        
        conn.close()
        logger.info(f"Database keepalive successful at {current_time}. Rating count: {rating_count}")
        
    except Exception as e:
        logger.error(f"Database keepalive failed: {e}")

def start_keepalive_thread():
    """Starts a background thread that runs database keepalive every 3 hours."""
    if not RATINGS_DB_URL:
        logger.info("Database keepalive thread: RATINGS_DB_URL not set, not starting keepalive thread")
        return
    
    def keepalive_worker():
        logger.info("Database keepalive thread started - will ping every 3 hours")
        while True:
            try:
                # Run keepalive immediately on start
                database_keepalive()
                # Sleep for configured interval
                time.sleep(DATABASE_CONFIG["keepalive_interval_seconds"])
            except Exception as e:
                logger.error(f"Keepalive thread error: {e}")
                # Sleep before retrying on error
                time.sleep(DATABASE_CONFIG["keepalive_retry_seconds"])
    
    # Start the keepalive thread as a daemon thread
    keepalive_thread = threading.Thread(target=keepalive_worker, daemon=True)
    keepalive_thread.start()
    logger.info("Database keepalive background thread started")

@app.route('/')
def index():
    try:
        # Get parameters from request
        params = DEFAULT_SCORING_PARAMETERS.copy()
        weights = DEFAULT_FEATURE_WEIGHTS.copy()
        
        # Override weights with any provided URL parameters
        for key in weights.keys():
            param_value = request.args.get(f"weight_{key}")
            if param_value:
                try:
                    weights[key] = float(param_value)
                except ValueError:
                    pass  # Keep default if invalid
        
        # Get filter parameters
        rating_filter = request.args.getlist('rating_filter')
        if not rating_filter:
            rating_filter = ['yes', 'maybe', 'blank']  # Default filters
            
        # Get financing eligibility filter
        financing_filter = request.args.getlist('financing_filter')
        if not financing_filter:
            financing_filter = ['eligible']  # Default to only eligible properties
        
        # Password check
        password = request.args.get('password', '')
        password_correct = password == CORRECT_PASSWORD
        
        # Generate and execute query
        norm_weights = normalize_weights(weights)
        scoring_sql = generate_scoring_sql(norm_weights, params, financing_filter)
        results_df = get_scored_properties(scoring_sql)
        
        # Load ratings and notes
        ratings_dict = {}
        notes_dict = {}
        if password_correct:
            try:
                ratings_conn = get_ratings_db_connection()
                init_ratings_db(ratings_conn)
                init_notes_db(ratings_conn)
                ratings_dict = load_ratings_dict(ratings_conn)
                notes_dict = load_notes_dict(ratings_conn)
                ratings_conn.close()
            except Exception as e:
                logger.error(f"Could not load ratings and notes: {e}")
        
        # Add ratings and notes to results
        results_df['rating'] = results_df['zpid'].map(lambda x: ratings_dict.get(x, ''))
        results_df['note'] = results_df['zpid'].map(lambda x: notes_dict.get(x, ''))
        
        # Add city info for listing column
        results_df['city_state'] = results_df['full_address'].apply(extract_city_from_address)
        
        # Add drive time color coding
        results_df['drive_time_color'] = results_df['drive_time'].apply(get_drive_time_color)
        
        # Add school rating color coding
        results_df['school_rating_color'] = results_df['avg_school_rating'].apply(get_school_rating_color)
        
        # Add risk and crime emojis
        results_df['risk_emoji'] = results_df['avg_risk_severity'].apply(get_risk_emoji)
        
        # Calculate crime icon ranges from actual data (using raw values to preserve NULLs)
        valid_crime_scores = results_df['avg_crime_severity_raw'].dropna()
        if len(valid_crime_scores) > 0:
            crime_min = valid_crime_scores.min()
            crime_max = valid_crime_scores.max()
            crime_range = crime_max - crime_min
            
            # Create 5 equal ranges based on raw data (preserving NULLs for display)
            # Higher crime rates should get LOWER level numbers (worse emojis)
            if crime_range > 0:
                results_df['crime_icon_level'] = results_df['avg_crime_severity_raw'].apply(
                    lambda x: None if pd.isna(x) else 4 - min(4, int((x - crime_min) / crime_range * 5))
                )
            else:
                # All values are the same
                results_df['crime_icon_level'] = results_df['avg_crime_severity_raw'].apply(
                    lambda x: None if pd.isna(x) else 2  # Middle level
                )
        else:
            # No valid crime data
            results_df['crime_icon_level'] = None
        
        # Apply rating filter
        allowed_ratings = set(rating_filter)
        if 'blank' in allowed_ratings:
            allowed_ratings.add('')
        results_df = results_df[results_df['rating'].isin(allowed_ratings)]
        
        # Default sorting by total_score descending (no user sorting controls)
        results_df = results_df.sort_values(by='total_score', ascending=False)
        
        # Add crime emojis
        results_df['crime_emoji'] = results_df['crime_icon_level'].apply(get_crime_emoji)
        
        # Convert to dict for template, handling NaN values properly
        properties = results_df.to_dict('records')
        
        # Convert pandas NaN to None for proper template handling
        for prop in properties:
            if pd.isna(prop.get('crime_icon_level')):
                prop['crime_icon_level'] = None
            if pd.isna(prop.get('avg_crime_severity_raw')):
                prop['avg_crime_severity_raw'] = None
            if pd.isna(prop.get('drive_time')):
                prop['drive_time'] = None
        
        # Debug: Check if we have any properties with missing crime data
        missing_crime_count = sum(1 for p in properties if p.get('crime_icon_level') is None)
        logger.info(f"Properties with missing crime data: {missing_crime_count} out of {len(properties)}")
        
        return render_template('index.html', 
                             properties=properties,
                             password_correct=password_correct,
                             rating_filter=rating_filter,
                             financing_filter=financing_filter,
                             current_weights=weights,
                             request=request)
                             
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('error.html', error=str(e))

@app.route('/update_rating', methods=['POST'])
def update_rating():
    password = request.form.get('password', '')
    if password != CORRECT_PASSWORD:
        return jsonify({'success': False, 'error': 'Invalid password'})
    
    zpid = request.form.get('zpid')
    rating = request.form.get('rating')
    
    try:
        ratings_conn = get_ratings_db_connection()
        init_ratings_db(ratings_conn)
        save_rating(ratings_conn, zpid, rating)
        ratings_conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating rating: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_note', methods=['POST'])
def update_note():
    password = request.form.get('password', '')
    if password != CORRECT_PASSWORD:
        return jsonify({'success': False, 'error': 'Invalid password'})
    
    zpid = request.form.get('zpid')
    note = request.form.get('note', '')
    
    try:
        notes_conn = get_ratings_db_connection()
        init_notes_db(notes_conn)
        save_note(notes_conn, zpid, note)
        notes_conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating note: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/settings')
def settings():
    """Settings page for adjusting feature weights"""
    try:
        # Get current weights from URL parameters or use defaults
        current_weights = DEFAULT_FEATURE_WEIGHTS.copy()
        
        # Override with any provided URL parameters
        for key in current_weights.keys():
            param_value = request.args.get(f"weight_{key}")
            if param_value:
                try:
                    current_weights[key] = float(param_value)
                except ValueError:
                    pass  # Keep default if invalid
        
        return render_template('settings.html', 
                             weights=current_weights,
                             default_weights=DEFAULT_FEATURE_WEIGHTS,
                             DEFAULT_SCORING_PARAMETERS=DEFAULT_SCORING_PARAMETERS)
                             
    except Exception as e:
        logger.error(f"Error in settings route: {e}")
        return render_template('error.html', error=str(e))

@app.route('/health')
def health_check():
    """Lightweight health check endpoint for Railway deployment"""
    try:
        # Just check if we can connect to the database quickly
        # Don't run the full property scoring query
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database_url_configured': bool(RATINGS_DB_URL)
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/keepalive')
def manual_keepalive():
    """Manual endpoint to test database keepalive - useful for debugging"""
    try:
        database_keepalive()
        return jsonify({
            'success': True, 
            'message': 'Database keepalive executed successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Manual keepalive failed: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

# Start the database keepalive thread when the app starts
start_keepalive_thread()

if __name__ == '__main__':
    app.run(debug=True)
