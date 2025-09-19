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
from weight_optimizer import WeightOptimizer, create_weight_url_params

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
            g.user_ratings_total as grocery_ratings_count,
            l.latitude,
            l.longitude
        FROM properties p
        LEFT JOIN property_features pf ON p.zpid = pf.zpid
        LEFT JOIN crime c ON p.zpid = c.zpid
        LEFT JOIN grocery g ON p.zpid = g.zpid
        LEFT JOIN location l ON p.zpid = l.zpid
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
        
        -- Individual feature scores for weight optimization
        ns.price_score,
        ns.beds_score,
        ns.baths_score,
        ns.home_size_sqft_score,
        ns.lot_size_acres_score,
        ns.kitchen_quality_score,
        ns.bathroom_quality_score,
        ns.general_interior_quality_score,
        ns.general_exterior_quality_score,
        ns.house_style_score,
        ns.privacy_level_score,
        ns.view_rating_score,
        ns.land_usability_score,
        ns.waterfront_quality_score,
        ns.road_exposure_score,
        ns.vegetation_density_score,
        ns.positive_features_score_score,
        ns.dedicated_office_score,
        ns.drive_time_score,
        ns.avg_risk_severity_score,
        ns.avg_school_rating_score,
        ns.avg_school_distance_score,
        ns.avg_crime_severity_score,
        
        -- Original property data
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
        bf.financing_eligibility_explanation,
        bf.latitude,
        bf.longitude
        
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
        logger.info("=== DUCKDB QUERY EXECUTION DEBUG ===")
        logger.info(f"DuckDB version: {duckdb.__version__}")
        logger.info(f"Database path: {DB_PATH}")
        logger.info(f"Database file size: {os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 'File not found'}")
        
        # Log environment info
        import platform
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python version: {platform.python_version()}")
        
        logger.info("Executing DuckDB query...")
        result_df = conn.execute(sql_query).df()
        logger.info(f"DuckDB query executed successfully, returned {len(result_df)} rows")
        return result_df
    except Exception as e:
        logger.error(f"=== DUCKDB QUERY EXECUTION FAILED ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error args: {e.args}")
        
        # Log the full query for debugging
        logger.error("=== FULL QUERY THAT FAILED ===")
        logger.error(sql_query)
        logger.error("=== END QUERY ===")
        
        # Try to identify the specific issue by testing parts of the query
        try:
            logger.info("=== DIAGNOSTIC TESTS ===")
            
            # Test basic table access
            tables_to_test = ['properties', 'property_features', 'crime', 'grocery', 'location']
            for table in tables_to_test:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    logger.info(f"✓ Table {table} accessible with {count} records")
                    
                    # Get table schema
                    schema = conn.execute(f"DESCRIBE {table}").fetchall()
                    logger.info(f"✓ Table {table} schema: {schema}")
                    
                except Exception as table_error:
                    logger.error(f"✗ Table {table} access failed: {table_error}")
            
            # Test progressive complexity
            test_queries = [
                "SELECT COUNT(*) FROM properties",
                "SELECT p.zpid FROM properties p LIMIT 5",
                "SELECT p.zpid, pf.full_address FROM properties p LEFT JOIN property_features pf ON p.zpid = pf.zpid LIMIT 5",
                "SELECT p.zpid, pf.full_address, c.violent_100k FROM properties p LEFT JOIN property_features pf ON p.zpid = pf.zpid LEFT JOIN crime c ON p.zpid = c.zpid LIMIT 5",
                "SELECT p.zpid, pf.full_address, c.violent_100k, g.drive_time FROM properties p LEFT JOIN property_features pf ON p.zpid = pf.zpid LEFT JOIN crime c ON p.zpid = c.zpid LEFT JOIN grocery g ON p.zpid = g.zpid LIMIT 5",
                "SELECT p.zpid, pf.full_address, c.violent_100k, g.drive_time, l.latitude FROM properties p LEFT JOIN property_features pf ON p.zpid = pf.zpid LEFT JOIN crime c ON p.zpid = c.zpid LEFT JOIN grocery g ON p.zpid = g.zpid LEFT JOIN location l ON p.zpid = l.zpid LIMIT 5"
            ]
            
            for i, test_query in enumerate(test_queries):
                try:
                    result = conn.execute(test_query).fetchall()
                    logger.info(f"✓ Test query {i+1} succeeded: {len(result)} rows")
                except Exception as test_error:
                    logger.error(f"✗ Test query {i+1} failed: {test_error}")
                    logger.error(f"✗ Failed query: {test_query}")
                    break
            
        except Exception as debug_error:
            logger.error(f"Diagnostic testing failed: {debug_error}")
        
        raise e  # Re-raise the original exception
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

def init_for_review_db(_conn):
    """Ensures the for_review table exists."""
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS for_review (
            zpid INTEGER PRIMARY KEY,
            zillow_url TEXT NOT NULL,
            added_at TEXT DEFAULT (datetime('now'))
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
        
        # Tables are already initialized at startup, just run keepalive queries
        
        # Get a count of ratings as a meaningful keepalive query
        count_result = conn.execute("SELECT COUNT(*) FROM rating").fetchone()
        rating_count = count_result[0] if count_result else 0
        
        conn.close()
        logger.info(f"Database keepalive successful at {current_time}. Rating count: {rating_count}")
        
    except Exception as e:
        logger.error(f"Database keepalive failed: {e}")

def initialize_databases():
    """Initialize all database tables once at startup."""
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
    except Exception as e:
        logger.error(f"Failed to initialize cloud database tables: {e}")

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
        logger.info("=== INDEX ROUTE DEBUG START ===")
        
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
        
        # Get minimum score threshold filter
        min_score_threshold = request.args.get('min_score_threshold', '0')
        try:
            min_score_threshold = float(min_score_threshold)
        except ValueError:
            min_score_threshold = 0.0
        
        # Get aggressiveness parameter
        aggressiveness = request.args.get('optimizer_aggressiveness', '0.4')
        try:
            aggressiveness = float(aggressiveness)
            # Clamp to valid range
            aggressiveness = max(0.1, min(0.9, aggressiveness))
        except ValueError:
            aggressiveness = 0.4
        
        # Password check
        password = request.args.get('password', '')
        password_correct = password == CORRECT_PASSWORD
        
        logger.info(f"Database path: {DB_PATH}")
        logger.info(f"Database exists: {os.path.exists(DB_PATH)}")
        
        # Debug: Check database tables and structure
        try:
            conn = duckdb.connect(DB_PATH, read_only=True)
            logger.info("Successfully connected to DuckDB")
            
            # Check available tables
            tables_result = conn.execute("SHOW TABLES").fetchall()
            available_tables = [table[0] for table in tables_result]
            logger.info(f"Available tables: {available_tables}")
            
            # Check if location table exists and has data
            if 'location' in available_tables:
                location_count = conn.execute("SELECT COUNT(*) FROM location").fetchone()[0]
                logger.info(f"Location table has {location_count} records")
                
                # Check a sample of location data
                sample_location = conn.execute("SELECT zpid, latitude, longitude FROM location LIMIT 5").fetchall()
                logger.info(f"Sample location data: {sample_location}")
            else:
                logger.warning("Location table does not exist!")
            
            # Check other key tables
            for table in ['properties', 'property_features', 'crime', 'grocery']:
                if table in available_tables:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    logger.info(f"Table {table} has {count} records")
                else:
                    logger.warning(f"Table {table} does not exist!")
            
            conn.close()
        except Exception as db_debug_error:
            logger.error(f"Database debug error: {db_debug_error}")
        
        # Generate and execute query
        logger.info("Generating scoring SQL...")
        norm_weights = normalize_weights(weights)
        scoring_sql = generate_scoring_sql(norm_weights, params, financing_filter)
        
        # Log the first 500 characters of the SQL for debugging
        logger.info(f"Generated SQL (first 500 chars): {scoring_sql[:500]}...")
        
        logger.info("Executing scoring query...")
        results_df = get_scored_properties(scoring_sql)
        logger.info(f"Query executed successfully, got {len(results_df)} results")
        
        # Load ratings and notes
        ratings_dict = {}
        notes_dict = {}
        if password_correct:
            try:
                ratings_conn = get_ratings_db_connection()
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
        
        # Apply minimum score threshold filter (only for blank ratings)
        if min_score_threshold > 0:
            # Keep all properties with ratings, but filter blank ratings by score
            mask = (results_df['rating'] != '') | (results_df['total_score'] >= min_score_threshold)
            results_df = results_df[mask]
        
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
        
        # Calculate score range for slider
        if len(results_df) > 0:
            score_min = float(results_df['total_score'].min())
            score_max = float(results_df['total_score'].max())
        else:
            score_min, score_max = 0.0, 100.0
        
        return render_template('index.html', 
                             properties=properties,
                             password_correct=password_correct,
                             rating_filter=rating_filter,
                             financing_filter=financing_filter,
                             min_score_threshold=min_score_threshold,
                             score_min=score_min,
                             score_max=score_max,
                             current_weights=weights,
                             aggressiveness=aggressiveness,
                             request=request)
                             
    except Exception as e:
        logger.error(f"=== ERROR IN INDEX ROUTE ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error args: {e.args}")
        
        # Log full traceback
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Enhanced error information for template
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'error_args': str(e.args),
            'traceback': traceback.format_exc()
        }
        
        return render_template('error.html', error=str(e), error_info=error_info)

@app.route('/add_property', methods=['POST'])
def add_property():
    """Add a new property URL for review"""
    password = request.form.get('password', '')
    if password != CORRECT_PASSWORD:
        return jsonify({'success': False, 'message': '❌ Invalid password'})
    
    zillow_url = request.form.get('zillow_url', '').strip()
    zpid = request.form.get('zpid', '').strip()
    
    # Validate inputs
    if not zillow_url or not zpid:
        return jsonify({'success': False, 'message': '❌ Missing URL or property ID'})
    
    # Validate ZPID is numeric
    try:
        zpid_int = int(zpid)
    except ValueError:
        return jsonify({'success': False, 'message': '❌ Invalid property ID format'})
    
    try:
        # Connect to both databases
        ratings_conn = get_ratings_db_connection()
        
        # Check if ZPID already exists in properties table (local DuckDB)
        local_conn = duckdb.connect(DB_PATH, read_only=True)
        existing_property = local_conn.execute(
            "SELECT p.zpid, c.violent_100k, c.property_100k, g.drive_time FROM properties p LEFT JOIN crime c ON p.zpid = c.zpid LEFT JOIN grocery g ON p.zpid = g.zpid WHERE p.zpid = ?", 
            (zpid_int,)
        ).fetchone()
        local_conn.close()
        
        if existing_property:
            # Property already exists - return crime and grocery info using existing functions
            zpid_val, violent_crime, property_crime, drive_time = existing_property
            
            # Calculate average crime for emoji (same logic as main route)
            if violent_crime is not None and property_crime is not None:
                avg_crime_raw = (violent_crime * 2.0 + property_crime) / 3.0
            else:
                avg_crime_raw = None
            
            # Use existing crime emoji function - we need to calculate icon level
            # This is simplified - in a real scenario you'd want to get the full dataset ranges
            # For now, use a reasonable default mapping
            if avg_crime_raw is not None:
                # Simple mapping based on typical crime ranges (this could be improved)
                if avg_crime_raw <= 1000:
                    crime_icon_level = 4  # Low crime
                elif avg_crime_raw <= 2000:
                    crime_icon_level = 3
                elif avg_crime_raw <= 3000:
                    crime_icon_level = 2
                elif avg_crime_raw <= 4000:
                    crime_icon_level = 1
                else:
                    crime_icon_level = 0  # High crime
            else:
                crime_icon_level = None
                
            crime_emoji = get_crime_emoji(crime_icon_level)
            
            # Format drive time using existing color logic
            if drive_time is not None:
                drive_time_str = f"{drive_time}min"
            else:
                drive_time_str = "🚫 No data"
            
            message = f"✅ Property already exists! {crime_emoji} Crime | 🛒 {drive_time_str}"
            ratings_conn.close()
            return jsonify({'success': True, 'message': message})
        
        # Check if already in for_review table
        existing_review = ratings_conn.execute(
            "SELECT zpid FROM for_review WHERE zpid = ?", (zpid_int,)
        ).fetchone()
        
        if existing_review:
            ratings_conn.close()
            return jsonify({'success': True, 'message': '✅ Property already added for review'})
        
        # Add to for_review table
        ratings_conn.execute(
            "INSERT INTO for_review (zpid, zillow_url, added_at) VALUES (?, ?, datetime('now'))",
            (zpid_int, zillow_url)
        )
        ratings_conn.commit()
        ratings_conn.close()
        
        return jsonify({
            'success': True, 
            'message': '✅ Property added for review at next update!'
        })
        
    except Exception as e:
        logger.error(f"Error adding property: {e}")
        return jsonify({
            'success': False, 
            'message': f'❌ Error: {str(e)}'
        })

@app.route('/update_rating', methods=['POST'])
def update_rating():
    password = request.form.get('password', '')
    if password != CORRECT_PASSWORD:
        return jsonify({'success': False, 'error': 'Invalid password'})
    
    zpid = request.form.get('zpid')
    rating = request.form.get('rating')
    
    try:
        ratings_conn = get_ratings_db_connection()
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

@app.route('/map')
def map_view():
    """Map view of all properties with same filtering as main page"""
    try:
        # Get filter parameters (same as index route)
        rating_filter = request.args.getlist('rating_filter')
        if not rating_filter:
            rating_filter = ['yes', 'maybe', 'blank']  # Default filters
            
        # Get financing eligibility filter
        financing_filter = request.args.getlist('financing_filter')
        if not financing_filter:
            financing_filter = ['eligible']  # Default to only eligible properties
        
        # Get minimum score threshold filter
        min_score_threshold = request.args.get('min_score_threshold', '0')
        try:
            min_score_threshold = float(min_score_threshold)
        except ValueError:
            min_score_threshold = 0.0
        
        # Password check
        password = request.args.get('password', '')
        password_correct = password == CORRECT_PASSWORD
        
        # Get current weights for scoring (use defaults since map doesn't have weight controls)
        weights = DEFAULT_FEATURE_WEIGHTS.copy()
        params = DEFAULT_SCORING_PARAMETERS.copy()
        
        # Generate and execute scoring query (same as table view but filter for location data)
        norm_weights = normalize_weights(weights)
        scoring_sql = generate_scoring_sql(norm_weights, params, financing_filter)
        results_df = get_scored_properties(scoring_sql)
        
        # Filter to only include properties with location data
        results_df = results_df[
            (results_df['latitude'].notna()) & 
            (results_df['longitude'].notna())
        ]
        
        # Load ratings and notes if authenticated
        ratings_dict = {}
        notes_dict = {}
        if password_correct:
            try:
                ratings_conn = get_ratings_db_connection()
                ratings_dict = load_ratings_dict(ratings_conn)
                notes_dict = load_notes_dict(ratings_conn)
                ratings_conn.close()
            except Exception as e:
                logger.error(f"Could not load ratings and notes: {e}")
        
        # Add ratings and notes to results
        results_df['rating'] = results_df['zpid'].map(lambda x: ratings_dict.get(x, ''))
        results_df['note'] = results_df['zpid'].map(lambda x: notes_dict.get(x, ''))
        
        # Add city info for display
        results_df['city_state'] = results_df['full_address'].apply(extract_city_from_address)
        
        # Add drive time color coding
        results_df['drive_time_color'] = results_df['drive_time'].apply(get_drive_time_color)
        
        # Add school rating color coding
        results_df['school_rating_color'] = results_df['avg_school_rating'].apply(get_school_rating_color)
        
        # Add risk emoji
        results_df['risk_emoji'] = results_df['avg_risk_severity'].apply(get_risk_emoji)
        
        # Calculate crime icon levels and emojis (same logic as index)
        valid_crime_scores = results_df['avg_crime_severity_raw'].dropna()
        if len(valid_crime_scores) > 0:
            crime_min = valid_crime_scores.min()
            crime_max = valid_crime_scores.max()
            crime_range = crime_max - crime_min
            
            if crime_range > 0:
                results_df['crime_icon_level'] = results_df['avg_crime_severity_raw'].apply(
                    lambda x: None if pd.isna(x) else 4 - min(4, int((x - crime_min) / crime_range * 5))
                )
            else:
                results_df['crime_icon_level'] = results_df['avg_crime_severity_raw'].apply(
                    lambda x: None if pd.isna(x) else 2  # Middle level
                )
        else:
            results_df['crime_icon_level'] = None
            
        results_df['crime_emoji'] = results_df['crime_icon_level'].apply(get_crime_emoji)
        
        # Apply rating filter
        allowed_ratings = set(rating_filter)
        if 'blank' in allowed_ratings:
            allowed_ratings.add('')
        results_df = results_df[results_df['rating'].isin(allowed_ratings)]
        
        # Apply minimum score threshold filter (only for blank ratings)
        if min_score_threshold > 0:
            # Keep all properties with ratings, but filter blank ratings by score
            mask = (results_df['rating'] != '') | (results_df['total_score'] >= min_score_threshold)
            results_df = results_df[mask]
        
        # Convert to dict for template, handling NaN values properly
        properties = results_df.to_dict('records')
        
        # Convert pandas NaN to None for proper JSON serialization
        for prop in properties:
            for key, value in prop.items():
                if pd.isna(value):
                    prop[key] = None
        
        # Calculate score range for slider
        if len(results_df) > 0:
            score_min = float(results_df['total_score'].min())
            score_max = float(results_df['total_score'].max())
        else:
            score_min, score_max = 0.0, 100.0
        
        return render_template('map.html', 
                             properties=properties,
                             password_correct=password_correct,
                             rating_filter=rating_filter,
                             financing_filter=financing_filter,
                             min_score_threshold=min_score_threshold,
                             score_min=score_min,
                             score_max=score_max,
                             request=request)
                             
    except Exception as e:
        logger.error(f"Error in map route: {e}")
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

@app.route('/optimize_weights', methods=['POST'])
def optimize_weights():
    """Optimize feature weights based on user ratings using pairwise ranking loss"""
    try:
        # Password check
        password = request.form.get('password', '')
        if password != CORRECT_PASSWORD:
            logger.warning("Unauthorized weight optimization attempt")
            return jsonify({
                'success': False, 
                'error': 'Invalid password'
            }), 403
        
        # Get current parameters and weights
        params = DEFAULT_SCORING_PARAMETERS.copy()
        current_weights = DEFAULT_FEATURE_WEIGHTS.copy()
        
        # Override weights with any provided parameters (for baseline)
        for key in current_weights.keys():
            param_value = request.form.get(f"weight_{key}")
            if param_value:
                try:
                    current_weights[key] = float(param_value)
                except ValueError:
                    pass  # Keep default if invalid
        
        # Get financing filter from request
        financing_filter = request.form.getlist('financing_filter')
        if not financing_filter:
            financing_filter = ['eligible']  # Default to eligible properties
            
        # Get rating filter from request
        rating_filter = request.form.getlist('rating_filter')
        if not rating_filter:
            rating_filter = ['yes', 'maybe', 'blank']  # Default filters
        
        # Get minimum score threshold filter
        min_score_threshold = request.form.get('min_score_threshold', '0')
        try:
            min_score_threshold = float(min_score_threshold)
        except ValueError:
            min_score_threshold = 0.0
        
        # Get aggressiveness parameter
        aggressiveness = request.form.get('optimizer_aggressiveness', '0.4')
        try:
            aggressiveness = float(aggressiveness)
            # Clamp to valid range
            aggressiveness = max(0.1, min(0.9, aggressiveness))
        except ValueError:
            aggressiveness = 0.4
        
        logger.info(f"Starting weight optimization process with aggressiveness={aggressiveness}...")
        
        # Generate scoring SQL and get property data
        norm_weights = normalize_weights(current_weights)
        scoring_sql = generate_scoring_sql(norm_weights, params, financing_filter)
        properties_df = get_scored_properties(scoring_sql)
        
        if len(properties_df) == 0:
            return jsonify({
                'success': False,
                'error': 'No properties found for optimization'
            })
        
        # Load ratings
        try:
            ratings_conn = get_ratings_db_connection()
            ratings_dict = load_ratings_dict(ratings_conn)
            ratings_conn.close()
        except Exception as e:
            logger.error(f"Could not load ratings for optimization: {e}")
            return jsonify({
                'success': False,
                'error': f'Could not load ratings: {str(e)}'
            })
        
        # Filter out blank ratings for optimization
        filtered_ratings = {zpid: rating for zpid, rating in ratings_dict.items() 
                          if rating in ['yes', 'maybe', 'no']}
        
        if len(filtered_ratings) < 10:
            return jsonify({
                'success': False,
                'error': f'Insufficient ratings for optimization: {len(filtered_ratings)} < 10'
            })
        
        # Run optimization with user-provided aggressiveness
        optimizer = WeightOptimizer(aggressiveness=aggressiveness)
        
        optimized_weights, info = optimizer.optimize_weights(properties_df, filtered_ratings)
        
        # Create URL parameters for optimized weights (no scaling - it breaks the optimization!)
        weight_params = create_weight_url_params(optimized_weights)
        
        # Generate new property data with optimized weights for immediate display
        new_properties_df = None
        if info['success']:
            try:
                # Re-score properties with optimized weights
                new_norm_weights = normalize_weights(optimized_weights)
                new_scoring_sql = generate_scoring_sql(new_norm_weights, params, financing_filter)
                new_properties_df = get_scored_properties(new_scoring_sql)
                
                # Add same enrichment as in main route
                new_properties_df['rating'] = new_properties_df['zpid'].map(lambda x: filtered_ratings.get(x, ''))
                new_properties_df['city_state'] = new_properties_df['full_address'].apply(extract_city_from_address)
                new_properties_df['drive_time_color'] = new_properties_df['drive_time'].apply(get_drive_time_color)
                new_properties_df['school_rating_color'] = new_properties_df['avg_school_rating'].apply(get_school_rating_color)
                new_properties_df['risk_emoji'] = new_properties_df['avg_risk_severity'].apply(get_risk_emoji)
                
                # Crime emoji logic (simplified)
                valid_crime_scores = new_properties_df['avg_crime_severity_raw'].dropna()
                if len(valid_crime_scores) > 0:
                    crime_min = valid_crime_scores.min()
                    crime_max = valid_crime_scores.max()
                    crime_range = crime_max - crime_min
                    if crime_range > 0:
                        new_properties_df['crime_icon_level'] = new_properties_df['avg_crime_severity_raw'].apply(
                            lambda x: None if pd.isna(x) else 4 - min(4, int((x - crime_min) / crime_range * 5))
                        )
                    else:
                        new_properties_df['crime_icon_level'] = new_properties_df['avg_crime_severity_raw'].apply(
                            lambda x: None if pd.isna(x) else 2
                        )
                else:
                    new_properties_df['crime_icon_level'] = None
                    
                new_properties_df['crime_emoji'] = new_properties_df['crime_icon_level'].apply(get_crime_emoji)
                
                # Apply rating filter - this should only affect display, not optimization
                # The optimization should work on ALL properties, but we filter for display
                allowed_ratings = set(rating_filter)
                if 'blank' in allowed_ratings:
                    allowed_ratings.add('')
                new_properties_df = new_properties_df[new_properties_df['rating'].isin(allowed_ratings)]
                
                # Apply minimum score threshold filter (only for blank ratings)
                if min_score_threshold > 0:
                    # Keep all properties with ratings, but filter blank ratings by score
                    mask = (new_properties_df['rating'] != '') | (new_properties_df['total_score'] >= min_score_threshold)
                    new_properties_df = new_properties_df[mask]
                
                # Sort by total_score descending
                new_properties_df = new_properties_df.sort_values(by='total_score', ascending=False)
                
                # Convert to dict for JSON response
                properties_list = new_properties_df.to_dict('records')
                
                # Convert pandas NaN to None for JSON serialization
                for prop in properties_list:
                    for key, value in prop.items():
                        if pd.isna(value):
                            prop[key] = None
                
                # Calculate score range for slider update
                if len(new_properties_df) > 0:
                    score_min = float(new_properties_df['total_score'].min())
                    score_max = float(new_properties_df['total_score'].max())
                else:
                    score_min, score_max = 0.0, 100.0
                            
            except Exception as e:
                logger.error(f"Error generating new property data: {e}")
                properties_list = None
                score_min, score_max = 0.0, 100.0
        
        # Prepare response
        response_data = {
            'success': info['success'],
            'optimized_weights': optimized_weights,
            'weight_url_params': weight_params,
            'optimization_info': info,
            'properties': properties_list if info['success'] else None,
            'score_min': score_min,
            'score_max': score_max
        }
        
        if info['success']:
            logger.info(f"Weight optimization completed successfully. "
                       f"Processed {info['n_ratings']} ratings, "
                       f"{info['n_comparisons']} comparisons, "
                       f"{info['n_iterations']} iterations.")
        else:
            logger.warning(f"Weight optimization failed: {info['message']}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Weight optimization error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'optimized_weights': DEFAULT_FEATURE_WEIGHTS.copy(),
            'weight_url_params': create_weight_url_params(DEFAULT_FEATURE_WEIGHTS),
            'redirect_url': request.url_root
        }), 500

# Initialize databases and start the keepalive thread when the app starts
initialize_databases()
start_keepalive_thread()

if __name__ == '__main__':
    app.run(debug=True)
