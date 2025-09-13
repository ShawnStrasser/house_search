#!/usr/bin/env python3
"""
Feature Extraction Module for Real Estate Properties using Google Gemini
This module handles the AI-powered feature extraction logic, including:
- Feature schema definitions
- Gemini API configuration and tool setup
- Rubric instructions for consistent feature extraction
- Database interaction functions for saving features
"""

import duckdb
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# --- FEATURE SCHEMA AND RUBRIC ---

FEATURE_PROPERTIES = {
    # Basic property information (extracted by AI from text/images)
    "full_address": {"type": "string"},
    "city": {"type": "string"},
    "county": {"type": "string"},
    "beds": {"type": "integer"},
    "baths": {"type": "number"},
    "home_size_sqft": {"type": "integer"},
    "lot_size_acres": {"type": "number"},
    
    # Financial and eligibility features
    "financing_eligibility": {"type": "boolean"},
    "financing_eligibility_explanation": {"type": "string"},
    
    # School information
    "elementary_school_distance_miles": {"type": "number"},
    "elementary_school_rating": {"type": "integer"},
    "middle_school_distance_miles": {"type": "number"},
    "middle_school_rating": {"type": "integer"},
    "high_school_distance_miles": {"type": "number"},
    "high_school_rating": {"type": "integer"},
    
    # Property characteristics
    "privacy_level": {"type": "integer"},
    "view_rating": {"type": "integer"},
    "view_type": {"type": "string"},
    "waterfront_quality": {"type": "integer"},
    "land_usability": {"type": "integer"},
    "vegetation_density": {"type": "integer"},
    "road_exposure": {"type": "integer"},
    "negative_features_severity": {"type": "integer"},
    "negative_features_list": {"type": "string"},
    "positive_features_score": {"type": "integer"},
    "positive_features_list": {"type": "string"},
    
    # Risk assessments
    "flood_risk_severity": {"type": "integer"},
    "fire_risk_severity": {"type": "integer"},
    "wind_risk_severity": {"type": "integer"},
    "heat_risk_severity": {"type": "integer"},
    "air_risk_severity": {"type": "integer"},
    
    # Property amenities and features
    "dedicated_office": {"type": "boolean"},
    
    # Quality ratings
    "kitchen_quality": {"type": "integer"},
    "bathroom_quality": {"type": "integer"},
    "general_interior_quality": {"type": "integer"},
    "general_exterior_quality": {"type": "integer"},
    "house_style": {"type": "integer"},
    
    # Overall assessment
    "general_assessment": {"type": "string"}
}

# Define the function declaration for Gemini's tool calling
house_features_func = FunctionDeclaration(
    name="extract_house_features",
    description="Extracts comprehensive house features from listing text and images based on a detailed rubric. For school distances and ratings, use null if not available. For risk severity ratings, use null if not available or not applicable.",
    parameters={
        "type": "object",
        "properties": FEATURE_PROPERTIES,
        "required": list(FEATURE_PROPERTIES.keys())
    },
)

# Create a Tool object for Gemini
feature_extraction_tool = Tool(function_declarations=[house_features_func])

RUBRIC_INSTRUCTIONS = """You are an expert real estate listing analyst. Extract structured features from the provided listing text and images with high accuracy.
Your task is to call the `extract_house_features` function with the data you extract.

CRITICAL RULES:
- Use BOTH text and images. Text is authoritative for numbers; images provide visual context.
- For numeric fields: return null if not explicitly stated (never guess or use 0).
- Be conservative in ratings - when uncertain, choose the middle value (3).
- Return ONLY the function call with the exact schema; no extra keys or commentary.

MISSING PHOTO REQUIREMENTS:
- For kitchen_quality: If no kitchen photos are visible, return 1 (lowest score)
- For bathroom_quality: If no bathroom photos are visible, return 1 (lowest score)  
- For view_rating: If no exterior/view photos are visible, return 1 (lowest score)
- For general_interior_quality: If no interior photos are visible, return 1 (lowest score)
- For general_exterior_quality: If no exterior photos are visible, return 1 (lowest score)
- For house_style: If no exterior photos are visible, return 1 (lowest score)
- DO NOT GUESS when required photos are missing - always return the lowest score (1)

BASIC PROPERTY INFORMATION:
- full_address: Complete street address including city, state, ZIP if available
- city: City name only (no state or ZIP)
- county: County name if explicitly mentioned (otherwise null)
- beds: Integer count of bedrooms
- baths: Total bathrooms (can be decimal, e.g., 2.5 for 2 full + 1 half)
- home_size_sqft: Interior living space in square feet (integer, not lot size)
- lot_size_acres: Lot/land size in acres (if given in sq ft, divide by 43,560)

FINANCING ELIGIBILITY:
- financing_eligibility: Default TRUE unless these red flags appear:
  * "cash only", "property only", "as-is sale"
  * "investor special", "non-habitable", "tear down"
  * "boat-only access", "no road access"
  * "off-grid" without proper power system
  * "no utilities", "no water/septic/sewer"
  * "liens", "back taxes", "title issues"
  * "unpermitted additions", "code violations"
  * "incomplete construction", "shell only"
  * "major structural damage", "foundation issues"
- financing_eligibility_explanation: Quote the specific problematic phrase if False; else "Standard financing eligible"

SCHOOL INFORMATION:
- Distance in miles (decimal ok) and rating (1-10 integer) for each school level
- ONLY extract if explicitly stated in listing text
- Return null for any missing values (do not guess or use 0)

PRIVACY LEVEL (1-5):
- 1: Busy road frontage, neighbors very close on multiple sides, no privacy
- 2: Close neighbors, minimal screening, urban/dense setting  
- 3: Standard suburban privacy, some trees/fencing, neighbors visible but not intrusive
- 4: Good privacy, natural screening, neighbors distant or blocked
- 5: Very secluded, gated/long driveway, heavily wooded, no visible neighbors in any of the photos

VIEW RATING (1-5):
- IMPORTANT: ONLY use images for view analysis - do not rely on text descriptions
- If no exterior/view photos are visible, return 1 (lowest score)
- 1: No view, blocked by structures, or undesirable view (parking lot, industrial)
- 2: Limited view, mostly neighboring houses
- 3: Pleasant yard/garden/trees view, typical residential
- 4: Nice nature views (woods, hills, fields)
- 5: Spectacular panoramic views (mountains, water, valley)

VIEW TYPE:
- Use ONE concise term: "forest", "mountain", "water", "valley", "pasture", "garden", "neighborhood", "none"
- IMPORTANT: ONLY use images for view type analysis - do not rely on text descriptions

WATERFRONT QUALITY (1-5):
- 1: No water features
- 2: Seasonal creek, drainage, distant water view
- 3: Small pond, year-round creek, non-recreational water
- 4: Nice stream, small lake access, recreational potential
- 5: Prime lakefront/riverfront with dock potential or beach access

LAND USABILITY (1-5):
- 1: Very steep, rocky, unusable for activities
- 2: Mostly sloped, limited flat areas
- 3: Mixed terrain, some usable flat areas
- 4: Mostly level with gentle slopes
- 5: Flat, cleared, ready for any use (lawn, pasture, gardens)

VEGETATION DENSITY (1-5):
- 1: Barren, dead vegetation, desert-like
- 2: Sparse vegetation, needs significant landscaping
- 3: Average suburban landscaping, some trees and lawn
- 4: Well-established trees, good landscaping
- 5: Lush mature trees, excellent gardens, park-like setting

ROAD EXPOSURE (1-5):
- 1: Highway/arterial frontage, heavy traffic, noise issues
- 2: Busy collector road, moderate traffic
- 3: Standard residential street, local traffic only
- 4: Quiet cul-de-sac or low-traffic street
- 5: Private road, gated, or very remote location

NEGATIVE FEATURES SEVERITY (1-5):
- 5: No unusual negative features identified.
- 4: Minor unusual issues (e.g., shared driveway, some deferred cosmetic maintenance).
- 3: Moderate concerns (e.g., power lines on property, cistern-only water, adjacent to commercial zone, private septic system).
- 2: Significant problems (e.g., known water quality issues, very steep/difficult access, major visible structural concerns, railroad tracks adjacent).
- 1: Severe issues that dramatically impact value/livability (e.g., boat-only access).

NEGATIVE FEATURES LIST:
- Provide a concise explanation of any unusual, property-specific negative features observed from text or images.
- Keep this as justification for the selected severity score.
- Avoid repeating items already scored elsewhere (e.g., climate risks, privacy, view, road exposure, quality ratings).
- If none, state "None identified".

POSITIVE FEATURES SCORE (1-5):
- 1: No special features beyond basics
- 2: A few nice additions (deck, basic landscaping)
- 3: Several desirable features (fireplace, nice deck, mature fruit trees)
- 4: Many premium features (pool, workshop, RV parking, solar, attached garage)
- 5: Exceptional amenities (guest house, pool/spa, large shop, premium finishes throughout)

POSITIVE FEATURES LIST:
- Provide a concise explanation of any unusual, property-specific positive features that stand out.
- Keep this as justification for the selected positive features score.
- Avoid repeating items already scored elsewhere (e.g., kitchen/bathroom/interior/exterior quality, waterfront, vegetation, privacy).
- If none, state "Standard features only".

CLIMATE RISK SEVERITIES (1-10):
- ONLY extract if explicitly stated with numeric rating in listing
- Look for phrases like "flood risk: 3/10" or "fire risk: minimal"
- Return null if not mentioned (do not guess based on location)
- Integer scale: 1=minimal, 5=moderate, 10=severe

DEDICATED OFFICE:
- TRUE if: office, den, study, library, or workspace explicitly mentioned
- Also TRUE if: 4+ bedrooms (spare room likely)
- FALSE if: only 1-3 bedrooms and no dedicated workspace mentioned

QUALITY RATINGS (1-5) - Be conservative, use 3 when uncertain:

KITCHEN QUALITY:
- CRITICAL: If no kitchen photos are visible, return 1 (lowest score)
- 1: Very dated, poor condition, needs full remodel
- 2: Dated but functional, cramped, limited counter/storage space, worn surfaces
- 3: Standard/average, clean, functional, some updates, adequate workspace for cooking
- 4: Recently updated, solid surface countertops, quality cabinets, ample counter space and storage
- 5: High-end remodel, stone/quartz counters, custom cabinetry, very spacious, premium storage solutions, walk-in pantry

BATHROOM QUALITY:
- CRITICAL: If no bathroom photos are visible, return 1 (lowest score)
- 1: Very dated, poor condition, needs renovation
- 2: Dated but clean and functional, small or cramped, basic fixtures
- 3: Average condition, some updates, clean, adequate size and functionality
- 4: Recently renovated, modern fixtures, solid surface countertops or quality tile, good space and layout
- 5: Luxury finishes, stone/quartz countertops, very spacious, high-end fixtures, spa-like amenities

GENERAL INTERIOR QUALITY:
- CRITICAL: If no interior photos are visible, return 1 (lowest score)
- 1: Needs major work, dated throughout, poor condition
- 2: Dated but livable, needs cosmetic updates
- 3: Average, mix of original and updated, clean
- 4: Well-maintained, consistent updates, move-in ready
- 5: Extensively remodeled, designer finishes, exceptional condition

GENERAL EXTERIOR QUALITY:
- CRITICAL: If no exterior photos are visible, return 1 (lowest score)
- 1: Deferred maintenance visible, needs paint/siding/roof work
- 2: Some wear visible but structurally sound
- 3: Average maintenance, typical wear for age
- 4: Well-maintained, recent paint/roof, good curb appeal
- 5: Excellent condition, premium materials, exceptional curb appeal

HOUSE STYLE (1-5):
- CRITICAL: If no exterior photos are visible, return 1 (lowest score)
- 1: Mobile home, or very basic box design.
- 2: Basic builder-grade, minimal architectural interest.
- 3: Standard suburban style, some architectural details
- 4: Attractive style with character (craftsman, colonial, ranch with charm)
- 5: Exceptional architecture (custom design, historical, unique in positive way)

GENERAL ASSESSMENT:
- Provide a 2-3 sentence overall assessment of the property
- Mention the strongest selling points and any significant concerns
- Be objective and balanced, like a professional inspector's summary
- Example: "Well-maintained ranch home with recent kitchen updates and large workshop. Property offers good privacy and usable land, though proximity to highway may be a concern for some buyers. Overall presents as move-in ready with strong potential for hobby farming or home business."
"""

# --- DATABASE FUNCTIONS ---

def create_property_features_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the property_features table with the updated schema."""
    
    # Build the CREATE TABLE SQL dynamically from FEATURE_PROPERTIES
    columns = ["zpid INTEGER PRIMARY KEY"]
    
    for field_name, field_info in FEATURE_PROPERTIES.items():
        field_type = field_info["type"]
        if field_type == "string":
            sql_type = "TEXT"
        elif field_type == "integer":
            sql_type = "INTEGER"
        elif field_type == "number":
            sql_type = "REAL"
        elif field_type == "boolean":
            sql_type = "BOOLEAN"
        else:
            sql_type = "TEXT"  # fallback
        
        columns.append(f"{field_name} {sql_type}")
    
    create_sql = f"CREATE TABLE IF NOT EXISTS property_features ({', '.join(columns)})"
    conn.execute(create_sql)
    conn.commit()

def save_features_to_db(conn: duckdb.DuckDBPyConnection, features: Dict[str, Any], verbose: bool = True) -> bool:
    """Save extracted features to the DuckDB database."""
    try:
        # Ensure zpid is present for the operation
        if 'zpid' not in features:
            print("❌ CRITICAL: Cannot save features: 'zpid' is missing.")
            print("🛑 Feature validation failure - stopping execution!")
            raise Exception("CRITICAL FAILURE: Features missing required 'zpid' field")
            
        columns = ', '.join(features.keys())
        placeholders = ', '.join(['?' for _ in features])
        
        # Delete existing record first, then insert fresh data
        conn.execute("DELETE FROM property_features WHERE zpid = ?", [features['zpid']])
        sql = f"INSERT INTO property_features ({columns}) VALUES ({placeholders})"
        conn.execute(sql, list(features.values()))
        conn.commit()
        
        if verbose:
            print(f"    -> 💾 Features for ZPID {features['zpid']} saved successfully.")
        return True
    except Exception as e:
        print(f"❌ CRITICAL DATABASE ERROR saving features for ZPID {features.get('zpid')}: {e}")
        print(f"🔍 DEBUG INFO:")
        print(f"    - Features keys: {list(features.keys())}")
        print(f"    - SQL: {sql}")
        print(f"    - Values: {list(features.values())}")
        # FAIL FAST - raise the error to stop execution
        raise e
