#!/usr/bin/env python3
"""
AI Ranker: Gemini-powered holistic property ranking.

Runs offline to produce a non-linear, interaction-aware ranking of properties,
stored in the local DuckDB database as `ai_rankings`.  The Flask app reads
these rankings and lets the user toggle between weighted-score order and
AI-recommendation order.

Usage:
    python ai_ranker.py                # rank with defaults
    python ai_ranker.py --batch-size 25 --passes 3  # override batch/passes
    python ai_ranker.py --dry-run      # preview without saving
"""

import os
import sys
import json
from pathlib import Path
import math
import random
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

import duckdb
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from house_paths import LOCAL_DB_PATH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = str(LOCAL_DB_PATH)
MODEL = "gemini-pro-latest"
BATCH_SIZE = 30          # sweet-spot for ranking accuracy
NUM_PASSES = 2           # tournament rounds
BOTTOM_THIRD_CUTOFF = True  # filter out bottom third by weighted score

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not set")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Function-calling schema for guaranteed structured output
# ---------------------------------------------------------------------------

# Define function declaration that forces Gemini to return structured JSON
rank_properties_func = FunctionDeclaration(
    name="rank_properties",
    description="Returns property ZPIDs ranked from best (most desirable) to worst (least desirable) for the buyer.",
    parameters={
        "type": "object",
        "properties": {
            "ranked_zpids": {
                "type": "array",
                "description": "Array of property ZPID integers, ordered from best (index 0) to worst (last index).",
                "items": {"type": "integer"}
            }
        },
        "required": ["ranked_zpids"]
    },
)

ranking_tool = Tool(function_declarations=[rank_properties_func])

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert real-estate analyst helping a buyer find their ideal rural/semi-rural home in the Pacific Northwest.

The buyer's core priorities (in rough order):

1. PRIVACY & quiet setting — high privacy level, low road exposure (5 = private road)
2. USABLE LAND — lot acreage only matters when land_usability is ≥ 3
3. QUALITY — kitchen, bathroom, interior, exterior, and house_style quality matter
3. LOW CRIME — safer areas strongly preferred
4. GOOD SCHOOLS — higher school ratings preferred
5. CLOSE TO GROCERY — shorter drive time to a decent grocery store
6. DEDICATED HOME OFFICE — a must-have for remote work, if there is a 4th bedroom that qualifies as a dedicated office
7. GOOD VALUE — moderate price relative to quality & size
8. FINANCING ELIGIBLE — properties that can't get financing are disqualified. If the property has unique features like off grid, boat access only, special water system etc that could impact financing then rank it lower.
9. WATER FEATURE - a USEABLE water feature is a bonus, but not a deal-breaker, but it has to be useable for fishing, swimming, wading or playing in the water etc.

KEY NON-LINEAR INTERACTIONS — do NOT treat features independently:
• Lot size and Land usability: 10 steep unusable acres < 2 flat usable acres.
• Lot size and Vegetation density: Large forested lot is good, but there also needs to be some useable land for a garden, etc.
• Privacy and Road exposure: These multiply. High-privacy + quiet-road = retreat. Low-privacy + busy-road = deal-breaker.
• View and Land usability: Great views often mean steep terrain — a real trade-off.
• Price and Overall quality: Value ratio matters.
• Drive time and School distance and Crime: These form a "location composite." All three bad = fundamentally bad location, even if the house itself is nice.
• Kitchen/Bath/Interior coherence: Consistent 3/3/3 quality is better than 5/1/3 (partial reno = more investment needed).
• Home size and Dedicated office: Office in a small house = sacrificed living space. Office in a larger house = genuine bonus.
• Waterfront and Flood risk: Waterfront is great but offset by high flood risk.
• Positive features and Negative features: They don't simply cancel. Negative features (structural, access issues) are more impactful than positives.

SCORING RUBRIC CONTEXT (so you understand the feature scales):
• Quality ratings (kitchen, bathroom, interior, exterior, house_style): 1-5 scale (1=poor, 5=exceptional)
• Privacy level: 1-5 (1=no privacy, 5=very secluded)
• Road exposure: 1-5 (1=highway frontage, 5=private road)
• View rating: 1-5 (1=no view, 5=spectacular panoramic)
• Land usability: 1-5 (1=steep/rocky, 5=flat/cleared)
• Waterfront quality: 1-5 (1=none, 5=prime lakefront)
• Vegetation density: 1-5 (1=barren, 5=lush/park-like)
• Negative features severity: 1-5 (1=severe issues, 5=no negatives)
• Positive features score: 1-5 (1=no special features, 5=exceptional amenities)
• Climate risks (flood, fire, wind, heat, air): 1-10 (1=minimal, 10=severe) — NULL means unknown
• School ratings: 1-10 per school level (higher = better) — NULL if not available
• Crime rates: violent_100k and property_100k (lower = safer)
"""

RANKING_INSTRUCTIONS = """\
Below is a set of {count} property listings with their extracted features.
Rank them from BEST (most desirable for the buyer) to WORST.

IMPORTANT:
- Consider ALL features holistically — do NOT just sum up individual scores.
- Pay special attention to the non-linear interactions described above.
- Think about value-for-money, livability trade-offs, and deal-breakers.
- Properties with financing_eligibility=False should be ranked near the bottom.

Call the `rank_properties` function with the ranked_zpids array containing ALL {count} ZPIDs ordered from best to worst.

PROPERTY LISTINGS:
{listings}
"""


# ---------------------------------------------------------------------------
# Feature summary builder
# ---------------------------------------------------------------------------

def build_property_summary(row: Dict[str, Any]) -> str:
    """Build a concise text summary of a property's features for the prompt."""
    parts = [f"ZPID: {row['zpid']}"]

    # Basic info
    if row.get("full_address"):
        parts.append(f"Address: {row['full_address']}")
    parts.append(f"Price: ${row.get('price', 'N/A'):,}" if isinstance(row.get('price'), (int, float)) else f"Price: {row.get('price', 'N/A')}")
    parts.append(f"Beds: {row.get('beds', '?')} | Baths: {row.get('baths', '?')}")
    if row.get("home_size_sqft"):
        parts.append(f"Home size: {row['home_size_sqft']:,} sqft")
    if row.get("lot_size_acres"):
        parts.append(f"Lot: {row['lot_size_acres']:.1f} acres")

    # Financing
    fin = row.get("financing_eligibility")
    if fin is not None:
        parts.append(f"Financing eligible: {fin}")
        if not fin and row.get("financing_eligibility_explanation"):
            parts.append(f"  Reason: {row['financing_eligibility_explanation']}")

    # Quality ratings
    quality_fields = [
        ("kitchen_quality", "Kitchen"),
        ("bathroom_quality", "Bathroom"),
        ("general_interior_quality", "Interior"),
        ("general_exterior_quality", "Exterior"),
        ("house_style", "Style"),
    ]
    quality_parts = []
    for field, label in quality_fields:
        val = row.get(field)
        if val is not None:
            quality_parts.append(f"{label}={val}")
    if quality_parts:
        parts.append(f"Quality: {', '.join(quality_parts)}")

    # Property characteristics
    char_fields = [
        ("privacy_level", "Privacy"),
        ("road_exposure", "Road exposure"),
        ("view_rating", "View"),
        ("view_type", "View type"),
        ("land_usability", "Land usability"),
        ("waterfront_quality", "Waterfront"),
        ("vegetation_density", "Vegetation"),
    ]
    char_parts = []
    for field, label in char_fields:
        val = row.get(field)
        if val is not None:
            char_parts.append(f"{label}={val}")
    if char_parts:
        parts.append(f"Characteristics: {', '.join(char_parts)}")

    # Dedicated office
    if row.get("dedicated_office") is not None:
        parts.append(f"Dedicated office: {row['dedicated_office']}")

    # Positive/negative features
    neg_sev = row.get("negative_features_severity")
    if neg_sev is not None:
        parts.append(f"Negative features severity: {neg_sev}/5")
    neg_list = row.get("negative_features_list")
    if neg_list and neg_list != "None identified":
        parts.append(f"  Negatives: {neg_list}")

    pos_score = row.get("positive_features_score")
    if pos_score is not None:
        parts.append(f"Positive features score: {pos_score}/5")
    pos_list = row.get("positive_features_list")
    if pos_list and pos_list != "Standard features only":
        parts.append(f"  Positives: {pos_list}")

    # Climate risks
    risk_fields = [
        ("flood_risk_severity", "Flood"),
        ("fire_risk_severity", "Fire"),
        ("wind_risk_severity", "Wind"),
        ("heat_risk_severity", "Heat"),
        ("air_risk_severity", "Air"),
    ]
    risk_parts = []
    for field, label in risk_fields:
        val = row.get(field)
        if val is not None:
            risk_parts.append(f"{label}={val}")
    if risk_parts:
        parts.append(f"Climate risks (1-10): {', '.join(risk_parts)}")

    # Schools
    school_fields = [
        ("elementary_school_rating", "elementary_school_distance_miles", "Elementary"),
        ("middle_school_rating", "middle_school_distance_miles", "Middle"),
        ("high_school_rating", "high_school_distance_miles", "High"),
    ]
    school_parts = []
    for rating_f, dist_f, label in school_fields:
        rating = row.get(rating_f)
        dist = row.get(dist_f)
        if rating is not None or dist is not None:
            s = f"{label}:"
            if rating is not None:
                s += f" rating={rating}"
            if dist is not None:
                s += f" dist={dist:.1f}mi"
            school_parts.append(s)
    if school_parts:
        parts.append(f"Schools: {', '.join(school_parts)}")

    # Crime
    violent = row.get("violent_100k")
    prop_crime = row.get("property_100k")
    if violent is not None or prop_crime is not None:
        crime_str = "Crime per 100k:"
        if violent is not None:
            crime_str += f" violent={violent:.0f}"
        if prop_crime is not None:
            crime_str += f" property={prop_crime:.0f}"
        parts.append(crime_str)

    # Drive time / grocery
    dt = row.get("drive_time")
    if dt is not None:
        gn = row.get("grocery_name", "")
        parts.append(f"Grocery: {gn} ({dt} min drive)" if gn else f"Grocery drive time: {dt} min")

    # General assessment
    assessment = row.get("general_assessment")
    if assessment:
        parts.append(f"Assessment: {assessment}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_properties(db_path: str, bottom_third_cutoff: bool = True) -> List[Dict[str, Any]]:
    """Load properties with features and optionally filter bottom third by weighted score."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        query = """
        SELECT
            p.zpid,
            p.price,
            p.status,
            p.source_url,
            pf.*,
            c.violent_100k,
            c.property_100k,
            g.drive_time,
            g.name as grocery_name
        FROM properties p
        LEFT JOIN property_features pf ON p.zpid = pf.zpid
        LEFT JOIN crime c ON p.zpid = c.zpid
        LEFT JOIN grocery g ON p.zpid = g.zpid
        WHERE pf.zpid IS NOT NULL
        """
        df = conn.execute(query).df()
        logger.info(f"Loaded {len(df)} properties with features")

        if len(df) == 0:
            return []

        # Filter out listings that are not actively for sale.
        EXCLUDED_STATUSES = {
            'Active w/contingency', 'Auction', 'Bumpablebuyer', 'Closed',
            'Contingent', 'House for rent', 'Off market', 'Pending',
            'Pending inspection', 'Pending short sale', 'Sold', 'Under contract'
        }
        if "status" in df.columns:
            status_clean = df["status"].fillna("").astype(str).str.strip()
            exclude_mask = status_clean.isin(EXCLUDED_STATUSES) | status_clean.str.startswith("Sold for ")
            excluded = exclude_mask.sum()
            if excluded > 0:
                df = df[~exclude_mask].copy()
                logger.info(f"Status filter: excluded {excluded} non-for-sale properties")

        if bottom_third_cutoff and len(df) > 3:
            df["_quick_score"] = 0.0

            if df["price"].notna().any():
                p_min, p_max = df["price"].min(), df["price"].max()
                if p_max > p_min:
                    df["_quick_score"] += (1 - (df["price"].fillna(p_max) - p_min) / (p_max - p_min)) * 20

            score_fields = {
                "kitchen_quality": 6, "bathroom_quality": 5,
                "general_interior_quality": 5, "privacy_level": 7,
                "road_exposure": 8, "land_usability": 4,
                "view_rating": 3, "positive_features_score": 2,
                "house_style": 3, "general_exterior_quality": 5,
            }
            for field, weight in score_fields.items():
                if field in df.columns:
                    df["_quick_score"] += df[field].fillna(3) * weight

            if "lot_size_acres" in df.columns:
                df["_quick_score"] += df["lot_size_acres"].fillna(0).clip(upper=8) / 8 * 8 * 5
            if "home_size_sqft" in df.columns:
                df["_quick_score"] += df["home_size_sqft"].fillna(0).clip(upper=3000) / 3000 * 7 * 5
            if "dedicated_office" in df.columns:
                df["_quick_score"] += df["dedicated_office"].fillna(False).astype(int) * 10 * 5

            cutoff = df["_quick_score"].quantile(1/3)
            before = len(df)
            df = df[df["_quick_score"] >= cutoff].copy()
            logger.info(f"Bottom-third filter: {before} → {len(df)} properties (cutoff score: {cutoff:.1f})")
            df.drop(columns=["_quick_score"], inplace=True)

        records = df.to_dict("records")

        import math as _math
        for rec in records:
            for k, v in rec.items():
                try:
                    if isinstance(v, float) and _math.isnan(v):
                        rec[k] = None
                except (TypeError, ValueError):
                    pass

        return records
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Gemini ranking (via function calling for guaranteed structured output)
# ---------------------------------------------------------------------------

def rank_batch(properties: List[Dict[str, Any]], batch_num: int, pass_num: int) -> List[int]:
    """
    Send a batch of properties to Gemini and get a ranked list of zpids.
    Uses function calling (same pattern as feature_extraction.py) to guarantee
    structured JSON output — no raw text parsing needed.
    """
    listings_text = "\n\n---\n\n".join(
        build_property_summary(p) for p in properties
    )

    prompt = RANKING_INSTRUCTIONS.format(
        count=len(properties),
        listings=listings_text,
    )

    zpids_in_batch = {p["zpid"] for p in properties}

    logger.info(f"  Pass {pass_num}, Batch {batch_num}: Sending {len(properties)} properties to Gemini...")

    model = genai.GenerativeModel(
        MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(
        prompt,
        tools=[ranking_tool],
        tool_config={'function_calling_config': 'ANY'},
        generation_config=genai.GenerationConfig(
            temperature=0,
        ),
    )

    # Extract structured response from function call (guaranteed format)
    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if not function_call or function_call.name != "rank_properties":
            raise ValueError(f"Unexpected function call: {function_call.name if function_call else 'None'}")

        # Extract the ranked_zpids array from the function call args
        ranked_zpids_raw = function_call.args.get("ranked_zpids")
        if ranked_zpids_raw is None:
            raise ValueError("Function call missing 'ranked_zpids' argument")

        # Convert protobuf repeated field to Python list of ints
        ranked_zpids = [int(z) for z in ranked_zpids_raw]

    except (IndexError, AttributeError) as e:
        # If function calling fails entirely, check if there's text we can parse
        logger.error(f"  Function call extraction failed: {e}")
        if hasattr(response, 'text'):
            logger.error(f"  Raw text response: {response.text[:500]}")
        raise RuntimeError(
            f"Gemini did not return a valid function call for batch {batch_num} pass {pass_num}. "
            f"Error: {e}"
        )

    # Validate: ensure all returned zpids are in the batch
    valid_ranked = [z for z in ranked_zpids if z in zpids_in_batch]
    missing = zpids_in_batch - set(valid_ranked)
    if missing:
        logger.warning(f"  {len(missing)} zpids missing from Gemini response, appending at end")
        valid_ranked.extend(sorted(missing))

    extra = set(ranked_zpids) - zpids_in_batch
    if extra:
        logger.warning(f"  {len(extra)} unexpected zpids in response (ignored): {extra}")

    logger.info(f"  Pass {pass_num}, Batch {batch_num}: Got ranking for {len(valid_ranked)} properties ✅")
    return valid_ranked


def create_batches(properties: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """Split properties into batches of the given size."""
    batches = []
    for i in range(0, len(properties), batch_size):
        batches.append(properties[i:i + batch_size])
    return batches


def assign_tier_ranks(batch_rankings: List[List[int]]) -> Dict[int, int]:
    """
    Convert per-batch rankings into global tier-based ranks.
    Returns dict mapping zpid → global rank (1-based, lower is better).
    """
    zpid_to_batch_rank = {}
    for ranked_zpids in batch_rankings:
        for rank_within_batch, zpid in enumerate(ranked_zpids, start=1):
            zpid_to_batch_rank[zpid] = rank_within_batch

    tiers: Dict[int, List[int]] = {}
    for zpid, rank in zpid_to_batch_rank.items():
        tiers.setdefault(rank, []).append(zpid)

    global_rank = 1
    zpid_to_global_rank = {}
    for tier_num in sorted(tiers.keys()):
        tier_zpids = tiers[tier_num]
        random.shuffle(tier_zpids)
        for zpid in tier_zpids:
            zpid_to_global_rank[zpid] = global_rank
            global_rank += 1

    return zpid_to_global_rank


def tournament_rank(properties: List[Dict[str, Any]],
                    batch_size: int = BATCH_SIZE,
                    num_passes: int = NUM_PASSES) -> Dict[int, int]:
    """
    Run a multi-pass tournament ranking.

    Pass 1: Random batches → rank → tier-based global ranks.
    Pass 2+: Re-batch using round-robin on previous ranking → rank → new tiers.
    Final rank = average of all passes, sorted.

    FAIL-FAST: Any Gemini API error stops execution immediately.
    """
    zpid_to_prop = {p["zpid"]: p for p in properties}
    all_zpids = list(zpid_to_prop.keys())
    n = len(all_zpids)

    logger.info(f"🏆 Starting tournament ranking: {n} properties, batch size {batch_size}, {num_passes} passes")

    zpid_rank_sums: Dict[int, float] = {zpid: 0.0 for zpid in all_zpids}

    for pass_num in range(1, num_passes + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Pass {pass_num}/{num_passes}")
        logger.info(f"{'='*60}")

        if pass_num == 1:
            shuffled = list(all_zpids)
            random.shuffle(shuffled)
            ordered_props = [zpid_to_prop[z] for z in shuffled]
        else:
            sorted_by_rank = sorted(all_zpids, key=lambda z: zpid_rank_sums[z])
            num_batches = math.ceil(n / batch_size)
            new_batch_order = [None] * n
            for i, zpid in enumerate(sorted_by_rank):
                batch_idx = i % num_batches
                slot = (i // num_batches)
                pos = batch_idx * batch_size + slot
                if pos < n:
                    new_batch_order[pos] = zpid
                else:
                    new_batch_order[new_batch_order.index(None)] = zpid
            new_batch_order = [z for z in new_batch_order if z is not None]
            ordered_props = [zpid_to_prop[z] for z in new_batch_order]

        batches = create_batches(ordered_props, batch_size)
        batch_rankings = []

        for batch_idx, batch in enumerate(batches, start=1):
            # FAIL FAST — any error stops everything
            ranked_zpids = rank_batch(batch, batch_idx, pass_num)
            batch_rankings.append(ranked_zpids)

        pass_ranks = assign_tier_ranks(batch_rankings)

        for zpid, rank in pass_ranks.items():
            zpid_rank_sums[zpid] += rank

        current_sorted = sorted(all_zpids, key=lambda z: zpid_rank_sums[z])
        logger.info(f"\n  Top 10 after pass {pass_num}:")
        for i, zpid in enumerate(current_sorted[:10], start=1):
            p = zpid_to_prop[zpid]
            avg = zpid_rank_sums[zpid] / pass_num
            city = p.get("city") or p.get("full_address", "?")
            logger.info(f"    #{i}: ZPID {zpid} ({city}) - avg rank: {avg:.1f}, price: ${p.get('price', 0):,}")

    final_sorted = sorted(all_zpids, key=lambda z: zpid_rank_sums[z])
    final_ranks = {zpid: rank for rank, zpid in enumerate(final_sorted, start=1)}

    return final_ranks


# ---------------------------------------------------------------------------
# Incremental ranking
# ---------------------------------------------------------------------------

def load_existing_rankings(db_path: str) -> Dict[int, int]:
    """Load existing AI rankings from the database. Returns {zpid: ai_rank}."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # Check if table exists
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "ai_rankings" not in tables:
            return {}
            
        rows = conn.execute("SELECT zpid, ai_rank FROM ai_rankings").fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        logger.warning(f"Failed to load existing rankings: {e}")
        return {}
    finally:
        conn.close()


def prune_and_compact_rankings(
    existing_rankings: Dict[int, int],
    valid_zpids: set[int],
) -> Dict[int, int]:
    """
    Drop rankings for listings that are no longer in the current filtered pool and
    resequence survivors to a dense 1..N range while preserving relative order.
    """
    surviving = [
        (rank, zpid)
        for zpid, rank in existing_rankings.items()
        if zpid in valid_zpids
    ]
    surviving.sort(key=lambda item: (item[0], item[1]))

    return {
        zpid: new_rank
        for new_rank, (_, zpid) in enumerate(surviving, start=1)
    }


def select_reference_properties(
    existing_rankings: Dict[int, int],
    all_properties: List[Dict[str, Any]],
    count: int = 30,
) -> List[Dict[str, Any]]:
    """
    Pick `count` properties evenly spaced across the existing ranking
    to serve as calibration points in a mini-tournament.
    """
    zpid_to_prop = {p["zpid"]: p for p in all_properties}
    ranked_zpids = sorted(existing_rankings.keys(), key=lambda z: existing_rankings[z])
    # Only include zpids that we have feature data for
    ranked_zpids = [z for z in ranked_zpids if z in zpid_to_prop]

    if len(ranked_zpids) <= count:
        return [zpid_to_prop[z] for z in ranked_zpids]

    # Evenly space across the ranking
    step = len(ranked_zpids) / count
    indices = [int(i * step) for i in range(count)]
    selected = [zpid_to_prop[ranked_zpids[i]] for i in indices]
    return selected


def map_to_global_ranks(
    mini_tournament_ranks: Dict[int, int],
    reference_global_ranks: Dict[int, int],
    total_existing: int,
) -> Dict[int, int]:
    """
    Convert mini-tournament ranks into global insertion positions.

    Uses reference properties (whose global ranks are known) to figure out
    where new properties should be inserted in the global ranking.
    """
    # Separate reference and new properties from mini-tournament results
    ref_zpids = set(reference_global_ranks.keys())
    new_zpids = [z for z in mini_tournament_ranks if z not in ref_zpids]

    if not new_zpids:
        return {}

    # Build mapping: mini_rank -> global_rank for reference properties
    ref_mini_to_global = []
    for zpid in ref_zpids:
        if zpid in mini_tournament_ranks:
            ref_mini_to_global.append(
                (mini_tournament_ranks[zpid], reference_global_ranks[zpid])
            )
    ref_mini_to_global.sort(key=lambda x: x[0])  # sort by mini rank

    if not ref_mini_to_global:
        # No reference overlap — just append at end
        return {z: total_existing + i + 1 for i, z in enumerate(new_zpids)}

    # For each new property, interpolate its global rank
    new_global_ranks = {}
    for zpid in new_zpids:
        mini_rank = mini_tournament_ranks[zpid]

        # Find surrounding reference points
        below = None  # reference with mini_rank <= this property's mini_rank
        above = None  # reference with mini_rank > this property's mini_rank
        for ref_mini, ref_global in ref_mini_to_global:
            if ref_mini <= mini_rank:
                below = (ref_mini, ref_global)
            else:
                above = (ref_mini, ref_global)
                break

        if below is None and above is not None:
            # New property ranked above all references — insert at top
            new_global_ranks[zpid] = max(1, above[1] - 1)
        elif above is None and below is not None:
            # New property ranked below all references — insert at bottom
            new_global_ranks[zpid] = below[1] + 1
        elif below is not None and above is not None:
            # Interpolate between the two reference points
            mini_range = above[0] - below[0]
            if mini_range == 0:
                new_global_ranks[zpid] = below[1]
            else:
                frac = (mini_rank - below[0]) / mini_range
                interp = below[1] + frac * (above[1] - below[1])
                new_global_ranks[zpid] = max(1, round(interp))
        else:
            # No reference points at all (shouldn't happen)
            new_global_ranks[zpid] = total_existing + 1

    return new_global_ranks


def merge_rankings(
    existing_rankings: Dict[int, int],
    new_global_ranks: Dict[int, int],
) -> Dict[int, int]:
    """
    Insert new properties into the existing ranking, shifting existing
    ranks down as needed to make room.
    """
    # Combine and sort all properties by their rank
    all_items = []
    for zpid, rank in existing_rankings.items():
        all_items.append((rank, 0, zpid))  # 0 = existing (sort first at same rank)
    for zpid, rank in new_global_ranks.items():
        all_items.append((rank, 1, zpid))  # 1 = new (sort after existing at same rank)

    all_items.sort()

    # Reassign sequential ranks
    merged = {}
    for new_rank, (_, _, zpid) in enumerate(all_items, start=1):
        merged[zpid] = new_rank

    return merged


def incremental_rank(
    new_properties: List[Dict[str, Any]],
    all_properties: List[Dict[str, Any]],
    existing_rankings: Dict[int, int],
    batch_size: int = BATCH_SIZE,
    num_passes: int = NUM_PASSES,
    num_references: int = 30,
) -> Dict[int, int]:
    """
    Rank only new properties by running a mini-tournament against
    a sample of already-ranked reference properties.

    Returns the complete merged ranking (existing + new).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"INCREMENTAL MODE: {len(new_properties)} new properties")
    logger.info(f"{'='*60}")

    # Select reference properties evenly spaced across existing ranking
    references = select_reference_properties(existing_rankings, all_properties, num_references)
    ref_zpids = {p["zpid"] for p in references}
    ref_global_ranks = {z: existing_rankings[z] for z in ref_zpids if z in existing_rankings}

    logger.info(f"Selected {len(references)} reference properties as calibration points")
    logger.info(f"Reference rank range: {min(ref_global_ranks.values())} - {max(ref_global_ranks.values())}")

    # Combine into one pool for the mini-tournament
    pool = new_properties + references
    logger.info(f"Mini-tournament pool: {len(pool)} properties ({len(new_properties)} new + {len(references)} reference)")

    # Run the same tournament on the smaller pool
    mini_ranks = tournament_rank(pool, batch_size=batch_size, num_passes=num_passes)

    # Map mini-tournament results to global ranks
    new_global_ranks = map_to_global_ranks(
        mini_ranks, ref_global_ranks, total_existing=len(existing_rankings)
    )

    # Log where new properties landed
    for zpid in sorted(new_global_ranks, key=lambda z: new_global_ranks[z]):
        prop = next((p for p in new_properties if p["zpid"] == zpid), {})
        city = prop.get("city") or prop.get("full_address", "?")
        logger.info(
            f"  NEW: ZPID {zpid} ({city}) -> global rank ~{new_global_ranks[zpid]}"
        )

    # Merge into existing rankings
    merged = merge_rankings(existing_rankings, new_global_ranks)
    logger.info(f"Merged ranking: {len(merged)} total properties")

    return merged


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def save_rankings(db_path: str, rankings: Dict[int, int]) -> None:
    """Save AI rankings to the database (full replace)."""
    conn = duckdb.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_rankings (
                zpid INTEGER PRIMARY KEY,
                ai_rank INTEGER NOT NULL,
                ranked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("DELETE FROM ai_rankings")
        for zpid, rank in rankings.items():
            conn.execute(
                "INSERT INTO ai_rankings (zpid, ai_rank, ranked_at) VALUES (?, ?, ?)",
                [zpid, rank, datetime.now().isoformat()]
            )
        conn.commit()
        logger.info(f"Saved {len(rankings)} AI rankings to database")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI-powered property ranking using Gemini")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Properties per batch (default: {BATCH_SIZE})")
    parser.add_argument("--passes", type=int, default=NUM_PASSES,
                        help=f"Tournament passes (default: {NUM_PASSES})")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable bottom-third filtering")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help=f"Database path (default: {DB_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without saving to database")
    parser.add_argument("--full", action="store_true",
                        help="Force full re-rank (ignore existing rankings)")
    args = parser.parse_args()

    logger.info("AI Ranker - Gemini-powered property ranking")
    logger.info(f"   Model: {MODEL}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Passes: {args.passes}")
    logger.info(f"   Bottom-third filter: {not args.no_filter}")
    logger.info(f"   Database: {args.db}")

    properties = load_properties(args.db, bottom_third_cutoff=not args.no_filter)
    if not properties:
        logger.error("No properties found. Exiting.")
        sys.exit(1)

    logger.info(f"{len(properties)} properties loaded")

    # Check for existing rankings to decide mode
    existing_rankings = {} if args.full else load_existing_rankings(args.db)
    all_zpids = {p["zpid"] for p in properties}
    if existing_rankings:
        raw_existing_count = len(existing_rankings)
        existing_rankings = prune_and_compact_rankings(existing_rankings, all_zpids)
        removed_count = raw_existing_count - len(existing_rankings)
        if removed_count > 0:
            logger.info(
                f"Pruned {removed_count} stale rankings for filtered-out/inactive listings"
            )
        if existing_rankings:
            logger.info(
                f"Compacted existing rankings to sequential range 1-{len(existing_rankings)}"
            )

    ranked_zpids = set(existing_rankings.keys())
    new_zpids = all_zpids - ranked_zpids

    if existing_rankings and new_zpids and not args.full:
        # --- INCREMENTAL MODE ---
        logger.info(f"INCREMENTAL MODE: {len(new_zpids)} new, {len(ranked_zpids)} existing")
        new_properties = [p for p in properties if p["zpid"] in new_zpids]

        rankings = incremental_rank(
            new_properties=new_properties,
            all_properties=properties,
            existing_rankings=existing_rankings,
            batch_size=args.batch_size,
            num_passes=args.passes,
        )
    else:
        # --- FULL TOURNAMENT MODE / COMPACT-ONLY MODE ---
        if args.full:
            logger.info("FULL MODE (forced via --full flag)")
        elif not existing_rankings:
            logger.info("FULL MODE (no existing rankings)")
        elif not new_zpids:
            logger.info("COMPACT-ONLY MODE (all current listings already ranked)")
            rankings = existing_rankings
        else:
            logger.info("FULL MODE (fallback)")

        if args.full or not existing_rankings:
            rankings = tournament_rank(
                properties,
                batch_size=args.batch_size,
                num_passes=args.passes,
            )

    zpid_to_prop = {p["zpid"]: p for p in properties}
    sorted_zpids = sorted(rankings.keys(), key=lambda z: rankings[z])

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RANKINGS (top 20 of {len(rankings)})")
    logger.info(f"{'='*60}")
    for zpid in sorted_zpids[:20]:
        p = zpid_to_prop.get(zpid, {})
        rank = rankings[zpid]
        city = p.get("city") or p.get("full_address", "?")
        price = p.get("price", 0)
        privacy = p.get("privacy_level", "?")
        land = p.get("land_usability", "?")
        lot = p.get("lot_size_acres", "?")
        logger.info(
            f"  #{rank:>3}: ZPID {zpid} | {city} | ${price:,} | "
            f"privacy={privacy} land={land} lot={lot}"
        )

    if not args.dry_run:
        save_rankings(args.db, rankings)
        logger.info("Rankings saved. Toggle 'AI Recommendations' in the app to use them.")
    else:
        logger.info("Dry run - rankings NOT saved to database.")


if __name__ == "__main__":
    main()
