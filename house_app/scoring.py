import logging

import duckdb
import pandas as pd

from .settings import DB_PATH


logger = logging.getLogger(__name__)


def normalize_weights(weights: dict) -> dict:
    total_positive_weight = sum(weight for weight in weights.values() if weight > 0)
    if total_positive_weight == 0:
        return weights

    normalized = {}
    for key, weight in weights.items():
        normalized[key] = weight / total_positive_weight
    return normalized


def ensure_local_database_schema():
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS properties (
                zpid INTEGER PRIMARY KEY,
                source_url TEXT,
                price INTEGER,
                status TEXT
            )
            """
        )

        columns = {row[1] for row in conn.execute("PRAGMA table_info('properties')").fetchall()}
        if "status" not in columns:
            conn.execute("ALTER TABLE properties ADD COLUMN status TEXT")
            logger.info("Added status column to local properties table")
        else:
            logger.info("Local properties table already has status column. Columns: %s", sorted(columns))

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_rankings (
                zpid INTEGER PRIMARY KEY,
                ai_rank INTEGER NOT NULL,
                ranked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_available_statuses():
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT status
            FROM properties
            WHERE status IS NOT NULL AND TRIM(status) <> ''
            ORDER BY status
            """
        ).fetchall()
        return [row[0] for row in rows]
    except Exception as exc:
        logger.warning("Could not load available statuses: %s", exc)
        return []
    finally:
        conn.close()


def apply_status_filter(results_df, status_filter: list):
    if not status_filter:
        return results_df

    excluded_statuses = {
        "Active w/contingency", "Auction", "Bumpablebuyer", "Closed",
        "Contingent", "House for rent", "Off market", "Pending",
        "Pending inspection", "Pending short sale", "Sold", "Under contract",
    }

    normalized_status = results_df["status"].fillna("").astype(str).str.strip()
    is_excluded = normalized_status.isin(excluded_statuses) | normalized_status.str.startswith("Sold for ")
    mask = pd.Series(False, index=results_df.index)

    if "Active" in status_filter:
        mask = mask | (~is_excluded)
    if "Inactive" in status_filter:
        mask = mask | is_excluded

    return results_df[mask]


def apply_threshold_filter(results_df, ranking_mode: str, min_score_threshold: float, ai_rank_threshold: int):
    if ranking_mode == "ai":
        if ai_rank_threshold <= 0:
            return results_df
        mask = results_df["ai_rank"].notna() & (results_df["ai_rank"] <= ai_rank_threshold)
        return results_df[mask]

    if min_score_threshold <= 0:
        return results_df

    mask = (results_df["rating"] != "") | (results_df["total_score"] >= min_score_threshold)
    return results_df[mask]


def generate_scoring_sql(weights: dict, params: dict, financing_filter: list = None) -> str:
    sql = """
    WITH base_features AS (
        SELECT
            p.zpid,
            p.source_url,
            p.price,
            p.status,
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
            l.longitude,
            ar.ai_rank
        FROM properties p
        LEFT JOIN property_features pf ON p.zpid = pf.zpid
        LEFT JOIN crime c ON p.zpid = c.zpid
        LEFT JOIN grocery g ON p.zpid = g.zpid
        LEFT JOIN location l ON p.zpid = l.zpid
        LEFT JOIN ai_rankings ar ON p.zpid = ar.zpid
    ),

    price_stats AS (
        SELECT
            quantile_cont(price, {price_floor_quantile}) AS price_floor,
            quantile_cont(price, {price_ceiling_quantile}) AS price_ceiling
        FROM base_features
        WHERE price IS NOT NULL
    ),

    normalized_scores AS (
        SELECT
            zpid,

            (
                1 - (
                    (
                        LEAST(GREATEST(price, ps.price_floor), ps.price_ceiling) - ps.price_floor
                    ) / NULLIF(ps.price_ceiling - ps.price_floor, 0)
                )
            ) * 100 AS price_score,
            LEAST(beds, 5) / 5.0 * 100 AS beds_score,
            LEAST(baths, 4) / 4.0 * 100 AS baths_score,

            CASE
                WHEN home_size_sqft >= {home_size_tier2_sqft} THEN 100.0
                WHEN home_size_sqft > {home_size_tier1_sqft} THEN 80.0 + ((home_size_sqft - {home_size_tier1_sqft}) / ({home_size_tier2_sqft} - {home_size_tier1_sqft})) * 20.0
                ELSE (COALESCE(home_size_sqft, 0) / {home_size_tier1_sqft}) * 80.0
            END AS home_size_sqft_score,

            -- Large lots only help when the land is actually usable.
            LEAST(COALESCE(lot_size_acres, 0) / {lot_size_cap_acres}, 1.0) * 100 *
            CASE
                WHEN land_usability >= 5 THEN 1.0
                WHEN land_usability >= 4 THEN 0.85
                WHEN land_usability >= 3 THEN 0.60
                WHEN land_usability >= 2 THEN 0.20
                WHEN land_usability >= 1 THEN 0.0
                ELSE 0.35
            END AS lot_size_acres_score,

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
            CASE
                -- Moderate "rural normal" issues should stay near neutral, while
                -- genuine red flags like railroad adjacency or flood-zone problems
                -- should materially lower the score.
                WHEN negative_features_severity >= 5 THEN 100.0
                WHEN negative_features_severity >= 4 THEN 80.0
                WHEN negative_features_severity >= 3 THEN 55.0
                WHEN negative_features_severity >= 2 THEN 15.0
                WHEN negative_features_severity >= 1 THEN 0.0
                ELSE 55.0
            END AS negative_features_severity_score,

            CAST(dedicated_office AS INTEGER) * 100 AS dedicated_office_score,
            (1 - (COALESCE(drive_time, {drive_time_max_minutes}) / {drive_time_max_minutes})) * 100 AS drive_time_score,
            (1 - ( (
                COALESCE(flood_risk_severity, 5) +
                COALESCE(fire_risk_severity, 5) +
                COALESCE(wind_risk_severity, 5) +
                COALESCE(heat_risk_severity, 5) +
                COALESCE(air_risk_severity, 5)
            ) / 5.0 - 1) / 9.0) * 100 AS avg_risk_severity_score,
            ( (
                COALESCE(elementary_school_rating, 5) +
                COALESCE(middle_school_rating, 5) +
                COALESCE(high_school_rating, 5)
            ) / 3.0) * 10 AS avg_school_rating_score,

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

            (1 - (COALESCE(violent_100k, {max_violent_crime_100k} / 2.0) / {max_violent_crime_100k})) * 100 AS violent_crime_score,
            (1 - (COALESCE(property_100k, {max_property_crime_100k} / 2.0) / {max_property_crime_100k})) * 100 AS property_crime_score,

            ((1 - (COALESCE(violent_100k, {max_violent_crime_100k} / 2.0) / {max_violent_crime_100k})) * 100 +
             (1 - (COALESCE(property_100k, {max_property_crime_100k} / 2.0) / {max_property_crime_100k})) * 100) / 2.0 AS avg_crime_severity_score
        FROM base_features
        CROSS JOIN price_stats ps
    )
    """.format(**params)

    weighted_sum_parts = []
    valid_features = list(weights.keys())
    for feature in valid_features:
        weight = weights[feature]
        if weight < 0:
            weighted_sum_parts.append(f"((100 - {feature}_score) / 100.0 * {abs(weight)})")
        else:
            weighted_sum_parts.append(f"({feature}_score * {weight})")

    weighted_sum_sql = " + ".join(
        part for feature, part in zip(valid_features, weighted_sum_parts) if weights[feature] > 0
    )
    penalty_sum_sql = " + ".join(
        part for feature, part in zip(valid_features, weighted_sum_parts) if weights[feature] < 0
    )

    school_dist_zero = params["school_dist_zero_points_miles"]
    financing_conditions = []
    if financing_filter:
        if "eligible" in financing_filter:
            financing_conditions.append("bf.financing_eligibility = TRUE")
        if "not_eligible" in financing_filter:
            financing_conditions.append("bf.financing_eligibility = FALSE")
        if "unknown" in financing_filter:
            financing_conditions.append("bf.financing_eligibility IS NULL")

    financing_where = ""
    if financing_conditions:
        financing_where = "WHERE " + " OR ".join(financing_conditions)

    final_select = f"""
    SELECT
        ns.zpid,
        ({weighted_sum_sql}) - ({penalty_sum_sql if penalty_sum_sql else '0'}) AS total_score,
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
        ns.negative_features_severity_score,
        ns.dedicated_office_score,
        ns.drive_time_score,
        ns.avg_risk_severity_score,
        ns.avg_school_rating_score,
        ns.avg_school_distance_score,
        ns.avg_crime_severity_score,
        bf.source_url,
        bf.county,
        bf.full_address,
        bf.price,
        bf.status,
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
        bf.longitude,
        bf.ai_rank
    FROM normalized_scores ns
    JOIN base_features bf ON ns.zpid = bf.zpid
    {financing_where}
    ORDER BY total_score DESC
    """

    return sql + final_select


def get_scored_properties(sql_query):
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        logger.info("DuckDB version: %s", duckdb.__version__)
        result_df = conn.execute(sql_query).df()
        logger.info("Query success: %s rows", len(result_df))
        return result_df
    except Exception as exc:
        logger.error("=== DUCKDB QUERY EXECUTION FAILED ===")
        logger.error("Error type: %s", type(exc).__name__)
        logger.error("Error message: %s", str(exc))
        logger.error("Error args: %s", exc.args)
        logger.error("=== FULL QUERY THAT FAILED ===")
        logger.error(sql_query)
        logger.error("=== END QUERY ===")

        try:
            conn.execute("SELECT p.zpid, p.price FROM properties p LIMIT 5").fetchall()
            logger.error("Simple query works - issue is in complex expression")
            conn.execute(
                """
                WITH base_features AS (
                    SELECT p.zpid, p.price, pf.beds, pf.baths
                    FROM properties p
                    LEFT JOIN property_features pf ON p.zpid = pf.zpid
                    LIMIT 5
                ) SELECT * FROM base_features
                """
            ).fetchall()
            logger.error("Base CTE works - issue is in scoring expressions")
        except Exception as test_error:
            logger.error("Simplified test failed: %s", test_error)

        raise exc
    finally:
        conn.close()


def parse_weight_overrides(request_data, base_weights: dict) -> dict:
    weights = base_weights.copy()
    for key in weights.keys():
        param_value = request_data.get(f"weight_{key}")
        if param_value:
            try:
                weights[key] = float(param_value)
            except ValueError:
                pass
    return weights


def parse_ranking_mode(request_data) -> str:
    ranking_mode = request_data.get("ranking_mode", "score")
    if ranking_mode not in ("score", "ai"):
        return "score"
    return ranking_mode


def parse_float_value(request_data, key: str, default: float = 0.0) -> float:
    raw_value = request_data.get(key, str(default))
    try:
        return float(raw_value)
    except ValueError:
        return default


def parse_non_negative_int_value(request_data, key: str, default: int = 0) -> int:
    raw_value = request_data.get(key, str(default))
    try:
        value = int(float(raw_value))
    except ValueError:
        value = default
    return max(0, value)


def parse_aggressiveness(request_data, key: str = "optimizer_aggressiveness", default: float = 0.4) -> float:
    value = parse_float_value(request_data, key, default)
    return max(0.1, min(0.9, value))


def parse_common_filters(request_data):
    rating_filter = request_data.getlist("rating_filter")
    if not rating_filter:
        rating_filter = ["yes", "maybe", "blank"]

    status_filter = request_data.getlist("status_filter")

    financing_filter = request_data.getlist("financing_filter")
    if not financing_filter:
        financing_filter = ["eligible"]

    return rating_filter, status_filter, financing_filter


def fetch_scored_properties(weights: dict, params: dict, financing_filter: list):
    norm_weights = normalize_weights(weights)
    scoring_sql = generate_scoring_sql(norm_weights, params, financing_filter)
    return get_scored_properties(scoring_sql)


def apply_rating_filter(results_df, rating_filter: list):
    allowed_ratings = set(rating_filter)
    if "blank" in allowed_ratings:
        allowed_ratings.add("")
    return results_df[results_df["rating"].isin(allowed_ratings)]


def calculate_threshold_ranges(results_df):
    if len(results_df) > 0:
        score_min = 0.0
        score_max = float(results_df["total_score"].max())
        ai_ranks = results_df["ai_rank"].dropna()
        ai_rank_min = 1
        ai_rank_max = int(ai_ranks.max()) if len(ai_ranks) > 0 else 100
    else:
        score_min, score_max = 0.0, 100.0
        ai_rank_min, ai_rank_max = 1, 100
    return score_min, score_max, ai_rank_min, ai_rank_max
