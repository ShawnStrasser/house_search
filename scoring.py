import duckdb
from config import DEFAULT_SCORING_PARAMETERS, DEFAULT_FEATURE_WEIGHTS

# --- Configuration ---
DB_PATH = r"C:\Users\shawn\House\property_data.db"
OUTPUT_TABLE = "property_scores"

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
    """Generate SQL query for property scoring - matches flask_app.py implementation"""
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


def main():
    """Main function to run the scoring process."""
    print("🏠 Starting property scoring process...")
    
    # Use defaults from config.py
    params = DEFAULT_SCORING_PARAMETERS.copy()
    weights = DEFAULT_FEATURE_WEIGHTS.copy()
    
    # Normalize weights so the positive ones sum to 1.0
    norm_weights = normalize_weights(weights)
    
    print("📊 Using normalized weights:")
    for feature, weight in norm_weights.items():
        print(f"  - {feature}: {weight:.3f}")
    
    # Generate the full SQL query - default to eligible properties only for scoring table
    scoring_sql = generate_scoring_sql(norm_weights, params, financing_filter=['eligible'])
    
    try:
        with duckdb.connect(DB_PATH) as conn:
            print(f"\n🔄 Calculating scores and creating table '{OUTPUT_TABLE}'...")
            
            # Execute the query and create/replace the output table
            conn.execute(f"CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS {scoring_sql}")
            
            print(f"✅ Successfully created/updated '{OUTPUT_TABLE}' table.")
            
            # Get the 50th percentile score
            percentile_50 = conn.execute(f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY total_score) as p50 FROM {OUTPUT_TABLE}").fetchone()[0]
            print(f"\n📊 50th percentile score: {percentile_50:.2f}")
            
            # Show the top 10 results
            print("\n🏆 Top 10 Properties:")
            
            # Get all columns from the output table to display them
            output_cols_df = conn.execute(f"PRAGMA table_info('{OUTPUT_TABLE}')").df()
            output_cols = output_cols_df['name'].tolist()
            
            # Select key columns for display (zpid, total_score, source_url, and a few key features)
            key_display_cols = ['total_score', 'source_url', 'full_address', 'price', 'beds', 'baths', 'home_size_sqft', 'drive_time']
            display_cols = [col for col in key_display_cols if col in output_cols]
            
            top_10 = conn.execute(f"SELECT {', '.join(display_cols)} FROM {OUTPUT_TABLE} LIMIT 10").df()
            print(top_10.to_string())

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please ensure 'properties' and 'property_features' tables exist and are populated.")

if __name__ == "__main__":
    main()
