import logging
import traceback
from datetime import datetime

import duckdb
from flask import Flask, jsonify, render_template, request

from config import DEFAULT_FEATURE_WEIGHTS, DEFAULT_SCORING_PARAMETERS
from house_app.cloud_db import (
    database_keepalive,
    get_ratings_db_connection,
    initialize_databases,
    load_ratings_dict,
    save_note,
    save_rating,
    start_keepalive_thread,
)
from house_app.presentation import (
    add_crime_emoji_column,
    add_crime_icon_levels,
    add_display_columns,
    dataframe_to_property_records,
    get_crime_emoji,
)
from house_app.scoring import (
    apply_rating_filter,
    apply_status_filter,
    ensure_local_database_schema,
    fetch_scored_properties,
    parse_aggressiveness,
    parse_common_filters,
    parse_float_value,
    parse_weight_overrides,
)
from house_app.settings import CORRECT_PASSWORD, DB_PATH, RATINGS_DB_URL, SECRET_KEY
from house_app.view_data import build_property_view_context
from weight_optimizer import WeightOptimizer, create_weight_url_params
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY

@app.route('/')
def index():
    try:
        context = build_property_view_context(request.args, "table")
        return render_template("index.html", **context)
                             
    except Exception as e:
        logger.exception("Error in index route")
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "error_args": str(e.args),
            "traceback": traceback.format_exc(),
        }

        return render_template("error.html", error=str(e), error_info=error_info)

@app.route('/add_property', methods=['POST'])
def add_property():
    password = request.form.get("password", "")
    if password != CORRECT_PASSWORD:
        return jsonify({"success": False, "message": "❌ Invalid password"})

    zillow_url = request.form.get("zillow_url", "").strip()
    zpid = request.form.get("zpid", "").strip()
    if not zillow_url or not zpid:
        return jsonify({"success": False, "message": "❌ Missing URL or property ID"})

    try:
        zpid_int = int(zpid)
    except ValueError:
        return jsonify({"success": False, "message": "❌ Invalid property ID format"})

    try:
        ratings_conn = get_ratings_db_connection()
        local_conn = duckdb.connect(DB_PATH, read_only=True)
        existing_property = local_conn.execute(
            "SELECT p.zpid, c.violent_100k, c.property_100k, g.drive_time FROM properties p LEFT JOIN crime c ON p.zpid = c.zpid LEFT JOIN grocery g ON p.zpid = g.zpid WHERE p.zpid = ?",
            (zpid_int,)
        ).fetchone()
        local_conn.close()

        if existing_property:
            _, violent_crime, property_crime, drive_time = existing_property

            if violent_crime is not None and property_crime is not None:
                avg_crime_raw = (violent_crime * 2.0 + property_crime) / 3.0
            else:
                avg_crime_raw = None

            if avg_crime_raw is not None:
                if avg_crime_raw <= 1000:
                    crime_icon_level = 4
                elif avg_crime_raw <= 2000:
                    crime_icon_level = 3
                elif avg_crime_raw <= 3000:
                    crime_icon_level = 2
                elif avg_crime_raw <= 4000:
                    crime_icon_level = 1
                else:
                    crime_icon_level = 0
            else:
                crime_icon_level = None

            crime_emoji = get_crime_emoji(crime_icon_level)
            drive_time_str = f"{drive_time}min" if drive_time is not None else "🚫 No data"
            current_rating = ratings_conn.execute(
                "SELECT rating FROM rating WHERE zpid = ?", (zpid_int,)
            ).fetchone()
            current_rating = current_rating[0] if current_rating else ""

            message = f"✅ Property already exists! {crime_emoji} Crime | 🛒 {drive_time_str}"
            ratings_conn.close()
            return jsonify({
                "success": True,
                "message": message,
                "property_exists": True,
                "current_rating": current_rating,
                "zpid": zpid_int,
            })

        existing_review = ratings_conn.execute(
            "SELECT zpid FROM for_review WHERE zpid = ?", (zpid_int,)
        ).fetchone()

        if existing_review:
            ratings_conn.close()
            return jsonify({"success": True, "message": "✅ Property already added for review"})

        ratings_conn.execute(
            "INSERT INTO for_review (zpid, zillow_url, added_at) VALUES (?, ?, datetime('now'))",
            (zpid_int, zillow_url)
        )
        ratings_conn.commit()
        ratings_conn.close()
        return jsonify({
            "success": True,
            "message": "✅ Property added for review at next update!",
        })

    except Exception as e:
        logger.error("Error adding property: %s", e)
        return jsonify({
            "success": False,
            "message": f"❌ Error: {str(e)}",
        })

@app.route('/update_rating', methods=['POST'])
def update_rating():
    password = request.form.get("password", "")
    if password != CORRECT_PASSWORD:
        return jsonify({"success": False, "error": "Invalid password"})

    zpid = request.form.get("zpid")
    rating = request.form.get("rating")

    try:
        ratings_conn = get_ratings_db_connection()
        save_rating(ratings_conn, zpid, rating)
        ratings_conn.close()

        return jsonify({"success": True})
    except Exception as e:
        logger.error("Error updating rating: %s", e)
        return jsonify({"success": False, "error": str(e)})

@app.route('/update_note', methods=['POST'])
def update_note():
    password = request.form.get("password", "")
    if password != CORRECT_PASSWORD:
        return jsonify({"success": False, "error": "Invalid password"})

    zpid = request.form.get("zpid")
    note = request.form.get("note", "")

    try:
        notes_conn = get_ratings_db_connection()
        save_note(notes_conn, zpid, note)
        notes_conn.close()

        return jsonify({"success": True})
    except Exception as e:
        logger.error("Error updating note: %s", e)
        return jsonify({"success": False, "error": str(e)})

@app.route('/settings')
def settings():
    try:
        current_weights = parse_weight_overrides(request.args, DEFAULT_FEATURE_WEIGHTS)
        return render_template(
            "settings.html",
            weights=current_weights,
            default_weights=DEFAULT_FEATURE_WEIGHTS,
            DEFAULT_SCORING_PARAMETERS=DEFAULT_SCORING_PARAMETERS,
        )
    except Exception as e:
        logger.error("Error in settings route: %s", e)
        return render_template("error.html", error=str(e))

@app.route('/map')
def map_view():
    try:
        context = build_property_view_context(request.args, "map")
        return render_template("map.html", **context)
    except Exception as e:
        logger.exception("Error in map route: %s", e)
        return render_template("error.html", error=str(e))


@app.route("/api/properties")
def properties_api():
    try:
        view_mode = request.args.get("view", "table")
        if view_mode not in {"table", "map"}:
            view_mode = "table"

        context = build_property_view_context(request.args, view_mode)
        results_template = "_table_results.html" if view_mode == "table" else "_map_results.html"
        results_html = render_template(results_template, **context)

        return jsonify({
            **context,
            "results_html": results_html,
        })
    except Exception as e:
        logger.exception("Error in properties API: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health')
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database_url_configured": bool(RATINGS_DB_URL),
        })
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }), 500

@app.route('/keepalive')
def manual_keepalive():
    try:
        database_keepalive()
        return jsonify({
            "success": True,
            "message": "Database keepalive executed successfully",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error("Manual keepalive failed: %s", e)
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        })

@app.route('/optimize_weights', methods=['POST'])
def optimize_weights():
    try:
        password = request.form.get("password", "")
        if password != CORRECT_PASSWORD:
            logger.warning("Unauthorized weight optimization attempt")
            return jsonify({
                "success": False,
                "error": "Invalid password",
            }), 403

        params = DEFAULT_SCORING_PARAMETERS.copy()
        current_weights = parse_weight_overrides(request.form, DEFAULT_FEATURE_WEIGHTS)
        rating_filter, status_filter, financing_filter = parse_common_filters(request.form)
        min_score_threshold = parse_float_value(request.form, "min_score_threshold", 0.0)
        aggressiveness = parse_aggressiveness(request.form)

        logger.info(f"Starting weight optimization process with aggressiveness={aggressiveness}...")

        properties_df = fetch_scored_properties(current_weights, params, financing_filter)
        if len(properties_df) == 0:
            return jsonify({
                "success": False,
                "error": "No properties found for optimization",
            })

        try:
            ratings_conn = get_ratings_db_connection()
            ratings_dict = load_ratings_dict(ratings_conn)
            ratings_conn.close()
        except Exception as e:
            logger.error("Could not load ratings for optimization: %s", e)
            return jsonify({
                "success": False,
                "error": f"Could not load ratings: {str(e)}",
            })

        properties_df["rating"] = properties_df["zpid"].map(lambda x: ratings_dict.get(x, ""))
        properties_df = apply_status_filter(properties_df, status_filter)

        if len(properties_df) == 0:
            return jsonify({
                "success": False,
                "error": "No properties found for the selected status filters",
            })

        filtered_zpids = set(properties_df["zpid"])
        filtered_ratings = {
            zpid: rating
            for zpid, rating in ratings_dict.items()
            if rating in ["yes", "maybe", "no"] and zpid in filtered_zpids
        }

        if len(filtered_ratings) < 10:
            return jsonify({
                "success": False,
                "error": f"Insufficient ratings for optimization: {len(filtered_ratings)} < 10",
            })

        optimizer = WeightOptimizer(aggressiveness=aggressiveness)
        optimized_weights, info = optimizer.optimize_weights(properties_df, filtered_ratings)
        weight_params = create_weight_url_params(optimized_weights)

        properties_list = None
        score_min, score_max = 0.0, 100.0

        if info["success"]:
            try:
                new_properties_df = fetch_scored_properties(optimized_weights, params, financing_filter)
                new_properties_df["rating"] = new_properties_df["zpid"].map(lambda x: filtered_ratings.get(x, ""))
                new_properties_df = apply_status_filter(new_properties_df, status_filter)
                new_properties_df = add_display_columns(new_properties_df)
                new_properties_df = add_crime_icon_levels(new_properties_df)
                new_properties_df = add_crime_emoji_column(new_properties_df)

                new_properties_df = apply_rating_filter(new_properties_df, rating_filter)
                if min_score_threshold > 0:
                    mask = (new_properties_df["rating"] != "") | (
                        new_properties_df["total_score"] >= min_score_threshold
                    )
                    new_properties_df = new_properties_df[mask]

                new_properties_df = new_properties_df.sort_values(by="total_score", ascending=False)
                properties_list = dataframe_to_property_records(new_properties_df, normalize_all_nans=True)
                if len(new_properties_df) > 0:
                    score_min = 0.0
                    score_max = float(new_properties_df["total_score"].max())
            except Exception as e:
                logger.exception("Error generating new property data: %s", e)

        response_data = {
            "success": info["success"],
            "optimized_weights": optimized_weights,
            "weight_url_params": weight_params,
            "optimization_info": info,
            "properties": properties_list if info["success"] else None,
            "score_min": score_min,
            "score_max": score_max,
        }

        if info["success"]:
            logger.info(
                "Weight optimization completed successfully. Processed %s ratings, %s comparisons, %s iterations.",
                info["n_ratings"],
                info["n_comparisons"],
                info["n_iterations"],
            )
        else:
            logger.warning("Weight optimization failed: %s", info["message"])

        return jsonify(response_data)

    except Exception as e:
        logger.exception("Weight optimization error: %s", e)
        return jsonify({
            "success": False,
            "error": str(e),
            "optimized_weights": DEFAULT_FEATURE_WEIGHTS.copy(),
            "weight_url_params": create_weight_url_params(DEFAULT_FEATURE_WEIGHTS),
            "redirect_url": request.url_root,
        }), 500

ensure_local_database_schema()
initialize_databases()
start_keepalive_thread()

if __name__ == "__main__":
    app.run(debug=True)
