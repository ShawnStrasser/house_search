from config import DEFAULT_FEATURE_WEIGHTS, DEFAULT_SCORING_PARAMETERS

from .cloud_db import load_user_data
from .presentation import (
    add_crime_emoji_column,
    add_crime_icon_levels,
    add_display_columns,
    add_rating_and_notes_columns,
    dataframe_to_property_records,
)
from .scoring import (
    add_score_ranks,
    apply_rating_filter,
    apply_status_filter,
    apply_threshold_filter,
    calculate_threshold_ranges,
    fetch_scored_properties,
    get_available_statuses,
    parse_aggressiveness,
    parse_common_filters,
    parse_non_negative_int_value,
    parse_ranking_mode,
    parse_weight_overrides,
)
from .settings import CORRECT_PASSWORD


def build_property_view_context(request_data, view_mode: str):
    params = DEFAULT_SCORING_PARAMETERS.copy()
    weights = parse_weight_overrides(request_data, DEFAULT_FEATURE_WEIGHTS)
    rating_filter, status_filter, financing_filter = parse_common_filters(request_data)
    ranking_mode = parse_ranking_mode(request_data)
    rank_threshold = max(1, parse_non_negative_int_value(request_data, "rank_threshold", 200))
    aggressiveness = parse_aggressiveness(request_data)
    password_value = request_data.get("password", "")
    password_correct = password_value == CORRECT_PASSWORD

    results_df = fetch_scored_properties(weights, params, financing_filter)
    if view_mode == "map":
        results_df = results_df[
            (results_df["latitude"].notna()) &
            (results_df["longitude"].notna())
        ]

    ratings_dict, notes_dict = load_user_data(password_correct)
    results_df = add_rating_and_notes_columns(results_df, ratings_dict, notes_dict)
    results_df = add_display_columns(results_df)
    results_df = add_crime_icon_levels(results_df)
    results_df = apply_rating_filter(results_df, rating_filter)
    results_df = apply_status_filter(results_df, status_filter)
    results_df = add_score_ranks(results_df)

    rank_min, rank_max, score_min, score_max, ai_rank_min, ai_rank_max = calculate_threshold_ranges(results_df)
    rank_threshold = max(rank_min, min(rank_max, rank_threshold))

    results_df = apply_threshold_filter(
        results_df,
        ranking_mode=ranking_mode,
        rank_threshold=rank_threshold,
    )

    if ranking_mode == "ai":
        results_df = results_df.sort_values(
            by=["ai_rank", "total_score"],
            ascending=[True, False],
            na_position="last",
        )
    else:
        results_df = results_df.sort_values(
            by=["score_rank", "total_score"],
            ascending=[True, False],
            na_position="last",
        )

    results_df = add_crime_emoji_column(results_df)
    properties = dataframe_to_property_records(results_df, normalize_all_nans=True)

    return {
        "properties": properties,
        "password_correct": password_correct,
        "password_value": password_value,
        "rating_filter": rating_filter,
        "status_filter": status_filter,
        "available_statuses": get_available_statuses(),
        "financing_filter": financing_filter,
        "ranking_mode": ranking_mode,
        "rank_threshold": rank_threshold,
        "rank_min": rank_min,
        "rank_max": rank_max,
        "score_min": score_min,
        "score_max": score_max,
        "ai_rank_min": ai_rank_min,
        "ai_rank_max": ai_rank_max,
        "current_weights": weights,
        "aggressiveness": aggressiveness,
        "view_mode": view_mode,
    }
