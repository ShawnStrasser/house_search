import pandas as pd

from config import CRIME_EMOJIS, RISK_EMOJIS


def get_drive_time_color(drive_time_minutes):
    if pd.isna(drive_time_minutes) or drive_time_minutes is None:
        return "#999999"

    drive_time = float(drive_time_minutes)
    if drive_time < 5:
        return "#006400"
    if drive_time < 10:
        return "#228B22"
    if drive_time < 20:
        return "#000000"
    if drive_time < 40:
        return "#FF8C00"
    return "#DC143C"


def get_school_rating_color(school_rating):
    if pd.isna(school_rating) or school_rating is None:
        return "#999999"

    rating = float(school_rating)
    if rating >= 8.5:
        return "#006400"
    if rating >= 7.0:
        return "#228B22"
    if rating >= 5.5:
        return "#000000"
    if rating >= 4.0:
        return "#FF8C00"
    return "#DC143C"


def get_risk_emoji(risk_level):
    if pd.isna(risk_level) or risk_level is None:
        return "❓"

    risk_int = min(4, max(0, int(round(float(risk_level)))))
    return RISK_EMOJIS.get(risk_int, "❓")


def get_crime_emoji(crime_icon_level):
    if pd.isna(crime_icon_level) or crime_icon_level is None:
        return "❓"

    crime_level = min(4, max(0, int(round(float(crime_icon_level)))))
    return CRIME_EMOJIS.get(crime_level, "❓")


def extract_city_from_address(address):
    try:
        if address is None:
            return ""
        text = str(address).strip()
        if not text or text.lower() == "nan":
            return ""
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) >= 3:
            city = parts[1]
            state_token = parts[2].split()[0] if parts[2] else ""
            letters_only = "".join(char for char in state_token if char.isalpha())
            state_abbrev = (letters_only[:2] if letters_only else state_token[:2]).upper()
            return f"{city}, {state_abbrev}".strip(", ")
        if len(parts) >= 2:
            return parts[1]
        return ""
    except Exception:
        return ""


def add_rating_and_notes_columns(results_df, ratings_dict: dict, notes_dict: dict):
    results_df["rating"] = results_df["zpid"].map(lambda x: ratings_dict.get(x, ""))
    results_df["note"] = results_df["zpid"].map(lambda x: notes_dict.get(x, ""))
    return results_df


def add_display_columns(results_df):
    results_df["city_state"] = results_df["full_address"].apply(extract_city_from_address)
    results_df["drive_time_color"] = results_df["drive_time"].apply(get_drive_time_color)
    results_df["school_rating_color"] = results_df["avg_school_rating"].apply(get_school_rating_color)
    results_df["risk_emoji"] = results_df["avg_risk_severity"].apply(get_risk_emoji)
    return results_df


def add_crime_icon_levels(results_df):
    valid_crime_scores = results_df["avg_crime_severity_raw"].dropna()
    if len(valid_crime_scores) > 0:
        crime_min = valid_crime_scores.min()
        crime_max = valid_crime_scores.max()
        crime_range = crime_max - crime_min
        if crime_range > 0:
            results_df["crime_icon_level"] = results_df["avg_crime_severity_raw"].apply(
                lambda x: None if pd.isna(x) else 4 - min(4, int((x - crime_min) / crime_range * 5))
            )
        else:
            results_df["crime_icon_level"] = results_df["avg_crime_severity_raw"].apply(
                lambda x: None if pd.isna(x) else 2
            )
    else:
        results_df["crime_icon_level"] = None
    return results_df


def add_crime_emoji_column(results_df):
    results_df["crime_emoji"] = results_df["crime_icon_level"].apply(get_crime_emoji)
    return results_df


def dataframe_to_property_records(results_df, normalize_all_nans: bool = False):
    properties = results_df.to_dict("records")
    if normalize_all_nans:
        for prop in properties:
            for key, value in prop.items():
                if pd.isna(value):
                    prop[key] = None
        return properties

    for prop in properties:
        if pd.isna(prop.get("crime_icon_level")):
            prop["crime_icon_level"] = None
        if pd.isna(prop.get("avg_crime_severity_raw")):
            prop["avg_crime_severity_raw"] = None
        if pd.isna(prop.get("drive_time")):
            prop["drive_time"] = None
    return properties
