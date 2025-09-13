# Configuration settings for the Property Rating Flask App

# Default scoring parameters
DEFAULT_SCORING_PARAMETERS = {
    "home_size_tier1_sqft": 2000,
    "home_size_tier2_sqft": 3000,
    "lot_size_cap_acres": 8.0,
    "school_dist_full_points_miles": 1.0,
    "school_dist_zero_points_miles": 20.0,
    "max_violent_crime_100k": 720.0,
    "max_property_crime_100k": 5526.0,
    "drive_time_max_minutes": 60.0,  # For scaling drive time feature
}

# Default feature weights - positive weights mean higher values are better
# Negative weights mean higher values are worse
DEFAULT_FEATURE_WEIGHTS = {
    # Core property features
    "price": 20,                    # Higher price = worse (negative scoring in SQL)
    "home_size_sqft": 7,           # Handled with tier system
    "lot_size_acres": 8,           # Handled with cap system
    "beds": 3,
    "baths": 2,
    
    # NEW: Drive time to grocery (negative feature - higher is worse)
    "drive_time": 8,               # Relatively important, shown before bed/bath
    
    # Quality ratings (1-5 scales)
    "kitchen_quality": 6,
    "bathroom_quality": 5,
    "general_interior_quality": 5,
    "general_exterior_quality": 5,
    "house_style": 3,
    
    # Property characteristics
    "privacy_level": 7,
    "view_rating": 3,
    "land_usability": 4,
    "waterfront_quality": 3,
    "road_exposure": 8,            # Lower road exposure is better
    "vegetation_density": 2,
    "positive_features_score": 2,  # Renamed from unusual_features_rating
    
    # Boolean features
    "dedicated_office": 10,
    
    # Aggregate scores
    "avg_risk_severity": 5,        # Negative feature - higher risk is worse
    "avg_crime_severity": 6,       # Negative feature - higher crime is worse
    "avg_school_rating": 10,       # Higher school rating is better
    "avg_school_distance": 3,      # Negative feature - farther distance is worse
}

# Drive time emoji categories with funny/expressive icons
DRIVE_TIME_EMOJIS = {
    "under_5": "🚗💨",      # Fast car with speed lines - super quick
    "under_10": "🛒✨",     # Shopping cart with sparkles - convenient
    "under_20": "🚙",       # SUV - reasonable drive
    "under_40": "🐌🛣️",    # Snail on road - getting slow
    "over_40": "🦴🏚️"      # Bone and abandoned house - food desert vibes
}

# Risk emoji categories 
RISK_EMOJIS = {
    0: "😇",    # Angel - no risk
    1: "😊",    # Happy - very low risk
    2: "😐",    # Neutral - moderate risk  
    3: "😰",    # Anxious - high risk
    4: "💀"     # Skull - extreme risk
}

# Crime emoji categories (icon level 0-4, where 0 is worst crime)
CRIME_EMOJIS = {
    0: "😱",     # Very high crime
    1: "😬",     # High crime
    2: "😐",     # Moderate crime
    3: "😊",     # Low crime
    4: "😇"      # Very low crime
}

# App configuration
APP_CONFIG = {
    "secret_key_env": "SECRET_KEY",
    "ratings_db_url_env": "RATINGS_DB_URL",
    "correct_password_env": "APP_PASSWORD",
}

# Database configuration
DATABASE_CONFIG = {
    "local_db_path": "property_data.db",  # Development database
    "keepalive_interval_seconds": 10800,       # 3 hours
    "keepalive_retry_seconds": 1800,          # 30 minutes on error
}
