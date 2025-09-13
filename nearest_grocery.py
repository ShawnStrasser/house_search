import os
import re
import time
import duckdb
import googlemaps

DB_NAME = 'property_data.db'
REVIEW_THRESHOLD = 200
SLEEP_BETWEEN_CALLS = 0.0  # increase if you hit rate limits

CHAIN_GROCERS = [
    "Albertsons",
    "Safeway",
    "Fred Meyer",
    "QFC",
    "WinCo Foods",
    "Walmart",
    "Costco",
    "Target",
    "Sam's Club",
    "Trader Joe's",
    "Whole Foods Market",
    "Grocery Outlet",
    "New Seasons Market",
    "Market of Choice",
    "PCC Community Markets",
    "Haggen",
    "Yoke's Fresh Market",
    "Rosauers",
    "Bi-Mart",
    "Sprouts Farmers Market",
    "Natural Grocers",
    "Raley's",
    "Uwajimaya",
    "99 Ranch Market",
    "H Mart",
    "IGA",
]

def compile_chain_patterns(chain_list):
    pats = []
    for chain in chain_list:
        chain_clean = re.sub(r'[^\w\s]', '', chain.lower()).strip()
        if not chain_clean:
            continue
        words = chain_clean.split()
        if len(words) > 1:
            pattern = r'\b' + r'\b.*?\b'.join(re.escape(w) for w in words) + r'\b'
        else:
            pattern = r'\b' + re.escape(chain_clean) + r'\b'
        pats.append(re.compile(pattern))
    return pats

CHAIN_PATTERNS = compile_chain_patterns(CHAIN_GROCERS)

def is_chain_store(store_name: str) -> bool:
    if not store_name:
        return False
    store_name_clean = re.sub(r'[^\w\s]', '', store_name.lower())
    for p in CHAIN_PATTERNS:
        if p.search(store_name_clean):
            return True
    return False

# create grocery table only if it doesn't exist
conn = duckdb.connect(DB_NAME)
conn.execute("""
CREATE TABLE IF NOT EXISTS grocery (
    zpid INTEGER,
    name VARCHAR,
    drive_time INTEGER,
    address VARCHAR,
    rating DOUBLE,
    user_ratings_total INTEGER
)
""")
conn.commit()
conn.close()

# fetch zpids + addresses that are NOT already in the grocery table
conn = duckdb.connect(DB_NAME)
zpids_addresses = conn.execute("""
SELECT p.zpid, pf.full_address
FROM properties p
JOIN property_features pf ON p.zpid = pf.zpid
LEFT JOIN grocery g ON p.zpid = g.zpid
WHERE pf.full_address IS NOT NULL 
  AND g.zpid IS NULL
""").fetchall()
conn.close()

# Validate API key is present
MAPS_API_KEY = os.getenv('MAPS_API_KEY')
if not MAPS_API_KEY:
    raise RuntimeError("MAPS_API_KEY environment variable is required but not set")

gmaps = googlemaps.Client(key=MAPS_API_KEY)

def upsert_grocery_row(zpid, name, drive_time, address, rating, user_ratings_total):
    conn = duckdb.connect(DB_NAME)
    # duckdb upsert: delete then insert (portable & simple)
    conn.execute("DELETE FROM grocery WHERE zpid = ?", [zpid])
    conn.execute(
        "INSERT INTO grocery (zpid, name, drive_time, address, rating, user_ratings_total) VALUES (?, ?, ?, ?, ?, ?)",
        [zpid, name, drive_time, address, rating, user_ratings_total]
    )
    conn.commit()
    conn.close()

for zpid, address in zpids_addresses:
    print(f"Processing {zpid}: {address}")
    try:
        geocode_result = gmaps.geocode(address)
        if not geocode_result:
            print("  Geocode returned no results")
            continue
        loc = geocode_result[0]['geometry']['location']
        origin_str = f"{loc['lat']},{loc['lng']}"
    except Exception as e:
        print("  Geocode error:", e)
        continue

    try:
        places_result = gmaps.places_nearby(location=origin_str, rank_by='distance', type='grocery_or_supermarket')
    except Exception as e:
        print("  Places API error:", e)
        time.sleep(SLEEP_BETWEEN_CALLS)
        continue

    results = places_result.get('results', [])
    if not results:
        print("  No nearby grocery_or_supermarket results")
        continue

    qualifying_store = None
    for store in results:
        store_name = store.get('name', '')
        store_place_id = store.get('place_id')
        if not store_place_id:
            continue

        try:
            place_details = gmaps.place(place_id=store_place_id, fields=['formatted_address', 'rating', 'user_ratings_total'])
            place_info = place_details.get('result', {}) or {}
        except Exception as e:
            print("  Place details error:", e)
            continue

        user_ratings_total = int(place_info.get('user_ratings_total') or 0)
        rating = place_info.get('rating')

        # qualification rule: >= threshold OR matches known chain
        if user_ratings_total >= REVIEW_THRESHOLD or is_chain_store(store_name):
            qualifying_store = {
                'name': store_name,
                'place_id': store_place_id,
                'address': place_info.get('formatted_address') or store.get('vicinity') or '',
                'rating': rating,
                'user_ratings_total': user_ratings_total
            }
            print(f"  Found qualifying store: {store_name} ({user_ratings_total} reviews, Chain: {is_chain_store(store_name)})")
            break
        else:
            print(f"  Skipping {store_name}: {user_ratings_total} reviews, not a known chain")

        time.sleep(0.05)  # tiny pause between place detail calls

    if not qualifying_store:
        print("  No qualifying grocery stores found near this address")
        continue

    # compute driving time (ensure origin is lat,lng string)
    try:
        matrix_result = gmaps.distance_matrix(origins=[origin_str],
                                              destinations=[f"place_id:{qualifying_store['place_id']}"],
                                              mode="driving")
        element = matrix_result['rows'][0]['elements'][0]
        if element.get('status') == 'OK' and 'duration' in element:
            drive_time_minutes = round(element['duration']['value'] / 60)
            upsert_grocery_row(zpid,
                               qualifying_store['name'],
                               drive_time_minutes,
                               qualifying_store['address'],
                               qualifying_store['rating'],
                               qualifying_store['user_ratings_total'])
            print(f"  -> {qualifying_store['name']}: {drive_time_minutes} minutes")
            if qualifying_store['rating'] is not None:
                print(f"     Rating: {qualifying_store['rating']}/5.0 ({qualifying_store['user_ratings_total']} reviews)")
        else:
            print("  Distance matrix status:", element.get('status'))
    except Exception as e:
        print("  Distance matrix error:", e)

    time.sleep(SLEEP_BETWEEN_CALLS)
