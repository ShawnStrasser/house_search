#!/usr/bin/env python3
"""
Populate the `crime` table in the local DuckDB database.

This script is the notebook-safe equivalent of `notebooks/crime.ipynb`, so it
can run unattended as part of the live scraper pipeline.
"""

from pathlib import Path
import re
import sys

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from house_paths import LOCAL_DB_PATH

DB_PATH = str(LOCAL_DB_PATH)
DATA_DIR = REPO_ROOT / "local_crime_data"
CITY_CRIME_PATH = DATA_DIR / "CIUS_Table_8_Offenses_Known_to_Law_Enforcement_by_State_by_City_2024.xlsx"
COUNTY_CRIME_PATH = DATA_DIR / "CIUS_Table_10_Offenses_Known_to_Law_Enforcement_by_State_by_Metropolitan_and_Nonmetropolitan_Counties_2024.xlsx"
WA_POPULATION_PATH = DATA_DIR / "co-est2024-pop-53.xlsx"
OR_POPULATION_PATH = DATA_DIR / "co-est2024-pop-41.xlsx"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace("\n", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    return df


def normalize_state(value):
    if not isinstance(value, str):
        return None
    token = re.sub(r"[^A-Za-z]", "", value).upper()
    if token in {"OR", "ORE", "OREGON"}:
        return "OREGON"
    if token in {"WA", "WASH", "WASHINGTON"}:
        return "WASHINGTON"
    return None


def normalize_city(value):
    if not isinstance(value, str):
        return None
    value = value.replace("_", " ").strip()
    return " ".join(word.capitalize() for word in re.split(r"\s+", value) if word)


def normalize_county(value):
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+County\s*$", "", value.strip(), flags=re.IGNORECASE)
    return value.strip() or None


def extract_city_state(address):
    if not isinstance(address, str):
        return pd.Series({"city": None, "state": None})

    parts = [part.strip() for part in address.split(",")]
    if len(parts) < 3:
        return pd.Series({"city": None, "state": None})

    for index, part in enumerate(parts):
        first_token = part.split()[0] if part.split() else ""
        state = normalize_state(first_token)
        if state and index > 0:
            return pd.Series({"city": normalize_city(parts[index - 1]), "state": state})

    return pd.Series({"city": None, "state": None})


def load_city_crime() -> pd.DataFrame:
    df = pd.read_excel(CITY_CRIME_PATH, header=3)
    df = df[df["City"].notna()].copy()
    df = normalize_columns(df)

    for column in ["population", "violent crime", "property crime"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    population = df["population"].replace({0: pd.NA})
    df["violent_100k"] = (df["violent crime"] / population) * 100_000
    df["property_100k"] = (df["property crime"] / population) * 100_000
    df["state"] = df["state"].apply(normalize_state)
    df["city"] = df["city"].apply(normalize_city)

    return df[["state", "city", "violent_100k", "property_100k"]].drop_duplicates()


def load_county_crime() -> pd.DataFrame:
    county_crime = pd.read_excel(COUNTY_CRIME_PATH, header=4)
    county_crime = county_crime[county_crime["County"].notna()].copy()
    county_crime = normalize_columns(county_crime)
    county_crime["state"] = county_crime["state"].apply(normalize_state)
    county_crime["county"] = county_crime["county"].apply(normalize_county)

    county_crime["violent crime"] = pd.to_numeric(county_crime["violent crime"], errors="coerce")
    county_crime["property crime"] = pd.to_numeric(county_crime["property crime"], errors="coerce")

    population = load_county_population()
    county_crime = county_crime.merge(population, on=["county", "state"], how="inner")
    county_crime["violent_100k"] = (county_crime["violent crime"] / county_crime["population"]) * 100_000
    county_crime["property_100k"] = (county_crime["property crime"] / county_crime["population"]) * 100_000

    return county_crime[["state", "county", "violent_100k", "property_100k"]].drop_duplicates()


def load_county_population() -> pd.DataFrame:
    def _load_population_sheet(path: Path) -> pd.DataFrame:
        frame = pd.read_excel(path, header=3)
        frame = frame.rename(
            columns={
                frame.columns[0]: "county_state",
                frame.columns[6]: "population",
            }
        ).dropna().iloc[1:]
        return frame[["county_state", "population"]]

    population = pd.concat(
        [_load_population_sheet(WA_POPULATION_PATH), _load_population_sheet(OR_POPULATION_PATH)],
        ignore_index=True,
    )
    population[["county", "state"]] = population["county_state"].str.extract(r"\.(.+) County, (.+)")
    population = population.drop(columns=["county_state"])
    population["county"] = population["county"].apply(normalize_county)
    population["state"] = population["state"].apply(normalize_state)
    population["population"] = pd.to_numeric(population["population"], errors="coerce")
    return population.dropna(subset=["county", "state", "population"])


def load_property_addresses() -> pd.DataFrame:
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = conn.execute(
            """
            SELECT p.zpid, pf.full_address, pf.county
            FROM properties p
            JOIN property_features pf ON p.zpid = pf.zpid
            """
        ).df()
    finally:
        conn.close()

    df[["city", "state"]] = df["full_address"].apply(extract_city_state)
    df["county"] = df["county"].apply(normalize_county)
    return df[["zpid", "city", "county", "state"]].drop_duplicates()


def build_crime_table() -> pd.DataFrame:
    property_addresses = load_property_addresses()
    city_crime = load_city_crime()
    county_crime = load_county_crime()

    city_matches = property_addresses.merge(city_crime, on=["city", "state"], how="left")
    county_matches = property_addresses[["zpid", "county", "state"]].merge(
        county_crime,
        on=["county", "state"],
        how="left",
    )

    final_df = city_matches.merge(
        county_matches,
        on="zpid",
        how="left",
        suffixes=("_city", "_county"),
    )
    final_df["violent_100k"] = final_df["violent_100k_city"].combine_first(final_df["violent_100k_county"])
    final_df["property_100k"] = final_df["property_100k_city"].combine_first(final_df["property_100k_county"])

    return final_df[["zpid", "violent_100k", "property_100k"]].drop_duplicates(subset=["zpid"])


def save_crime_table(crime_df: pd.DataFrame) -> None:
    conn = duckdb.connect(DB_PATH)
    try:
        conn.register("crime_df", crime_df)
        conn.execute("CREATE OR REPLACE TABLE crime AS SELECT * FROM crime_df")
        row_count = conn.execute("SELECT COUNT(*) FROM crime").fetchone()[0]
    finally:
        conn.close()

    print(f"✅ Wrote {row_count} rows to the crime table.")


def main() -> None:
    print("Loading property addresses and crime datasets...")
    crime_df = build_crime_table()
    print(f"Prepared crime data for {len(crime_df)} properties.")
    save_crime_table(crime_df)


if __name__ == "__main__":
    main()
