#!/usr/bin/env python3
"""Helpers for filtering properties using remote SQLite Cloud ratings."""

from __future__ import annotations

import os
from typing import Optional, Set


def get_remote_no_rated_zpids(verbose: bool = False) -> Set[int]:
    """
    Return ZPIDs that users explicitly rated as 'no' in SQLite Cloud.

    If RATINGS_DB_URL or sqlitecloud is unavailable, returns an empty set so
    caller behavior remains non-breaking.
    """
    try:
        ratings_db_url = os.getenv("RATINGS_DB_URL", "")
        if not ratings_db_url:
            if verbose:
                print("⚠️ RATINGS_DB_URL not set, skipping remote 'no' listing filter.")
            return set()

        try:
            import sqlitecloud
        except ImportError:
            if verbose:
                print("⚠️ sqlitecloud package not available, skipping remote 'no' listing filter.")
            return set()

        conn = sqlitecloud.connect(ratings_db_url)
        try:
            rows = conn.execute(
                "SELECT zpid FROM rating WHERE rating = 'no' AND zpid IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

        excluded = {
            int(row[0])
            for row in rows
            if row and row[0] is not None
        }
        if verbose and excluded:
            print(f"🚫 Loaded {len(excluded)} user-excluded ('no') listings from remote DB.")
        return excluded

    except Exception as e:
        if verbose:
            print(f"⚠️ Could not load remote 'no' listing filters: {e}")
        return set()
