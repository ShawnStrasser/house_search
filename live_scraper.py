#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "property_data.db"


def run_step(label: str, script_path: Path) -> None:
    print(f"\n=== Running {label} ===")
    subprocess.run([sys.executable, str(script_path)], cwd=REPO_ROOT, check=True)


def commit_and_push_db() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    db_rel_path = DB_PATH.relative_to(REPO_ROOT)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(db_rel_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        print("\n=== Database unchanged; skipping git commit/push ===")
        return

    print("\n=== Committing and pushing database ===")
    subprocess.run(["git", "add", "--", str(db_rel_path)], cwd=REPO_ROOT, check=True)
    subprocess.run(
        ["git", "commit", "--only", "-m", "new houses", "--", str(db_rel_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True)


def main() -> None:
    steps = [
        ("live scraper", REPO_ROOT / "scripts" / "live_scraper.py"),
        ("nearest grocery", REPO_ROOT / "nearest_grocery.py"),
        ("crime enrichment", REPO_ROOT / "scripts" / "crime.py"),
        ("AI ranker", REPO_ROOT / "ai_ranker.py"),
    ]

    for label, script_path in steps:
        run_step(label, script_path)

    commit_and_push_db()


if __name__ == "__main__":
    main()
