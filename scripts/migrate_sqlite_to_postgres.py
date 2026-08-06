from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def count_rows(sqlite_path: Path) -> dict[str, int]:
    tables = ["deer_info", "analysis_records", "daily_summary", "action_stream_log"]
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()
        counts: dict[str, int] = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(cursor.fetchone()[0])
            except sqlite3.Error:
                counts[table] = 0
        return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect legacy DeerUI SQLite data before migration")
    parser.add_argument("sqlite_path", type=Path)
    args = parser.parse_args()
    for table, count in count_rows(args.sqlite_path).items():
        print(f"{table}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
