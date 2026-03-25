"""
ingest.py — Load raw CSV files into SQLite and perform basic cleaning.

Why SQLite?
- Lets every downstream step (transform, train, evaluate) query data with SQL
  instead of re-reading 600MB of CSVs each time.
- Portable: the entire dataset lives in a single .db file.
- DuckDB (used in later steps) can query SQLite files directly for fast analytics.
"""

import sqlite3
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/articles.db")
TABLE_NAME = "articles"

# The three CSV files from Kaggle
CSV_FILES = [
    RAW_DIR / "articles1.csv",
    RAW_DIR / "articles2.csv",
    RAW_DIR / "articles3.csv",
]

# Columns we actually need — drop url (not useful for NLP) and the unnamed index
COLUMNS_TO_KEEP = ["id", "title", "publication", "author", "date", "year", "month", "content"]


def load_csvs() -> pd.DataFrame:
    """Read all three CSVs and concatenate into one DataFrame."""
    frames = []
    for path in CSV_FILES:
        print(f"  Loading {path.name}...")
        df = pd.read_csv(path, usecols=lambda c: c in COLUMNS_TO_KEEP)
        frames.append(df)
        print(f"    → {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows after concat: {len(combined):,}")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning rules. Each step is logged so you can see the impact.

    Rules:
    1. Drop rows where 'content' is null — no text = nothing to model.
    2. Drop rows where 'title' is null — title is used as a secondary feature.
    3. Drop exact duplicate rows — same article appearing in multiple files.
    4. Reset the index so row numbers are clean and sequential.
    """
    start = len(df)

    # 1. Drop missing content
    df = df.dropna(subset=["content"])
    print(f"  After dropping null content:  {len(df):,} rows  (removed {start - len(df):,})")

    # 2. Drop missing title
    after_content = len(df)
    df = df.dropna(subset=["title"])
    print(f"  After dropping null title:    {len(df):,} rows  (removed {after_content - len(df):,})")

    # 3. Drop exact duplicates (same id or same content)
    after_title = len(df)
    df = df.drop_duplicates(subset=["content"])
    print(f"  After dropping duplicates:    {len(df):,} rows  (removed {after_title - len(df):,})")

    # 4. Clean up the index
    df = df.reset_index(drop=True)

    return df


def write_to_sqlite(df: pd.DataFrame) -> None:
    """
    Write the cleaned DataFrame to a SQLite database.

    'if_exists="replace"' means: if we run ingest.py again, it rebuilds the
    table from scratch rather than appending duplicates.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    # Create an index on 'id' for fast lookups in later steps
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_id ON {TABLE_NAME}(id)")
    conn.commit()
    conn.close()
    print(f"\n  Saved {len(df):,} rows to {DB_PATH}  (table: '{TABLE_NAME}')")


def verify(df: pd.DataFrame) -> None:
    """Print a quick summary so we can sanity-check the result."""
    print("\n── Data Summary ──────────────────────────────────────────────")
    print(f"  Total articles:      {len(df):,}")
    print(f"  Columns:             {df.columns.tolist()}")
    print(f"\n  Publications:")
    print(df["publication"].value_counts().to_string())
    print(f"\n  Null counts:")
    print(df.isnull().sum().to_string())
    print(f"\n  Content length (chars) — min/median/max:")
    lengths = df["content"].str.len()
    print(f"    min={lengths.min():,}  median={lengths.median():,.0f}  max={lengths.max():,}")
    print("──────────────────────────────────────────────────────────────")


def run() -> None:
    print("── Step 1: Load CSVs ─────────────────────────────────────────")
    df = load_csvs()

    print("\n── Step 2: Clean ─────────────────────────────────────────────")
    df = clean(df)

    print("\n── Step 3: Write to SQLite ───────────────────────────────────")
    write_to_sqlite(df)

    print("\n── Step 4: Verify ────────────────────────────────────────────")
    verify(df)

    print("\nIngest complete.")


if __name__ == "__main__":
    run()
