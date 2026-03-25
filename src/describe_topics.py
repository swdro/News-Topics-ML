"""
describe_topics.py — Print all discovered topic labels from the current model run.

Run this any time to see the current topic list:
  python -m src.describe_topics

Output reflects whatever BERTopic discovered on the most recent run of
topic_model.py — always up to date with the actual model artifacts.
"""

import pandas as pd
from pathlib import Path

TOPIC_INFO_PATH = Path("data/processed/topic_info.csv")
ARTICLES_PATH   = Path("data/processed/articles_with_topics.parquet")


def load_topic_summary() -> pd.DataFrame:
    if not TOPIC_INFO_PATH.exists():
        raise FileNotFoundError(
            "topic_info.csv not found. Run topic_model.py first."
        )

    ti = pd.read_csv(TOPIC_INFO_PATH)
    ti = ti[ti["Topic"] != -1].copy()

    # Parse keywords out of the BERTopic name string (format: "ID_word1_word2_...")
    ti["keywords"] = ti["Name"].str.split("_").str[1:].str.join(", ")

    return ti[["Topic", "Count", "keywords"]].sort_values("Topic").reset_index(drop=True)


def print_topic_table(df: pd.DataFrame) -> None:
    total_articles = df["Count"].sum()

    print(f"\n{'─'*65}")
    print(f"  Discovered Topics  ({len(df)} topics, {total_articles:,} labeled articles)")
    print(f"{'─'*65}")
    print(f"  {'ID':>4}  {'Articles':>9}  {'  %':>5}  Keywords")
    print(f"  {'─'*4}  {'─'*9}  {'─'*5}  {'─'*40}")

    for _, row in df.iterrows():
        pct = 100 * row["Count"] / total_articles
        print(f"  {int(row['Topic']):>4}  {int(row['Count']):>9,}  {pct:>4.1f}%  {row['keywords']}")

    print(f"{'─'*65}")
    print(f"  Total labeled articles: {total_articles:,}")

    if ARTICLES_PATH.exists():
        outlier_info = pd.read_parquet(ARTICLES_PATH, columns=["topic_id"])
        # We can infer outlier count from the difference between full dataset and labeled
        print(f"  Source: {TOPIC_INFO_PATH}")
    print()


def run():
    df = load_topic_summary()
    print_topic_table(df)


if __name__ == "__main__":
    run()
