"""
topic_model.py — Unsupervised topic discovery using BERTopic.

How BERTopic works:
1. EMBED:   Each article is converted to a dense semantic vector using a
            sentence-transformer model. Unlike TF-IDF, these vectors capture
            meaning — "car" and "automobile" land near each other.
2. REDUCE:  UMAP compresses the high-dimensional embedding space down to
            5 dimensions, preserving local structure (similar articles stay close).
3. CLUSTER: HDBSCAN finds dense regions in the reduced space — these are topics.
4. LABEL:   c-TF-IDF extracts the most representative keywords per cluster.

Why fit on a sample?
  Embedding 141k articles end-to-end takes 30-60+ minutes even with GPU.
  The standard production pattern is: fit BERTopic on a representative sample
  (~20k articles), then use .transform() to assign topics to the full dataset.
  This is fast because .transform() skips UMAP/HDBSCAN re-fitting.

Apple Silicon note:
  sentence-transformers will automatically use MPS (Metal Performance Shaders)
  on Apple Silicon Macs when torch detects it — no extra config needed.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
ARTICLES_CLEAN_PATH = PROCESSED_DIR / "articles_clean.parquet"
TOPIC_MODEL_PATH = PROCESSED_DIR / "bertopic_model"
ARTICLES_WITH_TOPICS_PATH = PROCESSED_DIR / "articles_with_topics.parquet"
TOPIC_INFO_PATH = PROCESSED_DIR / "topic_info.csv"

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 20_000     # number of articles used to FIT BERTopic
MIN_TOPIC_SIZE = 50      # minimum articles to form a topic (smaller = more topics)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, high quality; 384-dim vectors
RANDOM_STATE = 42        # for reproducibility in UMAP


def load_articles() -> pd.DataFrame:
    df = pd.read_parquet(ARTICLES_CLEAN_PATH)
    print(f"  Loaded {len(df):,} cleaned articles")
    return df


def sample_for_fitting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Draw a stratified sample for fitting BERTopic.

    Stratified by publication ensures all 15 outlets are represented
    proportionally — we don't want the sample dominated by Breitbart just
    because it has the most articles.
    """
    sample = (
        df.groupby("publication", group_keys=False)
        .apply(lambda g: g.sample(
            min(len(g), int(SAMPLE_SIZE * len(g) / len(df))),
            random_state=RANDOM_STATE
        ))
    )
    # Top up to exactly SAMPLE_SIZE if rounding left us short
    if len(sample) < SAMPLE_SIZE:
        remaining = df[~df.index.isin(sample.index)]
        top_up = remaining.sample(SAMPLE_SIZE - len(sample), random_state=RANDOM_STATE)
        sample = pd.concat([sample, top_up])

    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Stratified sample: {len(sample):,} articles across {sample['publication'].nunique()} publications")
    return sample


def build_topic_model() -> BERTopic:
    """
    Instantiate BERTopic with explicit sub-models so each choice is visible
    and tunable.

    Why these UMAP settings?
    - n_components=5: reduce to 5 dims before clustering (BERTopic default;
      lower than 2D viz but better for clustering quality)
    - n_neighbors=15: controls local vs global structure balance
    - metric='cosine': appropriate for sentence embeddings (unit vectors)

    Why these HDBSCAN settings?
    - min_cluster_size=MIN_TOPIC_SIZE: topics must have at least this many
      articles — prevents noise clusters from being labeled as topics
    - metric='euclidean': works well in the UMAP-reduced space
    - prediction_data=True: required to call .transform() on new documents
    """
    from umap import UMAP
    from hdbscan import HDBSCAN

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,  # needed for .transform()
    )

    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,  # faster; we only need the top topic per article
        verbose=True,
    )

    return topic_model


def fit_model(topic_model: BERTopic, sample: pd.DataFrame):
    """
    Fit BERTopic on the sample. Returns topics and the fitted model.

    This is the slow step — it embeds SAMPLE_SIZE articles using
    sentence-transformers (MPS-accelerated on Apple Silicon).
    """
    print(f"\n  Fitting BERTopic on {len(sample):,} articles...")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print("  (This uses MPS on Apple Silicon — expect 3-8 minutes)\n")

    docs = sample["text_clean"].tolist()
    topics, _ = topic_model.fit_transform(docs)
    return topics


def transform_full_dataset(topic_model: BERTopic, df: pd.DataFrame) -> np.ndarray:
    """
    Assign topics to ALL 141k articles using the fitted model.

    .transform() only runs the embedding + nearest-centroid assignment —
    it skips re-fitting UMAP/HDBSCAN, so it's much faster than fit_transform.
    """
    print(f"\n  Transforming full dataset ({len(df):,} articles)...")
    docs = df["text_clean"].tolist()
    topics, _ = topic_model.transform(docs)
    return np.array(topics)


def summarize_topics(topic_model: BERTopic) -> pd.DataFrame:
    """Print and return a summary of discovered topics."""
    topic_info = topic_model.get_topic_info()

    # Topic -1 is BERTopic's "outlier" bucket — articles that didn't fit any cluster
    outliers = topic_info[topic_info["Topic"] == -1]["Count"].values[0]
    real_topics = topic_info[topic_info["Topic"] != -1]

    print(f"\n── Topic Discovery Results ───────────────────────────────────")
    print(f"  Topics discovered:  {len(real_topics)}")
    print(f"  Outlier articles:   {outliers:,}  (topic = -1, won't be used as labels)")
    print(f"\n  Top 20 topics by size:")
    print(f"  {'ID':>4}  {'Count':>7}  Keywords")
    print(f"  {'─'*4}  {'─'*7}  {'─'*50}")

    for _, row in real_topics.head(20).iterrows():
        keywords = ", ".join(row["Representation"][:6]) if "Representation" in row else row["Name"]
        print(f"  {row['Topic']:>4}  {row['Count']:>7,}  {keywords}")

    return topic_info


def save_artifacts(topic_model: BERTopic, df: pd.DataFrame,
                   all_topics: np.ndarray, topic_info: pd.DataFrame) -> None:
    """Save model, labeled articles, and topic metadata."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save the fitted BERTopic model
    topic_model.save(str(TOPIC_MODEL_PATH))
    print(f"\n  Saved BERTopic model     → {TOPIC_MODEL_PATH}")

    # 2. Add topic assignments to the article DataFrame
    df = df.copy()
    df["topic_id"] = all_topics

    # 3. Join topic keywords as a human-readable label column
    topic_labels = topic_model.get_topic_info().set_index("Topic")["Name"].to_dict()
    df["topic_label"] = df["topic_id"].map(topic_labels)

    # 4. Drop outlier articles (topic_id == -1) — they have no valid label
    before = len(df)
    df = df[df["topic_id"] != -1].reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} outlier articles (topic -1)")
    print(f"  Articles with valid topic labels: {len(df):,}")

    df.to_parquet(ARTICLES_WITH_TOPICS_PATH, index=False)
    print(f"  Saved labeled articles   → {ARTICLES_WITH_TOPICS_PATH}")

    # 5. Save topic info CSV (useful for EDA and model card)
    topic_info.to_csv(TOPIC_INFO_PATH, index=False)
    print(f"  Saved topic info         → {TOPIC_INFO_PATH}")


def run() -> None:
    print("── Step 1: Load articles ─────────────────────────────────────")
    df = load_articles()

    print("\n── Step 2: Sample for fitting ────────────────────────────────")
    sample = sample_for_fitting(df)

    print("\n── Step 3: Build topic model ─────────────────────────────────")
    topic_model = build_topic_model()

    print("\n── Step 4: Fit BERTopic on sample ────────────────────────────")
    fit_model(topic_model, sample)

    print("\n── Step 5: Transform full dataset ────────────────────────────")
    all_topics = transform_full_dataset(topic_model, df)

    print("\n── Step 6: Summarize topics ──────────────────────────────────")
    topic_info = summarize_topics(topic_model)

    print("\n── Step 7: Save artifacts ────────────────────────────────────")
    save_artifacts(topic_model, df, all_topics, topic_info)

    print("\nTopic modeling complete.")


if __name__ == "__main__":
    run()
