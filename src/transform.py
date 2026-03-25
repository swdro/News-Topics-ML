"""
transform.py — Text preprocessing and TF-IDF feature engineering.

Why preprocess before TF-IDF?
- Raw text contains noise: punctuation, numbers, stopwords ("the", "and", "is").
  These inflate the vocabulary and dilute the signal words that actually
  distinguish topics from each other.
- Lowercasing ensures "Policy" and "policy" are treated as the same word.
- Removing stopwords (via NLTK's 179-word list) cuts vocabulary size with no
  loss of topic signal.

Why TF-IDF?
- ML models need numbers, not strings. TF-IDF converts each article into a
  vector of word importance scores.
- TF  (Term Frequency):     how often a word appears in THIS article.
- IDF (Inverse Doc Freq):   penalizes words common across ALL articles.
- Result: words like "federal" score high in political articles; "the" scores
  near zero everywhere. This is exactly what a topic classifier needs.
"""

import re
import sqlite3
import pickle
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH = Path("data/processed/articles.db")
PROCESSED_DIR = Path("data/processed")

# Output files — saved so train.py doesn't need to re-run vectorization
TFIDF_MATRIX_PATH = PROCESSED_DIR / "tfidf_matrix.npz"
TFIDF_VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
ARTICLES_CLEAN_PATH = PROCESSED_DIR / "articles_clean.parquet"

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_CONTENT_LENGTH = 200   # drop articles shorter than this — likely stubs
MAX_FEATURES = 50_000      # keep only the top 50k most informative terms
MIN_DF = 5                 # ignore words appearing in fewer than 5 articles
MAX_DF = 0.95              # ignore words appearing in >95% of articles (too common)

# NLTK's English stopword list has 179 words — far more complete than a hand-rolled list
STOPWORDS = set(stopwords.words("english"))


def load_articles() -> pd.DataFrame:
    """Load cleaned articles from SQLite, filter out stubs."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT id, title, publication, content FROM articles", conn)
    conn.close()
    print(f"  Loaded {len(df):,} articles from SQLite")

    # Drop articles that are too short to be meaningful
    before = len(df)
    df = df[df["content"].str.len() >= MIN_CONTENT_LENGTH].reset_index(drop=True)
    print(f"  After filtering short articles (<{MIN_CONTENT_LENGTH} chars): {len(df):,}  (removed {before - len(df):,})")

    return df


def preprocess_text(text: str) -> str:
    """
    Clean a single article's text.

    Steps:
    1. Lowercase          — "Policy" and "policy" become the same token.
    2. Remove URLs        — they contribute noise, not topic signal.
    3. Remove non-alpha   — strip punctuation, numbers, special chars.
    4. Collapse whitespace — multiple spaces become one.
    5. Remove stopwords   — NLTK's 179 common English function words.
    6. Drop short tokens  — single characters add noise.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocess_text to every article."""
    print(f"  Preprocessing {len(df):,} articles...")

    # Combine title + content: titles are dense with topic signal and short,
    # so prepending them gives those keywords a slight frequency boost in TF-IDF.
    df["text_clean"] = (df["title"] + " " + df["content"]).apply(preprocess_text)

    # Sanity check — show a before/after example
    print("\n  Sample — original (first 200 chars):")
    print("  ", df["content"].iloc[0][:200])
    print("\n  Sample — cleaned (first 200 chars):")
    print("  ", df["text_clean"].iloc[0][:200])

    return df


def build_tfidf(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the cleaned corpus and transform it.

    Returns:
        tfidf_matrix : sparse matrix of shape (n_articles, MAX_FEATURES)
        vectorizer   : fitted TfidfVectorizer (saved to disk for inference later)
    """
    print(f"\n  Fitting TF-IDF vectorizer (max_features={MAX_FEATURES:,})...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,       # ignore rare words (appear in < 5 docs)
        max_df=MAX_DF,       # ignore near-universal words (in > 95% of docs)
        sublinear_tf=True,   # apply log(1 + tf) — dampens effect of very frequent words
        ngram_range=(1, 2),  # include unigrams AND bigrams ("white house", "interest rate")
    )

    tfidf_matrix = vectorizer.fit_transform(df["text_clean"])

    vocab_size = len(vectorizer.vocabulary_)
    sparsity = 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    print(f"  Vocabulary size:    {vocab_size:,} terms")
    print(f"  TF-IDF matrix:      {tfidf_matrix.shape[0]:,} articles × {tfidf_matrix.shape[1]:,} features")
    print(f"  Matrix sparsity:    {sparsity:.1f}% zeros  (expected — most words don't appear in most articles)")

    return tfidf_matrix, vectorizer


def save_artifacts(df: pd.DataFrame, tfidf_matrix, vectorizer) -> None:
    """Persist all outputs so downstream steps don't need to re-run this."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Cleaned article DataFrame — used by topic_model.py and train.py
    df.to_parquet(ARTICLES_CLEAN_PATH, index=False)
    print(f"\n  Saved cleaned articles  → {ARTICLES_CLEAN_PATH}")

    # Sparse TF-IDF matrix
    sp.save_npz(str(TFIDF_MATRIX_PATH), tfidf_matrix)
    print(f"  Saved TF-IDF matrix     → {TFIDF_MATRIX_PATH}")

    # Fitted vectorizer — needed at inference time to transform new articles
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved TF-IDF vectorizer → {TFIDF_VECTORIZER_PATH}")


def run() -> None:
    print("── Step 1: Load articles ─────────────────────────────────────")
    df = load_articles()

    print("\n── Step 2: Preprocess text ───────────────────────────────────")
    df = preprocess_corpus(df)

    print("\n── Step 3: Build TF-IDF features ─────────────────────────────")
    tfidf_matrix, vectorizer = build_tfidf(df)

    print("\n── Step 4: Save artifacts ────────────────────────────────────")
    save_artifacts(df, tfidf_matrix, vectorizer)

    print("\nTransform complete.")


if __name__ == "__main__":
    run()
