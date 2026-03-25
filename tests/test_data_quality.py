"""
test_data_quality.py — Validate that ingested and processed data meets
expected quality contracts.

Two kinds of tests here:
  1. Unit tests (no data files needed) — validate Pydantic schemas with
     synthetic records. These run in CI.
  2. Integration tests (require real data files) — check the actual SQLite
     database and parquet files produced by ingest.py and transform.py.
     These are skipped in CI if data files aren't present.

Why data quality tests?
  Data bugs are silent — a bad CSV row or a schema mismatch won't throw
  a Python exception; it'll just produce wrong features and quietly degrade
  model accuracy. These tests act as a contract: if the data doesn't look
  right, fail loudly before training.
"""

import sqlite3
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, field_validator, ValidationError
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH            = Path("data/processed/articles.db")
ARTICLES_CLEAN_PATH= Path("data/processed/articles_clean.parquet")
TFIDF_MATRIX_PATH  = Path("data/processed/tfidf_matrix.npz")

# ── Pydantic schema ────────────────────────────────────────────────────────────

class ArticleRecord(BaseModel):
    """
    Schema contract for a single article row.

    Pydantic validates types and constraints at instantiation time.
    If a field fails validation, a ValidationError is raised immediately
    rather than silently propagating bad data downstream.
    """
    id:          int
    title:       str
    publication: str
    content:     str
    author:      Optional[str] = None   # nullable — many wire stories lack bylines
    date:        Optional[str] = None   # nullable — some articles missing date

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v

    @field_validator("content")
    @classmethod
    def content_min_length(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError(f"content too short ({len(v)} chars) — likely stub")
        return v

    @field_validator("publication")
    @classmethod
    def publication_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("publication must not be empty")
        return v


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — run in CI, no data files needed
# ════════════════════════════════════════════════════════════════

class TestArticleSchema:
    """Test that the Pydantic schema correctly validates article records."""

    def test_valid_record_passes(self):
        """A well-formed record should pass schema validation."""
        record = ArticleRecord(
            id=1,
            title="Fed raises interest rates",
            publication="Reuters",
            content="The Federal Reserve raised interest rates by 25 basis points on Wednesday.",
            author="Jane Smith",
            date="2017-03-15",
        )
        assert record.id == 1
        assert record.publication == "Reuters"

    def test_null_author_allowed(self):
        """Author is optional — wire stories often lack bylines."""
        record = ArticleRecord(
            id=2,
            title="Market update",
            publication="Reuters",
            content="Stocks fell sharply on Thursday amid rising inflation concerns.",
            author=None,
        )
        assert record.author is None

    def test_null_date_allowed(self):
        """Date is optional — some articles are missing date metadata."""
        record = ArticleRecord(
            id=3,
            title="Tech layoffs continue",
            publication="CNN",
            content="Several major tech companies announced layoffs this quarter.",
            date=None,
        )
        assert record.date is None

    def test_empty_title_rejected(self):
        """Empty title should raise a ValidationError."""
        with pytest.raises(ValidationError):
            ArticleRecord(
                id=4,
                title="   ",
                publication="CNN",
                content="Some content here that is long enough to pass.",
            )

    def test_short_content_rejected(self):
        """Content under 10 chars is a stub and should be rejected."""
        with pytest.raises(ValidationError):
            ArticleRecord(
                id=5,
                title="Valid title",
                publication="CNN",
                content="Too short",
            )

    def test_empty_publication_rejected(self):
        """Empty publication name should raise a ValidationError."""
        with pytest.raises(ValidationError):
            ArticleRecord(
                id=6,
                title="Valid title",
                publication="",
                content="This is a valid content string that is long enough.",
            )

    def test_batch_validation(self):
        """All records in a synthetic batch should pass schema validation."""
        batch = [
            {"id": i, "title": f"Article {i}", "publication": "NYT",
             "content": "x" * 50, "author": None, "date": None}
            for i in range(100)
        ]
        records = [ArticleRecord(**row) for row in batch]
        assert len(records) == 100

    def test_duplicate_ids_detectable(self):
        """Duplicate IDs should be detectable (schema doesn't enforce uniqueness,
        but we can check programmatically)."""
        batch = [
            {"id": 1, "title": "Article A", "publication": "CNN",
             "content": "Content A is long enough for validation."},
            {"id": 1, "title": "Article B", "publication": "CNN",
             "content": "Content B is long enough for validation."},  # duplicate ID
        ]
        ids = [row["id"] for row in batch]
        assert len(ids) != len(set(ids)), "Expected duplicate IDs in this test batch"


# ════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — require real data files, skipped in CI
# ════════════════════════════════════════════════════════════════

requires_db = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="Requires ingested data (data/processed/articles.db). Run ingest.py first."
)
requires_processed = pytest.mark.skipif(
    not ARTICLES_CLEAN_PATH.exists(),
    reason="Requires processed data. Run transform.py first."
)


@requires_db
class TestDatabaseQuality:
    """Integration tests against the real SQLite database."""

    def setup_method(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.df   = pd.read_sql("SELECT * FROM articles", self.conn)

    def teardown_method(self):
        self.conn.close()

    def test_article_count_in_range(self):
        """Dataset should have between 100k and 200k articles after cleaning."""
        count = len(self.df)
        assert 100_000 <= count <= 200_000, \
            f"Unexpected article count: {count}. Expected 100k–200k."

    def test_no_null_content(self):
        """No article should have null content after ingest cleaning."""
        null_count = self.df["content"].isnull().sum()
        assert null_count == 0, f"Found {null_count} articles with null content"

    def test_no_null_title(self):
        """No article should have null title after ingest cleaning."""
        null_count = self.df["title"].isnull().sum()
        assert null_count == 0, f"Found {null_count} articles with null title"

    def test_no_duplicate_content(self):
        """No two articles should have identical content."""
        dup_count = self.df["content"].duplicated().sum()
        assert dup_count == 0, f"Found {dup_count} duplicate articles"

    def test_expected_publications_present(self):
        """All 15 expected publications should be present."""
        expected = {
            "Breitbart", "New York Post", "NPR", "CNN", "Washington Post",
            "Reuters", "Guardian", "New York Times", "Atlantic",
            "Business Insider", "National Review", "Talking Points Memo",
            "Vox", "Buzzfeed News", "Fox News",
        }
        actual = set(self.df["publication"].unique())
        missing = expected - actual
        assert not missing, f"Missing publications: {missing}"

    def test_content_length_distribution(self):
        """Median article length should be between 1,000 and 10,000 chars."""
        median_len = self.df["content"].str.len().median()
        assert 1_000 <= median_len <= 10_000, \
            f"Unexpected median content length: {median_len:.0f}"

    def test_schema_sample_validates(self):
        """A random sample of 100 rows should all pass the Pydantic schema."""
        sample = self.df.sample(100, random_state=42)
        errors = []
        for _, row in sample.iterrows():
            try:
                ArticleRecord(**row.to_dict())
            except ValidationError as e:
                errors.append(str(e))
        assert not errors, f"Schema violations found:\n" + "\n".join(errors[:3])


@requires_processed
class TestProcessedData:
    """Integration tests against processed parquet and TF-IDF artifacts."""

    def test_cleaned_parquet_has_text_clean(self):
        """articles_clean.parquet must have a text_clean column."""
        df = pd.read_parquet(ARTICLES_CLEAN_PATH)
        assert "text_clean" in df.columns, "Missing text_clean column"

    def test_cleaned_parquet_no_null_text(self):
        """text_clean column should have no null values."""
        df = pd.read_parquet(ARTICLES_CLEAN_PATH)
        null_count = df["text_clean"].isnull().sum()
        assert null_count == 0, f"Found {null_count} null text_clean values"

    def test_tfidf_matrix_shape(self):
        """TF-IDF matrix should have expected dimensions."""
        import scipy.sparse as sp
        matrix = sp.load_npz(str(TFIDF_MATRIX_PATH))
        n_articles, n_features = matrix.shape
        assert n_articles > 100_000, f"Too few rows in TF-IDF matrix: {n_articles}"
        assert n_features == 50_000,  f"Expected 50,000 features, got {n_features}"

    def test_tfidf_matrix_is_sparse(self):
        """TF-IDF matrix should be highly sparse (>95% zeros)."""
        import scipy.sparse as sp
        matrix = sp.load_npz(str(TFIDF_MATRIX_PATH))
        sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        assert sparsity > 0.95, f"Matrix not sparse enough: {sparsity:.2%} zeros"

    def test_tfidf_values_in_range(self):
        """All TF-IDF values should be between 0 and 1."""
        import scipy.sparse as sp
        matrix = sp.load_npz(str(TFIDF_MATRIX_PATH))
        # Sample a small dense slice for speed
        sample = matrix[:100].toarray()
        assert sample.min() >= 0.0, "TF-IDF values should be non-negative"
        assert sample.max() <= 1.0, "TF-IDF values should not exceed 1.0"
